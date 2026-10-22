"""English / Hindi / Hinglish, end to end: speak -> recognise -> wake phrase -> understood (right tool).

    .venv\\Scripts\\python.exe scripts\\language_eval.py [--models small,medium,large-v3-turbo] [--work DIR]

1. Each request below is spoken by the cloned-voice service (needs scripts/start_tts.ps1) in several voices: the active
   profile plus two synthetic speakers enrolled from Windows voices for the run and deleted afterwards.
2. Each Whisper model transcribes every clip, choosing between English and Hindi itself (as the assistant does).
3. The transcript must contain the wake phrase, and the request after it, sent to the real agent (actions declined),
   must lead to the right tool. Character error rate is also reported where it is meaningful.
Clips and transcripts are cached in --work, so a second run only redoes what changed.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from vox_assistant.agent import Agent, Say, ToolFinished
from vox_assistant.asr_local import LocalWhisperASR
from vox_assistant.audit import AuditLog
from vox_assistant.config import Settings
from vox_assistant.enroll import SCRIPT
from vox_assistant.llm_client import ToolChoiceLLM
from vox_assistant.permissions import PermissionGate
from vox_assistant.tools import build_registry
from vox_assistant.tts_neutral import NeutralVoiceTTS
from vox_assistant.voice import strip_wake_phrase
from vox_assistant.voices import REF_RATE, VoiceProfiles

# (language, what is said, acceptable first tools; empty = answered without a tool)
CASES = [
    ("en", "Hey Vox, what's the temperature of my graphics cards?", {"gpu_status"}),
    ("en", "Hey Vox, open Notepad.", {"open_app"}),
    ("en", "Hey Vox, remind me in ten minutes to stretch.", {"set_reminder"}),
    ("en", "Hey Vox, what is the capital of France?", set()),
    ("en", "Hey Vox, which models are loaded right now?", {"list_models", "gpu_status"}),
    ("en", "Hey Vox, how much memory is my computer using?", {"system_info"}),
    ("hi", "हे वॉक्स, मेरे ग्राफ़िक्स कार्ड का तापमान क्या है?", {"gpu_status"}),
    ("hi", "हे वॉक्स, नोटपैड खोलो।", {"open_app"}),
    ("hi", "हे वॉक्स, दस मिनट बाद मुझे पानी पीने की याद दिलाना।", {"set_reminder"}),
    ("hi", "हे वॉक्स, भारत की राजधानी क्या है?", set()),
    ("hi", "हे वॉक्स, अभी कौन से मॉडल चल रहे हैं?", {"list_models", "gpu_status"}),
    ("hi", "हे वॉक्स, कंप्यूटर की मेमोरी कितनी इस्तेमाल हो रही है?", {"system_info"}),
    ("hinglish", "Hey Vox, मेरे GPU का temperature check करो।", {"gpu_status"}),
    ("hinglish", "Hey Vox, Notepad open कर दो।", {"open_app"}),
    ("hinglish", "Hey Vox, ten minutes में मुझे stretch करने का reminder set करो।", {"set_reminder"}),
    ("hinglish", "Hey Vox, Cutline project को VS Code में खोलो।", {"open_project"}),
    ("hinglish", "Hey Vox, RAM कितनी use हो रही है?", {"system_info"}),
    ("hinglish", "Hey Vox, web पर ROCm release notes search करो।", {"web_search"}),
]
SPEAKERS = [("leval_zira", "Microsoft Zira Desktop"), ("leval_david", "Microsoft David Desktop")]
PROMPT = "Hey vox."  # what the voice loop primes Whisper with (spelling of the wake phrase)


def strip_wake(text: str, wakes: list[str]) -> str | None:
    """VoiceLoop._strip_wake without the loop: the wake phrase may be in either reading."""
    from vox_assistant.asr_gpu import join_readings, split_readings

    readings = split_readings(text)
    rests = [strip_wake_phrase(r, wakes) for r in readings]
    found = [r for r in rests if r is not None]
    if not found:
        return None
    if not any(found):
        return ""
    return join_readings([rest if rest is not None else reading for reading, rest in zip(readings, rests)])


def cer(ref: str, hyp: str) -> float:
    import re

    norm = lambda s: re.sub(r"[\s\W_]+", "", s.lower(), flags=re.UNICODE)  # noqa: E731
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / len(r)


async def make_speakers(profiles: VoiceProfiles) -> list[str]:
    made = []
    for name, sapi in SPEAKERS:
        tts = NeutralVoiceTTS(voice=sapi)
        raw = b"".join([c async for c in tts.synth(" ".join(SCRIPT[:5]))])
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
        made.append(profiles.create(name, audio, tts.sample_rate, " ".join(SCRIPT[:5]), permission_confirmed=True,
                                    activate=False, source="language_eval (synthetic)")["id"])
    return made


async def synthesize(work: Path, settings: Settings) -> list[dict]:
    """Speak every case in every voice; cached."""
    index = work / "clips.json"
    if index.exists():
        return json.loads(index.read_text(encoding="utf-8"))
    profiles = VoiceProfiles(settings.voices_dir)
    voices = [profiles.active_id()]
    made = await make_speakers(profiles)
    clips = []
    try:
        async with httpx.AsyncClient(base_url=settings.tts_url, timeout=300) as c:
            for vi, voice in enumerate(voices + made):
                for ci, (lang, text, want) in enumerate(CASES):
                    r = await c.post("/audio/speech", json={"input": text, "voice": voice, "response_format": "wav"})
                    r.raise_for_status()
                    audio, rate = sf.read(io.BytesIO(r.content), dtype="float32")
                    path = work / f"v{vi}_c{ci}.wav"
                    sf.write(path, resample_poly(audio, 16000, rate).astype(np.float32), 16000)
                    clips.append({"voice": voice, "case": ci, "lang": lang, "path": str(path)})
                    print(f"  spoke v{vi} c{ci} ({lang})", flush=True)
    finally:
        async with httpx.AsyncClient(base_url=settings.tts_url, timeout=30) as c:
            for pid in made:
                await c.delete(f"/voices/{pid}")
    index.write_text(json.dumps(clips, ensure_ascii=False), encoding="utf-8")
    return clips


async def transcribe_all(work: Path, clips: list[dict], model: str, settings: Settings) -> dict:
    out_path = work / f"asr_{model}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    if model == "service":  # the GPU recogniser in the voice service, exactly as the assistant uses it
        from vox_assistant.asr_remote import ServiceASR

        asr = ServiceASR(settings.tts_url)
    else:
        asr = LocalWhisperASR(model, "cpu", "int8", settings.whisper_threads)
    await asr.warm()
    out = {}
    for clip in clips:
        audio, _ = sf.read(clip["path"], dtype="float32")
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
        t0 = time.perf_counter()
        text = await asr.transcribe(pcm, language="auto", prompt=PROMPT)
        out[clip["path"]] = {"text": text, "seconds": time.perf_counter() - t0, "lang": asr.last_language,
                             "readings": getattr(asr, "last_readings", {})}
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="service", help="'service' (GPU, as used) and/or CPU faster-whisper model names")
    ap.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "voxa_language_eval"))
    args = ap.parse_args()
    settings = Settings.load()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    print("1. Speaking every request in every voice")
    clips = await synthesize(work, settings)

    tmp = Path(tempfile.mkdtemp(prefix="voxa_leval_"))
    s2 = Settings.load(tmp)
    audit = AuditLog(tmp / "a.sqlite3")
    registry, store = build_registry(s2)

    async def decline(intent, tool, a):
        return False

    llm = ToolChoiceLLM(settings.llm_base_url, settings.llm_model, temperature=0.3, timeout=120.0, first_token_timeout=90.0)
    agent = Agent(llm, registry, PermissionGate(decline, audit), audit)
    wakes = [settings.wake_phrase, *settings.wake_aliases]
    understood_cache: dict[str, str | None] = {}

    async def first_tool(request: str) -> str | None:
        if request not in understood_cache:
            agent.reset()
            tools = [ev.name async for ev in agent.run(request) if isinstance(ev, ToolFinished)]
            understood_cache[request] = tools[0] if tools else None
        return understood_cache[request]

    for model in args.models.split(","):
        print(f"\n2. {model}")
        res = await transcribe_all(work, clips, model, settings)
        by_lang: dict[str, dict] = {}
        for clip in clips:
            lang, text, want = CASES[clip["case"]]
            r = res[clip["path"]]
            rest = strip_wake(r["text"], wakes)  # per reading, as the voice loop does
            tool = await first_tool(rest if rest else r["text"])
            ok_intent = (tool in want) if want else tool is None
            s = by_lang.setdefault(lang, {"n": 0, "wake": 0, "intent": 0, "cer": [], "sec": [], "lang_ok": 0, "fails": []})
            s["n"] += 1
            s["wake"] += rest is not None
            s["intent"] += ok_intent
            s["sec"].append(r["seconds"])
            s["lang_ok"] += (r["lang"] in (("en",) if lang == "en" else ("hi", "both")))
            if lang in ("en", "hi"):  # each language's own reading, not the combined text the assistant sees
                s["cer"].append(cer(text, (r.get("readings") or {}).get(lang, r["text"])))
            if not ok_intent or rest is None:
                s["fails"].append(f"{r['text']!r} -> {tool}")
        for lang, s in by_lang.items():
            c = f"CER {np.mean(s['cer']) * 100:4.1f}%  " if s["cer"] else "              "
            print(f"  {lang:8s} wake {s['wake']}/{s['n']}  understood {s['intent']}/{s['n']}  {c}"
                  f"language {s['lang_ok']}/{s['n']}  {np.mean(s['sec']):.2f}s/clip")
            for f in s["fails"][:4]:
                print(f"      miss: {f}")
    await llm.aclose()
    audit.close()
    store.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
