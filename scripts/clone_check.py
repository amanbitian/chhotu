"""Enroll -> clone -> speak -> score -> revoke, against the running cloned-voice service (scripts/start_tts.ps1).

    .venv\\Scripts\\python.exe scripts\\clone_check.py

No human is needed: two Windows voices stand in for people. "Zira" reads the enrollment script into the real
recording path (record_take with Silero VAD, then the take quality checks and assembly); "David" is enrolled as a
second, different speaker. The service then clones Zira, and a speaker-verification model scores the clone against
Zira's reference and against David's. Both test profiles are deleted at the end through the service's revoke
endpoint, and your active voice is left as it was.
"""

from __future__ import annotations

import asyncio
import io
import sys
import time

import httpx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from voxvortex.audio import FRAME_BYTES, SAMPLE_RATE, SileroVAD

from vox_assistant.asr_local import LocalWhisperASR
from vox_assistant.config import Settings
from vox_assistant.disclosure import read_label
from vox_assistant.enroll import SCRIPT, TARGET_SECONDS, assemble, assess_take, record_take
from vox_assistant.tts_neutral import NeutralVoiceTTS
from vox_assistant.voices import REF_RATE, VoiceProfiles

RESULTS: list[tuple[str, str]] = []


def record(ok: bool, name: str, detail: str = "") -> None:
    RESULTS.append(("PASS" if ok else "FAIL", name))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}", flush=True)


async def speak16k(voice: str, text: str) -> bytes:
    tts = NeutralVoiceTTS(voice=voice)
    raw = b"".join([c async for c in tts.synth(text)])
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return resample_poly(x, SAMPLE_RATE, tts.sample_rate).astype(np.int16).tobytes()


def mic_frames(pcm: bytes):
    """What a microphone delivers: a little room silence, the speech, then silence until the take ends."""
    silence = b"\x00" * FRAME_BYTES
    for _ in range(10):
        yield silence
    pcm += b"\x00" * (-len(pcm) % FRAME_BYTES)
    for i in range(0, len(pcm), FRAME_BYTES):
        yield pcm[i:i + FRAME_BYTES]
    while True:
        yield silence


async def enroll(profiles: VoiceProfiles, vad: SileroVAD, name: str, sapi_voice: str) -> dict:
    takes = []
    for line in SCRIPT:
        if sum(len(t.audio) for t in takes) / REF_RATE >= TARGET_SECONDS[0]:
            break
        raw = record_take(mic_frames(await speak16k(sapi_voice, line)), vad)
        take = assess_take(line, raw, SAMPLE_RATE)
        if take.usable:
            takes.append(take)
    audio, transcript = assemble(takes)
    return profiles.create(name, audio, REF_RATE, transcript, permission_confirmed=True, activate=False,
                           source="clone_check (synthetic Windows voice)")


async def main() -> int:
    settings = Settings.load()
    profiles = VoiceProfiles(settings.voices_dir)
    active_before = profiles.active_id()
    vad = SileroVAD(settings.silero_model)
    asr = LocalWhisperASR(settings.whisper_model, settings.whisper_device, settings.whisper_compute, settings.whisper_threads)
    await asr.warm()
    made: list[str] = []
    async with httpx.AsyncClient(base_url=settings.tts_url, timeout=600.0) as c:
        try:
            h = (await c.get("/health")).json()
        except httpx.HTTPError:
            print("The cloned-voice service is not running: start scripts\\start_tts.ps1 first.")
            return 1
        print(f"Service: model_loaded={h['model_loaded']} on {(h.get('runtime') or {}).get('gpu')}")
        try:
            print("1. Enrollment through the recording path (Silero VAD end-of-take, quality checks, assembly)")
            zira = await enroll(profiles, vad, "vcheck_zira", "Microsoft Zira Desktop")
            made.append(zira["id"])
            q = zira["quality"]
            record(20 <= q["seconds"] <= 36.5 and q["snr_db"] > 12, "reference assembled",
                   f"{zira['id']}: {q['seconds']:.1f} s, SNR {q['snr_db']:.0f} dB, {len(zira['transcript'].split())} words")
            david = await enroll(profiles, vad, "vcheck_david", "Microsoft David Desktop")
            made.append(david["id"])
            record(profiles.active_id() == active_before, "enrolling did not change the active voice", str(active_before))

            print("2. Cloned speech")
            sentence = "Both graphics cards are below seventy degrees, and the reminder is set for three o'clock."
            t0 = time.perf_counter()
            r = await c.post("/audio/speech", json={"input": sentence, "voice": zira["id"], "response_format": "wav"})
            took = time.perf_counter() - t0
            record(r.status_code == 200, "synthesis", f"HTTP {r.status_code}, {took:.2f} s total, RTF {r.headers.get('x-rtf')}, "
                   f"{r.headers.get('x-audio-seconds')} s of audio")
            if r.status_code != 200:
                print("   ", r.text[:300])
                return 1
            record(r.headers.get("x-ai-generated") == "true" and "AI-generated" in (read_label(r.content) or ""),
                   "WAV is labelled as AI-generated", read_label(r.content) or "no label")
            audio, rate = sf.read(io.BytesIO(r.content), dtype="float32")
            text = await asr.transcribe((resample_poly(audio, SAMPLE_RATE, rate) * 32767).astype(np.int16).tobytes())
            expected = [("graphics",), ("seventy", "70"), ("reminder",), ("three", "3")]  # Whisper may write digits
            heard = [w for w in expected if any(v in text.lower() for v in w)]
            record(len(heard) == len(expected), "clone is intelligible (Whisper round trip)", repr(text))

            first = None
            t0 = time.perf_counter()
            async with c.stream("POST", "/audio/speech", json={"input": "Okay, opening it now.", "voice": zira["id"]}) as s:
                async for chunk in s.aiter_bytes():
                    if chunk and first is None:
                        first = time.perf_counter() - t0
            record(first is not None and first < 2.0, "first audio of a short sentence (streamed)", f"{first:.2f} s")

            print("3. Similarity (speaker-verification model, threshold 0.86)")
            t0 = time.perf_counter()
            r = await c.post(f"/voices/{zira['id']}/similarity", json={"compare_with": david["id"]})
            if r.status_code != 200:
                record(False, "similarity endpoint", f"HTTP {r.status_code}: {r.text[:200]}")
            else:
                d = r.json()
                record(d["same_speaker"], "clone matches its own speaker", f"{d['similarity']:.3f}")
                record(d["other_speaker_similarity"] < d["similarity"] - 0.1, "clone is clearly not the other speaker",
                       f"{d['other_speaker_similarity']:.3f} vs David ({time.perf_counter() - t0:.1f} s incl. model load)")

            r = await c.post("/warm", json={})
            record(r.status_code == 200, "warm endpoint", str(r.json()))
        finally:
            print("4. Revoke")
            for pid in made:
                r = await c.delete(f"/voices/{pid}")
                gone = not (settings.voices_dir / pid).exists()
                record(r.status_code == 200 and gone, f"revoked {pid}",
                       "recording deleted" if gone else f"STILL ON DISK (HTTP {r.status_code}: {r.text[:120]})")
            record(profiles.active_id() == active_before, "active voice unchanged", str(profiles.active_id()))
    fails = sum(1 for s, _ in RESULTS if s == "FAIL")
    print(f"\nSummary: {len(RESULTS) - fails} PASS, {fails} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
