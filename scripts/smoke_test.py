"""End-to-end smoke test with the real components (not fakes): live LLM, Dual GPU telemetry, Silero VAD, Whisper,
audio devices, a real HTTP server, and the full voice loop fed with *spoken* audio (Windows TTS stands in for a person).
Speaker output is captured, not played, so the room stays quiet.

    .venv\\Scripts\\python.exe scripts\\smoke_test.py

Pass --keep to leave the scratch data directory behind. Exit code is non-zero if any check FAILs. SKIP / WARN are reported but do not fail the run.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from scipy.signal import resample_poly

from voxvortex.audio import FRAME_BYTES, SAMPLE_RATE, SileroVAD

from vox_assistant.agent import Agent
from vox_assistant.asr_local import LocalWhisperASR
from vox_assistant.audit import AuditLog
from vox_assistant.config import Settings
from vox_assistant.permissions import PermissionGate
from vox_assistant.tools import build_registry
from vox_assistant.tts_neutral import NeutralVoiceTTS
from vox_assistant.voice import CHIME, VoiceLoop, strip_wake_phrase
from vox_assistant.voices import VoiceProfiles

RESULTS: list[tuple[str, str, str]] = []
AUDIT_LOGS: list[AuditLog] = []


def record(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    print(f"  [{status:^4}] {name}{(' - ' + detail) if detail else ''}", flush=True)


async def say_pcm16k(text: str) -> bytes:
    """Speech for `text` as 16 kHz PCM16, the way a microphone would deliver it."""
    tts = NeutralVoiceTTS()
    raw = b"".join([c async for c in tts.synth(text)])
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    y = resample_poly(x, SAMPLE_RATE, tts.sample_rate)
    loud = np.flatnonzero(np.abs(y) > 300)
    if len(loud):  # Windows TTS pads ~0.7 s of silence; latency is measured from the last voiced sample
        y = y[: loud[-1] + int(0.05 * SAMPLE_RATE)]
    return y.astype(np.int16).tobytes()


def frames_of(pcm: bytes) -> list[bytes]:
    pcm = pcm + b"\x00" * (-len(pcm) % FRAME_BYTES)
    return [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]


SILENCE = b"\x00" * FRAME_BYTES


class CapturingSpeaker:
    def __init__(self) -> None:
        self.plays: list[tuple[float, int, int]] = []
        self.pending_seconds = 0.0

    def play(self, pcm: bytes, rate: int) -> None:
        self.plays.append((time.perf_counter(), len(pcm), rate))

    def stop(self) -> None:
        pass

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass

    def seconds(self) -> float:
        return sum(n / 2 / r for _, n, r in self.plays)


class Feeder:
    """A microphone stand-in: silence until segments are queued, then the speech frames."""

    def __init__(self, realtime: bool = False) -> None:
        self.queue: list[bytes] = []
        self.stopped = False
        self.realtime = realtime  # real time: one 32 ms frame every 32 ms, like a microphone
        self.speech_fed_at: list[float] = []  # when the last frame of each queued utterance was delivered
        self._ends: list[int] = []
        self._fed = 0

    def say(self, pcm: bytes, lead_s: float = 0.5) -> None:
        self.queue += [SILENCE] * int(lead_s * 31.25) + frames_of(pcm)
        self._ends.append(self._fed + len(self.queue))

    async def frames(self):
        t0 = time.perf_counter()
        n = 0
        while not self.stopped:
            frame = self.queue.pop(0) if self.queue else SILENCE
            self._fed += 1
            if self._ends and self._fed >= self._ends[0]:
                self._ends.pop(0)
                self.speech_fed_at.append(time.perf_counter())
            yield frame
            n += 1
            if self.realtime:
                await asyncio.sleep(max(0.0, t0 + n * 0.032 - time.perf_counter()))
            else:
                await asyncio.sleep(0.001)


async def wait_for(cond, timeout: float, what: str) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        await asyncio.sleep(0.05)
    print(f"      (timed out waiting for {what})")
    return False


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --------------------------------------------------------------------------------------------------
async def check_services(settings: Settings) -> None:
    print("\n1. Services")
    async with httpx.AsyncClient(timeout=5.0) as c:
        try:
            r = await c.get(f"{settings.llm_base_url}/models")
            record("PASS" if r.status_code == 200 else "FAIL", "LLM endpoint", settings.llm_base_url)
        except httpx.HTTPError as exc:
            record("FAIL", "LLM endpoint", type(exc).__name__)
        try:
            hw = (await c.get(f"{settings.dual_gpu_url}/api/hardware")).json()["hardware"]
            lanes = [g["lane"] for g in hw["gpus"] if g.get("lane")]
            record("PASS" if lanes else "FAIL", "Dual GPU telemetry", f"lanes {lanes}")
        except Exception as exc:
            record("FAIL", "Dual GPU telemetry", type(exc).__name__)
        try:
            h = (await c.get(f"{settings.tts_url}/health")).json()
            record("PASS", "cloned-voice service", f"up, model_loaded={h['model_loaded']}")
        except httpx.HTTPError:
            record("SKIP", "cloned-voice service", "not running")


def check_gpu_torch(settings: Settings) -> None:
    print("\n2. PyTorch ROCm in the Audio Clone environment, loaded the way tts_service.py loads it")
    py = settings.audio_clone_root / ".venv" / "Scripts" / "python.exe"
    if not py.is_file():
        record("SKIP", "torch GPU probe", "Audio Clone venv not found")
        return
    probe = ("from vox_assistant.tts_service import use_driver_hip_runtime; print(use_driver_hip_runtime());"
             "import torch; x = torch.ones(256, 256, device='cuda:0'); print(torch.cuda.device_count(), float((x @ x).sum()))")
    env = {**__import__("os").environ, "PYTHONUTF8": "1", "ROCM_SDK_TARGET_FAMILY": "custom",
           "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    try:
        p = subprocess.run([str(py), "-c", probe], capture_output=True, text=True, timeout=180, env=env)
    except subprocess.TimeoutExpired:
        record("WARN", "torch GPU probe", "timed out")
        return
    if p.returncode == 0 and "16777216.0" in p.stdout:
        lines = p.stdout.strip().splitlines()
        record("PASS", "torch runs a GPU matmul with the driver's HIP runtime", f"{lines[0]}; {lines[-1].split()[0]} devices")
    else:
        record("FAIL", "torch GPU probe", f"exit {p.returncode & 0xFFFFFFFF:#x}: {(p.stderr or p.stdout).strip()[-200:]}")


async def check_audio_devices(settings: Settings) -> None:
    print("\n3. Audio devices (microphone is opened and read for 1 s; speaker gets 0.2 s of silence)")
    from vox_assistant.audio_io import Microphone, Speaker

    mic = Microphone(settings.input_device)
    try:
        mic.start()
        n = 0
        async def count():
            nonlocal n
            async for _ in mic.frames():
                n += 1
        task = asyncio.create_task(count())
        await asyncio.sleep(1.2)
        task.cancel()
        record("PASS" if n >= 20 else "FAIL", "microphone capture", f"{n} frames of 32 ms in 1.2 s")
    except Exception as exc:
        record("FAIL", "microphone capture", f"{type(exc).__name__}: {exc}")
    finally:
        mic.stop()
    spk = Speaker(settings.output_device)
    try:
        spk.play(b"\x00\x00" * 4800, 24000)
        await spk.drain()
        record("PASS", "speaker stream opens and drains")
    except Exception as exc:
        record("FAIL", "speaker stream", f"{type(exc).__name__}: {exc}")
    finally:
        spk.close()


async def check_speech_chain(settings: Settings, asr: LocalWhisperASR) -> dict:
    print("\n4. Speech chain: Windows TTS -> Silero VAD -> Whisper -> wake phrase")
    phrase = "Hey Vox, which models are running on the graphics cards, and are the temperatures safe?"
    speech = await say_pcm16k(phrase)
    vad = SileroVAD(settings.silero_model)
    probs = [vad.prob(f) for f in frames_of(speech)]
    voiced = sum(p > 0.5 for p in probs)
    vad.reset()
    quiet = max(vad.prob(SILENCE) for _ in range(30))
    record("PASS" if voiced > 20 and quiet < 0.2 else "FAIL", "Silero VAD", f"{voiced} voiced frames in speech, max {quiet:.2f} on silence")
    t0 = time.perf_counter()
    text = await asr.transcribe(speech, prompt="Hey vox.")
    took = time.perf_counter() - t0
    record("PASS" if "temperature" in text.lower() else "FAIL", "Whisper transcript", f"{text!r} ({took:.2f} s for {len(speech)/32000:.1f} s of speech)")
    rest = strip_wake_phrase(text, [settings.wake_phrase, *settings.wake_aliases])
    record("PASS" if rest else "FAIL", "wake phrase recognised", f"request = {rest!r}")
    return {"speech": speech}


async def run_loop(settings: Settings, asr: LocalWhisperASR, tmp: Path, scenario, *, cloned=None, spotter=None,
                   realtime: bool = False) -> dict:
    audit = AuditLog(tmp / "audit.sqlite3")
    AUDIT_LOGS.append(audit)
    registry, store = build_registry(settings)
    events: list[tuple[float, dict]] = []
    from vox_assistant.llm_client import ToolChoiceLLM

    llm = ToolChoiceLLM(settings.llm_base_url, settings.llm_model, temperature=0.3, timeout=90.0, first_token_timeout=60.0)
    agent = Agent(llm, registry, PermissionGate(None, audit), audit)
    spk = CapturingSpeaker()
    asr_calls: list[float] = []
    real = asr.transcribe

    async def timed(pcm, **kw):
        asr_calls.append(time.perf_counter())
        return await real(pcm, **kw)

    asr.transcribe = timed
    loop = VoiceLoop(agent=agent, asr=asr, tts=cloned, neutral_tts=NeutralVoiceTTS(), speaker=spk, vad=SileroVAD(settings.silero_model),
                     settings=settings, reminders=None, confirm_timeout_s=40.0, spotter=spotter,
                     on_event=lambda e: events.append((time.perf_counter(), e)))
    feeder = Feeder(realtime=realtime)
    runner = asyncio.create_task(loop.run(feeder.frames()))
    try:
        await scenario(feeder, loop, events, spk)
    finally:
        feeder.stopped = True
        await asyncio.wait_for(runner, 10)
        asr.transcribe = real
        await llm.aclose()
        store.close()
    return {"events": events, "speaker": spk, "asr_calls": asr_calls, "audit": audit, "feeder": feeder, "loop": loop}


async def check_voice_loop(settings: Settings, asr: LocalWhisperASR, speech: bytes, tmp: Path, cloned, quick: bool = False) -> None:
    voice = "cloned voice" if cloned else "neutral voice"
    print(f"\n5. Voice loop, hands-free, in real time: spoken question -> live LLM + gpu_status -> spoken answer ({voice})")
    spotter = LocalWhisperASR(settings.wake_spotter_model, settings.whisper_device, settings.whisper_compute, 4)
    await spotter.warm()

    async def ask(feeder, loop, events, spk):
        await wait_for(lambda: len(loop._acks) >= (2 if cloned else 1), 60, "acknowledgements to be prepared")
        feeder.say(speech)
        await wait_for(lambda: any(e["kind"] == "say" for _, e in events), 90, "the first words of the answer")
        await wait_for(lambda: loop._turn is not None and loop._turn.done(), 120, "the turn to finish")
        await wait_for(lambda: spk.seconds() > 1.0, 30, "audio")

    out = await run_loop(settings, asr, tmp, ask, cloned=cloned, spotter=spotter, realtime=True)
    wake = [(t, e) for t, e in out["events"] if e["kind"] == "wake"]
    record("PASS" if wake and wake[0][1]["via"] == "spotter" else "FAIL", "wake phrase spotted while still speaking",
           f"{(out['feeder'].speech_fed_at[0] - wake[0][0]):.2f} s before the utterance ended" if wake and out["feeder"].speech_fed_at else "")
    ev = out["events"]
    kinds = [e["kind"] for _, e in ev]
    said = "".join(e["text"] for _, e in ev if e["kind"] == "say")
    record("PASS" if "heard" in kinds else "FAIL", "utterance heard and transcribed")
    used = [e for _, e in ev if e["kind"] == "tool_result"]
    record("PASS" if any(e["name"] == "gpu_status" and e["ok"] for e in used) else "FAIL", "agent called gpu_status", str([(e["name"], e["decision"]) for e in used]))
    record("PASS" if len(said) > 20 else "FAIL", "answer produced", said[:160].replace("\n", " "))
    spk: CapturingSpeaker = out["speaker"]
    record("PASS" if spk.seconds() > 1.0 else "FAIL", "answer synthesized to audio", f"{spk.seconds():.1f} s")
    ack_sizes = {len(pcm) for clips in out["loop"]._acks.values() for pcm, _ in clips}
    ack_plays = [p for p in spk.plays if p[1] in ack_sizes]
    answer_plays = [p for p in spk.plays if p[1] != len(CHIME) and p[1] not in ack_sizes]  # chime and "let me check"
    if ack_plays and out["feeder"].speech_fed_at:
        record("PASS" if ack_plays[0][0] - out["feeder"].speech_fed_at[0] < 2.0 else "WARN",
               f"end of speech -> first sound (\"let me check\", {voice})", f"{ack_plays[0][0] - out['feeder'].speech_fed_at[0]:.2f} s")
    if answer_plays and out["feeder"].speech_fed_at:
        # From the last frame of speech the user said, through the VAD's end-of-turn silence, ASR, the LLM and its tool
        # call, to the first audio of the answer: what the user waits for.
        end = out["feeder"].speech_fed_at[0]
        lat = answer_plays[0][0] - end
        record("PASS" if lat < 3.0 else "WARN", f"end of speech -> first audio of the answer ({voice})", f"{lat:.2f} s (target 1-2 s)")
        steps = [f"asr start +{out['asr_calls'][0] - end:.2f}"] if out["asr_calls"] else []
        for t, e in out["events"]:
            if t >= end and e["kind"] in ("heard", "tool", "tool_result", "ack", "say"):
                steps.append(f"{e['kind']}{'(' + e['name'] + ')' if 'name' in e else ''} +{t - end:.2f}")
                if e["kind"] == "say":
                    break
        steps.append(f"first audio +{lat:.2f}")
        print("      timeline after end of speech: " + ", ".join(steps))
        first_say = next((e["text"] for _, e in out["events"] if e["kind"] == "say"), "")
        print(f"      first text chunk: {first_say[:100]!r}")
    record("PASS" if any(r["tool"] == "gpu_status" and r["decision"] == "auto" for r in out["audit"].recent()) else "FAIL", "audit log has the read-only call")
    if quick:
        return

    print("\n6. Voice loop: permission by voice (state-changing action, user says no)")
    ask_speech = await say_pcm16k("Hey Vox, please open Notepad.")
    no_speech = await say_pcm16k("No, cancel that.")

    async def decline(feeder, loop, events, spk):
        feeder.say(ask_speech)
        if await wait_for(lambda: any(e["kind"] == "confirm_request" for _, e in events), 90, "the spoken confirmation request"):
            await asyncio.sleep(0.5)
            feeder.say(no_speech)
        await wait_for(lambda: loop._turn is not None and loop._turn.done(), 90, "the turn to finish")

    out = await run_loop(settings, asr, tmp, decline)
    ev = [e for _, e in out["events"]]
    result = [e for e in ev if e["kind"] == "confirm_result"]
    record("PASS" if any(e["kind"] == "confirm_request" for e in ev) else "FAIL", "assistant asked before acting", next((e["intent"] for e in ev if e["kind"] == "confirm_request"), ""))
    record("PASS" if result and result[0]["approved"] is False else "FAIL", "spoken 'no' was understood as a refusal")
    record("PASS" if any(r["tool"] == "open_app" and r["decision"] == "denied" for r in out["audit"].recent()) else "FAIL", "denied action is audited and nothing was launched")

    print("\n7. Voice loop: privacy commands")
    mute = await say_pcm16k("Hey Vox, mute.")
    after = await say_pcm16k("Hey Vox, what time is it?")

    async def muting(feeder, loop, events, spk):
        feeder.say(mute)
        await wait_for(lambda: loop.muted, 30, "mute")
        feeder.say(after)
        await asyncio.sleep(8)

    out = await run_loop(settings, asr, tmp, muting)
    ev = [e for _, e in out["events"]]
    record("PASS" if any(e["kind"] == "muted" for e in ev) else "FAIL", "'mute' handled locally")
    record("PASS" if not any(e["kind"] in ("tool", "say") for e in ev) else "FAIL", "nothing reached the model while muted")


async def check_web(settings: Settings, tmp: Path) -> None:
    print("\n8. Web app on a real socket")
    import uvicorn

    from vox_assistant.webapp import create_app

    settings = Settings.load(tmp)  # fresh data dir: no demo voice, own audit log
    settings.llm_base_url, settings.llm_model = Settings().llm_base_url, Settings().llm_model
    port = free_port()
    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient(base_url=base, timeout=120.0) as c:
            r = await c.get("/")
            record("PASS" if r.status_code == 200 and "VoxVortex Desktop" in r.text else "FAIL", "GET /")
            s = (await c.get("/api/status")).json()
            record("PASS" if s["llm"]["ok"] else "FAIL", "/api/status sees the LLM", f"tts ok={s['tts'].get('ok')}, {len(s['tools'])} tools")
            record("PASS" if (await c.get("/api/voices", headers={"host": "evil.example"})).status_code == 403 else "FAIL", "foreign Host header rejected")
            record("PASS" if (await c.post("/api/reset", content="x", headers={"content-type": "text/plain"})).status_code == 415 else "FAIL", "non-JSON POST rejected")
            r = await c.post("/api/speak", json={"text": "Smoke test of the neutral voice.", "voice_mode": "neutral"})
            record("PASS" if r.status_code == 200 and r.content[:4] == b"RIFF" and len(r.content) > 20000 else "FAIL", "neutral speech as WAV", f"{len(r.content)} bytes")
            r = await c.post("/api/speak", json={"text": "hello", "voice_mode": "cloned"})
            record("PASS" if r.status_code in (200, 503, 409) else "FAIL", "cloned speech path", f"HTTP {r.status_code}: {(r.json().get('detail') if r.status_code != 200 else 'audio')[:90]}")
            evs = []
            async with c.stream("POST", "/api/chat", json={"text": "How much RAM is my PC using right now?"}) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        evs.append(json.loads(line))
            ok_tool = any(e["type"] == "tool_result" and e["name"] == "system_info" and e["ok"] for e in evs)
            text = "".join(e.get("text", "") for e in evs if e["type"] == "say")
            record("PASS" if ok_tool and text else "FAIL", "chat: system_info tool + answer", text[:120])
    finally:
        server.should_exit = True
        await task
        app.state.audit.close()
        app.state.reminders.close()


async def check_real_clone(settings: Settings, asr: LocalWhisperASR) -> None:
    print("\n9. Cloned voice (only if the service is up)")
    try:
        async with httpx.AsyncClient(base_url=settings.tts_url, timeout=300.0) as c:
            await c.get("/health")
            t0 = time.perf_counter()
            r = await c.post("/audio/speech", json={"input": "Please check both graphics cards.", "response_format": "wav"})
        if r.status_code != 200:
            record("FAIL", "clone synthesis", r.text[:150])
            return
        import io
        import soundfile as sf

        audio, rate = sf.read(io.BytesIO(r.content), dtype="float32")
        pcm = (resample_poly(audio, SAMPLE_RATE, rate) * 32767).astype(np.int16).tobytes()
        text = await asr.transcribe(pcm)
        record("PASS" if "graphics" in text.lower() else "FAIL", "clone speech is intelligible (round-trips through Whisper)", f"{text!r}; RTF {r.headers.get('x-rtf')}; {time.perf_counter() - t0:.1f} s")
    except httpx.HTTPError:
        record("SKIP", "clone synthesis", "service not running")


async def main() -> int:
    import tempfile

    settings = Settings.load()
    tmp = Path(tempfile.mkdtemp(prefix="voxa_smoke_", dir=settings.data_dir))
    print(f"VoxVortex Desktop smoke test (scratch data in {tmp})")
    quick = "--quick" in sys.argv  # services, speech chain and the timed voice loop only
    await check_services(settings)
    if not quick:
        check_gpu_torch(settings)
        await check_audio_devices(settings)
    asr = LocalWhisperASR(settings.whisper_model, settings.whisper_device, settings.whisper_compute, settings.whisper_threads)
    await asr.warm()
    chain = await check_speech_chain(settings, asr)
    from vox_assistant.tts_client import ClonedVoiceTTS

    cloned = ClonedVoiceTTS(settings.tts_url, preset=settings.tts_preset)
    try:
        await cloned.health()
    except httpx.HTTPError:
        cloned = None
    await check_voice_loop(settings, asr, chain["speech"], tmp, cloned, quick=quick)
    if not quick:
        await check_web(settings, tmp)
        await check_real_clone(settings, asr)
    for log in AUDIT_LOGS:
        log.close()
    if "--keep" not in sys.argv:
        shutil.rmtree(tmp, ignore_errors=True)
    counts = {k: sum(1 for s, *_ in RESULTS if s == k) for k in ("PASS", "FAIL", "WARN", "SKIP")}
    print("\nSummary: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
