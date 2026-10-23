"""Cloned-voice TTS service. Run it with the *Audio Clone* virtualenv's Python (see scripts/start_tts.ps1).

That environment owns the ROCm PyTorch + VoxCPM2 install, so this module imports `voice_studio` lazily and
only when a request needs the model. It keeps one resident model and a small LRU of acoustic prompts, and
resolves the active voice profile on every request, so switching or revoking a voice takes effect at once.

POST /audio/speech  {"input": str, "voice": profile id | null, "response_format": "pcm" | "wav"}
    pcm streams raw 16-bit mono as the model produces it, in ~0.16 s pieces (sample rate in X-Sample-Rate);
    wav returns one labelled file.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import speech_text
from .config import Settings
from .disclosure import labelled_wav
from .security import install_local_only
from .voices import VoiceError, VoiceProfiles

PRESETS = {
    "Draft": {"steps": 6, "cfg": 2.0, "attempts": 1},
    "Balanced": {"steps": 10, "cfg": 2.0, "attempts": 2},
    "Quality": {"steps": 16, "cfg": 2.0, "attempts": 3},
}
_SENTENCE = re.compile(r"(?<=[.!?।॥])\s+|\n+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
MAX_CHARS = 240
PROMPT_CACHE = 8


def split_for_tts(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Sentences, packed down to words only when one exceeds the limit."""
    chunks: list[str] = []
    for sentence in _SENTENCE.split(text.strip()):
        current = ""
        for word in sentence.split():
            candidate = f"{current} {word}".strip()
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current:
            chunks.append(current)
    return chunks


def pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def stream_generate(backend: Any, text: str, cache: Any, preset: dict, seed: int) -> Iterator[np.ndarray]:
    """Audio as the model produces it. VoxCPM2 decodes one latent patch at a time (~0.16 s at 48 kHz), so the first
    audio is ready ~0.07 s after the request instead of after the whole sentence (1.2-2.6 s measured). Backends without
    a streaming mode yield the sentence as one piece."""
    stream = getattr(getattr(backend, "model", None), "generate_with_prompt_cache_streaming", None)
    if stream is None:
        yield backend.generate(text, cache, preset, seed)
        return
    torch = backend.torch
    random.seed(seed)  # VoxCPM 2.0.3 has no seed argument; same seeding as VoxBackend.generate
    np.random.seed(seed)
    torch.manual_seed(seed)
    with torch.inference_mode():
        for wav, _, _ in stream(target_text=text, prompt_cache=cache, inference_timesteps=preset["steps"],
                                cfg_value=preset["cfg"], max_len=1024):
            yield wav.squeeze(0).float().cpu().numpy()


class SpeechRequest(BaseModel):
    input: str
    voice: str | None = None
    response_format: str = "pcm"
    preset: str | None = None
    seed: int = 42
    model: str | None = None  # accepted for OpenAI-client compatibility, ignored
    speed: float | None = None
    stream: bool | None = None


def use_driver_hip_runtime() -> str | None:
    """Load the display driver's HIP runtime before PyTorch can load the one bundled in its wheel.

    The rocm-sdk 7.2.1 wheel ships amdhip64_7.dll 10.0.3581, which crashes (0xC0000005) inside
    hipGetDeviceCount with Adrenalin 32.0.31036 (driver runtime 10.0.3679). Windows reuses an already
    loaded DLL of the same name, so torch and rocBLAS then bind to the driver's copy. Must run before
    `import torch`. Returns the path loaded, or None when there is nothing to do.

    TorchScript is switched off too: its GPU fuser compiles kernels with the wheel's hiprtc0702.dll, which
    does not match the driver runtime and crashes in VoxCPM2's `snake` activation (the driver ships no hipRTC).
    Without it `snake` runs as plain PyTorch ops.
    """
    import ctypes
    import os
    import sys

    if sys.platform != "win32" or "torch" in sys.modules:
        return None
    path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "amdhip64_7.dll")
    if not os.path.isfile(path):
        return None
    os.environ.setdefault("PYTORCH_JIT", "0")
    ctypes.WinDLL(path)
    return path


def pick_device(wanted: str, names: list[str]) -> str:
    """"cpu" / "cuda:N" pass through; anything else is a case-insensitive fragment of a GPU name."""
    if wanted == "cpu" or wanted.startswith("cuda"):
        return wanted
    for i, name in enumerate(names):
        if wanted.lower() in name.lower():
            return f"cuda:{i}"
    raise RuntimeError(f"No GPU matching {wanted!r} (found: {', '.join(names) or 'none'}). Set tts_device.")


def resolve_device(wanted: str) -> str:
    if wanted == "cpu" or wanted.startswith("cuda"):
        return wanted
    import torch

    return pick_device(wanted, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


class SimilarityBody(BaseModel):
    compare_with: str | None = None
    sentence: str | None = None  # e.g. a Hindi sentence: does the voice stay the same person in another language?


class AsrBody(BaseModel):
    pcm_b64: str  # 16 kHz mono PCM16, base64 (the request guard admits JSON bodies only)
    languages: list[str] = ["hi", "en"]


_runtime_ready = False


def prepare_runtime(settings: Settings) -> None:
    """Everything PyTorch needs on this machine before the first `import torch`; idempotent. Shared by the voice model
    and speech recognition."""
    global _runtime_ready
    if _runtime_ready:
        return
    import os
    import sys

    if settings.tts_hip_runtime == "driver":
        use_driver_hip_runtime()
    # Only matters if MIOpen is switched back on (see below): its default find mode benchmarks every new shape.
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    root = settings.audio_clone_root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("VOICE_STUDIO_DATA", str(root))
    from voice_studio.config import Settings as StudioSettings
    from voice_studio.runtime import configure_runtime

    configure_runtime(StudioSettings.from_env())  # caches and ROCm target family; must precede `import torch`
    import torch

    # PyTorch's own convolutions instead of MIOpen's. Measured here: MIOpen crashes (access violation, in any find
    # mode) on the tiny shapes of streaming decode, and in its default mode it spent 10-25 s tuning every new
    # sentence length. Without it, whole-sentence synthesis is as fast as before (RTF ~0.45) and streaming works.
    torch.backends.cudnn.enabled = settings.tts_miopen
    _runtime_ready = True


def _default_backend_factory(settings: Settings) -> Callable[[], Any]:
    def make():
        import os

        prepare_runtime(settings)
        if settings.tts_device != "auto":
            os.environ["VOICE_STUDIO_DEVICE"] = resolve_device(settings.tts_device)
        from voice_studio.config import Settings as StudioSettings
        from voice_studio.engine import VoxBackend

        return VoxBackend(StudioSettings.from_env())

    return make


def _default_asr_factory(settings: Settings) -> Callable[[], Any]:
    def make():
        from .asr_gpu import GpuWhisper

        prepare_runtime(settings)
        device = settings.asr_device or settings.tts_device
        return GpuWhisper(settings.asr_model, resolve_device(device) if device != "auto" else "cuda:0")

    return make


class TTSState:
    """Resident model + prompt cache. All GPU work goes through `lock`, one request at a time."""

    def __init__(self, settings: Settings, backend_factory: Callable[[], Any]) -> None:
        self.settings = settings
        self.profiles = VoiceProfiles(settings.voices_dir)
        self._factory = backend_factory
        self.backend: Any = None
        self.prompts: OrderedDict[tuple, Any] = OrderedDict()
        self.lock = threading.Lock()
        self.load_seconds: float | None = None
        self.embedder: Any = None  # voice_similarity.SpeakerEmbedder, created on first use

    def ensure_loaded(self) -> Any:
        if self.backend is None:
            t0 = time.perf_counter()
            self.backend = self._factory()
            self.load_seconds = time.perf_counter() - t0
        return self.backend

    def resolve(self, voice_id: str | None, text: str = "") -> dict:
        """The explicit voice, else the active voice for the text's language (Devanagari -> the Hindi reference if one
        is set), else the main voice."""
        lang = "hi" if _DEVANAGARI.search(text) else "en"
        profile = self.profiles.get(voice_id) if voice_id else self.profiles.active(lang)
        if profile is None:
            raise VoiceError("No active voice profile. Enroll one with `vox enroll` or pick one in the web UI.")
        return profile

    def prompt_for(self, backend: Any, profile: dict) -> tuple[Any, bool]:
        digest = hashlib.sha256(Path(profile["path"]).read_bytes()).hexdigest()
        key = (digest, profile["transcript"])
        if key in self.prompts:
            self.prompts.move_to_end(key)
            return self.prompts[key], True
        cache = backend.build_prompt({"path": profile["path"], "transcript": profile["transcript"]}, "Ultimate")
        self.prompts[key] = cache
        while len(self.prompts) > PROMPT_CACHE:
            self.prompts.popitem(last=False)
        return cache, False

    def forget_voice(self, profile_id: str) -> None:
        """Drop every cached prompt (cheap to rebuild) after a profile is revoked."""
        self.prompts.clear()

    def unload(self) -> None:
        with self.lock:
            self.prompts.clear()
            self.backend = None


def create_app(settings: Settings | None = None, backend_factory: Callable[[], Any] | None = None,
               asr_factory: Callable[[], Any] | None = None) -> FastAPI:
    settings = settings or Settings.load()
    state = TTSState(settings, backend_factory or _default_backend_factory(settings))
    asr_state: dict[str, Any] = {"factory": asr_factory or _default_asr_factory(settings), "model": None}

    def asr_model() -> Any:
        if asr_state["model"] is None:
            asr_state["model"] = asr_state["factory"]()
        return asr_state["model"]
    app = FastAPI(title="VoxVortex voice service (cloned speech + speech recognition)")
    app.state.tts = state
    app.state.asr_model = asr_model
    install_local_only(app)

    @app.get("/health")
    def health() -> dict:
        b = state.backend
        return {
            "ok": True,
            "model_loaded": b is not None,
            "sample_rate": getattr(b, "sample_rate", None),
            "runtime": getattr(b, "description", None),
            "load_seconds": state.load_seconds,
            "active_voice": state.profiles.active_id(),
            "asr_loaded": asr_state["model"] is not None,
        }

    @app.get("/voices")
    def voices() -> dict:
        return {"active": state.profiles.active_id(), "voices": state.profiles.list()}

    @app.post("/voices/{profile_id}/activate")
    def activate(profile_id: str) -> dict:
        try:
            state.profiles.set_active(profile_id)
        except VoiceError as exc:
            raise HTTPException(404, str(exc))
        return {"active": profile_id}

    @app.delete("/voices/{profile_id}")
    def revoke(profile_id: str) -> dict:
        try:
            state.profiles.revoke(profile_id)
        except VoiceError as exc:
            raise HTTPException(404, str(exc))
        state.forget_voice(profile_id)
        return {"revoked": profile_id, "active": state.profiles.active_id()}

    @app.post("/load")
    async def load() -> dict:
        def work():
            with state.lock:
                return state.ensure_loaded().description

        return {"runtime": await asyncio.to_thread(work)}

    @app.post("/asr")
    async def asr(body: AsrBody) -> dict:
        """Transcribe one utterance in each requested language (one GPU batch) and combine them for the assistant."""
        import base64

        from .asr_gpu import combine

        try:
            pcm = base64.b64decode(body.pcm_b64, validate=True)
        except ValueError:
            raise HTTPException(400, "pcm_b64 is not base64")
        if not pcm or len(pcm) > 16000 * 2 * 60:
            raise HTTPException(400, "send 16 kHz PCM16 audio, at most 60 s")
        audio = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0

        def work() -> dict:
            t0 = time.perf_counter()
            model = asr_model()
            readings = model.readings(audio, tuple(body.languages), {"en": settings.asr_prompt, "hi": settings.asr_prompt_hi})
            scores = getattr(model, "last_scores", {})
            text, used = combine(readings.get("hi", ""), readings.get("en", ""), scores) \
                if set(body.languages) == {"hi", "en"} else (next(iter(readings.values()), ""), body.languages[0])
            return {"text": text, "used": used, "readings": readings, "scores": scores,
                    "seconds": round(time.perf_counter() - t0, 3)}

        return await asyncio.to_thread(work)

    @app.post("/voices/{profile_id}/similarity")
    async def similarity(profile_id: str, body: SimilarityBody | None = None) -> dict:
        """Clone a fixed sentence with this profile and score it against the reference recording (NFR-05).
        With `compare_with`, also score the clone against another profile's reference: a working clone is much
        closer to its own speaker than to someone else."""
        from .voice_similarity import SAME_SPEAKER_THRESHOLD, TEST_SENTENCE, SpeakerEmbedder, cosine

        try:
            profile = state.resolve(profile_id)
            other = state.resolve(body.compare_with) if body and body.compare_with else None
        except VoiceError as exc:
            raise HTTPException(404, str(exc))
        if state.embedder is None:
            state.embedder = SpeakerEmbedder()

        def work() -> dict:
            import soundfile as sf

            sentence = speech_text.normalize((body.sentence if body and body.sentence else TEST_SENTENCE).strip())
            with state.lock:
                backend = state.ensure_loaded()
                cache, _ = state.prompt_for(backend, profile)
                audio = backend.generate(sentence, cache, PRESETS[settings.tts_preset], 7)
                backend.synchronize()
                rate = backend.sample_rate
            ref, ref_rate = sf.read(profile["path"], dtype="float32")
            clone = state.embedder.embed(audio, rate)
            score = cosine(state.embedder.embed(ref, ref_rate), clone)
            out = {"voice": profile["id"], "similarity": round(score, 4), "threshold": SAME_SPEAKER_THRESHOLD,
                   "same_speaker": score >= SAME_SPEAKER_THRESHOLD, "sentence": sentence}
            if other:
                o, o_rate = sf.read(other["path"], dtype="float32")
                out.update(compared_with=other["id"], other_speaker_similarity=round(cosine(state.embedder.embed(o, o_rate), clone), 4))
            return out

        return await asyncio.to_thread(work)

    @app.post("/warm")
    async def warm() -> dict:
        """Touch every weight with a tiny synthesis, so a model Windows paged out of VRAM is back before it is needed.

        The voice loop calls this when someone starts speaking. Skipped if the model is not loaded or is busy
        (busy means resident)."""
        def work() -> dict:
            if state.backend is None or not state.lock.acquire(blocking=False):
                return {"warmed": False}
            try:
                profile = state.profiles.active()
                if profile is None:
                    return {"warmed": False}
                t0 = time.perf_counter()
                cache, _ = state.prompt_for(state.backend, profile)
                state.backend.generate("Okay.", cache, {"steps": 2, "cfg": 2.0, "attempts": 1}, 0)
                state.backend.synchronize()
                return {"warmed": True, "seconds": round(time.perf_counter() - t0, 3)}
            finally:
                state.lock.release()

        return await asyncio.to_thread(work)

    @app.post("/forget")
    def forget() -> dict:
        """Drop cached acoustic prompts (voice-derived data held in memory) without unloading the model."""
        state.prompts.clear()
        return {"prompts_cleared": True}

    @app.post("/unload")
    async def unload() -> dict:
        await asyncio.to_thread(state.unload)
        return {"model_loaded": False}

    @app.post("/audio/speech")
    async def speech(body: SpeechRequest, request: Request):
        text = body.input.strip()
        if not text:
            raise HTTPException(400, "input is empty")
        if len(text) > 12000:
            raise HTTPException(400, "input exceeds 12,000 characters")
        if body.response_format not in ("pcm", "wav"):
            raise HTTPException(400, "response_format must be 'pcm' or 'wav'")
        preset_name = body.preset or settings.tts_preset
        if preset_name not in PRESETS:
            raise HTTPException(400, f"preset must be one of {sorted(PRESETS)}")
        preset = PRESETS[preset_name]
        # Each sentence as the model should read it: Hindi numbers as Hindi words, Hinglish ones in English, English as is,
        # romanized Hindi in Devanagari.
        chunks = [speech_text.normalize(c) for c in split_for_tts(text)]
        if not chunks:
            raise HTTPException(400, "input has nothing to speak")
        try:  # after normalizing: romanized Hinglish is now Devanagari, so it gets the Hindi reference if there is one
            profile = state.resolve(body.voice, " ".join(chunks))
        except VoiceError as exc:
            raise HTTPException(409, str(exc))

        def synth_chunk(index: int, chunk: str) -> tuple[np.ndarray, dict]:
            with state.lock:
                t0 = time.perf_counter()
                backend = state.ensure_loaded()
                cache, hit = state.prompt_for(backend, profile)
                begin = time.perf_counter()
                audio = backend.generate(chunk, cache, preset, (body.seed + index) % 2**32)
                backend.synchronize()
                took = time.perf_counter() - begin
            return audio, {"seconds": took, "prompt_cache_hit": hit, "audio_seconds": len(audio) / backend.sample_rate,
                           "sample_rate": backend.sample_rate, "overhead_seconds": begin - t0}

        if body.response_format == "pcm" and settings.tts_stream:
            return await stream_pcm(chunks, profile, preset, preset_name, body.seed)

        first_audio, first_meta = await asyncio.to_thread(synth_chunk, 0, chunks[0])
        rate = first_meta["sample_rate"]
        headers = {"X-Sample-Rate": str(rate), "X-Voice": profile["id"], "X-Preset": preset_name}

        if body.response_format == "wav":
            parts, metas = [first_audio], [first_meta]
            for i, chunk in enumerate(chunks[1:], start=1):
                a, m = await asyncio.to_thread(synth_chunk, i, chunk)
                parts.append(a)
                metas.append(m)
            gap = np.zeros(int(rate * 0.14), dtype=np.float32)
            merged = np.concatenate([p for a in parts for p in (a, gap)][:-1])
            gen = sum(m["seconds"] for m in metas)
            audio_s = sum(m["audio_seconds"] for m in metas)
            headers.update({"X-Generation-Seconds": f"{gen:.3f}", "X-Audio-Seconds": f"{audio_s:.3f}",
                            "X-RTF": f"{gen / audio_s:.3f}" if audio_s else "0",
                            "X-Chunks": str(len(chunks)), "X-AI-Generated": "true"})
            return Response(labelled_wav(pcm16(merged), rate, profile["id"]), media_type="audio/wav", headers=headers)

        async def stream():
            yield pcm16(first_audio)
            for i, chunk in enumerate(chunks[1:], start=1):
                if await request.is_disconnected():
                    return  # barge-in: the client dropped the stream, so stop spending GPU time
                a, _ = await asyncio.to_thread(synth_chunk, i, chunk)
                yield pcm16(a)

        return StreamingResponse(stream(), media_type="audio/L16", headers=headers)

    async def stream_pcm(chunks: list[str], profile: dict, preset: dict, preset_name: str, seed: int):
        """Generate on a worker thread, hand each piece to the response as soon as it exists.

        The first piece is awaited before the response starts, so a failure is still an HTTP error. When the client
        goes away (barge-in), the worker stops after the piece in progress, freeing the GPU within ~0.1 s."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        timing: dict = {}

        def put(item) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def generate(backend, cache) -> None:
            for i, chunk in enumerate(chunks):
                for piece in stream_generate(backend, chunk, cache, preset, (seed + i) % 2**32):
                    if stop.is_set():
                        return
                    put(pcm16(piece))

        def worker() -> None:
            try:
                t0 = time.perf_counter()
                with state.lock:  # one GPU request at a time; /warm never waits for it
                    timing["lock_wait"] = time.perf_counter() - t0
                    backend = state.ensure_loaded()
                    cache, _ = state.prompt_for(backend, profile)
                    timing["rate"] = backend.sample_rate
                    generate(backend, cache)
            except Exception as exc:  # reported as an HTTP error if nothing was sent yet
                put(exc)
            finally:
                put(None)

        loop.run_in_executor(None, worker)
        first = await queue.get()
        if isinstance(first, Exception) or first is None:
            stop.set()
            raise HTTPException(500, f"synthesis failed: {first}" if first else "synthesis produced no audio")
        headers = {"X-Sample-Rate": str(timing["rate"]), "X-Voice": profile["id"], "X-Preset": preset_name,
                   "X-Lock-Wait-Seconds": f"{timing.get('lock_wait', 0):.3f}", "X-AI-Generated": "true"}

        async def body():
            try:
                yield first
                while (item := await queue.get()) is not None:
                    if isinstance(item, Exception):
                        return  # mid-stream failure: end the audio; the client hears it stop
                    yield item
            finally:
                stop.set()  # the client disconnected, or the stream finished

        return StreamingResponse(body(), media_type="audio/L16", headers=headers)

    @app.exception_handler(VoiceError)
    async def voice_error(_, exc: VoiceError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    return app


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Cloned-voice TTS service")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--preload", action="store_true", help="load the model and warm it up before serving")
    args = parser.parse_args()
    settings = Settings.load()
    app = create_app(settings)
    if args.preload:
        st: TTSState = app.state.tts
        with st.lock:
            backend = st.ensure_loaded()
            profile = st.profiles.active()
            if profile:  # the first synthesis is slow (kernel loading); do it before the first real request
                cache, _ = st.prompt_for(backend, profile)
                for _ in stream_generate(backend, "Warming up the voice.", cache, PRESETS[settings.tts_preset], 42):
                    pass  # the streaming path, which is what the voice loop uses
                backend.synchronize()
            print(f"Voice model on {getattr(backend, 'description', {}).get('gpu')}", flush=True)
        if settings.asr_backend == "service":
            t0 = time.perf_counter()
            asr = app.state.asr_model()
            rng = np.random.default_rng(0)
            for seconds in (1, 3, 6, 10):  # kernels load per input shape: the first real 5 s utterance took 5 s otherwise
                asr.readings((rng.standard_normal(16000 * seconds) * 0.01).astype(np.float32),
                             ("hi", "en"), {"en": settings.asr_prompt, "hi": settings.asr_prompt_hi})
            print(f"Speech recognition ({settings.asr_model}) ready in {time.perf_counter() - t0:.1f} s", flush=True)
    port = args.port or int(settings.tts_url.rsplit(":", 1)[1])
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
