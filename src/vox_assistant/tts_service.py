"""Cloned-voice TTS service. Run it with the *Audio Clone* virtualenv's Python (see scripts/start_tts.ps1).

That environment owns the ROCm PyTorch + VoxCPM2 install, so this module imports `voice_studio` lazily and
only when a request needs the model. It keeps one resident model and a small LRU of acoustic prompts, and
resolves the active voice profile on every request, so switching or revoking a voice takes effect at once.

POST /audio/speech  {"input": str, "voice": profile id | null, "response_format": "pcm" | "wav"}
    pcm streams raw 16-bit mono sentence by sentence (sample rate in the X-Sample-Rate header).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

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


def _default_backend_factory(settings: Settings) -> Callable[[], Any]:
    def make():
        import os
        import sys

        if settings.tts_hip_runtime == "driver":
            use_driver_hip_runtime()
        # MIOpen's default find mode benchmarks kernels for every new tensor shape, and every new sentence length is a
        # new shape: measured 10-25 s for the first synthesis of each length (RTF 4-5). FAST picks a kernel at once:
        # first runs cost the same as repeats (RTF ~0.5-0.65), for ~20% slower warm synthesis.
        os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
        root = settings.audio_clone_root
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        os.environ.setdefault("VOICE_STUDIO_DATA", str(root))
        from voice_studio.config import Settings as StudioSettings
        from voice_studio.runtime import configure_runtime

        configure_runtime(StudioSettings.from_env())  # caches and ROCm target family; must precede `import torch`
        if settings.tts_device != "auto":
            os.environ["VOICE_STUDIO_DEVICE"] = resolve_device(settings.tts_device)
        from voice_studio.engine import VoxBackend

        return VoxBackend(StudioSettings.from_env())

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

    def resolve(self, voice_id: str | None) -> dict:
        profile = self.profiles.get(voice_id) if voice_id else self.profiles.active()
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


def create_app(settings: Settings | None = None, backend_factory: Callable[[], Any] | None = None) -> FastAPI:
    settings = settings or Settings.load()
    state = TTSState(settings, backend_factory or _default_backend_factory(settings))
    app = FastAPI(title="VoxVortex cloned-voice TTS")
    app.state.tts = state
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

            with state.lock:
                backend = state.ensure_loaded()
                cache, _ = state.prompt_for(backend, profile)
                audio = backend.generate(TEST_SENTENCE, cache, PRESETS[settings.tts_preset], 7)
                backend.synchronize()
                rate = backend.sample_rate
            ref, ref_rate = sf.read(profile["path"], dtype="float32")
            clone = state.embedder.embed(audio, rate)
            score = cosine(state.embedder.embed(ref, ref_rate), clone)
            out = {"voice": profile["id"], "similarity": round(score, 4), "threshold": SAME_SPEAKER_THRESHOLD,
                   "same_speaker": score >= SAME_SPEAKER_THRESHOLD, "sentence": TEST_SENTENCE}
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
        try:
            profile = state.resolve(body.voice)
        except VoiceError as exc:
            raise HTTPException(409, str(exc))
        chunks = split_for_tts(text)
        if not chunks:
            raise HTTPException(400, "input has nothing to speak")

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
            if profile:  # first audio decode compiles kernels (RTF ~4.5); do it before the first real request
                cache, _ = st.prompt_for(backend, profile)
                backend.generate("Warming up the voice.", cache, PRESETS[settings.tts_preset], 42)
                backend.synchronize()
    port = args.port or int(settings.tts_url.rsplit(":", 1)[1])
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
