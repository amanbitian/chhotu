"""In-process faster-whisper ASR (CPU by default), implementing VoxVortex's `ASR` protocol."""

from __future__ import annotations

import asyncio

import numpy as np


class LocalWhisperASR:
    def __init__(self, model: str = "small.en", device: str = "cpu", compute_type: str = "int8", threads: int = 0) -> None:
        self.model_name, self.device, self.compute_type, self.threads = model, device, compute_type, threads
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type,
                                       cpu_threads=self.threads)
        return self._model

    def _run(self, pcm: bytes, language: str, prompt: str | None) -> str:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        lang = None if language == "auto" else language
        if self.model_name.endswith(".en"):
            lang = "en"
        segments, _ = self._load().transcribe(
            audio, language=lang, initial_prompt=prompt, beam_size=1, temperature=0.0,
            vad_filter=False, condition_on_previous_text=False,
        )
        return " ".join(" ".join(s.text for s in segments).split())

    async def transcribe(self, pcm: bytes, *, language: str = "auto", prompt: str | None = None) -> str:
        return await asyncio.to_thread(self._run, pcm, language, prompt)

    async def warm(self) -> None:
        await asyncio.to_thread(self._load)

    async def aclose(self) -> None:
        self._model = None
