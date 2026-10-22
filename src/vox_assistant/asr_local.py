"""In-process faster-whisper ASR (CPU by default), implementing VoxVortex's `ASR` protocol.

`language="auto"` lets Whisper identify the language but keeps the answer within `languages` (English and Hindi by
default). Whisper often labels Hindi speech as Urdu (or Marathi, Nepali), which would also switch the output script;
when it does, the clip is decoded again in the most likely allowed language. Detection reuses the decoding pass, so
the common case costs nothing extra. `.en` models are English only.
"""

from __future__ import annotations

import asyncio

import numpy as np


class LocalWhisperASR:
    def __init__(self, model: str = "small.en", device: str = "cpu", compute_type: str = "int8", threads: int = 0,
                 languages: tuple[str, ...] = ("en", "hi")) -> None:
        self.model_name, self.device, self.compute_type, self.threads = model, device, compute_type, threads
        self.languages = languages
        self.last_language: str | None = None
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type,
                                       cpu_threads=self.threads)
        return self._model

    def _decode(self, audio: np.ndarray, lang: str | None, prompt: str | None):
        segments, info = self._load().transcribe(
            audio, language=lang, initial_prompt=prompt, beam_size=1, temperature=0.0,
            vad_filter=False, condition_on_previous_text=False,
        )
        return " ".join(" ".join(s.text for s in segments).split()), info

    def _run(self, pcm: bytes, language: str, prompt: str | None) -> str:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if self.model_name.endswith(".en"):
            lang: str | None = "en"
        elif language in ("auto", None, ""):
            lang = None
        else:
            lang = language
        text, info = self._decode(audio, lang, prompt)
        if lang is None and info.language not in self.languages:
            probs = dict(info.all_language_probs or [])
            best = max(self.languages, key=lambda code: probs.get(code, 0.0))
            text, info = self._decode(audio, best, prompt)
        self.last_language = info.language if lang is None else lang
        return text

    async def transcribe(self, pcm: bytes, *, language: str = "auto", prompt: str | None = None) -> str:
        return await asyncio.to_thread(self._run, pcm, language, prompt)

    async def warm(self) -> None:
        await asyncio.to_thread(self._load)

    async def aclose(self) -> None:
        self._model = None
