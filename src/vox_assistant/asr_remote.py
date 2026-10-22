"""Speech recognition through the voice service's GPU Whisper (`/asr`), with the CPU recogniser as fallback.

Same interface as LocalWhisperASR (VoxVortex's ASR protocol). Every utterance is read as Hindi and as English; the
returned text is what the assistant should see (see asr_gpu.combine), `last_readings` keeps both.
"""

from __future__ import annotations

import base64
import logging

import httpx

log = logging.getLogger(__name__)


class ServiceASR:
    def __init__(self, base_url: str, fallback=None, timeout: float = 20.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout, connect=2.0))
        self.fallback = fallback
        self.last_readings: dict[str, str] = {}
        self.last_language: str | None = None
        self.last_backend = "service"

    async def transcribe(self, pcm: bytes, *, language: str = "auto", prompt: str | None = None) -> str:
        try:
            r = await self._client.post("/asr", json={"pcm_b64": base64.b64encode(pcm).decode()})
            r.raise_for_status()
            d = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            if self.fallback is None:
                raise
            log.warning("GPU speech recognition unavailable (%s); using the CPU recogniser", exc)
            self.last_backend = "cpu"
            text = await self.fallback.transcribe(pcm, language=language, prompt=prompt)
            self.last_readings, self.last_language = {"cpu": text}, getattr(self.fallback, "last_language", None)
            return text
        self.last_backend = "service"
        self.last_readings, self.last_language = d.get("readings", {}), d.get("used")
        return d["text"]

    async def warm(self) -> None:
        if self.fallback is not None:
            await self.fallback.warm()

    async def aclose(self) -> None:
        await self._client.aclose()
