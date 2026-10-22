"""Client for tts_service.py, shaped like VoxVortex's `TTS` protocol (`sample_rate`, `synth(text)`)."""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from voxvortex.tts import normalize_for_speech


class ClonedVoiceTTS:
    speaks_hindi = True  # VoxCPM2 speaks Hindi; the service prepares Hindi text (speech_text) itself

    def __init__(self, base_url: str, *, sample_rate: int = 48000, voice: str | None = None,
                 preset: str | None = None, timeout: float = 120.0, normalize: bool = False) -> None:
        # normalize: VoxVortex's English normalizer, off by default: it mangles names ("R9700") and Hindi numbers, and
        # the service now prepares text per language (speech_text).
        self.sample_rate = sample_rate  # VoxCPM2's native rate; updated from the first response header
        self.voice, self.preset, self.normalize = voice, preset, normalize
        self._client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(timeout, connect=3.0))

    async def synth(self, text: str) -> AsyncIterator[bytes]:
        body: dict = {"input": normalize_for_speech(text) if self.normalize else text, "response_format": "pcm"}
        if self.voice:
            body["voice"] = self.voice
        if self.preset:
            body["preset"] = self.preset
        carry = b""
        async with self._client.stream("POST", "/audio/speech", json=body) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode("utf-8", "replace")[:300]
                raise RuntimeError(f"TTS service returned HTTP {resp.status_code}: {detail}")
            self.sample_rate = int(resp.headers.get("x-sample-rate", self.sample_rate))
            async for chunk in resp.aiter_bytes(8192):
                chunk = carry + chunk
                cut = len(chunk) - (len(chunk) % 2)
                carry = chunk[cut:]
                if cut:
                    yield chunk[:cut]

    async def warm(self) -> None:
        """Ask the service to touch its weights (pages them back into VRAM if Windows moved them out)."""
        try:
            await self._client.post("/warm", json={}, timeout=30.0)
        except httpx.HTTPError:
            pass

    async def health(self) -> dict:
        resp = await self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()
