"""Neutral (non-cloned) voice from Windows' built-in speech engine.

Used when the user wants a generic voice (other people present), and as the fallback when the cloned-voice
service is unavailable. The text travels in an environment variable, never inside the PowerShell source.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import wave
from pathlib import Path
from typing import AsyncIterator

_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($env:VOXA_SAPI_VOICE) { $s.SelectVoice($env:VOXA_SAPI_VOICE) }
$s.Rate = [int]$env:VOXA_SAPI_RATE
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($env:VOXA_SAPI_OUT, $fmt)
$s.Speak($env:VOXA_SAPI_TEXT)
$s.Dispose()
"""


_DEVANAGARI = re.compile("[ऀ-ॿ]")


class NeutralVoiceTTS:
    """`hindi`: an engine for text in Devanagari, since Windows here has no Hindi voice (installing the Hindi speech pack
    adds one; then pass its name as `hindi_voice`). Normally the voice service with a synthetic, non-cloned reference,
    so a neutral voice stays neutral in Hindi too. `sample_rate` follows whichever engine spoke last."""

    def __init__(self, voice: str | None = None, rate: int = 1, hindi=None, hindi_voice: str | None = None) -> None:
        self.voice, self.rate, self.hindi, self.hindi_voice = voice, rate, hindi, hindi_voice
        self.sample_rate = 22050

    @property
    def speaks_hindi(self) -> bool:
        return self.hindi is not None or bool(self.hindi_voice)

    async def synth(self, text: str) -> AsyncIterator[bytes]:
        if _DEVANAGARI.search(text):
            if self.hindi_voice:
                async for chunk in self._sapi(text, self.hindi_voice):
                    yield chunk
                return
            if self.hindi is not None:
                async for chunk in self.hindi.synth(text):
                    self.sample_rate = self.hindi.sample_rate
                    yield chunk
                return
        async for chunk in self._sapi(text, self.voice):
            yield chunk

    async def _sapi(self, text: str, voice: str | None) -> AsyncIterator[bytes]:
        self.sample_rate = 22050
        fd, out = tempfile.mkstemp(suffix=".wav", prefix="voxa_")
        os.close(fd)
        env = {**os.environ, "VOXA_SAPI_TEXT": text, "VOXA_SAPI_OUT": out, "VOXA_SAPI_RATE": str(self.rate),
               "VOXA_SAPI_VOICE": voice or ""}
        try:
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-NoProfile", "-NonInteractive", "-Command", _SCRIPT, env=env,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            _, err = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Windows speech synthesis failed: {err.decode('utf-8', 'replace')[:300]}")
            with wave.open(out, "rb") as w:
                pcm = w.readframes(w.getnframes())
            for i in range(0, len(pcm), 8192):
                yield pcm[i : i + 8192]
        finally:
            Path(out).unlink(missing_ok=True)

    async def aclose(self) -> None:
        pass
