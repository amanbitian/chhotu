"""Neutral (non-cloned) voice from Windows' built-in speech engine.

Used when the user wants a generic voice (other people present), and as the fallback when the cloned-voice
service is unavailable. The text travels in an environment variable, never inside the PowerShell source.
"""

from __future__ import annotations

import asyncio
import os
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


class NeutralVoiceTTS:
    sample_rate = 22050

    def __init__(self, voice: str | None = None, rate: int = 1) -> None:
        self.voice, self.rate = voice, rate

    async def synth(self, text: str) -> AsyncIterator[bytes]:
        fd, out = tempfile.mkstemp(suffix=".wav", prefix="voxa_")
        os.close(fd)
        env = {**os.environ, "VOXA_SAPI_TEXT": text, "VOXA_SAPI_OUT": out, "VOXA_SAPI_RATE": str(self.rate),
               "VOXA_SAPI_VOICE": self.voice or ""}
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
