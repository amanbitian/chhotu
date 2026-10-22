"""Microphone capture and interruptible speaker playback (sounddevice)."""

from __future__ import annotations

import asyncio
import threading
from collections import deque

import numpy as np
import sounddevice as sd

from voxvortex.audio import FRAME_SAMPLES, SAMPLE_RATE, Reframer


def resolve_device(device: str | int | None, kind: str) -> int | None:
    """Accept an index, or a case-insensitive name fragment (e.g. 'Realtek'). None = system default."""
    if device in (None, ""):
        return None
    if isinstance(device, int) or str(device).isdigit():
        return int(device)
    needle = str(device).lower()
    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    for i, d in enumerate(sd.query_devices()):
        if d[channels] > 0 and needle in d["name"].lower():
            return i
    raise ValueError(f"No {kind} audio device matching {device!r}. Run `vox devices`.")


def list_devices() -> list[dict]:
    default_in, default_out = sd.default.device
    return [
        {"index": i, "name": d["name"], "inputs": d["max_input_channels"], "outputs": d["max_output_channels"],
         "default_in": i == default_in, "default_out": i == default_out, "hostapi": sd.query_hostapis(d["hostapi"])["name"]}
        for i, d in enumerate(sd.query_devices())
    ]


class Microphone:
    """Mono 16 kHz PCM16 capture, delivered as 32 ms frames (the size Silero expects)."""

    def __init__(self, device: str | int | None = None) -> None:
        self.device = resolve_device(device, "input")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[bytes] | None = None
        self._reframer = Reframer()
        self._stream: sd.RawInputStream | None = None

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=500)
        self._stream = sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=self.device,
                                         blocksize=FRAME_SAMPLES, callback=self._on_audio)
        self._stream.start()

    def _on_audio(self, data, frames, time_info, status) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._offer, bytes(data))

    def _offer(self, raw: bytes) -> None:
        for frame in self._reframer.feed(raw):
            if self._queue.full():
                self._queue.get_nowait()  # drop the oldest rather than fall behind real time
            self._queue.put_nowait(frame)

    async def frames(self):
        while True:
            yield await self._queue.get()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


OUT_RATE = 48000  # VoxCPM2's native rate, so the cloned voice is never resampled


class Resampler:
    """Linear-interpolation resampler for PCM16 that carries its phase and last sample across calls, so audio that
    arrives in pieces (streamed speech) has no seams at the joins."""

    def __init__(self, src: int, dst: int = OUT_RATE) -> None:
        self.step = src / dst
        self.pos = 1.0  # position of the next output sample; index 0 is the previous call's last sample
        self.prev = 0.0

    def __call__(self, pcm: bytes) -> bytes:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if not len(x):
            return b""
        sig = np.concatenate([[self.prev], x])
        t = np.arange(self.pos, len(x), self.step)
        i = t.astype(np.int64)
        f = t - i
        out = sig[i] * (1 - f) + sig[np.minimum(i + 1, len(x))] * f
        self.pos = (t[-1] + self.step - len(x)) if len(t) else self.pos - len(x)
        self.prev = float(x[-1])
        return np.clip(np.round(out), -32768, 32767).astype("<i2").tobytes()


class Speaker:
    """Plays PCM16 from a queue. `stop()` empties it, so playback ends within one audio block.

    One output stream at a fixed rate; other rates are resampled on the way in. (Reopening the stream at each new
    rate, as before, played whatever was still queued at the wrong speed: a 24 kHz chime followed by 48 kHz speech
    finished the chime at double speed.)"""

    def __init__(self, device: str | int | None = None) -> None:
        self.device = resolve_device(device, "output")
        self._buf: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._stream: sd.RawOutputStream | None = None
        self._rate = OUT_RATE
        self._resamplers: dict[int, Resampler] = {}

    def _open(self) -> None:
        if self._stream is None:
            self._stream = sd.RawOutputStream(samplerate=OUT_RATE, channels=1, dtype="int16", device=self.device,
                                              callback=self._fill, blocksize=1024)
            self._stream.start()

    def _fill(self, outdata, frames, time_info, status) -> None:
        need = frames * 2
        out = bytearray()
        with self._lock:
            while len(out) < need and self._buf:
                chunk = self._buf.popleft()
                take = need - len(out)
                if len(chunk) > take:
                    self._buf.appendleft(chunk[take:])
                    chunk = chunk[:take]
                out += chunk
        outdata[:] = bytes(out) + b"\x00" * (need - len(out))

    def play(self, pcm: bytes, rate: int) -> None:
        self._open()
        if rate != OUT_RATE:
            if self.pending_seconds == 0:  # a new piece of audio, not the continuation of one: start fresh
                self._resamplers.pop(rate, None)
            pcm = self._resamplers.setdefault(rate, Resampler(rate))(pcm)
        with self._lock:
            self._buf.append(pcm)

    @property
    def pending_seconds(self) -> float:
        with self._lock:
            return sum(len(c) for c in self._buf) / 2 / self._rate

    def stop(self) -> None:
        with self._lock:
            self._buf.clear()
        self._resamplers.clear()

    async def drain(self) -> None:
        """Wait until everything queued has been handed to the device."""
        while self.pending_seconds > 0:
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.15)  # the device's own buffer

    def close(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
