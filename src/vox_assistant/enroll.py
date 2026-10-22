"""Guided voice enrollment: read a short script, keep clean takes, join them into one reference + exact transcript."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .voices import REF_RATE, Quality, VoiceProfiles, analyze, clipped_fraction, join_clips, to_reference, trim_silence

# Conversational, a question, numbers, names and technical words. The script is also the exact transcript.
SCRIPT = [
    "Good morning. I'm checking in on a few things before I start the day.",
    "Could you tell me which model is running on each graphics card right now?",
    "The temperature looks fine, about sixty-five degrees, and the fan is quiet.",
    "Please open the project folder and look for the latest failed extraction.",
    "I found the error, and it looks like a missing field in the validator.",
    "Remind me in thirty minutes to check on the print, and then summarize my notes.",
    "That's great news, thank you so much. Let's try that again, a little slower this time.",
    "Honestly, I wasn't sure it would work, but it sounds exactly like I'd say it.",
]
TARGET_SECONDS = (20.0, 36.0)  # stop once past the low end; never exceed the high end


@dataclass
class Take:
    text: str
    audio: np.ndarray  # 16 kHz float32, trimmed
    quality: Quality

    @property
    def usable(self) -> bool:
        # A single line is naturally short; only the whole reference has a minimum length.
        return all("short" in e.lower() for e in self.quality.errors)


def assess_take(text: str, raw: np.ndarray, rate: int) -> Take:
    audio = trim_silence(to_reference(raw, rate), REF_RATE)
    return Take(text, audio, analyze(audio, REF_RATE, clipped=clipped_fraction(raw)))


def assemble(takes: list[Take]) -> tuple[np.ndarray, str]:
    """Join accepted takes (script order) into a single reference and its transcript."""
    audio = join_clips([t.audio for t in takes], REF_RATE, gap_ms=350)
    return audio, " ".join(t.text for t in takes)


def record_take(frames, vad, *, max_seconds: float = 14.0, end_silence_ms: int = 1100, wait_seconds: float = 12.0):
    """Collect 32 ms PCM16 frames until speech has started and then stopped. Returns float32 audio at 16 kHz."""
    kept: list[bytes] = []
    started = False
    silence = 0
    t0 = time.monotonic()
    vad.reset()
    for frame in frames:
        p = vad.prob(frame)
        kept.append(frame)
        if p > 0.5:
            started, silence = True, 0
        elif started:
            silence += 32
            if silence >= end_silence_ms:
                break
        if not started:
            kept = kept[-20:]  # keep about 0.6 s of pre-roll only
            if time.monotonic() - t0 > wait_seconds:
                break
        elif len(kept) * 32 / 1000 > max_seconds:
            break
    if not started:
        return np.zeros(0, np.float32)
    return np.frombuffer(b"".join(kept), dtype=np.int16).astype(np.float32) / 32768.0


def run_interactive(profiles: VoiceProfiles, name: str, input_device, silero_path: Path) -> dict:
    """Terminal enrollment. Needs a microphone; the decisions live in the testable helpers above."""
    import queue

    import sounddevice as sd
    from voxvortex.audio import FRAME_SAMPLES, SAMPLE_RATE, SileroVAD

    from .audio_io import resolve_device

    vad = SileroVAD(silero_path)
    q: queue.Queue[bytes] = queue.Queue()
    stream = sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_SAMPLES,
                               device=resolve_device(input_device, "input"),
                               callback=lambda data, frames, t, status: q.put(bytes(data)))

    def frames():
        while True:
            yield q.get()

    print(f"\nEnrolling voice '{name}'. Speak in your normal voice, in a quiet room, at a steady distance from the mic.")
    print("For each line: press Enter, then read it aloud naturally. Recording stops when you pause.\n")
    takes: list[Take] = []
    with stream:
        for text in SCRIPT:
            total = sum(len(t.audio) for t in takes) / REF_RATE
            if total >= TARGET_SECONDS[0]:
                break
            while True:
                print(f"  [{len(takes) + 1}/{len(SCRIPT)}]  \"{text}\"")
                input("  Press Enter, then speak... ")
                while not q.empty():
                    q.get_nowait()
                raw = record_take(frames(), vad)
                if not len(raw):
                    print("  No speech heard - check the mic and try again.\n")
                    continue
                take = assess_take(text, raw, SAMPLE_RATE)
                for w in take.quality.warnings:
                    print(f"  note: {w}")
                if take.usable and (total + take.quality.seconds <= TARGET_SECONDS[1] or not takes):
                    takes.append(take)
                    print(f"  kept ({take.quality.seconds:.1f} s, SNR {take.quality.snr_db:.0f} dB)\n")
                    break
                print("  rejected: " + " ".join(take.quality.errors) + "\n  Try that line again.\n")
    audio, transcript = assemble(takes)
    print(f"Joined {len(takes)} takes into {len(audio) / REF_RATE:.1f} s of reference.")
    answer = input("I confirm this is my own voice, or I have the speaker's permission to clone it (yes/no): ").strip().lower()
    return profiles.create(name, audio, REF_RATE, transcript, permission_confirmed=answer in ("y", "yes"),
                           source="guided enrollment")
