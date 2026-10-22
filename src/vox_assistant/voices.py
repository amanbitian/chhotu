"""Local voice profiles: versioned reference recordings with one active profile.

No PyTorch here, so the same module serves the assistant and the TTS service. A profile is a 16 kHz mono
reference clip plus its exact transcript (VoxCPM2 "Ultimate" cloning needs both). Profiles are plain files
under `<data>/voices/<name>_v<N>/`, so revoking one deletes the recording and its derived data.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REF_RATE = 16000
MIN_SECONDS = 3.0
MAX_SECONDS = 40.0
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class VoiceError(ValueError):
    """A recording or request that cannot become / identify a voice profile."""


@dataclass
class Quality:
    seconds: float
    rms_dbfs: float
    peak: float
    clipped_fraction: float
    noise_floor_dbfs: float
    snr_db: float
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _db(x: float) -> float:
    return 20.0 * float(np.log10(max(x, 1e-7)))


def clipped_fraction(audio: np.ndarray) -> float:
    """Share of samples at full scale. Measure it on the raw input: peak limiting hides clipping afterwards."""
    audio = np.asarray(audio, dtype=np.float32)
    return float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 0.0


def analyze(audio: np.ndarray, rate: int, clipped: float | None = None) -> Quality:
    """Cheap signal checks on mono float audio: length, level, clipping and a frame-based SNR estimate.

    `clipped` overrides the clipping measurement with one taken on the original (pre-limiter) recording."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    seconds = len(audio) / rate if rate else 0.0
    if not len(audio) or not np.isfinite(audio).all():
        return Quality(seconds, -120, 0, 0, -120, 0, errors=["The recording is empty or contains invalid samples."])
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    clipped = clipped_fraction(audio) if clipped is None else clipped
    hop = max(int(rate * 0.03), 1)
    n = len(audio) // hop
    frames = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1)) if n else np.array([rms])
    noise = float(np.percentile(frames, 10))
    speech = float(np.percentile(frames, 90))
    q = Quality(seconds, _db(rms), peak, clipped, _db(noise), _db(speech) - _db(noise))
    if seconds < MIN_SECONDS:
        q.errors.append(f"Too short ({seconds:.1f} s); need at least {MIN_SECONDS:.0f} s of speech.")
    if peak < 1e-3 or q.rms_dbfs < -55:
        q.errors.append("Silent or far too quiet; check the microphone input.")
    if clipped > 0.01:
        q.errors.append(f"Clipping on {clipped:.1%} of samples; lower the input gain and re-record.")
    elif clipped > 0.001:
        q.warnings.append("Slight clipping detected; a cleaner take may clone better.")
    if q.snr_db < 12 and not q.errors:
        q.errors.append(f"Background noise is too high (about {q.snr_db:.0f} dB signal-to-noise); record somewhere quieter.")
    elif q.snr_db < 20:
        q.warnings.append(f"Moderate background noise (about {q.snr_db:.0f} dB signal-to-noise).")
    if q.rms_dbfs < -34 and not q.errors:
        q.warnings.append("Quiet recording; raise the microphone level if you can.")
    if seconds > MAX_SECONDS:
        q.errors.append(f"Too long ({seconds:.0f} s); keep the reference under {MAX_SECONDS:.0f} s.")
    elif seconds < 15:
        q.warnings.append("Under 15 s of reference; 20-30 s usually clones more faithfully.")
    return q


def to_reference(audio: np.ndarray, rate: int) -> np.ndarray:
    """Mono, DC-free, 16 kHz, peak-limited. Speech and pauses are kept so the transcript stays aligned."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio - float(audio.mean())
    if rate != REF_RATE:
        g = gcd(rate, REF_RATE)
        audio = resample_poly(audio, REF_RATE // g, rate // g).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.98:
        audio *= 0.98 / peak
    return audio


def trim_silence(audio: np.ndarray, rate: int, pad_ms: int = 150, threshold_db: float = -42.0) -> np.ndarray:
    """Drop leading/trailing silence from one clip, keeping a short pad so words are not clipped."""
    hop = max(int(rate * 0.01), 1)
    n = len(audio) // hop
    if n == 0:
        return audio
    level = 20 * np.log10(np.maximum(np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1)), 1e-7))
    voiced = np.flatnonzero(level > threshold_db)
    if not len(voiced):
        return audio
    pad = int(rate * pad_ms / 1000)
    return audio[max(voiced[0] * hop - pad, 0) : min((voiced[-1] + 1) * hop + pad, len(audio))]


def join_clips(clips: list[np.ndarray], rate: int, gap_ms: int = 350) -> np.ndarray:
    gap = np.zeros(int(rate * gap_ms / 1000), dtype=np.float32)
    out: list[np.ndarray] = []
    for clip in clips:
        if out:
            out.append(gap)
        out.append(np.asarray(clip, dtype=np.float32))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


class VoiceProfiles:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- reading --------------------------------------------------------------------------
    def list(self) -> list[dict]:
        out = []
        for meta in sorted(self.root.glob("*/profile.json")):
            out.append(json.loads(meta.read_text(encoding="utf-8")))
        return sorted(out, key=lambda p: p["created_at"])

    def get(self, profile_id: str) -> dict:
        meta = self._dir(profile_id) / "profile.json"
        if not meta.is_file():
            raise VoiceError(f"No voice profile {profile_id!r}.")
        profile = json.loads(meta.read_text(encoding="utf-8"))
        profile["path"] = str(self._dir(profile_id) / "reference.wav")
        return profile

    def active_id(self) -> str | None:
        state = self.root / "state.json"
        if state.is_file():
            active = json.loads(state.read_text(encoding="utf-8")).get("active")
            if active and (self._dir(active) / "profile.json").is_file():
                return active
        return None

    def active(self) -> dict | None:
        active = self.active_id()
        return self.get(active) if active else None

    # -- writing --------------------------------------------------------------------------
    def create(
        self,
        name: str,
        audio: np.ndarray,
        rate: int,
        transcript: str,
        *,
        permission_confirmed: bool,
        activate: bool = True,
        source: str = "recording",
    ) -> dict:
        """Validate, convert and store a reference as `<name>_v<N>`. Raises VoiceError on a bad recording."""
        name = name.strip().lower()
        if not _NAME.match(name):
            raise VoiceError("Voice names use 1-32 lowercase letters, digits, '-' or '_'.")
        if not permission_confirmed:
            raise VoiceError("Confirm that you own this voice or have the speaker's permission to clone it.")
        transcript = " ".join(transcript.split())
        if not transcript:
            raise VoiceError("The exact transcript of the recording is required.")
        raw = np.asarray(audio, dtype=np.float32)
        raw_clip = clipped_fraction(raw.mean(axis=1) if raw.ndim > 1 else raw)
        reference = to_reference(audio, rate)
        quality = analyze(reference, REF_RATE, clipped=raw_clip)
        if not quality.ok:
            raise VoiceError(" ".join(quality.errors))
        version = 1 + max(
            (int(p["id"].rsplit("_v", 1)[1]) for p in self.list() if p["name"] == name), default=0
        )
        profile_id = f"{name}_v{version}"
        folder = self._dir(profile_id)
        folder.mkdir(parents=True)
        sf.write(str(folder / "reference.wav"), reference, REF_RATE, subtype="PCM_16")
        profile = {
            "id": profile_id,
            "name": name,
            "version": version,
            "transcript": transcript,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "permission_confirmed": True,
            "source": source,
            "quality": quality.as_dict(),
            "uuid": uuid.uuid4().hex,
        }
        (folder / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        if activate:
            self.set_active(profile_id)
        return profile

    def set_active(self, profile_id: str) -> None:
        self.get(profile_id)
        (self.root / "state.json").write_text(json.dumps({"active": profile_id}), encoding="utf-8")

    def revoke(self, profile_id: str) -> None:
        """Delete a profile's recording and derived data. If it was active, fall back to the newest remaining one."""
        self.get(profile_id)
        was_active = self.active_id() == profile_id
        shutil.rmtree(self._dir(profile_id))
        if was_active:
            remaining = self.list()
            state = self.root / "state.json"
            if remaining:
                state.write_text(json.dumps({"active": remaining[-1]["id"]}), encoding="utf-8")
            elif state.exists():
                state.unlink()

    def _dir(self, profile_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*_v\d+", profile_id or ""):
            raise VoiceError(f"Invalid voice profile id {profile_id!r}.")
        return self.root / profile_id
