import numpy as np
import pytest

from vox_assistant.config import Settings


def tone(seconds=8.0, rate=16000, amp=0.3, noise=0.002, seed=0):
    """Syllable-like bursts over low noise: enough level contrast for the SNR estimate."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    envelope = (np.sin(2 * np.pi * 3 * t) > -0.3).astype(np.float32)
    return (amp * envelope * np.sin(2 * np.pi * 180 * t) + noise * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture
def settings(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    return s
