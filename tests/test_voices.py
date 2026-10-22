import numpy as np
import pytest
import soundfile as sf

from vox_assistant.voices import REF_RATE, VoiceError, VoiceProfiles, analyze, join_clips, to_reference, trim_silence

from conftest import tone


def make(profiles, name="me", **kw):
    return profiles.create(name, tone(), 16000, "hello there", permission_confirmed=True, **kw)


def test_versions_increment_and_latest_is_active(tmp_path):
    p = VoiceProfiles(tmp_path)
    assert make(p)["id"] == "me_v1"
    assert make(p)["id"] == "me_v2"
    assert make(p, "other")["id"] == "other_v1"
    assert p.active_id() == "other_v1"
    p.set_active("me_v1")
    assert p.active()["id"] == "me_v1"
    assert sf.info(p.get("me_v1")["path"]).samplerate == REF_RATE


def test_not_activating_keeps_previous_active(tmp_path):
    p = VoiceProfiles(tmp_path)
    make(p)
    make(p, activate=False)
    assert p.active_id() == "me_v1"


def test_revoke_deletes_files_and_falls_back(tmp_path):
    p = VoiceProfiles(tmp_path)
    make(p)
    second = make(p)
    p.revoke(second["id"])
    assert not (tmp_path / "me_v2").exists()
    assert p.active_id() == "me_v1"
    p.revoke("me_v1")
    assert p.active() is None and p.list() == []


def test_requires_permission_transcript_and_valid_name(tmp_path):
    p = VoiceProfiles(tmp_path)
    with pytest.raises(VoiceError, match="permission"):
        p.create("me", tone(), 16000, "hi", permission_confirmed=False)
    with pytest.raises(VoiceError, match="transcript"):
        p.create("me", tone(), 16000, "  ", permission_confirmed=True)
    with pytest.raises(VoiceError, match="names"):
        p.create("../evil", tone(), 16000, "hi", permission_confirmed=True)
    assert p.list() == []


@pytest.mark.parametrize("bad", ["../x", "me", "me_v", "ME_v1", "a/b_v1"])
def test_profile_ids_cannot_escape_the_voice_directory(tmp_path, bad):
    with pytest.raises(VoiceError):
        VoiceProfiles(tmp_path).get(bad)


def test_resamples_to_reference_rate(tmp_path):
    p = VoiceProfiles(tmp_path)
    prof = p.create("hi", tone(rate=48000), 48000, "x", permission_confirmed=True)
    assert abs(prof["quality"]["seconds"] - 8.0) < 0.05


def test_quality_rejects_bad_recordings():
    assert not analyze(np.zeros(16000 * 5, dtype=np.float32), 16000).ok
    assert any("short" in e.lower() for e in analyze(tone(1.0), 16000).errors)
    assert any("Clipping" in e for e in analyze(np.clip(tone(amp=3.0), -1, 1), 16000).errors)
    noisy = tone(noise=0.2, amp=0.3)
    assert any("noise" in e.lower() for e in analyze(noisy, 16000).errors)
    assert analyze(tone(), 16000).ok


def test_trim_and_join():
    clip = np.concatenate([np.zeros(16000), tone(1.0), np.zeros(16000)])
    trimmed = trim_silence(clip, 16000)
    assert 16000 < len(trimmed) < 16000 * 1.5
    joined = join_clips([trimmed, trimmed], 16000, gap_ms=500)
    assert len(joined) == 2 * len(trimmed) + 8000
    assert to_reference(np.ones(100) * 5, 16000).max() <= 0.98 + 1e-6


def test_clipping_is_detected_even_though_the_reference_is_peak_limited(tmp_path):
    loud = np.clip(tone(amp=3.0), -1, 1)
    with pytest.raises(VoiceError, match="Clipping"):
        VoiceProfiles(tmp_path).create("me", loud, 16000, "hello", permission_confirmed=True)
