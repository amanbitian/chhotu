"""Regression tests for gaps found in review: enrollment logic, request guard on the TTS service, executable
refusal, reminder delivery over HTTP, and wake-phrase spellings."""

import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vox_assistant.enroll import SCRIPT, assemble, assess_take, record_take
from vox_assistant.tts_service import create_app as create_tts
from vox_assistant.tools import build_registry
from vox_assistant.voice import strip_wake_phrase
from vox_assistant.voices import REF_RATE
from vox_assistant.webapp import create_app as create_web

from conftest import tone


# -- enrollment ---------------------------------------------------------------------------------
def test_take_assessment_accepts_a_short_clean_line_and_rejects_bad_ones():
    clean = np.concatenate([np.zeros(16000), tone(4.0), np.zeros(16000)])
    take = assess_take("hello", clean, 16000)
    assert take.usable and 4.0 < take.quality.seconds < 5.0  # padding trimmed, but a short line is not "too short"
    assert not assess_take("x", np.clip(tone(4.0, amp=3.0), -1, 1), 16000).usable
    assert not assess_take("x", tone(4.0, noise=0.2), 16000).usable
    assert not assess_take("x", np.zeros(32000, dtype=np.float32), 16000).usable


def test_assemble_joins_takes_in_order_with_gaps_and_transcript():
    a, b = assess_take("one", tone(3.0), 16000), assess_take("two", tone(3.0, seed=1), 16000)
    audio, text = assemble([a, b])
    assert text == "one two"
    assert len(audio) == len(a.audio) + len(b.audio) + int(REF_RATE * 0.35)


class StubVAD:
    def __init__(self):
        self.resets = 0

    def prob(self, frame):
        return frame[0] / 255

    def reset(self):
        self.resets += 1


def test_record_take_stops_after_speech_then_silence_and_ignores_leading_quiet():
    frames = [b"\x00" * 1024] * 50 + [b"\xff" * 1024] * 40 + [b"\x00" * 1024] * 60 + [b"\xff" * 1024] * 10
    audio = record_take(iter(frames), StubVAD())
    seconds = len(audio) / 16000
    assert 1.2 < seconds < 3.2  # pre-roll + speech + the silence that ended it; not the late burst


def test_record_take_returns_nothing_without_speech():
    assert len(record_take(iter([b"\x00" * 1024] * 400), StubVAD(), wait_seconds=0.0)) == 0


def test_enrollment_script_is_long_enough_for_a_reference():
    assert len(SCRIPT) >= 6 and all(len(line) > 20 for line in SCRIPT)


# -- request guard on the TTS service -----------------------------------------------------------
def test_tts_service_rejects_foreign_hosts_origins_and_form_posts(settings):
    c = TestClient(create_tts(settings, backend_factory=lambda: None))
    assert c.get("/health").status_code == 200
    assert c.get("/health", headers={"host": "evil.example"}).status_code == 403
    assert c.post("/unload", json={}, headers={"origin": "https://evil.example"}).status_code == 403
    assert c.post("/unload").status_code == 415  # body-less POST: a browser can send this cross-site without a preflight
    assert c.post("/forget", json={}).status_code == 200


# -- tools / web --------------------------------------------------------------------------------
async def test_open_path_refuses_programs_even_inside_the_allowed_roots(settings, tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    for name in ("run.bat", "tool.exe", "x.ps1", "shortcut.lnk", "m.py"):
        (root / name).write_text("echo hi")
    settings.search_roots = [str(root)]
    reg, _ = build_registry(settings)
    for name in ("run.bat", "tool.exe", "x.ps1", "shortcut.lnk", "m.py"):
        r = await reg.get("open_path").handler({"path": str(root / name)})
        assert r["ok"] is False and "run a program" in r["error"]


async def test_due_reminders_are_delivered_once_over_http(settings):
    app = create_web(settings, llm=object())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765") as c:
        app.state.reminders.add("stretch", 0.0001)
        time.sleep(0.05)
        assert [r["text"] for r in (await c.get("/api/reminders/due")).json()["due"]] == ["stretch"]
        assert (await c.get("/api/reminders/due")).json()["due"] == []


# -- wake phrase --------------------------------------------------------------------------------
def test_wake_phrase_accepts_known_asr_spellings():
    phrases = ["hey vox", "hey box", "hey vaux"]
    assert strip_wake_phrase("Hey, box. What's the GPU temperature?", phrases) == "what's the gpu temperature"
    assert strip_wake_phrase("Hey Vaux", phrases) == ""
    assert strip_wake_phrase("hey fax open notepad", phrases) is None
    assert strip_wake_phrase("they box of things", phrases) is None
