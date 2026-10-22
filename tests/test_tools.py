import os
import time
from pathlib import Path

import pytest

from vox_assistant.permissions import Risk
from vox_assistant.tools import build_registry
from vox_assistant.tools.apps import match_app
from vox_assistant.tools.files import inside_roots, search
from vox_assistant.tools.reminders import ReminderStore
from vox_assistant.tools.system import summarize_gpus


def test_registry_risk_levels(settings):
    reg, _ = build_registry(settings)
    risks = {n: reg.get(n).risk for n in reg.names()}
    assert risks["gpu_status"] is Risk.READ and risks["find_files"] is Risk.READ
    assert risks["open_app"] is Risk.ACT and risks["set_clipboard"] is Risk.ACT and risks["set_reminder"] is Risk.ACT
    assert all(s["type"] == "function" and s["function"]["parameters"]["type"] == "object" for s in reg.specs())


def test_gpu_summary_joins_telemetry_and_models():
    hw = {"hardware": {"gpus": [
        {"name": "R9700", "lane": "r9700", "vram_total_bytes": 32 * 1024**3,
         "latest": {"temp_junction_c": 71.0, "power_w": 250.0, "gpu_busy_pct": 90, "vram_used_bytes": 20 * 1024**3}},
        {"name": "9070 XT", "lane": "9070xt", "vram_total_bytes": 16 * 1024**3,
         "latest": {"temp_junction_c": 95.0, "power_w": None, "vram_used_bytes": 1024**3}},
    ]}}
    st = {"servers": [{"target": "r9700", "lane_keys": ["r9700"], "model": "Qwen", "alive": True, "context_window": 4096}]}
    a, b = summarize_gpus(hw, st, 90)
    assert a["model"] == "Qwen" and a["vram_used_gb"] == 20.0 and a["within_temp_limit"] is True
    assert b["model"] is None and b["within_temp_limit"] is False and b["power_w"] is None


def test_file_search_words_newest_first_and_skips(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "eval_report.txt").write_text("x")
    old = tmp_path / "eval_report_old.txt"
    new = tmp_path / "sub" / "eval_report_new.txt"
    new.parent.mkdir()
    old.write_text("x")
    new.write_text("x")
    os.utime(old, (time.time() - 5000,) * 2)
    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / "token_eval_report.txt").write_text("x")
    names = [Path(h["path"]).name for h in search([tmp_path], "eval report")]
    assert names == ["eval_report_new.txt", "eval_report_old.txt"]
    assert search([tmp_path], "env") == []
    assert search([tmp_path], "   ") == []


def test_roots_confine_paths(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    assert inside_roots(root / "a.txt", [root.resolve()])
    assert not inside_roots(root / ".." / "other.txt", [root.resolve()])
    assert not inside_roots(Path("C:/Windows/system32/cmd.exe"), [root.resolve()])


async def test_read_text_file_rules(settings, tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "notes.txt").write_text("hello")
    (root / "secrets.txt").write_text("hunter2")
    settings.search_roots = [str(root)]
    reg, _ = build_registry(settings)
    read = reg.get("read_text_file").handler
    good = await read({"path": str(root / "notes.txt")})
    assert good["ok"] and good["content"] == "hello" and good["untrusted_content"]
    assert not (await read({"path": str(root / "secrets.txt")}))["ok"]
    assert not (await read({"path": str(tmp_path / "outside.txt")}))["ok"]


def test_app_matching_uses_the_allowlist_only():
    apps = {"vs code": "code", "firefox": "firefox", "file explorer": "explorer"}
    assert match_app("VS Code", apps) == ("vs code", "code")
    assert match_app("open firefox please", apps) == ("firefox", "firefox")
    assert match_app("powershell", apps) is None
    assert match_app("", apps) is None


async def test_open_app_refuses_unknown_names(settings):
    reg, _ = build_registry(settings)
    result = await reg.get("open_app").handler({"name": "format c:"})
    assert result["ok"] is False and "only know" in result["error"]


async def test_open_url_rejects_non_http(settings):
    reg, _ = build_registry(settings)
    assert not (await reg.get("open_url").handler({"url": "file:///c:/windows/win.ini"}))["ok"]
    assert not (await reg.get("open_url").handler({"url": "javascript:alert(1)"}))["ok"]


def test_reminders_fire_once_in_order(tmp_path):
    now = [1000.0]
    store = ReminderStore(tmp_path / "r.sqlite3", clock=lambda: now[0])
    a = store.add("first", 1)
    b = store.add("second", 5)
    assert [r["text"] for r in store.pending()] == ["first", "second"]
    assert store.due() == []
    now[0] += 90
    assert [r["text"] for r in store.due()] == ["first"]
    assert store.due() == []
    assert store.cancel(b["id"]) and not store.cancel(b["id"])
    now[0] += 3600
    assert store.due() == [] and store.pending() == []


async def test_set_reminder_validates(settings):
    reg, store = build_registry(settings)
    h = reg.get("set_reminder").handler
    assert not (await h({"text": "x", "in_minutes": "soon"}))["ok"]
    assert not (await h({"text": "", "in_minutes": 5}))["ok"]
    assert not (await h({"text": "x", "in_minutes": 10**7}))["ok"]
    good = await h({"text": "check the print", "in_minutes": 30})
    assert good["ok"] and good["verified_stored"]
