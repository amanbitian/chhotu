import sys
import time

import httpx
import pytest

import main as launcher
from vox_assistant.services import ServiceManager
from vox_assistant.tasks import TaskManager
from vox_assistant.webapp import create_app

from conftest import tone
from test_webapp import ScriptedLLM, tts_stub, wav_b64


def wait_for(cond, seconds=10.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def manager(settings):
    m = ServiceManager(settings)
    # Harmless stand-ins for the real processes: one that prints and waits, one that fails at once.
    m.services["llm"].command = [sys.executable, "-u", "-c", "print('hello from llm'); import time; time.sleep(60)"]
    m.services["llm"].health_url = None
    m.services["voice"].command = [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]
    m.services["voice"].health_url = "http://127.0.0.1:9/health"  # nothing listens there
    yield m
    m.shutdown()


def test_start_stop_and_log_capture(manager):
    assert manager.state("llm")["state"] == "stopped"
    assert manager.start("llm")["state"] in ("starting", "running")
    assert wait_for(lambda: "hello from llm" in manager.logs("llm"))
    assert manager.state("llm")["state"] == "running" and manager.state("llm")["pid"]
    assert manager.stop("llm")["state"] == "stopped"


def test_structured_hands_free_events_are_separate_from_human_logs(manager):
    manager.start("llm")
    manager.services["listen"].command = [
        sys.executable, "-u", "-c",
        "import time; print('@@VOXA_EVENT@@{\"kind\":\"request\",\"text\":\"hello\"}'); "
        "print('you: hello'); time.sleep(60)",
    ]
    manager.start("listen")
    assert wait_for(lambda: bool(manager.events("listen")["events"]))
    data = manager.events("listen")
    assert data["events"][0]["kind"] == "request" and data["events"][0]["text"] == "hello"
    assert "you: hello" in manager.logs("listen")
    assert not any("VOXA_EVENT" in line for line in manager.logs("listen"))
    assert manager.events("listen", after=data["events"][0]["seq"], run_id=data["run_id"])["events"] == []


def test_a_service_that_exits_with_an_error_is_reported_failed(manager):
    manager.start("voice")
    assert wait_for(lambda: manager.state("voice")["state"] == "failed")
    assert manager.state("voice")["exit_code"] == 3 and "boom" in manager.logs("voice")


def test_start_is_idempotent_and_hands_free_needs_the_llm(manager, monkeypatch):
    with pytest.raises(RuntimeError, match="first"):
        manager.start("listen")
    manager.start("llm")
    pid = manager.state("llm")["pid"]
    assert manager.start("llm")["pid"] == pid

    class Reply:
        status_code = 200

        @staticmethod
        def json():
            return {"api_version": launcher.API_VERSION}

    with monkeypatch.context() as patch:
        patch.setattr(launcher.httpx, "get", lambda *args, **kwargs: Reply())
        assert launcher.compatible_panel("http://127.0.0.1:8765")
        patch.setattr(launcher, "port_in_use", lambda port: port < 8768)
        assert launcher.next_available_port(8765) == 8768


def test_an_already_running_service_is_external_and_not_controllable(settings):
    m = ServiceManager(settings)
    m._healthy = lambda svc: svc.name == "llm"
    s = m.state("llm")
    assert s["state"] == "external" and not s["controllable"]
    assert m.start("llm")["state"] == "external"


@pytest.fixture
async def client(settings, manager):
    app = create_app(settings, llm=ScriptedLLM(), tts_transport=tts_stub(), services=manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765") as c:
        yield c


async def test_services_api_start_logs_stop(client):
    assert (await client.get("/api/version")).json()["api_version"] == launcher.API_VERSION
    assert {s["name"] for s in (await client.get("/api/services")).json()["services"]} == {"llm", "voice", "listen"}
    assert (await client.post("/api/services/llm/start", json={})).json()["state"] in ("starting", "running")
    assert (await client.post("/api/services/nope/start", json={})).status_code == 404
    assert (await client.post("/api/services/llm/explode", json={})).status_code == 404
    assert wait_for(lambda: "hello from llm" in client._transport.app.state.services.logs("llm"))
    assert "hello from llm" in (await client.get("/api/services/llm/logs")).json()["lines"]
    assert (await client.get("/api/services/listen/events")).json()["events"] == []
    assert (await client.post("/api/services/llm/stop", json={})).json()["state"] == "stopped"


async def test_hands_free_without_llm_is_a_clear_409(client):
    r = await client.post("/api/services/listen/start", json={"always": True, "language": "hi"})
    assert r.status_code == 409 and "first" in r.json()["detail"]


async def test_service_control_needs_json(client):
    assert (await client.post("/api/services/llm/stop", content="x", headers={"content-type": "text/plain"})).status_code == 415


async def test_browser_enrollment_end_to_end(client):
    d = (await client.post("/api/enroll/start", json={"language": "en"})).json()
    assert len(d["script"]) == 8 and not d["ready"]
    sid = d["session"]
    for i in range(3):  # 3 x 8 s of tone-speech is past the 20 s minimum
        t = (await client.post("/api/enroll/take", json={"session": sid, "index": i, "wav_base64": wav_b64(8.0)})).json()
        assert t["accepted"], t
    assert t["ready"] and t["taken"] == [0, 1, 2]
    no = await client.post("/api/enroll/finish", json={"session": sid, "name": "me", "permission_confirmed": False})
    assert no.status_code == 400 and "permission" in no.json()["detail"]
    p = (await client.post("/api/enroll/finish", json={"session": sid, "name": "me", "permission_confirmed": True})).json()
    assert p["id"] == "me_v1" and p["language"] == "en" and p["transcript"].startswith("Good morning.")
    assert (await client.get("/api/voices")).json()["active"] == "me_v1"
    assert (await client.post("/api/enroll/take", json={"session": sid, "index": 0, "wav_base64": wav_b64()})).status_code == 404


async def test_enrollment_rejects_a_silent_take_and_a_hindi_session_uses_the_hindi_script(client):
    import base64
    import io

    import numpy as np
    import soundfile as sf

    d = (await client.post("/api/enroll/start", json={"language": "hi"})).json()
    assert d["language"] == "hi" and d["script"][0].startswith("नमस्ते")
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, np.float32), 16000, format="WAV", subtype="PCM_16")
    t = (await client.post("/api/enroll/take", json={"session": d["session"], "index": 0,
                                                       "wav_base64": base64.b64encode(buf.getvalue()).decode()})).json()
    assert not t["accepted"] and t["errors"] and t["taken"] == []
    bad = await client.post("/api/enroll/take", json={"session": d["session"], "index": 99, "wav_base64": wav_b64()})
    assert bad.status_code == 404
    junk = await client.post("/api/enroll/take", json={"session": d["session"], "index": 0, "wav_base64": "bm90IGEgd2F2"})
    assert junk.status_code == 400


async def test_transcribe_sends_16k_pcm_to_the_recogniser(settings, manager, monkeypatch):
    from vox_assistant.asr_remote import ServiceASR

    seen = {}

    async def fake(self, pcm, *, language="auto", prompt=None):
        seen["bytes"] = len(pcm)
        return " open notepad "

    monkeypatch.setattr(ServiceASR, "transcribe", fake)
    app = create_app(settings, llm=ScriptedLLM(), tts_transport=tts_stub(), services=manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765") as c:
        r = await c.post("/api/transcribe", json={"wav_base64": wav_b64(2.0)})
        short = await c.post("/api/transcribe", json={"wav_base64": wav_b64(0.1)})
    assert r.status_code == 200 and r.json() == {"text": "open notepad"}
    assert seen["bytes"] == 2 * 32000  # 2 s of 16 kHz PCM16
    assert short.status_code == 400


async def test_voice_check_proxies_the_similarity_score(settings, manager):
    def handler(request):
        assert request.url.path == "/voices/me_v1/similarity"
        return httpx.Response(200, json={"voice": "me_v1", "similarity": 0.91, "threshold": 0.86, "same_speaker": True})

    app = create_app(settings, llm=ScriptedLLM(), tts_transport=httpx.MockTransport(handler), services=manager)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765") as c:
        assert (await c.post("/api/voices/me_v1/check", json={})).json()["same_speaker"] is True


def test_tracked_task_captures_progress_and_completion(settings, tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr("vox_assistant.tasks.shutil.which", lambda _: r"C:\Tools\opencode.CMD")
        command = TaskManager(settings)._command('fix "tests" & report', allow_changes=True)
        assert command == [r"C:\Tools\opencode.CMD", "run", "--auto", "fix tests report"]
    project = tmp_path / "SampleProject"
    project.mkdir()
    settings.coding_agent = [sys.executable, "-u", "-c",
                             "import sys,time; print('analysing', flush=True); time.sleep(.1); print(sys.argv[1], flush=True)",
                             "{task}"]
    tasks = TaskManager(settings)
    job = tasks.start(project, "fix the tests")
    assert job["status"] == "running" and job["project"] == "SampleProject"
    assert wait_for(lambda: tasks.state(job["id"])["status"] == "completed")
    done = tasks.state(job["id"])
    assert done["exit_code"] == 0 and "analysing" in done["output"] and "fix the tests" in done["output"]


def test_tracked_task_can_be_stopped(settings, tmp_path):
    project = tmp_path / "SampleProject"
    project.mkdir()
    settings.coding_agent = [sys.executable, "-u", "-c", "import time; print('working', flush=True); time.sleep(60)"]
    tasks = TaskManager(settings)
    job = tasks.start(project, "wait")
    assert wait_for(lambda: "working" in tasks.state(job["id"])["output"])
    assert tasks.stop(job["id"])["status"] == "cancelled"


async def test_tasks_api_lists_projects_starts_and_reports_output(settings, manager, tmp_path):
    root = tmp_path / "projects"
    project = root / "DemoApp"
    project.mkdir(parents=True)
    settings.search_roots = [str(root)]
    settings.coding_agent = [sys.executable, "-u", "-c", "print('task output', flush=True)"]
    tasks = TaskManager(settings)
    app = create_app(settings, llm=ScriptedLLM(), tts_transport=tts_stub(), services=manager, tasks=tasks)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765") as c:
        assert (await c.get("/api/projects")).json()["projects"][0]["name"] == "DemoApp"
        bad = await c.post("/api/tasks", json={"project": "missing", "task": "do it"})
        assert bad.status_code == 404
        started = (await c.post("/api/tasks", json={"project": "demo app", "task": "do it"})).json()
        assert wait_for(lambda: tasks.state(started["id"])["status"] == "completed")
        detail = (await c.get(f"/api/tasks/{started['id']}")).json()
        assert detail["status"] == "completed" and "task output" in detail["output"]
        assert (await c.get("/api/tasks")).json()["tasks"][0]["id"] == started["id"]
