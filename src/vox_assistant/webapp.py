"""Local web app: type-and-generate in your voice, chat with the agent, manage voices, read the audit log.

Binds to 127.0.0.1 only. Because it can drive the PC, every request must carry a localhost Host header (blocks DNS
rebinding) and any Origin must be this same server (blocks other web pages from posting to it).
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import io
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from .llm_client import ToolChoiceLLM

from .agent import Agent, Say, ToolFinished, ToolStarted
from .audit import AuditLog
from . import language
from .config import Settings
from .disclosure import labelled_wav
from .enroll import SCRIPT, SCRIPT_HI, TARGET_SECONDS, Take, assemble, assess_take
from .permissions import PermissionGate
from .security import install_local_only
from .services import ServiceManager
from .tasks import TaskBusy, TaskManager
from .tools import build_registry
from .tools.projects import list_projects, match_project
from .tools.reminders import is_late, show_toast
from .tts_neutral import NeutralVoiceTTS
from .voices import REF_RATE, VoiceError, VoiceProfiles, to_reference

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
CONFIRM_TIMEOUT_S = 90.0
API_VERSION = 4  # main.py checks it to tell this panel from an older one still running
REMINDER_POLL_S = 3.0


class SpeakBody(BaseModel):
    text: str
    voice_mode: str = "cloned"  # or "neutral"
    preset: str | None = None


class ChatBody(BaseModel):
    text: str
    language: str | None = None  # auto / en / hi / hinglish; None = setting `language`


class ConfirmBody(BaseModel):
    id: str
    approve: bool


class ServiceBody(BaseModel):
    always: bool = False  # hands-free: treat every utterance as a request (no wake phrase)
    language: str | None = None


class EnrollStart(BaseModel):
    language: str = "en"


class EnrollTake(BaseModel):
    session: str
    index: int  # which script line was read
    wav_base64: str


class EnrollFinish(BaseModel):
    session: str
    name: str
    permission_confirmed: bool


class Recording(BaseModel):
    wav_base64: str


class TaskBody(BaseModel):
    project: str
    task: str
    allow_changes: bool = False


class VoiceUpload(BaseModel):
    name: str
    transcript: str
    permission_confirmed: bool
    wav_base64: str
    language: str = "en"  # what the recording is spoken in: "en" or "hi"


def read_wav(b64: str) -> tuple[np.ndarray, int]:
    """A browser recording (base64 WAV) as mono float32 and its sample rate."""
    try:
        audio, rate = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32", always_2d=True)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, f"That is not a readable WAV recording ({exc}).")
    return audio.mean(axis=1), rate


def create_app(settings: Settings | None = None, llm: Any = None, tts_transport: httpx.AsyncBaseTransport | None = None,
               services: ServiceManager | None = None, tasks: TaskManager | None = None) -> FastAPI:
    settings = settings or Settings.load()
    services = services or ServiceManager(settings)
    tasks = tasks or TaskManager(settings)
    enrolling: dict[str, dict] = {}  # one guided enrollment at a time: id -> {language, takes}
    audit = AuditLog(settings.audit_db)
    registry, reminders = build_registry(settings, task_manager=tasks, audit=audit)
    profiles = VoiceProfiles(settings.voices_dir)
    pending: dict[str, asyncio.Future] = {}
    outbox: dict[str, asyncio.Queue] = {}
    turn_lock = asyncio.Lock()
    llm = llm or ToolChoiceLLM(settings.llm_base_url, settings.llm_model, temperature=0.3, timeout=90.0,
                                 first_token_timeout=60.0)
    tts_http = httpx.AsyncClient(base_url=settings.tts_url, timeout=httpx.Timeout(300.0, connect=2.0), transport=tts_transport)
    state: dict[str, Any] = {"turn": None}

    async def confirm(intent: str, tool, args: dict) -> bool:
        cid = uuid.uuid4().hex
        fut = asyncio.get_running_loop().create_future()
        pending[cid] = fut
        await state["emit"]({"type": "confirm_request", "id": cid, "intent": intent, "risk": tool.risk.value, "tool": tool.name})
        try:
            return await asyncio.wait_for(fut, CONFIRM_TIMEOUT_S)
        except asyncio.TimeoutError:
            return False
        finally:
            pending.pop(cid, None)

    agent = Agent(llm, registry, PermissionGate(confirm, audit, set(settings.auto_approve)), audit,
                  name=settings.assistant_name, gender=settings.hindi_self_gender)

    def listening() -> bool:
        listen = services.services.get("listen")
        return listen is not None and listen.alive()

    def guest_active() -> bool:
        """Guest mode from the setting, or switched on by voice in the running hands-free mode."""
        return settings.guest_mode or (listening() and services.services["listen"].flags.get("guest_mode", False))

    # Reminders announced to the page (it speaks them) and as Windows notifications, so one is not missed while the
    # page is closed. Hands-free mode speaks them itself while it runs, so nothing is taken here then.
    announced: collections.deque = collections.deque(maxlen=50)
    announced_lock = threading.Lock()  # appended on a worker thread, read by request threads

    def announce_due() -> list[dict]:
        if listening():
            return []
        out = []
        for r in reminders.due():
            with announced_lock:
                item = {"seq": (announced[-1]["seq"] + 1) if announced else 1, "id": r["id"], "text": r["text"],
                        "due_local": r["due_local"], "late": is_late(r), "at": time.time()}
                announced.append(item)
            out.append(item)
            if settings.reminder_toast:
                show_toast(f"{settings.assistant_name} reminder" + (f" (due {item['due_local']})" if item["late"] else ""),
                           r["text"])
        return out

    async def reminder_announcer() -> None:
        while True:
            await asyncio.sleep(REMINDER_POLL_S)
            try:
                await asyncio.to_thread(announce_due)
            except Exception as exc:  # keep announcing later ones
                audit.record("reminder", "announce", "", {}, "error", ok=False, summary=f"{type(exc).__name__}: {exc}")

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        announcer = asyncio.ensure_future(reminder_announcer())
        try:
            yield
        finally:
            announcer.cancel()

    def neutral_tts() -> NeutralVoiceTTS:
        """Windows voice; Hindi through the voice service with the synthetic neutral reference (no Windows Hindi voice)."""
        from .tts_client import ClonedVoiceTTS

        hindi = None
        if settings.neutral_voice_profile and (settings.voices_dir / settings.neutral_voice_profile).is_dir():
            hindi = ClonedVoiceTTS(settings.tts_url, voice=settings.neutral_voice_profile, preset=settings.tts_preset)
        return NeutralVoiceTTS(hindi=hindi, hindi_voice=settings.neutral_hindi_voice or None)
    app = FastAPI(title="VoxVortex Desktop", lifespan=lifespan)
    app.state.agent, app.state.audit, app.state.reminders = agent, audit, reminders
    app.state.services, app.state.tasks = services, tasks
    app.state.announce_due = announce_due

    install_local_only(app)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/version")
    def version():
        return {"api_version": API_VERSION}

    @app.get("/api/status")
    async def status():
        async def probe(coro):
            try:
                return await coro
            except Exception as exc:
                return {"ok": False, "error": type(exc).__name__}

        async def tts():
            r = await tts_http.get("/health", timeout=2.0)
            return {"ok": True, **r.json()}

        async def llm_up():
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{settings.llm_base_url}/models")
                served = [m.get("id") for m in r.json().get("data", [])] if r.status_code == 200 else []
                # `vox llm-server` serves under the alias "voice"; the model file's name says which model that is.
                name = settings.llm_gguf.stem if served == ["voice"] else (served[0] if served else settings.llm_model)
                return {"ok": r.status_code == 200, "model": name, "served": served}

        t, l = await asyncio.gather(probe(tts()), probe(llm_up()))
        return {"api_version": API_VERSION, "tts": t, "llm": l, "active_voice": profiles.active_id(), "active_voice_hi": profiles.active_id("hi"),
                "voices": profiles.list(), "tools": registry.names(), "pending_reminders": reminders.pending(),
                "language": settings.language, "guest_mode": guest_active()}

    @app.post("/api/speak")
    async def speak(body: SpeakBody):
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "Enter some text to speak.")
        if body.voice_mode == "neutral" or guest_active():  # guest mode: others can hear, never the cloned voice
            tts = neutral_tts()
            pcm = b"".join([c async for c in tts.synth(text)])
            return Response(labelled_wav(pcm, tts.sample_rate, "neutral"), media_type="audio/wav",
                            headers={"X-Voice": "neutral", "X-AI-Generated": "true"})
        try:
            r = await tts_http.post("/audio/speech", json={"input": text, "response_format": "wav", "preset": body.preset})
        except httpx.HTTPError:
            raise HTTPException(503, f"The cloned-voice service at {settings.tts_url} is not running. "
                                     "Start it with scripts\\start_tts.ps1, or use the neutral voice.")
        if r.status_code != 200:
            raise HTTPException(r.status_code, r.json().get("detail", "TTS error") if r.headers.get("content-type", "").startswith("application/json") else "TTS error")
        keep = {k: v for k, v in r.headers.items() if k.lower().startswith("x-")}
        return Response(r.content, media_type="audio/wav", headers=keep)

    @app.post("/api/speak/stream")
    async def speak_stream(body: SpeakBody):
        """Raw PCM16 as it is synthesized (rate in X-Sample-Rate), for the page to play while the rest is generated.
        `/api/speak` (a whole labelled WAV) stays for Generate/download."""
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "Nothing to speak.")
        headers = {"X-AI-Generated": "true"}
        if body.voice_mode == "neutral" or guest_active():
            tts = neutral_tts()
            pcm = b"".join([c async for c in tts.synth(text)])  # whole, so the rate header matches the engine used
            return Response(pcm, media_type="audio/L16",
                            headers={**headers, "X-Sample-Rate": str(tts.sample_rate), "X-Voice": "neutral"})
        req = tts_http.build_request("POST", "/audio/speech", json={"input": text, "response_format": "pcm",
                                                                   "preset": body.preset})
        try:
            resp = await tts_http.send(req, stream=True)
        except httpx.HTTPError:
            raise HTTPException(503, f"The cloned-voice service at {settings.tts_url} is not running.")
        if resp.status_code != 200:
            detail = (await resp.aread()).decode("utf-8", "replace")[:200]
            await resp.aclose()
            raise HTTPException(resp.status_code, detail)
        headers.update({k: v for k, v in resp.headers.items() if k.lower() in ("x-sample-rate", "x-voice")})

        async def relay():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()  # the page stopped listening: closing tells the service to stop generating

        return StreamingResponse(relay(), media_type="audio/L16", headers=headers)

    @app.post("/api/chat")
    async def chat(body: ChatBody):
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "Say or type something.")
        if turn_lock.locked():
            raise HTTPException(409, "I am still working on the previous request.")
        # Taken here, not inside the task: a second request arriving before the task first ran would pass the check
        # above and queue behind this turn. Released when the task ends, even if it is cancelled before it starts.
        await turn_lock.acquire()
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(ev: dict) -> None:
            await queue.put(ev)

        mode = body.language if body.language in language.MODES else settings.language
        reply_lang = language.reply_language(mode, text)

        async def work() -> None:
            state["emit"] = emit
            await emit({"type": "language", "mode": mode, "reply": reply_lang})
            try:
                async for ev in agent.run(text, reply_lang):
                    if isinstance(ev, Say):
                        await emit({"type": "say", "text": ev.text})
                    elif isinstance(ev, ToolStarted):
                        await emit({"type": "tool", "name": ev.name, "args": ev.args, "risk": ev.risk})
                    elif isinstance(ev, ToolFinished):
                        await emit({"type": "tool_result", "name": ev.name, "decision": ev.decision,
                                    "ok": ev.result.get("ok"), "detail": ev.result.get("error") or ev.result.get("note")})
            except Exception as exc:
                await emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                await emit({"type": "done"})

        task = asyncio.ensure_future(work())
        task.add_done_callback(lambda _: turn_lock.release())

        async def stream():
            try:
                while True:
                    ev = await queue.get()
                    yield json.dumps(ev) + "\n"
                    if ev["type"] == "done":
                        return
            finally:
                if not task.done():  # browser closed the stream: cancel the turn and any open confirmation
                    task.cancel()

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    @app.post("/api/confirm")
    async def confirm_endpoint(body: ConfirmBody):
        fut = pending.get(body.id)
        if fut is None or fut.done():
            raise HTTPException(404, "That confirmation is no longer pending.")
        fut.set_result(body.approve)
        return {"ok": True}

    @app.post("/api/reset")
    async def reset():
        agent.reset()
        return {"ok": True}

    @app.get("/api/voices")
    def voices():
        return {"active": profiles.active_id(), "active_hi": profiles.active_id("hi"), "voices": profiles.list()}

    @app.post("/api/voices")
    def add_voice(body: VoiceUpload):
        try:
            audio, rate = sf.read(io.BytesIO(base64.b64decode(body.wav_base64)), dtype="float32", always_2d=True)
            profile = profiles.create(body.name, audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0], rate,
                                      body.transcript, permission_confirmed=body.permission_confirmed, source="web upload",
                                      language="hi" if body.language == "hi" else "en")
        except (VoiceError, RuntimeError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        audit.record("voice", "create", "act", {"id": profile["id"]}, "user", ok=True)
        return profile

    @app.post("/api/voices/{profile_id}/activate")
    def activate(profile_id: str):
        try:
            profiles.set_active(profile_id)
        except VoiceError as exc:
            raise HTTPException(404, str(exc))
        audit.record("voice", "activate", "act", {"id": profile_id}, "user", ok=True)
        return {"active": profile_id}

    @app.post("/api/voices/{profile_id}/revoke")
    async def revoke(profile_id: str):
        try:
            profiles.revoke(profile_id)
        except VoiceError as exc:
            raise HTTPException(404, str(exc))
        audit.record("voice", "revoke", "destructive", {"id": profile_id}, "user", ok=True)
        try:  # drop any cached acoustic prompt in the TTS service; the file itself is already gone
            await tts_http.post("/forget", json={}, timeout=5.0)
        except httpx.HTTPError:
            pass
        return {"revoked": profile_id, "active": profiles.active_id()}

    @app.get("/api/services")
    async def services_state():
        return {"services": await asyncio.to_thread(services.states)}

    @app.post("/api/services/{name}/{action}")
    async def service_control(name: str, action: str, body: ServiceBody | None = None):
        if name not in services.services or action not in ("start", "stop", "restart"):
            raise HTTPException(404, "Unknown service or action.")
        body = body or ServiceBody()
        args: list[str] = []
        if name == "listen":
            if body.always:
                args.append("--always")
            if body.language in language.MODES:
                args += ["--language", body.language]
        try:
            if action in ("stop", "restart"):
                await asyncio.to_thread(services.stop, name)
            if action in ("start", "restart"):
                await asyncio.to_thread(services.start, name, args)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        audit.record("service", f"{action} {name}", "act", {}, "user", ok=True)
        return await asyncio.to_thread(services.state, name)

    @app.get("/api/services/{name}/logs")
    def service_logs(name: str, lines: int = 120):
        if name not in services.services:
            raise HTTPException(404, "Unknown service.")
        return {"lines": services.logs(name, lines)}

    @app.get("/api/services/{name}/events")
    def service_events(name: str, after: int = 0, run_id: int | None = None):
        if name not in services.services:
            raise HTTPException(404, "Unknown service.")
        return services.events(name, max(0, after), run_id)

    # ---- coding tasks -----------------------------------------------------------------------------------------

    @app.get("/api/projects")
    def projects_state():
        return {"projects": [{"name": p.name, "path": str(p)} for p in sorted(list_projects(settings), key=lambda p: p.name.lower())]}

    @app.get("/api/tasks")
    def tasks_state():
        return {"tasks": tasks.states()}

    @app.post("/api/tasks")
    def task_start(body: TaskBody):
        found = match_project(body.project, list_projects(settings))
        if found is None:
            raise HTTPException(404, "That project folder was not found.")
        if isinstance(found, list):
            raise HTTPException(409, "Several projects match: " + ", ".join(p.name for p in found[:10]))
        try:
            job = tasks.start(found, body.task, allow_changes=body.allow_changes)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except TaskBusy as exc:
            raise HTTPException(409, str(exc))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))
        audit.record("coding_task", body.task[:160], "act", {"project": found.name, "task_id": job["id"]},
                     "user", ok=True)
        return job

    @app.get("/api/tasks/{task_id}")
    def task_state(task_id: str):
        try:
            return tasks.state(task_id)
        except KeyError:
            raise HTTPException(404, "Task not found.")

    @app.post("/api/tasks/{task_id}/stop")
    def task_stop(task_id: str):
        try:
            result = tasks.stop(task_id)
        except KeyError:
            raise HTTPException(404, "Task not found.")
        audit.record("coding_task", "stop", "act", {"task_id": task_id}, "user", ok=True)
        return result

    @app.post("/api/transcribe")
    async def transcribe(body: Recording):
        """Speech from the page's microphone to text (the recogniser hands-free mode uses); the page sends it as chat."""
        audio, rate = read_wav(body.wav_base64)
        pcm = (np.clip(to_reference(audio, rate), -1.0, 1.0) * 32767).astype("<i2").tobytes()
        if len(pcm) < REF_RATE * 2 * 0.3:
            raise HTTPException(400, "That recording is too short.")
        from .asr_local import LocalWhisperASR

        cpu = LocalWhisperASR(settings.whisper_model, settings.whisper_device, settings.whisper_compute, settings.whisper_threads)
        if settings.asr_backend == "service":
            from .asr_remote import ServiceASR

            asr = ServiceASR(settings.tts_url, fallback=cpu)
        else:
            asr = cpu
        try:
            text = await asr.transcribe(pcm, language="auto", prompt=settings.asr_prompt)
        except Exception as exc:
            raise HTTPException(503, f"Speech recognition is not available ({type(exc).__name__}). Start the voice service.")
        finally:
            if hasattr(asr, "aclose"):
                await asr.aclose()
        return {"text": text.strip()}

    # ---- guided voice enrollment from the page's microphone (the browser twin of `vox enroll`) ----

    def enroll_summary(sid: str) -> dict:
        e = enrolling[sid]
        total = sum(t.quality.seconds for t in e["takes"].values())
        return {"session": sid, "language": e["language"], "taken": sorted(e["takes"]), "seconds": round(total, 1),
                "ready": total >= TARGET_SECONDS[0], "target": list(TARGET_SECONDS)}

    @app.post("/api/enroll/start")
    def enroll_start(body: EnrollStart):
        lang = "hi" if body.language == "hi" else "en"
        enrolling.clear()
        sid = uuid.uuid4().hex[:12]
        enrolling[sid] = {"language": lang, "takes": {}}
        return {**enroll_summary(sid), "script": SCRIPT_HI if lang == "hi" else SCRIPT}

    @app.post("/api/enroll/take")
    def enroll_take(body: EnrollTake):
        e = enrolling.get(body.session)
        script = (SCRIPT_HI if e["language"] == "hi" else SCRIPT) if e else []
        if e is None or not 0 <= body.index < len(script):
            raise HTTPException(404, "That enrollment is no longer open; start again.")
        raw, rate = read_wav(body.wav_base64)
        take: Take = assess_take(script[body.index], raw, rate)
        others = sum(t.quality.seconds for i, t in e["takes"].items() if i != body.index)
        accepted = take.usable and others + take.quality.seconds <= TARGET_SECONDS[1]
        if accepted:
            e["takes"][body.index] = take
        reasons = take.quality.errors if not take.usable else (
            [] if accepted else ["That would make the reference longer than needed; you can finish now."])
        return {"accepted": accepted, "seconds": round(take.quality.seconds, 1), "snr_db": round(take.quality.snr_db),
                "warnings": take.quality.warnings, "errors": reasons, **enroll_summary(body.session)}

    @app.post("/api/enroll/finish")
    def enroll_finish(body: EnrollFinish):
        e = enrolling.get(body.session)
        if e is None or not e["takes"]:
            raise HTTPException(404, "Record at least one line first.")
        audio, transcript = assemble([e["takes"][i] for i in sorted(e["takes"])])
        try:
            profile = profiles.create(body.name, audio, REF_RATE, transcript, permission_confirmed=body.permission_confirmed,
                                      source="guided enrollment (web)", language=e["language"])
        except VoiceError as exc:
            raise HTTPException(400, str(exc))
        enrolling.pop(body.session, None)
        audit.record("voice", "create", "act", {"id": profile["id"]}, "user", ok=True)
        return profile

    @app.post("/api/enroll/cancel")
    def enroll_cancel():
        enrolling.clear()
        return {"ok": True}

    @app.post("/api/voices/{profile_id}/check")
    async def voice_check(profile_id: str):
        """Clone a fixed sentence with this voice and score it against the recording (slow the first time)."""
        try:
            r = await tts_http.post(f"/voices/{profile_id}/similarity", json={}, timeout=600.0)
        except httpx.HTTPError:
            raise HTTPException(503, "The cloned-voice service is not running; start it first.")
        if r.status_code != 200:
            raise HTTPException(r.status_code, r.json().get("detail", "check failed"))
        return r.json()

    @app.get("/api/reminders/announced")
    def reminders_announced(after: int | None = None):
        """What the panel announced after `after` (read-only; the page polls it and speaks new ones). Without `after`,
        a page that just opened gets the last two minutes'."""
        with announced_lock:
            items = [r for r in announced if r["seq"] > after] if after is not None else \
                [r for r in announced if r["at"] > time.time() - 120]
            return {"seq": announced[-1]["seq"] if announced else 0, "reminders": items}

    @app.get("/api/audit")
    def audit_log(limit: int = 50):
        return {"events": audit.recent(min(max(limit, 1), 500))}

    return app


def main(settings: Settings | None = None) -> None:
    """`vox serve`: the panel alone (main.py also starts the services). Stops what it started on exit."""
    import uvicorn

    from .tools.knowledge import warm_index

    settings = settings or Settings.load()
    url = f"http://127.0.0.1:{settings.web_port}"
    services, tasks = ServiceManager(settings), TaskManager(settings)
    services.panel_url = url
    threading.Thread(target=warm_index, args=(settings,), name="notes-index", daemon=True).start()
    print(f"VoxVortex Desktop: {url}")
    try:
        uvicorn.run(create_app(settings, services=services, tasks=tasks), host="127.0.0.1", port=settings.web_port,
                    log_level="info")
    finally:
        tasks.shutdown()
        services.shutdown()


if __name__ == "__main__":
    main()
