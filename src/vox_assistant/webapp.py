"""Local web app: type-and-generate in your voice, chat with the agent, manage voices, read the audit log.

Binds to 127.0.0.1 only. Because it can drive the PC, every request must carry a localhost Host header (blocks DNS
rebinding) and any Origin must be this same server (blocks other web pages from posting to it).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
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
from .permissions import PermissionGate
from .security import install_local_only
from .tools import build_registry
from .tts_neutral import NeutralVoiceTTS
from .voices import VoiceError, VoiceProfiles

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
CONFIRM_TIMEOUT_S = 90.0


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


class VoiceUpload(BaseModel):
    name: str
    transcript: str
    permission_confirmed: bool
    wav_base64: str
    language: str = "en"  # what the recording is spoken in: "en" or "hi"


def create_app(settings: Settings | None = None, llm: Any = None, tts_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings.load()
    audit = AuditLog(settings.audit_db)
    registry, reminders = build_registry(settings)
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

    agent = Agent(llm, registry, PermissionGate(confirm, audit), audit, name=settings.assistant_name,
                  gender=settings.hindi_self_gender)

    def neutral_tts() -> NeutralVoiceTTS:
        """Windows voice; Hindi through the voice service with the synthetic neutral reference (no Windows Hindi voice)."""
        from .tts_client import ClonedVoiceTTS

        hindi = None
        if settings.neutral_voice_profile and (settings.voices_dir / settings.neutral_voice_profile).is_dir():
            hindi = ClonedVoiceTTS(settings.tts_url, voice=settings.neutral_voice_profile, preset=settings.tts_preset)
        return NeutralVoiceTTS(hindi=hindi, hindi_voice=settings.neutral_hindi_voice or None)
    app = FastAPI(title="VoxVortex Desktop")
    app.state.agent, app.state.audit, app.state.reminders = agent, audit, reminders

    install_local_only(app)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

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
        return {"tts": t, "llm": l, "active_voice": profiles.active_id(), "active_voice_hi": profiles.active_id("hi"),
                "voices": profiles.list(), "tools": registry.names(), "pending_reminders": reminders.pending(),
                "language": settings.language}

    @app.post("/api/speak")
    async def speak(body: SpeakBody):
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "Enter some text to speak.")
        if body.voice_mode == "neutral":
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
        if body.voice_mode == "neutral":
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
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(ev: dict) -> None:
            await queue.put(ev)

        mode = body.language if body.language in language.MODES else settings.language
        reply_lang = language.reply_language(mode, text)

        async def work() -> None:
            async with turn_lock:
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

    @app.get("/api/reminders/due")
    def reminders_due():
        """The page polls this; a due reminder is returned once, then marked fired."""
        return {"due": reminders.due()}

    @app.get("/api/audit")
    def audit_log(limit: int = 50):
        return {"events": audit.recent(min(max(limit, 1), 500))}

    return app


def main() -> None:
    import uvicorn

    import threading

    from .tools.knowledge import warm_index

    settings = Settings.load()
    threading.Thread(target=warm_index, args=(settings,), name="notes-index", daemon=True).start()
    print(f"VoxVortex Desktop: http://127.0.0.1:{settings.web_port}")
    uvicorn.run(create_app(settings), host="127.0.0.1", port=settings.web_port, log_level="info")


if __name__ == "__main__":
    main()
