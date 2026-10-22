import asyncio
import base64
import io
import json

import httpx
import numpy as np
import pytest
import soundfile as sf
from voxvortex.llm import TextDelta, ToolCall

from vox_assistant.webapp import create_app

from conftest import tone


class ScriptedLLM:
    def __init__(self, *rounds):
        self.rounds = list(rounds)

    async def stream(self, messages, tools=None, *, max_tokens=120):
        for ev in self.rounds.pop(0):
            yield ev


def tts_stub():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/audio/speech":
            return httpx.Response(200, content=b"RIFFfake", headers={"x-voice": "me_v1", "x-rtf": "0.5", "x-audio-seconds": "1.0"})
        if request.url.path == "/forget":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"model_loaded": True})

    return httpx.MockTransport(handler)


@pytest.fixture
async def client(settings):
    llm = ScriptedLLM()
    app = create_app(settings, llm=llm, tts_transport=tts_stub())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765") as c:
        c.llm, c.app = llm, app
        yield c


def wav_b64(seconds=8.0):
    buf = io.BytesIO()
    sf.write(buf, tone(seconds), 16000, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


async def events(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


async def test_index_served(client):
    r = await client.get("/")
    assert r.status_code == 200 and "VoxVortex Desktop" in r.text


async def test_rejects_foreign_host_origin_and_form_posts(client):
    assert (await client.get("/api/voices", headers={"host": "evil.example"})).status_code == 403
    assert (await client.post("/api/reset", json={}, headers={"origin": "https://evil.example"})).status_code == 403
    assert (await client.post("/api/reset", content="x=1", headers={"content-type": "text/plain"})).status_code == 415
    assert (await client.post("/api/reset", json={}, headers={"origin": "http://127.0.0.1:8765"})).status_code == 200


async def test_speak_proxies_cloned_voice_and_reports_timing(client):
    r = await client.post("/api/speak", json={"text": "hello", "voice_mode": "cloned"})
    assert r.status_code == 200 and r.headers["x-rtf"] == "0.5"
    assert (await client.post("/api/speak", json={"text": "  "})).status_code == 400


async def test_speak_stream_relays_pcm_with_its_rate_and_asks_for_pcm(settings):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=b"\x01\x00" * 4800, headers={"x-sample-rate": "48000", "x-voice": "me_v1"})

    app = create_app(settings, llm=ScriptedLLM(), tts_transport=httpx.MockTransport(handler))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765") as c:
        r = await c.post("/api/speak/stream", json={"text": "Hello there.", "voice_mode": "cloned"})
        assert r.status_code == 200 and r.headers["x-sample-rate"] == "48000" and r.headers["x-ai-generated"] == "true"
        assert len(r.content) == 9600 and seen[0]["response_format"] == "pcm"
        assert (await c.post("/api/speak/stream", json={"text": " "})).status_code == 400


async def test_speak_reports_a_stopped_tts_service(settings):
    def down(request):
        raise httpx.ConnectError("refused")

    app = create_app(settings, llm=ScriptedLLM(), tts_transport=httpx.MockTransport(down))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765") as c:
        r = await c.post("/api/speak", json={"text": "hi"})
    assert r.status_code == 503 and "not running" in r.json()["detail"]


async def test_voice_upload_activate_revoke(client):
    r = await client.post("/api/voices", json={"name": "me", "transcript": "hello there", "permission_confirmed": True,
                                               "wav_base64": wav_b64()})
    assert r.status_code == 200 and r.json()["id"] == "me_v1"
    bad = await client.post("/api/voices", json={"name": "me", "transcript": "x", "permission_confirmed": False,
                                                 "wav_base64": wav_b64()})
    assert bad.status_code == 400 and "permission" in bad.json()["detail"]
    assert (await client.get("/api/voices")).json()["active"] == "me_v1"
    assert (await client.post("/api/voices/me_v1/revoke", json={})).json()["active"] is None
    assert (await client.get("/api/voices")).json()["voices"] == []


async def test_chat_streams_tool_and_text(client):
    client.llm.rounds = [[ToolCall("c1", "system_info", {}, "{}")], [TextDelta("Your PC is fine.")]]
    r = await client.post("/api/chat", json={"text": "how is my pc"})
    evs = await events(r)
    types = [e["type"] for e in evs]
    assert types == ["language", "tool", "tool_result", "say", "done"]
    assert evs[0]["reply"] == "en"
    assert evs[2]["ok"] is True and evs[2]["decision"] == "auto"


async def test_chat_reply_language_follows_the_selector_or_the_user(client):
    client.llm.rounds = [[TextDelta("दिल्ली।")], [TextDelta("Delhi.")]]
    evs = await events(await client.post("/api/chat", json={"text": "भारत की राजधानी क्या है?", "language": "auto"}))
    assert evs[0] == {"type": "language", "mode": "auto", "reply": "hi"}
    evs = await events(await client.post("/api/chat", json={"text": "भारत की राजधानी क्या है?", "language": "en"}))
    assert evs[0]["reply"] == "en"


@pytest.fixture
async def live(settings):
    """A real server on a free port: httpx's ASGI transport buffers streams, which would hide the confirm round trip."""
    import socket

    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    llm = ScriptedLLM()
    app = create_app(settings, llm=llm, tts_transport=tts_stub())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=15.0) as c:
        c.llm, c.app = llm, app
        yield c
    server.should_exit = True
    await task


async def run_chat(c, text, approve):
    seen = []
    async with c.stream("POST", "/api/chat", json={"text": text}) as resp:
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            ev = json.loads(line)
            seen.append(ev)
            if ev["type"] == "confirm_request":
                assert "reminder" in ev["intent"]
                assert (await c.post("/api/confirm", json={"id": ev["id"], "approve": approve})).status_code == 200
            if ev["type"] == "done":
                break
    return seen


async def test_chat_confirmation_approved_runs_the_action(live):
    live.llm.rounds = [[ToolCall("c1", "set_reminder", {"text": "water", "in_minutes": 5}, "{}")], [TextDelta("Set.")]]
    seen = await run_chat(live, "remind me", True)
    result = [e for e in seen if e["type"] == "tool_result"][0]
    assert result["decision"] == "approved" and result["ok"] is True
    assert len(live.app.state.reminders.pending()) == 1


async def test_chat_confirmation_declined_does_nothing(live):
    live.llm.rounds = [[ToolCall("c1", "set_reminder", {"text": "water", "in_minutes": 5}, "{}")], [TextDelta("Okay.")]]
    await run_chat(live, "remind me", False)
    assert live.app.state.reminders.pending() == []
    assert (await live.post("/api/confirm", json={"id": "nope", "approve": True})).status_code == 404
    assert live.app.state.audit.recent()[0]["decision"] == "denied"
