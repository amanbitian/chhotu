"""VoxVortex's streaming LLM client plus extra request fields: `tool_choice`, `parallel_tool_calls` and a `</think>` stop.

`OpenAICompatLLM.stream` builds its JSON body internally, so rather than copy it (or edit the sibling project) the
fields are added by a transport wrapper; `tool_choice` comes from a ContextVar set for the duration of one `stream()` call.

Why the stop sequence: with reasoning switched off, Qwen3.8 sometimes writes its answer, then a stray `</think>`, then the
same answer again (seen in about 1 reply in 5). Stopping at the marker leaves exactly one copy.
"""

from __future__ import annotations

import contextvars
import json
from typing import Any, AsyncIterator

import httpx
from voxvortex.llm import LLMEvent, OpenAICompatLLM

_tool_choice: contextvars.ContextVar[str | None] = contextvars.ContextVar("voxa_tool_choice", default=None)


class _InjectToolChoice(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            body["stop"] = sorted({*(body.get("stop") or []), "</think>"})
            choice = _tool_choice.get()
            if body.get("tools"):
                # llama-server defaults to one tool call per reply; "GPU temperatures and loaded models" then needs two
                # LLM rounds instead of one (~0.3 s more before the answer).
                body.setdefault("parallel_tool_calls", True)
                if choice:
                    body["tool_choice"] = choice
            headers = {k: v for k, v in request.headers.items() if k.lower() != "content-length"}
            request = httpx.Request(request.method, request.url, headers=headers, content=json.dumps(body).encode(),
                                    extensions=request.extensions)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


class ToolChoiceLLM(OpenAICompatLLM):
    supports_tool_choice = True

    def __init__(self, *args: Any, transport: httpx.AsyncBaseTransport | None = None, **kwargs: Any) -> None:
        super().__init__(*args, transport=_InjectToolChoice(transport or httpx.AsyncHTTPTransport()), **kwargs)

    async def stream(self, messages: list[dict], tools: list[dict] | None = None, *, max_tokens: int = 120,
                     tool_choice: str | None = None) -> AsyncIterator[LLMEvent]:
        token = _tool_choice.set(tool_choice)
        try:
            async for ev in super().stream(messages, tools, max_tokens=max_tokens):
                yield ev
        finally:
            _tool_choice.reset(token)
