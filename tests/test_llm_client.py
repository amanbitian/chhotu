import json

import httpx
from voxvortex.llm import TextDelta, ToolCall

from vox_assistant.agent import Agent
from vox_assistant.audit import AuditLog
from vox_assistant.llm_client import ToolChoiceLLM
from vox_assistant.permissions import PermissionGate, Risk, Tool, ToolRegistry, ok

SSE = ('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n' "data: [DONE]\n\n")


def llm_with(bodies):
    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=SSE.encode(), headers={"content-type": "text/event-stream"})

    return ToolChoiceLLM("http://llm.test/v1", "m", transport=httpx.MockTransport(handler))


TOOLS = [{"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}]


async def test_tool_choice_is_injected_only_when_asked():
    bodies = []
    llm = llm_with(bodies)
    assert [e.text async for e in llm.stream([{"role": "user", "content": "x"}], TOOLS) if isinstance(e, TextDelta)] == ["hi"]
    [_ async for _ in llm.stream([{"role": "user", "content": "x"}], TOOLS, tool_choice="required")]
    [_ async for _ in llm.stream([{"role": "user", "content": "x"}], TOOLS)]
    assert all(b["stop"] == ["</think>"] for b in bodies)
    assert "tool_choice" not in bodies[0] and bodies[1]["tool_choice"] == "required" and "tool_choice" not in bodies[2]
    assert bodies[1]["tools"] == TOOLS and bodies[1]["stream"] is True  # the rest of the body is untouched


async def test_tool_choice_not_added_when_no_tools_are_sent():
    bodies = []
    [_ async for _ in llm_with(bodies).stream([{"role": "user", "content": "x"}], None, tool_choice="required")]
    assert "tool_choice" not in bodies[0] and "parallel_tool_calls" not in bodies[0]


async def test_parallel_tool_calls_are_requested_and_both_calls_are_parsed():
    def tc(i, cid, name):
        return {"choices": [{"delta": {"tool_calls": [{"index": i, "id": cid, "type": "function",
                                                         "function": {"name": name, "arguments": "{}"}}]}}]}

    sse = "".join(f"data: {json.dumps(d)}\n\n" for d in (tc(0, "a", "gpu_status"), tc(1, "b", "list_models")))
    sse += 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\ndata: [DONE]\n\n'
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})

    llm = ToolChoiceLLM("http://llm.test/v1", "m", transport=httpx.MockTransport(handler))
    calls = [e for e in [ev async for ev in llm.stream([{"role": "user", "content": "x"}], TOOLS)] if isinstance(e, ToolCall)]
    assert bodies[0]["parallel_tool_calls"] is True
    assert [c.name for c in calls] == ["gpu_status", "list_models"]


async def test_agent_forces_a_tool_call_on_the_guard_retry(tmp_path):
    class Fake:
        supports_tool_choice = True

        def __init__(self):
            self.choices = []
            self.rounds = [[TextDelta("Poke is done.")],
                           [ToolCall("c", "poke", {}, "{}")],
                           [TextDelta("Poked.")]]

        async def stream(self, messages, tools=None, *, max_tokens=120, tool_choice=None):
            self.choices.append(tool_choice)
            for ev in self.rounds.pop(0):
                yield ev

    ran = []

    async def act(args):
        ran.append(1)
        return ok()

    async def yes(i, t, a):
        return True

    audit = AuditLog(tmp_path / "a.sqlite3")
    llm = Fake()
    agent = Agent(llm, ToolRegistry([Tool("poke", "d", {"type": "object", "properties": {}}, Risk.ACT, act)]),
                  PermissionGate(yes, audit), audit)
    out = "".join([e.text async for e in agent.run("poke") if hasattr(e, "text")])
    assert out == "Poked." and ran == [1]
    assert llm.choices == [None, "required", None]
