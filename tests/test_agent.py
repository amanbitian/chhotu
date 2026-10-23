import asyncio
import json

import pytest

from voxvortex.llm import TextDelta, ToolCall

from vox_assistant.agent import Agent, Say, ToolFinished, ToolStarted
from vox_assistant.audit import AuditLog
from vox_assistant.permissions import PermissionGate, Risk, Tool, ToolRegistry, fail, ok


class ScriptedLLM:
    """Each call to stream() plays the next scripted list of events; records what it was sent."""

    def __init__(self, *rounds):
        self.rounds = list(rounds)
        self.seen: list[list[dict]] = []

    async def stream(self, messages, tools=None, *, max_tokens=120):
        self.seen.append([dict(m) for m in messages])
        for ev in self.rounds.pop(0):
            yield ev


def call(name, args=None, id="c1"):
    return ToolCall(id=id, name=name, arguments=args or {}, raw_arguments=json.dumps(args or {}))


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.sqlite3")


def make(audit, llm, confirm=None, auto=None):
    calls = []

    async def read_tool(args):
        calls.append(("read", args))
        return ok(value=42)

    async def act_tool(args):
        calls.append(("act", args))
        return ok(done=True)

    async def boom(args):
        raise RuntimeError("kaboom")

    reg = ToolRegistry([
        Tool("peek", "read", {"type": "object", "properties": {}}, Risk.READ, read_tool),
        Tool("poke", "act", {"type": "object", "properties": {}}, Risk.ACT, act_tool, describe=lambda a: "poke it"),
        Tool("nuke", "destructive", {"type": "object", "properties": {}}, Risk.DESTRUCTIVE, act_tool),
        Tool("boom", "broken", {"type": "object", "properties": {}}, Risk.READ, boom),
    ])
    gate = PermissionGate(confirm, audit, auto_approve=auto)
    return Agent(llm, reg, gate, audit), calls


async def collect(agent, text):
    return [ev async for ev in agent.run(text)]


async def test_plain_answer_streams_text_and_updates_history(audit):
    agent, _ = make(audit, ScriptedLLM([TextDelta("Hello "), TextDelta("there.")]))
    events = await collect(agent, "hi")
    assert "".join(e.text for e in events if isinstance(e, Say)) == "Hello there."
    assert [m["role"] for m in agent.history] == ["user", "assistant"]


async def test_read_tool_runs_without_asking_and_result_reaches_the_model(audit):
    llm = ScriptedLLM([call("peek")], [TextDelta("It is forty two.")])
    agent, calls = make(audit, llm, confirm=None)
    events = await collect(agent, "what is it")
    assert calls == [("read", {})]
    finished = [e for e in events if isinstance(e, ToolFinished)][0]
    assert finished.decision == "auto" and finished.result["value"] == 42
    tool_msg = [m for m in llm.seen[1] if m["role"] == "tool"][0]
    assert json.loads(tool_msg["content"])["value"] == 42


async def test_act_tool_needs_confirmation_and_denial_is_reported(audit):
    asked = []

    async def deny(intent, tool, args):
        asked.append(intent)
        return False

    llm = ScriptedLLM([call("poke")], [TextDelta("Okay, I will not.")])
    agent, calls = make(audit, llm, confirm=deny)
    events = await collect(agent, "poke it")
    assert asked == ["poke it"] and calls == []
    assert [e for e in events if isinstance(e, ToolFinished)][0].decision == "denied"
    assert "did not approve" in llm.seen[1][-1]["content"]


async def test_approved_act_runs_and_is_audited(audit):
    async def yes(intent, tool, args):
        return True

    agent, calls = make(audit, ScriptedLLM([call("poke", {"x": 1})], [TextDelta("Done.")]), confirm=yes)
    await collect(agent, "poke")
    assert calls == [("act", {"x": 1})]
    row = audit.recent()[0]
    assert (row["tool"], row["decision"], row["ok"]) == ("poke", "approved", 1)


async def test_no_confirmer_means_act_is_refused(audit):
    agent, calls = make(audit, ScriptedLLM([call("poke")], [TextDelta("I cannot.")]), confirm=None)
    await collect(agent, "poke")
    assert calls == []
    assert audit.recent()[0]["decision"] == "denied-no-confirmer"


async def test_destructive_is_never_pre_approved(audit):
    asked = []

    async def deny(intent, tool, args):
        asked.append(tool.name)
        return False

    agent, calls = make(audit, ScriptedLLM([call("nuke")], [TextDelta("no")]), confirm=deny, auto={"nuke", "poke"})
    await collect(agent, "nuke")
    assert asked == ["nuke"] and calls == []


async def test_pre_approved_act_skips_the_prompt(audit):
    agent, calls = make(audit, ScriptedLLM([call("poke")], [TextDelta("ok")]), confirm=None, auto={"poke"})
    await collect(agent, "poke")
    assert calls == [("act", {})]


async def test_tool_exception_becomes_an_error_result(audit):
    llm = ScriptedLLM([call("boom")], [TextDelta("That failed.")])
    agent, _ = make(audit, llm)
    events = await collect(agent, "boom")
    result = [e for e in events if isinstance(e, ToolFinished)][0].result
    assert result["ok"] is False and "kaboom" in result["error"]


async def test_unknown_tool_and_bad_arguments(audit):
    bad = ToolCall(id="c2", name="peek", arguments={"__error": "x"}, raw_arguments="{")
    llm = ScriptedLLM([call("nonexistent"), bad], [TextDelta("sorry")])
    agent, calls = make(audit, llm)
    events = await collect(agent, "x")
    assert [e.decision for e in events if isinstance(e, ToolFinished)] == ["unknown-tool", "bad-arguments"]
    assert calls == []


async def test_round_limit_stops_a_tool_loop(audit):
    llm = ScriptedLLM(*[[call("peek")] for _ in range(10)])
    agent, _ = make(audit, llm)
    events = await collect(agent, "loop")
    assert "could not finish" in "".join(e.text for e in events if isinstance(e, Say))
    assert len(llm.seen) == 6


async def test_interrupt_during_confirmation_keeps_history_consistent(audit):
    started = asyncio.Event()

    async def slow(intent, tool, args):
        started.set()
        await asyncio.sleep(10)
        return True

    agent, calls = make(audit, ScriptedLLM([TextDelta("Let me "), call("poke")]), confirm=slow)

    async def go():
        async for _ in agent.run("poke"):
            pass

    task = asyncio.create_task(go())
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == []
    assert agent.history[0]["role"] == "user"


async def test_history_is_trimmed_at_a_user_boundary(audit):
    agent, _ = make(audit, ScriptedLLM(*[[TextDelta("a")] for _ in range(20)]))
    for i in range(20):
        await collect(agent, f"q{i}")
    assert len(agent.history) <= 24 and agent.history[0]["role"] == "user"


def test_system_prompt_mentions_untrusted_tool_text_and_time():
    from datetime import datetime

    from vox_assistant.agent import system_prompt

    p = system_prompt("Vox", datetime(2026, 10, 22, 9, 5))
    assert "untrusted" in p and "22 October 2026" in p
    assert "09:05" not in p and system_prompt("Vox", datetime(2026, 10, 22, 23, 59)) == p  # stable all day: cacheable


async def test_user_messages_carry_their_time_and_history_is_not_rewritten(audit):
    from datetime import datetime

    llm = ScriptedLLM([TextDelta("a")], [TextDelta("b")])
    agent, _ = make(audit, llm)
    agent.clock = lambda: datetime(2026, 10, 22, 9, 5)
    await collect(agent, "first")
    agent.clock = lambda: datetime(2026, 10, 22, 9, 6)
    async for _ in agent.run("दूसरा", "hi"):
        pass
    assert agent.history[0]["content"] == "[09:05, reply in English] first"
    assert agent.history[2]["content"] == "[09:06, reply in Hindi] दूसरा"
    assert llm.seen[1][:2] == llm.seen[0][:2]  # system prompt and first message unchanged a minute later


async def test_several_read_calls_run_at_the_same_time_but_actions_stay_sequential(audit):
    import time

    async def slow_read(args):
        await asyncio.sleep(0.3)
        return ok(n=args["n"])

    order = []

    async def act(args):
        order.append(args["n"])
        await asyncio.sleep(0.05)
        return ok()

    async def yes(intent, tool, args):
        return True

    reg = ToolRegistry([Tool("peek", "r", {"type": "object", "properties": {}}, Risk.READ, slow_read),
                        Tool("poke", "a", {"type": "object", "properties": {}}, Risk.ACT, act)])
    llm = ScriptedLLM([call("peek", {"n": 1}, "a"), call("peek", {"n": 2}, "b"), call("peek", {"n": 3}, "c")],
                      [TextDelta("Done.")])
    agent = Agent(llm, reg, PermissionGate(yes, audit), audit)
    t0 = time.perf_counter()
    events = await collect(agent, "look")
    assert time.perf_counter() - t0 < 0.75  # three 0.3 s reads together, not 0.9 s one after another
    tool_msgs = [json.loads(m["content"])["n"] for m in agent.history if m["role"] == "tool"]
    assert tool_msgs == [1, 2, 3] and [e.decision for e in events if isinstance(e, ToolFinished)] == ["auto"] * 3
    assert [r["tool"] for r in audit.recent()].count("peek") == 3

    # Interrupted mid-read: every call still gets a (failed) result in the history, so the next request is valid.
    agent3 = Agent(ScriptedLLM([call("peek", {"n": 1}, "a"), call("peek", {"n": 2}, "b")]), reg, PermissionGate(yes, audit), audit)

    async def go():
        async for _ in agent3.run("look"):
            pass

    task = asyncio.create_task(go())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    calls_made = [tc["id"] for m in agent3.history if m.get("tool_calls") for tc in m["tool_calls"]]
    answered = [m["tool_call_id"] for m in agent3.history if m["role"] == "tool"]
    assert calls_made == answered == ["a", "b"]

    llm2 = ScriptedLLM([call("poke", {"n": 1}, "a"), call("poke", {"n": 2}, "b")], [TextDelta("ok")])
    agent2 = Agent(llm2, reg, PermissionGate(yes, audit), audit)
    await collect(agent2, "poke twice")
    assert order == [1, 2]  # actions: one at a time, each through the gate


async def test_a_declined_action_is_not_asked_again_in_the_same_turn(audit):
    asked = []

    async def deny(intent, tool, args):
        asked.append(tool.name)
        return False

    # The model keeps re-requesting the refused action (measured with the 4B on a Hindi reminder).
    llm = ScriptedLLM(*[[call("poke", {}, f"c{i}")] for i in range(6)])
    agent, calls = make(audit, llm, confirm=deny)
    events = []
    async for ev in agent.run("याद दिलाना", "hi"):
        events.append(ev)
    assert asked == ["poke"] and calls == []  # asked once, never again
    assert [e.decision for e in events if isinstance(e, ToolFinished)] == ["denied", "declined-earlier"]
    assert "".join(e.text for e in events if isinstance(e, Say)) == "ठीक है, यह नहीं करेंगे।"  # in the reply language


async def test_prewarm_sends_the_prompt_for_one_token_and_swallows_errors(audit):
    class Boom:
        async def stream(self, messages, tools=None, *, max_tokens=120):
            raise RuntimeError("down")
            yield

    agent, _ = make(audit, ScriptedLLM([TextDelta("x")]))
    await agent.prewarm()
    assert agent.llm.seen[0][-1] == {"role": "user", "content": "."} and agent.history == []
    agent.llm = Boom()
    await agent.prewarm()  # no exception


async def test_a_running_tool_finishes_and_is_recorded_when_the_turn_is_cancelled(audit):
    started, finished = asyncio.Event(), []

    async def slow(args):
        started.set()
        await asyncio.sleep(0.3)
        finished.append(1)
        return ok(done=True)

    reg = ToolRegistry([Tool("slow", "s", {"type": "object", "properties": {}}, Risk.READ, slow)])
    agent = Agent(ScriptedLLM([call("slow")]), reg, PermissionGate(None, audit), audit)

    async def go():
        async for _ in agent.run("do it"):
            pass

    task = asyncio.create_task(go())
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished == [1]                                   # not abandoned half way
    assert agent.history[-1]["role"] == "tool"               # and its result is in the history
    assert audit.recent()[0]["tool"] == "slow" and audit.recent()[0]["ok"] == 1


async def test_interrupted_while_waiting_for_approval_answers_every_open_call(audit):
    asking = asyncio.Event()

    async def confirm(intent, tool, args):
        asking.set()
        await asyncio.sleep(3600)  # the user has not answered yet

    llm = ScriptedLLM([call("poke", id="c1"), call("poke", id="c2")], [TextDelta("Okay.")])
    agent, calls = make(audit, llm, confirm=confirm)

    async def go():
        async for _ in agent.run("poke twice"):
            pass

    task = asyncio.create_task(go())
    await asyncio.wait_for(asking.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == []  # nothing ran
    results = [m for m in agent.history if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["c1", "c2"]  # each call has a result, so the next request is valid
    assert all(json.loads(m["content"])["ok"] is False for m in results)
    assert audit.recent()[0]["decision"] == "interrupted"

    async for _ in agent.run("never mind"):  # the conversation carries on
        pass
    sent = llm.seen[-1]
    open_ids = {c["id"] for m in sent if m.get("tool_calls") for c in m["tool_calls"]}
    assert open_ids == {m["tool_call_id"] for m in sent if m["role"] == "tool"}
