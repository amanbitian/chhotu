"""The false-claim guard: the assistant must never speak 'it is done' for an action it did not perform."""

import json

import pytest
from voxvortex.llm import TextDelta, ToolCall

from vox_assistant.agent import Agent, Say, claims_action
from vox_assistant.audit import AuditLog
from vox_assistant.permissions import PermissionGate, Risk, Tool, ToolRegistry, ok


class ScriptedLLM:
    def __init__(self, *rounds):
        self.rounds, self.seen = list(rounds), []

    async def stream(self, messages, tools=None, *, max_tokens=120):
        self.seen.append([dict(m) for m in messages])
        for ev in self.rounds.pop(0):
            yield ev


def call(name):
    return ToolCall(id="c1", name=name, arguments={}, raw_arguments="{}")


def build(tmp_path, llm, approve=True):
    audit = AuditLog(tmp_path / "a.sqlite3")
    ran = []

    async def act(args):
        ran.append(1)
        return ok(done=True)

    async def read(args):
        return ok(value=42)

    reg = ToolRegistry([Tool("poke", "act", {"type": "object", "properties": {}}, Risk.ACT, act),
                        Tool("peek", "read", {"type": "object", "properties": {}}, Risk.READ, read)])

    async def confirm(i, t, a):
        return approve

    return Agent(llm, reg, PermissionGate(confirm, audit), audit), ran, audit


async def said(agent, text):
    return "".join([e.text async for e in agent.run(text) if isinstance(e, Say)])


@pytest.mark.parametrize("text,expected", [
    ("Notepad is open now.", True), ("I've opened Notepad for you.", True), ("Done.", True),
    ("Your reminder is set for nine.", True), ("I copied it to the clipboard.", True),
    ("Sure, let me do that.", False), ("The R9700 is at fifty nine degrees.", False), ("I can't delete files.", False),
])
def test_claims_action_detector(text, expected):
    assert claims_action(text) is expected


async def test_false_claim_without_a_tool_call_is_never_spoken_and_is_retried(tmp_path):
    llm = ScriptedLLM([TextDelta("Poke is done now.")], [call("poke")], [TextDelta("Poked it.")])
    agent, ran, audit = build(tmp_path, llm)
    assert await said(agent, "poke it") == "Poked it."
    assert ran == [1]
    assert "you called no tool" in llm.seen[1][0]["content"]
    assert [m["role"] for m in llm.seen[1]].count("system") == 1  # the chat template allows only a leading system message
    assert [r["kind"] for r in audit.recent()].count("guard") == 1


async def test_claim_is_dropped_when_the_same_reply_also_calls_the_tool(tmp_path):
    llm = ScriptedLLM([TextDelta("Opened it for you. "), call("poke")], [TextDelta("I did not do it.")])
    agent, ran, _ = build(tmp_path, llm, approve=False)
    assert await said(agent, "poke it") == "I did not do it."
    assert ran == []


async def test_after_a_tool_has_run_a_confirming_sentence_is_spoken_normally(tmp_path):
    agent, _, _ = build(tmp_path, ScriptedLLM([call("peek")], [TextDelta("It is open now and the value is 42.")]))
    assert await said(agent, "what is it") == "It is open now and the value is 42."


async def test_a_second_unsupported_claim_is_accepted_not_looped_forever(tmp_path):
    llm = ScriptedLLM([TextDelta("All set.")], [TextDelta("All set.")])
    agent, _, _ = build(tmp_path, llm)
    assert await said(agent, "hi") == "All set."
    assert len(llm.seen) == 2


async def test_ordinary_short_reply_is_delivered_whole(tmp_path):
    agent, _, _ = build(tmp_path, ScriptedLLM([TextDelta("Hello"), TextDelta(" there")]))
    assert await said(agent, "hi") == "Hello there"


async def test_long_first_sentence_without_punctuation_is_released(tmp_path):
    long = "this is a long reply without any sentence ending punctuation at all, it just keeps going and going"
    agent, _, _ = build(tmp_path, ScriptedLLM([TextDelta(long[:50]), TextDelta(long[50:]), TextDelta(" more")]))
    assert await said(agent, "hi") == long + " more"
