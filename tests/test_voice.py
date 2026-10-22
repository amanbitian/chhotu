import asyncio
import json
import re

import pytest

from voxvortex.llm import TextDelta, ToolCall

from vox_assistant.agent import Agent
from vox_assistant.audit import AuditLog
from vox_assistant.permissions import PermissionGate, Risk, Tool, ToolRegistry, ok
from vox_assistant.voice import VoiceLoop, parse_yes_no, strip_wake_phrase

VOICED = b"\xff" * 1024
QUIET = b"\x00" * 1024


class FakeVAD:
    def prob(self, frame):
        return frame[0] / 255

    def reset(self):
        pass


class FakeASR:
    def __init__(self, *texts):
        self.texts = list(texts)

    async def transcribe(self, pcm, *, language="auto", prompt=None):
        return self.texts.pop(0) if self.texts else ""


class FakeTTS:
    sample_rate = 16000

    def __init__(self, fail=False):
        self.spoken, self.fail = [], fail

    async def synth(self, text):
        if self.fail:
            raise RuntimeError("service down")
        self.spoken.append(text)
        yield b"\x00\x10" * 1600  # audible (4096), so leading-silence trimming keeps it


class FakeSpeaker:
    def __init__(self):
        self.played, self.stops, self.pending_seconds = [], 0, 0.0

    def play(self, pcm, rate):
        self.played.append(len(pcm))

    def stop(self):
        self.stops += 1

    async def drain(self):
        pass


class ScriptedLLM:
    def __init__(self, *rounds, delay=0.0):
        self.rounds, self.delay, self.users = list(rounds), delay, []

    async def stream(self, messages, tools=None, *, max_tokens=120):
        if max_tokens == 1:  # a pre-warm, not a turn
            self.warms = getattr(self, "warms", 0) + 1
            return
        last = [m["content"] for m in messages if m["role"] == "user"][-1]
        self.langs = getattr(self, "langs", []) + [re.match(r"^\[\d\d:\d\d, reply in (\w+)\]", last).group(1)]
        self.users.append(re.sub(r"^\[\d\d:\d\d, reply in \w+\] ", "", last))
        if self.delay:
            await asyncio.sleep(self.delay)
        for ev in self.rounds.pop(0) if self.rounds else [TextDelta("ok.")]:
            yield ev


def utterance(n_voiced=12, n_quiet=18):
    return [VOICED] * n_voiced + [QUIET] * n_quiet


async def feed(frames):
    for f in frames:
        yield f
        await asyncio.sleep(0.0005)
    await asyncio.sleep(0.15)


def build(settings, tmp_path, *, asr, llm, tts=None, neutral=None, mode="wake", confirm_timeout=2.0, acks=False):
    audit = AuditLog(tmp_path / "a.sqlite3")
    ran = []
    if not acks:
        settings.ack_phrases = []

    async def poke(args):
        ran.append(args)
        return ok(done=True)

    async def peek(args):
        await asyncio.sleep(0.05)
        return ok(value=42)

    reg = ToolRegistry([Tool("poke", "act", {"type": "object", "properties": {}}, Risk.ACT, poke,
                             describe=lambda a: "poke the thing"),
                        Tool("peek", "read", {"type": "object", "properties": {}}, Risk.READ, peek)])
    agent = Agent(llm, reg, PermissionGate(None, audit), audit)
    events = []
    loop = VoiceLoop(agent=agent, asr=asr, tts=tts or FakeTTS(), neutral_tts=neutral or FakeTTS(), speaker=FakeSpeaker(),
                     vad=FakeVAD(), settings=settings, mode=mode, confirm_timeout_s=confirm_timeout, on_event=events.append)
    return loop, events, ran


def kinds(events):
    return [e["kind"] for e in events]


def test_parse_yes_no():
    assert parse_yes_no("Yes, go ahead") is True
    assert parse_yes_no("no don't do it") is False
    assert parse_yes_no("yes no") is False
    assert parse_yes_no("hmm maybe") is None
    assert parse_yes_no("") is None


def test_first_chunk_ends_at_the_first_clause_or_sentence_whichever_is_sooner():
    from vox_assistant.voice import EagerSegmenter

    def run(*parts):
        seg, out = EagerSegmenter(), []
        for p in parts:
            out += seg.push(p)
        tail = seg.flush()
        return out + ([tail] if tail else [])

    assert run("Two models are ", "running: the R9700 has Qwen. ", "Both are cool.") == \
        ["Two models are running:", "the R9700 has Qwen.", "Both are cool."]
    assert run("It is fine. Both cards run cool, and quiet. ") == ["It is fine.", "Both cards run cool, and quiet."]
    assert run("Yes, ", "it is open.") == ["Yes, it is open."]  # a comma after one word is too early to cut
    assert run("Version 3.8 is loaded.") == ["Version 3.8 is loaded."]  # no split inside a number


def test_wake_phrase_extraction():
    assert strip_wake_phrase("Hey Vox, what time is it?", "hey vox") == "what time is it"
    assert strip_wake_phrase("hey vox", "hey vox") == ""
    assert strip_wake_phrase("okay hey vox open notepad", "hey vox") == "open notepad"
    assert strip_wake_phrase("hey fox open notepad", "hey vox") is None
    assert strip_wake_phrase("what time is it", "hey vox") is None


async def test_ignores_speech_without_the_wake_phrase(settings, tmp_path):
    llm = ScriptedLLM()
    loop, events, _ = build(settings, tmp_path, asr=FakeASR("what time is it"), llm=llm)
    await loop.run(feed(utterance()))
    assert llm.users == [] and "say" not in kinds(events)


async def test_wake_phrase_starts_a_turn_and_follow_up_needs_no_wake(settings, tmp_path):
    llm = ScriptedLLM([TextDelta("It is nine. ")], [TextDelta("Sure thing.")])
    tts = FakeTTS()
    loop, events, _ = build(settings, tmp_path, asr=FakeASR("Hey Vox, what time is it", "and tomorrow"), llm=llm, tts=tts)
    await loop.run(feed(utterance() + [QUIET] * 5 + utterance()))
    assert llm.users == ["what time is it", "and tomorrow"]
    assert tts.spoken == ["It is nine.", "Sure thing."]


async def test_wake_phrase_alone_just_acknowledges(settings, tmp_path):
    llm = ScriptedLLM()
    tts = FakeTTS()
    loop, _, _ = build(settings, tmp_path, asr=FakeASR("hey vox"), llm=llm, tts=tts)
    await loop.run(feed(utterance()))
    assert tts.spoken == ["Yes?"] and llm.users == []


async def test_always_mode_needs_no_wake_phrase(settings, tmp_path):
    llm = ScriptedLLM([TextDelta("Hi.")])
    loop, _, _ = build(settings, tmp_path, asr=FakeASR("hello there"), llm=llm, mode="always")
    await loop.run(feed(utterance()))
    assert llm.users == ["hello there"]


async def test_short_noise_is_discarded(settings, tmp_path):
    asr = FakeASR("hey vox hello")
    loop, _, _ = build(settings, tmp_path, asr=asr, llm=ScriptedLLM(), mode="always")
    await loop.run(feed([VOICED] * 3 + [QUIET] * 20))
    assert asr.texts == ["hey vox hello"]  # never transcribed


async def test_barge_in_cancels_the_reply_and_clears_the_speaker(settings, tmp_path):
    llm = ScriptedLLM([TextDelta("A very long answer.")], delay=5.0)
    loop, events, _ = build(settings, tmp_path, asr=FakeASR("hey vox tell me a story", "stop that"), llm=llm)
    # first utterance starts a (slow) turn; second is spoken over it
    await loop.run(feed(utterance() + [QUIET] * 3 + utterance(n_voiced=20)))
    assert "interrupted" in kinds(events)
    assert loop.speaker.stops >= 1
    assert loop._turn.cancelled() or loop._turn.done()


async def test_voice_confirmation_yes_runs_the_action(settings, tmp_path):
    llm = ScriptedLLM([ToolCall("c1", "poke", {}, "{}")], [TextDelta("Done.")])
    tts = FakeTTS()
    loop, events, ran = build(settings, tmp_path, asr=FakeASR("hey vox poke it", "yes please"), llm=llm, tts=tts)
    await loop.run(feed(utterance() + [QUIET] * 40 + utterance()))
    assert ran == [{}]
    assert any("poke the thing" in s for s in tts.spoken)
    assert [e["approved"] for e in events if e["kind"] == "confirm_result"] == [True]


async def test_voice_confirmation_no_and_timeout_deny(settings, tmp_path):
    for second, expect in (("no, cancel that", "denied"), (None, "timeout")):
        llm = ScriptedLLM([ToolCall("c1", "poke", {}, "{}")], [TextDelta("Okay.")])
        asr = FakeASR("hey vox poke it", *([second] if second else []))
        loop, events, ran = build(settings, tmp_path, asr=asr, llm=llm, confirm_timeout=0.3)
        frames = utterance() + [QUIET] * 40 + (utterance() if second else [QUIET] * 5)
        await loop.run(feed(frames))
        assert ran == []
        result = [e for e in events if e["kind"] == "confirm_result"][0]
        assert result["approved"] is False


async def test_mute_and_unmute_by_voice(settings, tmp_path):
    llm = ScriptedLLM([TextDelta("Back.")])
    asr = FakeASR("hey vox mute", "hey vox what is up", "hey vox unmute please")
    loop, events, _ = build(settings, tmp_path, asr=asr, llm=llm)
    gap = [QUIET] * 40
    await loop.run(feed(utterance() + gap + utterance() + gap + utterance()))
    assert llm.users == []  # nothing reached the model while muted
    assert "muted" in kinds(events) and "unmuted" in kinds(events)
    assert not loop.muted


async def test_neutral_voice_switch_and_fallback(settings, tmp_path):
    cloned, neutral = FakeTTS(), FakeTTS()
    loop, events, _ = build(settings, tmp_path, asr=FakeASR("hey vox use a neutral voice", "hello"),
                            llm=ScriptedLLM([TextDelta("Hi there.")]), tts=cloned, neutral=neutral)
    await loop.run(feed(utterance() + [QUIET] * 40 + utterance()))
    assert loop.voice_mode == "neutral"
    assert neutral.spoken[-1] == "Hi there." and "Hi there." not in cloned.spoken

    broken, neutral2 = FakeTTS(fail=True), FakeTTS()
    loop, events, _ = build(settings, tmp_path, asr=FakeASR("hey vox hello"), llm=ScriptedLLM([TextDelta("Hi.")]),
                            tts=broken, neutral=neutral2)
    await loop.run(feed(utterance()))
    assert neutral2.spoken == ["Hi."] and "tts_fallback" in kinds(events)


async def test_spotter_hears_the_wake_phrase_mid_utterance_chimes_and_prewarms(settings, tmp_path):
    from vox_assistant.voice import CHIME

    spotter = FakeASR("hey vox what", "hey vox what time")
    llm = ScriptedLLM([TextDelta("Nine.")])
    tts = FakeTTS()
    loop, events, _ = build(settings, tmp_path, asr=FakeASR("Hey Vox, what time is it"), llm=llm, tts=tts)
    loop.spotter = spotter
    await loop.run(feed(utterance(n_voiced=40)))
    k = kinds(events)
    assert k.index("wake") < k.index("heard")  # acknowledged before the utterance was over
    assert [e["via"] for e in events if e["kind"] == "wake"] == ["spotter"]
    assert len(CHIME) in loop.speaker.played and llm.warms == 1
    assert llm.users == ["what time is it"] and tts.spoken == ["Nine."]


async def test_spotter_ignores_other_speech_and_is_not_used_when_awake(settings, tmp_path):
    spotter = FakeASR("what a nice day", "what a nice day today")
    llm = ScriptedLLM()
    loop, events, _ = build(settings, tmp_path, asr=FakeASR("what a nice day today"), llm=llm)
    loop.spotter = spotter
    await loop.run(feed(utterance(n_voiced=40)))
    assert "wake" not in kinds(events) and llm.users == [] and getattr(llm, "warms", 0) == 0

    spotter = FakeASR("anything")
    loop, events, _ = build(settings, tmp_path, asr=FakeASR("hello"), llm=ScriptedLLM([TextDelta("Hi.")]), mode="always")
    loop.spotter = spotter
    await loop.run(feed(utterance(n_voiced=40)))
    assert spotter.texts == ["anything"]  # always listening: nothing to spot


async def test_hotkey_wake_chimes_and_the_next_utterance_needs_no_wake_phrase(settings, tmp_path):
    llm = ScriptedLLM([TextDelta("Opening it.")])
    loop, events, _ = build(settings, tmp_path, asr=FakeASR("what is the gpu temperature"), llm=llm)
    loop.wake("hotkey")
    await loop.run(feed(utterance()))
    assert [e["via"] for e in events if e["kind"] == "wake"] == ["hotkey"]
    assert llm.users == ["what is the gpu temperature"]


async def test_wake_phrase_alone_after_a_chime_is_not_answered_twice(settings, tmp_path):
    tts = FakeTTS()
    loop, _, _ = build(settings, tmp_path, asr=FakeASR("hey vox"), llm=ScriptedLLM(), tts=tts)
    loop.spotter = FakeASR("hey vox")
    await loop.run(feed(utterance(n_voiced=25)))
    assert tts.spoken == []  # the chime was the acknowledgement; no "Yes?"
    assert loop._listening()


async def test_guest_mode_uses_the_neutral_voice_and_discloses(settings, tmp_path):
    cloned, neutral = FakeTTS(), FakeTTS()
    asr = FakeASR("hey vox guest mode on", "hey vox hello", "hey vox guest mode off", "hey vox hello again")
    llm = ScriptedLLM([TextDelta("Hi.")], [TextDelta("Hello.")])
    loop, events, _ = build(settings, tmp_path, asr=asr, llm=llm, tts=cloned, neutral=neutral)
    gap = [QUIET] * 40
    await loop.run(feed(utterance() + gap + utterance() + gap + utterance() + gap + utterance()))
    assert [e["on"] for e in events if e["kind"] == "guest_mode"] == [True, False]
    assert "AI assistant" in neutral.spoken[0] and "Hi." in neutral.spoken and "Hi." not in cloned.spoken
    assert cloned.spoken == ["Guest mode is off.", "Hello."]  # back in the cloned voice after guest mode


async def test_guest_mode_from_settings_discloses_before_the_first_reply(settings, tmp_path):
    from vox_assistant.disclosure import GUEST_NOTICE

    settings.guest_mode = True
    neutral = FakeTTS()
    loop, _, _ = build(settings, tmp_path, asr=FakeASR("hey vox hello"), llm=ScriptedLLM([TextDelta("Hi.")]), neutral=neutral)
    await loop.run(feed(utterance()))
    assert neutral.spoken == [GUEST_NOTICE, "Hi."]


async def test_keep_warm_touches_the_models_while_idle_and_not_while_muted(settings, tmp_path):
    class WarmTTS(FakeTTS):
        warms = 0

        async def warm(self):
            WarmTTS.warms += 1

    async def idle(seconds):
        for _ in range(int(seconds / 0.02)):
            yield QUIET
            await asyncio.sleep(0.02)

    settings.keep_warm_s = 0.1
    llm = ScriptedLLM()
    loop, _, _ = build(settings, tmp_path, asr=FakeASR(), llm=llm, tts=WarmTTS())
    await loop.run(idle(0.55))
    assert llm.warms >= 3 and WarmTTS.warms >= 3

    llm = ScriptedLLM()
    loop, _, _ = build(settings, tmp_path, asr=FakeASR(), llm=llm, tts=WarmTTS())
    loop.muted = True
    await loop.run(idle(0.35))
    assert getattr(llm, "warms", 0) == 0


async def test_a_read_tool_is_covered_by_a_cached_acknowledgement_but_an_action_is_not(settings, tmp_path):
    settings.ack_phrases = ["Let me check."]
    tts = FakeTTS()
    llm = ScriptedLLM([ToolCall("c1", "peek", {}, "{}")], [TextDelta("It is forty two.")],
                      [ToolCall("c2", "poke", {}, "{}")], [TextDelta("Okay.")])
    loop, events, _ = build(settings, tmp_path, asr=FakeASR("hey vox what is it", "and poke it", "no"), llm=llm, tts=tts,
                            acks=True)
    gap = [QUIET] * 40
    await loop.run(feed([QUIET] * 20 + utterance() + gap + utterance() + gap + utterance()))
    assert tts.spoken[0] == "Let me check."  # prepared once at start, not during the turn
    assert tts.spoken.count("Let me check.") == 1 and kinds(events).count("ack") == 1
    k = kinds(events)
    assert k.index("ack") < k.index("say")  # heard before the answer
    assert len(tts.spoken[0]) and any("poke the thing" in s for s in tts.spoken)  # the action got its question, no ack


async def test_speculative_transcript_is_used_when_the_pause_becomes_the_end(settings, tmp_path):
    asr = FakeASR("hey vox what time is it")
    llm = ScriptedLLM([TextDelta("Nine.")])
    loop, _, _ = build(settings, tmp_path, asr=asr, llm=llm)
    await loop.run(feed(utterance()))
    assert llm.users == ["what time is it"] and asr.texts == []  # transcribed exactly once

    # A pause shorter than the end of turn: the early transcript is thrown away and the whole utterance is used.
    asr = FakeASR("hey vox what", "hey vox what time is it")
    llm = ScriptedLLM([TextDelta("Nine.")])
    loop, _, _ = build(settings, tmp_path, asr=asr, llm=llm)
    await loop.run(feed([VOICED] * 12 + [QUIET] * 9 + [VOICED] * 12 + [QUIET] * 18))
    assert llm.users == ["what time is it"]


def test_lead_trim_drops_opening_silence_across_pieces_and_keeps_a_little():
    import numpy as np

    from vox_assistant.voice import LeadTrim

    rate = 1000
    silence, voice = np.zeros(300, np.int16), np.full(200, 5000, np.int16)
    trim = LeadTrim(rate, keep_s=0.02)
    out = trim(silence[:150].tobytes()) + trim(silence[150:].tobytes()) + trim(voice.tobytes()) + trim(silence.tobytes())
    a = np.frombuffer(out, np.int16)
    assert len(a) == 20 + 200 + 300 and (a[:20] == 0).all() and (a[20:220] == 5000).all()  # later silence untouched
    quiet = LeadTrim(rate, max_s=0.5)  # never more than max_s is dropped, even if the clip is all silence
    assert quiet(silence.tobytes()) == b"" and len(quiet(silence.tobytes())) == (30 + 300) * 2  # 30 ms lead kept


def test_resampler_is_seamless_across_pieces():
    import numpy as np

    from vox_assistant.audio_io import Resampler

    t = np.arange(22050) / 22050
    x = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    whole = np.frombuffer(Resampler(22050)(x.tobytes()), np.int16)
    r = Resampler(22050)
    pieces = np.frombuffer(b"".join(r(x[i:i + 977].tobytes()) for i in range(0, len(x), 977)), np.int16)
    assert abs(len(whole) - 48000) <= 3 and abs(len(pieces) - len(whole)) <= 1
    n = min(len(whole), len(pieces))
    assert np.abs(whole[:n].astype(int) - pieces[:n]).max() <= 1  # identical whether it arrives whole or in pieces


async def test_reminder_is_marked_fired_only_after_it_was_spoken(settings, tmp_path):
    from vox_assistant.tools.reminders import ReminderStore

    now = [1000.0]
    store = ReminderStore(tmp_path / "r.sqlite3", clock=lambda: now[0])
    store.add("check the print", 0)
    assert [r["text"] for r in store.claim_due()] == ["check the print"]
    assert store.claim_due() == []  # claimed: not handed out twice
    now[0] += 121  # the process died before speaking it: after two minutes it is due again
    r = store.claim_due()
    assert [x["text"] for x in r] == ["check the print"]
    store.mark_fired(r[0]["id"])
    now[0] += 500
    assert store.claim_due() == [] and store.due() == []
    store.close()


async def test_hindi_wake_phrase_and_auto_language_reply(settings, tmp_path):
    from vox_assistant.asr_gpu import join_readings

    llm = ScriptedLLM([TextDelta("ठीक है।")])
    asr = FakeASR(join_readings(["हे वाक्स, भारत की राजधानी क्या है?", "Hey Vox, Bharat ki rajdhani kya hai?"]))
    loop, _, _ = build(settings, tmp_path, asr=asr, llm=llm)
    await loop.run(feed(utterance()))
    assert llm.users == [join_readings(["भारत की राजधानी क्या है", "bharat ki rajdhani kya hai"])]  # wake stripped from both
    assert llm.langs == ["Hindi"]  # auto: replies in the language spoken


async def test_switch_language_by_voice_then_confirm_in_hindi(settings, tmp_path):
    llm = ScriptedLLM([ToolCall("c1", "poke", {}, "{}")], [TextDelta("हो गया।")])
    tts = FakeTTS()
    asr = FakeASR("hey vox hindi mein baat karo", "hey vox poke it", "हाँ")
    loop, events, ran = build(settings, tmp_path, asr=asr, llm=llm, tts=tts)
    gap = [QUIET] * 40
    await loop.run(feed(utterance() + gap + utterance() + gap + utterance()))
    assert [e["mode"] for e in events if e["kind"] == "language"] == ["hi"]
    assert tts.spoken[0] == "ठीक है, अब हिंदी में बात करते हैं।"
    assert llm.langs == ["Hindi", "Hindi"]  # English words, Hindi reply: the mode decides
    assert any("करूँ?" in s for s in tts.spoken) and ran == [{}]  # asked in Hindi, "हाँ" approved


async def test_hindi_no_refuses(settings, tmp_path):
    settings.language = "hi"
    llm = ScriptedLLM([ToolCall("c1", "poke", {}, "{}")], [TextDelta("ठीक है।")])
    loop, events, ran = build(settings, tmp_path, asr=FakeASR("hey vox poke it", "नहीं, मत करो"), llm=llm)
    await loop.run(feed(utterance() + [QUIET] * 40 + utterance()))
    assert ran == [] and [e["approved"] for e in events if e["kind"] == "confirm_result"] == [False]


async def test_due_reminders_are_spoken(settings, tmp_path):
    class Store:
        def __init__(self):
            self.items = [{"id": 1, "text": "check the print", "due": 0}]

        def due(self):
            out, self.items = self.items, []
            return out

    tts = FakeTTS()
    loop, events, _ = build(settings, tmp_path, asr=FakeASR(), llm=ScriptedLLM(), tts=tts)
    loop.reminders = Store()

    async def slow():
        for _ in range(60):
            yield QUIET
            await asyncio.sleep(0.05)

    await loop.run(slow())
    assert tts.spoken == ["Reminder: check the print"] and "reminder" in kinds(events)
