"""The hands-free loop: mic frames -> VAD -> utterance -> ASR -> (wake gate) -> agent -> TTS -> speaker.

Turn-taking follows VoxVortex's rules: 3 voiced frames start an utterance, `end_silence_ms` of quiet ends it, and
speech sustained for `barge_in_ms` while the assistant is talking cancels the reply and clears the speaker.
Everything it talks to is injected, so the loop is tested with scripted audio and no hardware.

Reacting quickly: while asleep, a small `spotter` ASR reads the first seconds of an utterance *while it is still being
spoken*; when it hears the wake phrase the loop chimes at once and pre-warms the LLM and voice model, so they are in
VRAM with the prompt cached by the time the full transcript is ready. A hotkey (`wake()`) does the same.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from typing import AsyncIterator, Awaitable, Callable

import numpy as np

from voxvortex.asr import is_hallucination
from voxvortex.audio import FRAME_MS, pcm_duration_ms, SAMPLE_RATE
from voxvortex.tts import SentenceSegmenter

from .agent import Agent, Say, ToolFinished, ToolStarted
from .disclosure import GUEST_NOTICE

log = logging.getLogger(__name__)

WARM_EVERY_MS = 15000  # at most one pre-warm per 15 s
SPECULATE_AFTER_MS = 200  # start transcribing at this much silence; used if the utterance then ends (450 ms)
SPOT_FROM_MS, SPOT_UNTIL_MS, SPOT_STEP_MS = 600, 3000, 250  # the wake phrase is at the start of an utterance


def _chime(rate: int = 24000) -> bytes:
    """Two short rising tones: 'I heard the wake phrase'."""
    t = np.arange(int(rate * 0.07)) / rate
    tone = np.concatenate([np.sin(2 * np.pi * 880 * t), np.sin(2 * np.pi * 1320 * t)])
    ramp = np.minimum(1.0, np.minimum(np.arange(len(tone)), len(tone) - 1 - np.arange(len(tone))) / (rate * 0.008))
    return (tone * ramp * 0.2 * 32767).astype("<i2").tobytes()


CHIME_RATE = 24000
CHIME = _chime(CHIME_RATE)

# The first chunk of a reply ends at the first clause break after a few words ("Two models are running:"), not at the
# end of the sentence: speech synthesis is not streamed within a chunk, so a long first sentence delayed the first
# audio by ~7 s (measured). Later chunks are whole sentences, synthesized while earlier ones play.
_CLAUSE_END = re.compile(r"^((?:\S+\s+){2,}?\S*?[,;:–—])\s")  # at least three words, then , ; : or a dash
_SENTENCE_END = re.compile(r"^(.*?[.!?])\s")


class EagerSegmenter:
    def __init__(self) -> None:
        self.inner = SentenceSegmenter()
        self.first = True
        self.buf = ""

    def push(self, text: str) -> list[str]:
        if not self.first:
            return self.inner.push(text)
        self.buf += text
        found = [m for m in (_CLAUSE_END.match(self.buf), _SENTENCE_END.match(self.buf)) if m]
        if not found:
            return []
        m = min(found, key=lambda x: x.end())  # whichever boundary comes first
        head, rest = m.group(1).strip(), self.buf[m.end():]
        self.first, self.buf = False, ""
        return [head, *self.inner.push(rest)]

    def flush(self) -> str | None:
        if self.first:
            self.first, text, self.buf = False, self.buf.strip(), ""
            return text or None
        return self.inner.flush()

YES = {"yes", "yeah", "yep", "sure", "okay", "ok", "go", "proceed", "confirm", "do"}
NO = {"no", "nope", "don't", "dont", "cancel", "stop", "never", "negative", "nah"}
_WORDS = re.compile(r"[a-z']+")


def parse_yes_no(text: str) -> bool | None:
    words = _WORDS.findall(text.lower())
    if not words:
        return None
    if any(w in NO for w in words):  # "no, don't" must never read as consent
        return False
    return True if any(w in YES for w in words) else None


def strip_wake_phrase(text: str, phrase: str | list[str]) -> str | None:
    """Return the request after the wake phrase, '' if only the phrase was said, None if it is absent.

    `phrase` may be a list (the phrase plus spellings Whisper tends to produce for it)."""
    words = " ".join(_WORDS.findall(text.lower())).split()
    for candidate in ([phrase] if isinstance(phrase, str) else phrase):
        want = " ".join(_WORDS.findall(candidate.lower())).split()
        n = len(want)
        for start in range(0, min(3, max(len(words) - n + 1, 0))):  # allow a word or two before it ("okay hey vox")
            if words[start : start + n] == want:
                return " ".join(words[start + n :])
    return None


class VoiceLoop:
    def __init__(self, *, agent: Agent, asr, tts, neutral_tts, speaker, vad, settings, reminders=None, spotter=None,
                 mode: str = "wake", end_silence_ms: int = 450, min_speech_ms: int = 250, barge_in_ms: int = 320,
                 awake_seconds: float = 30.0, confirm_timeout_s: float = 15.0, on_event: Callable[[dict], None] | None = None) -> None:
        self.agent, self.asr, self.speaker, self.vad, self.settings, self.reminders = agent, asr, speaker, vad, settings, reminders
        self.spotter = spotter
        self.cloned_tts, self.neutral_tts = tts, neutral_tts
        self.voice_mode = "cloned"  # or "neutral"
        # Guest mode: other people can hear. Neutral voice, and the next reply starts by saying this is an AI.
        self.guest = bool(getattr(settings, "guest_mode", False))
        self._notice_due = self.guest
        self._last_warm_ms = -1e12
        self._spot_task: asyncio.Task | None = None
        self._spot_len_ms = 0.0
        self._chimed = False  # the current wake was already acknowledged with a chime
        self._spec: asyncio.Task | None = None  # speculative transcript of the utterance so far
        self._acks: dict[int, list[tuple[bytes, int]]] = {}  # id(tts engine) -> pre-synthesized acknowledgements
        self._ack_next = 0
        self.mode = mode  # "wake": wait for the wake phrase; "always": every utterance is for the assistant
        self.end_silence_ms, self.min_speech_ms, self.barge_in_ms = end_silence_ms, min_speech_ms, barge_in_ms
        self.awake_seconds, self.confirm_timeout_s = awake_seconds, confirm_timeout_s
        self.on_event = on_event or (lambda e: None)
        self.muted = False
        self.clock_ms = 0.0
        self._awake_until = -1.0
        self._turn: asyncio.Task | None = None
        self._confirm_future: asyncio.Future | None = None
        self._tasks: set[asyncio.Task] = set()
        agent.gate.confirm = self._confirm
        self._warned_fallback = False

    # -- helpers --------------------------------------------------------------------------
    @property
    def tts(self):
        if self.guest or self.voice_mode == "neutral" or self.cloned_tts is None:
            return self.neutral_tts
        return self.cloned_tts

    def _listening(self) -> bool:
        """Is the next utterance meant for the assistant without a wake phrase?"""
        return not self.muted and (self.mode == "always" or self.clock_ms < self._awake_until)

    def wake(self, via: str = "hotkey") -> None:
        """Wake as if the wake phrase was heard: chime, stay awake, warm the models. Call on the event loop thread."""
        if self.muted:
            return
        self._awake_until = self.clock_ms + self.awake_seconds * 1000
        self._chimed = True
        self.emit("wake", via=via)
        self.speaker.play(CHIME, CHIME_RATE)
        self._warm()

    def _warm(self, force: bool = False) -> None:
        if not force and self.clock_ms - self._last_warm_ms < WARM_EVERY_MS:
            return
        self._last_warm_ms = self.clock_ms
        self._spawn(self.agent.prewarm())
        if self.tts is self.cloned_tts and hasattr(self.cloned_tts, "warm"):
            self._spawn(self.cloned_tts.warm())

    async def _keep_warm_loop(self, every_s: float) -> None:
        """Windows lets an idle GPU go cold within ~30 s (first token / first audio then +1.5-2.6 s)."""
        while True:
            await asyncio.sleep(every_s)
            if not self.muted and not self._busy_speaking() and self._turn_idle():
                self._warm(force=True)

    def _turn_idle(self) -> bool:
        return self._turn is None or self._turn.done()

    def _maybe_spot(self, utter: list[bytes]) -> None:
        if self.spotter is None or self.muted or self._listening():
            return
        length = len(utter) * FRAME_MS
        if not SPOT_FROM_MS <= length <= SPOT_UNTIL_MS or length - self._spot_len_ms < SPOT_STEP_MS:
            return
        if self._spot_task is not None and not self._spot_task.done():
            return
        self._spot_len_ms = length
        self._spot_task = self._spawn(self._spot(b"".join(utter)))

    async def _spot(self, pcm: bytes) -> None:
        try:
            # No prompt here: priming tiny.en with the phrase made it "hear" it in 1 of 10 unrelated sentences
            # (0 of 10 without, and still 4 of 4 real ones). The full transcript keeps the prompt for spelling only.
            text = await self.spotter.transcribe(pcm, language="en", prompt=None)
        except Exception as exc:
            log.debug("wake spotter failed: %s", exc)
            return
        if strip_wake_phrase(text, self._wake_phrases()) is not None and not self._listening():
            self.wake(via="spotter")

    def emit(self, kind: str, **data) -> None:
        self.on_event({"kind": kind, **data})

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _busy_speaking(self) -> bool:
        return (self._turn is not None and not self._turn.done() and self._confirm_future is None) or \
            self.speaker.pending_seconds > 0

    # -- main loop ------------------------------------------------------------------------
    async def run(self, frames: AsyncIterator[bytes]) -> None:
        pre: deque[bytes] = deque(maxlen=10)
        utter: list[bytes] = []
        voiced = silence = 0
        speaking_ms = 0.0  # sustained speech while the assistant talks
        in_utt = False
        reminder_task = self._spawn(self._reminder_loop()) if self.reminders else None
        keep_warm_s = getattr(self.settings, "keep_warm_s", 0)
        if keep_warm_s > 0:
            self._spawn(self._keep_warm_loop(keep_warm_s))
        if getattr(self.settings, "ack_phrases", None):
            self._spawn(self._prepare_acks())
        try:
            async for frame in frames:
                self.clock_ms += FRAME_MS
                p = self.vad.prob(frame)
                busy = self._busy_speaking()
                hot = p > (0.7 if busy else 0.5)
                if not in_utt:
                    pre.append(frame)
                    voiced = voiced + 1 if hot else 0
                    speaking_ms = speaking_ms + FRAME_MS if hot else 0.0
                    if busy and speaking_ms >= self.barge_in_ms:
                        self.interrupt()
                        in_utt, utter, silence, voiced = True, list(pre), 0, 0
                    elif not busy and voiced >= 3:
                        in_utt, utter, silence = True, list(pre), 0
                    if in_utt:
                        self._spot_len_ms = 0.0
                        if self._listening():  # this utterance is for us: get the models ready while it is spoken
                            self._warm()
                else:
                    utter.append(frame)
                    silence = 0 if hot else silence + FRAME_MS
                    if not silence:
                        self._maybe_spot(utter)
                        if self._spec is not None:  # they kept talking: the early transcript is incomplete
                            self._spec.cancel()
                            self._spec = None
                    elif silence >= SPECULATE_AFTER_MS and self._spec is None \
                            and pcm_duration_ms(b"".join(utter), SAMPLE_RATE) - silence >= self.min_speech_ms:
                        self._spec = self._spawn(self._transcribe(b"".join(utter)))
                    if silence >= self.end_silence_ms:
                        pcm = b"".join(utter)
                        in_utt, utter, voiced, speaking_ms = False, [], 0, 0.0
                        pre.clear()
                        self.vad.reset()
                        early, self._spec = self._spec, None
                        if pcm_duration_ms(pcm, SAMPLE_RATE) - silence >= self.min_speech_ms:
                            self._spawn(self._on_utterance(pcm, early))
                        elif early is not None:
                            early.cancel()
        finally:
            if reminder_task:
                reminder_task.cancel()
            for t in list(self._tasks):
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def interrupt(self) -> None:
        self.speaker.stop()
        if self._turn and not self._turn.done():
            self._turn.cancel()
        self.emit("interrupted")

    # -- utterance handling ---------------------------------------------------------------
    async def _transcribe(self, pcm: bytes) -> str:
        # Priming Whisper with the phrase makes it spell the wake word the way we match it.
        return await self.asr.transcribe(pcm, language="auto", prompt=f"{self.settings.wake_phrase.capitalize()}.")

    async def _on_utterance(self, pcm: bytes, early: asyncio.Task | None = None) -> None:
        """`early` is a transcript started at the first 200 ms of silence; the speech it covers is the same, and it
        saves ~0.25 s on every turn."""
        try:
            text = await early if early is not None else await self._transcribe(pcm)
        except Exception as exc:
            log.warning("ASR failed: %s", exc)
            self.emit("error", where="asr", message=str(exc))
            return
        if not text or is_hallucination(text, pcm_duration_ms(pcm, SAMPLE_RATE)):
            return
        self.emit("heard", text=text)
        if self._confirm_future is not None and not self._confirm_future.done():
            self._confirm_future.set_result(text)
            return
        chimed, self._chimed = self._chimed, False
        request = self._gate_by_wake(text)
        if request is None:
            return
        if not request:
            if not chimed:  # the chime already said "I'm listening"
                self._spawn(self._say_plain("Yes?"))
            return
        if self._local_command(request):
            return
        if self._turn and not self._turn.done():
            self._turn.cancel()
        self._turn = self._spawn(self._run_turn(request))

    def _wake_phrases(self) -> list[str]:
        return [self.settings.wake_phrase, *getattr(self.settings, "wake_aliases", [])]

    def _gate_by_wake(self, text: str) -> str | None:
        if self.muted:
            rest = strip_wake_phrase(text, self._wake_phrases())
            if rest is not None and re.search(r"\bunmute\b|\bwake up\b|\blisten\b", rest):
                self.muted = False
                self._awake_until = self.clock_ms + self.awake_seconds * 1000
                self.emit("unmuted")
                self._spawn(self._say_plain("I'm listening again."))
            return None
        rest = strip_wake_phrase(text, self._wake_phrases())
        if rest is not None:
            self._awake_until = self.clock_ms + self.awake_seconds * 1000
            return rest
        if self.mode == "always" or self.clock_ms < self._awake_until:
            self._awake_until = self.clock_ms + self.awake_seconds * 1000
            return text
        return None

    def _local_command(self, text: str) -> bool:
        """Privacy controls handled in code, so they work even when the LLM is down or mistaken."""
        t = " ".join(_WORDS.findall(text.lower()))
        if re.fullmatch(r"(please )?(mute|stop listening|go to sleep)( now)?", t):
            self.muted = True
            self.emit("muted")
            self._spawn(self._say_plain("Muted. Say the wake phrase and unmute when you want me back."))
            return True
        if re.fullmatch(r"(please )?(use|switch to) (a |the )?(neutral|default|generic) voice", t):
            self.voice_mode = "neutral"
            self.emit("voice_mode", mode="neutral")
            self._spawn(self._say_plain("Using the neutral voice."))
            return True
        if re.fullmatch(r"(please )?(use|switch to) my voice", t):
            self.voice_mode = "cloned"
            self.emit("voice_mode", mode="cloned")
            self._spawn(self._say_plain("Guest mode is on, so I'll keep the neutral voice until you turn it off."
                                        if self.guest else "Using your cloned voice."))
            return True
        if re.fullmatch(r"(please )?((turn off|disable|end|stop|exit) guest mode|guest mode off)( now)?|(i'?m|i am) alone( now)?", t):
            self.guest = self._notice_due = False
            self.emit("guest_mode", on=False)
            self._spawn(self._say_plain("Guest mode is off."))
            return True
        if re.fullmatch(r"(please )?((turn on|enable|start) )?guest mode( on)?|(we|i) have (guests|visitors|company)( here)?", t):
            self.guest, self._notice_due = True, False
            self.emit("guest_mode", on=True)
            self._spawn(self._say_plain(f"Guest mode is on. {GUEST_NOTICE}"))
            return True
        return False

    # -- speaking -------------------------------------------------------------------------
    async def _prepare_acks(self) -> None:
        """Synthesize the acknowledgement phrases once per voice, so they can play the instant a tool starts."""
        for engine in (self.cloned_tts, self.neutral_tts):
            if engine is None or id(engine) in self._acks:
                continue
            clips = []
            for phrase in self.settings.ack_phrases:
                try:
                    pcm = b"".join([c async for c in engine.synth(phrase)])
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.debug("could not prepare %r: %s", phrase, exc)
                    break
                clips.append((pcm, engine.sample_rate))
            if clips:
                self._acks[id(engine)] = clips

    def _ack(self) -> tuple[bytes, int] | None:
        clips = self._acks.get(id(self.tts))
        if not clips:
            return None  # never synthesize one live: that would delay the answer instead of covering for it
        self._ack_next = (self._ack_next + 1) % len(clips)
        return clips[self._ack_next]

    async def _say_plain(self, text: str) -> None:
        await self._speak_sentences(_one(text))

    async def _speak_sentences(self, sentences: AsyncIterator[str]) -> None:
        async for sentence in sentences:
            await self._speak(sentence)

    async def _speak(self, sentence: str) -> None:
        for engine in ([self.tts] if self.tts is self.neutral_tts else [self.tts, self.neutral_tts]):
            try:
                async for chunk in engine.synth(sentence):
                    self.speaker.play(chunk, engine.sample_rate)
                if engine is self.neutral_tts and self.tts is not self.neutral_tts and not self._warned_fallback:
                    self._warned_fallback = True
                    self.emit("tts_fallback")
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("TTS engine %s failed: %s", type(engine).__name__, exc)
                self.emit("error", where="tts", message=str(exc))

    async def _run_turn(self, request: str) -> None:
        segmenter = EagerSegmenter()
        queue: asyncio.Queue[str | tuple[bytes, int] | None] = asyncio.Queue()
        said_something = False

        async def talker():
            while (s := await queue.get()) is not None:
                if isinstance(s, tuple):  # a pre-synthesized acknowledgement
                    self.speaker.play(*s)
                else:
                    await self._speak(s)

        talk = asyncio.ensure_future(talker())
        if self._notice_due:  # guest mode was on from the start: disclose before the first reply
            self._notice_due = False
            queue.put_nowait(GUEST_NOTICE)
        try:
            async for ev in self.agent.run(request):
                if isinstance(ev, Say):
                    said_something = True
                    self.emit("say", text=ev.text)
                    for s in segmenter.push(ev.text):
                        queue.put_nowait(s)
                elif isinstance(ev, ToolStarted):
                    self.emit("tool", name=ev.name, args=ev.args, risk=ev.risk)
                    # Reading something takes a tool round before the answer can start: say "let me check" meanwhile.
                    # Not for actions: those are about to be asked about, and that question is the first thing said.
                    if not said_something and ev.risk == "read" and (clip := self._ack()):
                        said_something = True
                        self.emit("ack")
                        queue.put_nowait(clip)
                elif isinstance(ev, ToolFinished):
                    self.emit("tool_result", name=ev.name, decision=ev.decision, ok=ev.result.get("ok"))
            if (tail := segmenter.flush()):
                queue.put_nowait(tail)
            queue.put_nowait(None)
            await talk
            await self.speaker.drain()
        except asyncio.CancelledError:
            talk.cancel()
            self.speaker.stop()
            raise
        except Exception as exc:
            log.exception("turn failed")
            self.emit("error", where="turn", message=str(exc))
            talk.cancel()
            await self._say_plain("Sorry, something went wrong on my side.")

    # -- confirmation by voice ------------------------------------------------------------
    async def _confirm(self, intent: str, tool, args: dict) -> bool:
        for attempt in range(2):
            self.emit("confirm_request", intent=intent)
            await self._say_plain(f"I'm about to {intent}. Say yes to go ahead, or no to cancel.")
            await self.speaker.drain()
            self._confirm_future = asyncio.get_running_loop().create_future()
            try:
                answer = await asyncio.wait_for(self._confirm_future, self.confirm_timeout_s)
            except asyncio.TimeoutError:
                self.emit("confirm_result", approved=False, reason="timeout")
                return False
            finally:
                self._confirm_future = None
            verdict = parse_yes_no(answer)
            if verdict is not None:
                self.emit("confirm_result", approved=verdict)
                return verdict
            await self._say_plain("Sorry, I only need a yes or a no.")
            await self.speaker.drain()
        self.emit("confirm_result", approved=False, reason="unclear")
        return False

    # -- reminders ------------------------------------------------------------------------
    async def _reminder_loop(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            for r in self.reminders.due():
                self.emit("reminder", text=r["text"])
                while self._busy_speaking():
                    await asyncio.sleep(0.5)
                await self._say_plain(f"Reminder: {r['text']}")


async def _one(text: str):
    yield text
