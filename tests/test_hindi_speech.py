"""Hindi / Hinglish as it is spoken: romanized Hindi read from Devanagari, clock times said the Hindi way, the assistant's
own phrases free of Latin-letter Hindi, the reply prompt, and the echoed message stamp."""

import re

from voxvortex.llm import TextDelta

from vox_assistant import language
from vox_assistant.agent import Agent, Say
from vox_assistant.audit import AuditLog
from vox_assistant.permissions import PermissionGate, ToolRegistry
from vox_assistant.speech_text import ROMAN_TO_DEVANAGARI, devanagari_hindi, english_time, hindi_time, normalize


def test_romanized_hindi_is_spoken_from_devanagari_and_english_stays():
    # replies the 4B gave when the user typed Hinglish in English letters
    assert normalize("Aapke dono GPUs ka temperature limit se neeche hai. Dono safe hain.") == \
        "आपके दोनों GPUs का temperature limit से नीचे है. दोनों safe हैं."
    assert normalize("Notepad open karne ke liye aapko usse approve karna hoga.") == \
        "Notepad open करने के लिए आपको उससे approve करना होगा."
    assert devanagari_hindi("Notepad open kar do, phir mujhe bata do.") == "Notepad open कर दो, फिर मुझे बता दो."
    assert devanagari_hindi("यह to बढ़िया है") == "यह तो बढ़िया है"  # "to" only straight after a Hindi word
    # English text is never touched, even when it has words that are Hindi elsewhere
    for english in ("Switch to the main GPU and do it now.", "The R9700 is at 45 degrees, use the GPU.",
                    "Hi, I can open the log in a new tab."):
        assert normalize(english) == english


def test_no_mapped_word_is_an_everyday_english_word():
    english = {"to", "me", "main", "the", "do", "hi", "use", "log", "mile", "tab", "banana", "is", "us", "in", "on",
               "so", "a", "an", "it", "be", "go", "no", "say", "mat", "lo", "le"}
    assert not english & set(ROMAN_TO_DEVANAGARI)


def test_hindi_clock_times_are_said_the_way_people_say_them():
    assert hindi_time(19, 30) == "शाम साढ़े सात बजे"
    assert hindi_time(8, 15) == "सवा आठ बजे"
    assert hindi_time(20, 45) == "रात पौने नौ बजे"
    assert hindi_time(13, 30) == "दोपहर डेढ़ बजे" and hindi_time(14, 30) == "दोपहर ढाई बजे"
    assert hindi_time(14, 5) == "दोपहर दो बजकर पाँच मिनट"
    assert hindi_time(12, 0) == "बारह बजे" and hindi_time(9, 5) == "नौ बजकर पाँच मिनट"  # 12:00 / 9:05: no guessed part
    assert hindi_time(0, 10) == "रात बारह बजकर दस मिनट"
    assert normalize("मीटिंग 12:00 बजे से 12:01 बजे तक थी।") == "मीटिंग बारह बजे से बारह बजकर एक मिनट तक थी।"
    assert normalize("रिमाइंडर शाम 7:45 बजे है।") == "रिमाइंडर शाम पौने आठ बजे है।"  # the part of the day said once
    assert english_time(19, 30) == "seven thirty PM" and english_time(9, 5) == "nine oh five"
    assert english_time(12, 0) == "twelve o'clock"
    assert normalize("Meeting 9:05 baje hai.") == "Meeting nine oh five है."


def test_the_assistants_own_hindi_and_hinglish_phrases_have_no_hindi_in_english_letters():
    for lang in ("hi", "hinglish"):
        for key, text in language.PHRASES[lang].items():
            latin = {w.lower() for w in re.findall(r"[A-Za-z]+", text)}
            assert not latin & set(ROMAN_TO_DEVANAGARI), (lang, key, text)  # the voice would read them as English
    assert language.phrase("hi", "reminder_late", time="14:05", text="दवाई") == "14:05 का रिमाइंडर: दवाई"
    assert normalize(language.phrase("hi", "reminder_late", time="14:05", text="दवाई")) == \
        "दोपहर दो बजकर पाँच मिनट का रिमाइंडर: दवाई"


def test_the_prompt_asks_for_devanagari_hinglish_and_gender_free_self_reference():
    rules = language.rules("neutral")
    assert "every Hindi word in Devanagari" in rules and "even when the user typed Hindi in English letters" in rules
    assert "हम यह कर सकते हैं" in rules and "सकता हूँ" in rules
    assert "सकता हूँ" in language.rules("male") and "सकती हूँ" in language.rules("female")


class Echo:
    """The 4B starting its reply with the stamp of the user's message (seen once in 22 Hinglish replies)."""

    async def stream(self, messages, tools=None, **kw):
        for piece in ("[10:24, reply ", "in Hinglish] ", "आज का दिन ", "अच्छा है। ", "आपका?"):
            yield TextDelta(piece)


async def test_an_echoed_message_stamp_is_never_spoken_or_kept(tmp_path):
    audit = AuditLog(tmp_path / "a.sqlite3")
    agent = Agent(Echo(), ToolRegistry(), PermissionGate(None, audit), audit)
    said = "".join([ev.text async for ev in agent.run("Aaj ka din kaisa hai?", "hinglish") if isinstance(ev, Say)])
    assert said == "आज का दिन अच्छा है। आपका?"
    assert agent.history[-1]["content"] == "आज का दिन अच्छा है। आपका?"
