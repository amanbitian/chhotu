"""English / Hindi / Hinglish: detection, switching, confirmations, recognition plumbing, speech text, voices."""

import base64

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import tone
from vox_assistant import language
from vox_assistant.agent import claims_action
from vox_assistant.asr_gpu import combine, join_readings, looping, split_readings
from vox_assistant.speech_text import english_number, hindi_number, normalize
from vox_assistant.voice import parse_yes_no
from vox_assistant.voices import VoiceProfiles


def test_detect_reply_language():
    assert language.detect("What's the temperature of my GPU?") == "en"
    assert language.detect("मेरे ग्राफ़िक्स कार्ड का तापमान क्या है?") == "hi"
    assert language.detect("mere GPU ka temperature kya hai yaar") == "hinglish"  # typed, Latin letters
    assert language.detect("Notepad open kar do") == "hinglish" and language.detect("aaj ka weather batao") == "hinglish"
    for english in ("Remind me in ten minutes to stretch.", "Is the main car parked?", "I have a band rehearsal",
                    "Please launch the calculator for me."):  # words like me / to / main / band are English too
        assert language.detect(english) == "en", english
    assert language.detect("मेरे GPU का temperature check करो, और memory भी बताओ") == "hinglish"
    spoken = join_readings(["नोटपैड खोलो", "Notepad Kholo"])  # two readings: judged by the Hindi one
    assert language.detect(spoken) == "hi"
    assert language.reply_language("en", "नोटपैड खोलो") == "en" and language.reply_language("auto", "नोटपैड खोलो") == "hi"


@pytest.mark.parametrize("said, mode", [
    ("hindi mein baat karo", "hi"), ("हिंदी में बात करो", "hi"), ("speak hindi", "hi"), ("please hindi me bolo", "hi"),
    ("english mein baat karo", "en"), ("switch to english", "en"), ("अंग्रेज़ी में बोलो", "en"),
    ("hinglish mein baat karo", "hinglish"), ("talk hinglish", "hinglish"),
    ("automatic language", "auto"), ("match my language", "auto"),
    ("हिंदी में बेत करो", "hi"), ("hindi main bait caro", "hi"),  # as Whisper heard it from an English voice
    ("what is hindi", None), ("hindi", None), ("how do you say water in hindi", None),
    ("translate this to english", None), ("open notepad", None), ("hindi english hinglish mein", None),
])
def test_spoken_language_switches(said, mode):
    assert language.switch_command(said) == mode


def test_hindi_and_hinglish_yes_no_with_no_winning():
    assert parse_yes_no("हाँ") is True and parse_yes_no("haan ji") is True and parse_yes_no("ठीक है") is True
    assert parse_yes_no("नहीं") is False and parse_yes_no("nahi rehne do") is False
    assert parse_yes_no("mat karo") is False and parse_yes_no("मत करो") is False  # "karo" is there, "mat" wins
    assert parse_yes_no("हाँ नहीं") is False
    assert parse_yes_no(join_readings(["नहीं", "Nahin"])) is False  # both readings are read


def test_false_claims_are_caught_in_hindi_and_hinglish_too():
    for claim in ("Notepad खोल दिया है।", "reminder set कर दिया।", "ठीक है, हो गया।", "Notepad khol diya.", "Done."):
        assert claims_action(claim), claim
    assert not claims_action("Notepad खोलने से पहले पूछ लेते हैं।") and not claims_action("आपका GPU 45 डिग्री पर है।")


def test_two_readings_are_combined_only_when_they_add_something():
    assert combine("Hey Vox, open notepad.", "Hey Vox, open notepad.") == ("Hey Vox, open notepad.", "en")
    assert combine("है वक्स, वाट वाट वाट वाट वाट वाट वाट", "Hey Vox, what's the time?")[1] == "en"  # looping Hindi
    text, used = combine("नोटपैड खोलो", "Notepad Kholo")
    assert used == "both" and split_readings(text) == ["नोटपैड खोलो", "Notepad Kholo"]
    # English speech whose Hindi decode is Devanagari gibberish: Whisper is far surer of the English reading
    gib = "है वोच्स, वाइज वाइज अज़ अरुट इन"
    assert combine(gib, "Hey Vox, what is the capital of France?", {"hi": -0.45, "en": -0.04})[1] == "en"
    assert combine(gib, "Hey Vox, what is the capital?", {"hi": -0.45, "en": -0.35})[1] == "both"  # not confident
    assert combine("है वाक्स, राम कितनी यूज हो रही है", "Hey Vox, RAM kidney use", {"hi": -0.34, "en": -0.72})[1] == "both"
    assert looping("रोड़स रोड़स रोड़स रोड़स रोड़स रोड़स") and not looping("मेरे GPU का तापमान क्या है")


def test_speech_text_numbers_per_language():
    assert hindi_number(52) == "बावन" and hindi_number(9700) == "नौ हज़ार सात सौ"
    assert hindi_number(1250000) == "बारह लाख पचास हज़ार" and hindi_number(105) == "एक सौ पाँच"
    assert english_number(45) == "forty-five" and english_number(9700) == "nine thousand seven hundred"
    assert normalize("तापमान 52 डिग्री है।") == "तापमान बावन डिग्री है।"
    assert normalize("मेमोरी 64.6 GB है, 93% भरी।") == "मेमोरी चौंसठ दशमलव छह GB है, तिरानवे प्रतिशत भरी।"
    assert "forty-five" in normalize("आपके GPU का temperature 45 degree है, सब safe है।")
    assert normalize("The R9700 is at 45 degrees.") == "The R9700 is at 45 degrees."  # English: as written
    assert "R9700" in normalize("R9700 का तापमान 45 डिग्री है।")  # names keep their digits


def test_voice_per_language(tmp_path):
    profiles = VoiceProfiles(tmp_path)
    en = profiles.create("me", tone(), 16000, "hello there", permission_confirmed=True)
    hi = profiles.create("me-hi", tone(seed=1), 16000, "नमस्ते", permission_confirmed=True, language="hi")
    assert profiles.active_id() == en["id"] and profiles.active_id("hi") == hi["id"]  # Hindi did not replace English
    assert profiles.active_id("en") == en["id"]
    profiles.revoke(hi["id"])
    assert profiles.active_id("hi") == en["id"]  # back to the main voice
    lone = VoiceProfiles(tmp_path / "x")
    only_hi = lone.create("me-hi", tone(), 16000, "नमस्ते", permission_confirmed=True, language="hi")
    assert lone.active_id() == only_hi["id"]  # the only voice is the main voice too


def test_service_picks_the_hindi_reference_for_devanagari_text(settings):
    from test_tts_service import FakeBackend

    from vox_assistant.tts_service import create_app

    backend = FakeBackend()
    c = TestClient(create_app(settings, backend_factory=lambda: backend), base_url="http://127.0.0.1")
    profiles = VoiceProfiles(settings.voices_dir)
    profiles.create("me", tone(), 16000, "hello there", permission_confirmed=True)
    profiles.create("me-hi", tone(seed=1), 16000, "नमस्ते", permission_confirmed=True, language="hi")
    assert c.post("/audio/speech", json={"input": "Hello."}).headers["x-voice"] == "me_v1"
    assert c.post("/audio/speech", json={"input": "नमस्ते, कैसे हैं?"}).headers["x-voice"] == "me-hi_v1"
    c.post("/audio/speech", json={"input": "तापमान 52 डिग्री है।"})
    assert backend.generated[-1] == "तापमान बावन डिग्री है।"  # Hindi numbers spoken as Hindi words


def test_asr_endpoint_and_client_fallback(settings):
    from vox_assistant.tts_service import create_app

    class FakeAsr:
        def readings(self, audio, languages, prompts=None):
            return {"hi": "नोटपैड खोलो", "en": "Notepad Kholo"}

    c = TestClient(create_app(settings, backend_factory=lambda: None, asr_factory=FakeAsr), base_url="http://127.0.0.1")
    pcm = (np.zeros(16000, np.int16)).tobytes()
    d = c.post("/asr", json={"pcm_b64": base64.b64encode(pcm).decode()}).json()
    assert d["used"] == "both" and d["readings"]["en"] == "Notepad Kholo" and "नोटपैड खोलो" in d["text"]
    assert c.post("/asr", json={"pcm_b64": "not base64!"}).status_code == 400


async def test_service_asr_falls_back_to_the_cpu_recogniser():
    from vox_assistant.asr_remote import ServiceASR

    class Cpu:
        async def transcribe(self, pcm, *, language="auto", prompt=None):
            return "hello from the cpu"

    asr = ServiceASR("http://127.0.0.1:9", fallback=Cpu())  # nothing listens there
    assert await asr.transcribe(b"\x00\x00" * 100) == "hello from the cpu" and asr.last_backend == "cpu"
    with pytest.raises(httpx.HTTPError):
        await ServiceASR("http://127.0.0.1:9").transcribe(b"\x00\x00")
    await asr.aclose()


async def test_neutral_voice_speaks_hindi_through_another_engine():
    from vox_assistant.tts_neutral import NeutralVoiceTTS

    class Hindi:
        sample_rate = 48000
        said = []

        async def synth(self, text):
            Hindi.said.append(text)
            yield b"\x00\x10" * 10

    tts = NeutralVoiceTTS(hindi=Hindi())
    assert tts.speaks_hindi and not NeutralVoiceTTS().speaks_hindi
    out = b"".join([c async for c in tts.synth("नमस्ते")])
    assert Hindi.said == ["नमस्ते"] and len(out) == 20 and tts.sample_rate == 48000
