"""English, Hindi and Hinglish: which language to answer in, and the assistant's own fixed phrases in each.

Every mode *understands* all three languages; the mode decides how the assistant replies:

    auto      reply in the language the user just used
    en        English
    hi        Hindi in Devanagari, technical terms and names kept in English letters
    hinglish  Hindi and English mixed the way people in India speak: Hindi words in Devanagari, English in English letters

Devanagari plus English in Latin letters is also what the voice model pronounces best: Hindi written in Latin letters is
read with English pronunciation.
"""

from __future__ import annotations

import re

MODES = ("auto", "en", "hi", "hinglish")
NAMES = {"auto": "Auto", "en": "English", "hi": "Hindi", "hinglish": "Hinglish"}
_DEVANAGARI_WORD = re.compile(r"[ऀ-ॣ०-ॿ]+")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']*")
# Common Hindi words as people type them in Latin letters (romanized Hinglish). Only words that are not also English:
# "me", "to", "the", "do", "main", "band", "par"... were here at first and made plain English look like Hinglish.
ROMAN_HINDI = {
    "hai", "hain", "hoga", "hogi", "kya", "kyu", "kyun", "kaise", "kaun", "kab", "kahan", "kitna", "kitni", "kitne",
    "ka", "ki", "ke", "ko", "mein", "mera", "meri", "mere", "tera", "teri", "aap", "aapka", "aapki", "aapke",
    "humara", "karo", "karna", "karke", "kardo", "dena", "batao", "bolo", "dekho", "kholo", "chalao", "nahi", "nahin",
    "haan", "aur", "bhi", "abhi", "toh", "wala", "wali", "raha", "rahi", "rahe", "chahiye", "accha", "acha", "theek",
    "thik", "yaar", "kuch", "baat", "baad", "pehle", "wahi", "yeh", "woh", "hoga", "gaya", "gayi", "diya", "liye",
    "kar", "pe",
}
# One of these alone marks Hinglish (Hindi verb forms and question words no English sentence contains).
_STRONG = re.compile(r"\b(karo|kardo|kar do|kar dijiye|kar dena|kholo|batao|bolo|chalao|dekho|kya|hai|hain|nahi|nahin)\b")
_TECH = {"gpu", "gpus", "ram", "cpu", "vram", "ai", "ok", "okay"}  # English words that do not indicate a language


def detect(text: str) -> str:
    """'en', 'hi' or 'hinglish' for what the user said or typed."""
    first = text.split("\n(the same speech heard as English:")[0]  # the Hindi reading when there are two
    dev = len(_DEVANAGARI_WORD.findall(first))
    latin = [w.lower() for w in _LATIN_WORD.findall(first)]
    roman_hi = sum(w in ROMAN_HINDI for w in latin)
    english = sum(w not in ROMAN_HINDI and w not in _TECH for w in latin)
    if dev:
        return "hinglish" if english >= max(2, dev // 2) else "hi"
    if _STRONG.search(" ".join(latin)) or (roman_hi >= 2 and roman_hi >= len(latin) // 4):
        return "hinglish"
    return "en"


def reply_language(mode: str, text: str) -> str:
    return detect(text) if mode == "auto" else mode


def tag(lang: str) -> str:
    """Goes into the time stamp of the user's message: '[09:05, reply in Hindi]'."""
    return f"reply in {NAMES[lang]}"


def rules(gender: str = "neutral") -> str:
    """The language part of the system prompt (stable, so the prompt prefix stays cached)."""
    self_ref = {"male": "When you speak about yourself in Hindi, use masculine verb forms (मैं देख रहा हूँ).",
                "female": "When you speak about yourself in Hindi, use feminine verb forms (मैं देख रही हूँ).",
                }.get(gender, "When you speak about yourself in Hindi, prefer forms that do not mark gender, such as "
                              "देखते हैं or करूँ?, over रहा हूँ / रही हूँ.")
    return (
        "Languages: the user may speak English, Hindi or Hinglish; understand all three. A spoken message may carry two "
        "transcriptions of the same speech, a Hindi one and '(the same speech heard as English: ...)'; understand the user "
        "from whichever makes sense and never mention the transcriptions. Speech transcripts often contain phonetic "
        "misspellings (for example 'बैटरी चेकरो' for 'battery check करो', 'Chrome Kolo' for 'Chrome खोलो'); work out the "
        "words the user meant before deciding. Reply in the language named in the brackets at "
        "the start of the user's message: English; Hindi = Hindi in Devanagari script, keeping technical terms, app, "
        "product and model names, file names and units in English letters (GPU, RAM, Notepad, Qwen 3.8); Hinglish = "
        "natural Indian Hindi-English mixing, Hindi words in Devanagari and English words in English letters. Write "
        f"numbers as digits. Tool names and tool arguments always stay in English. {self_ref}"
    )


# The assistant's own sentences (spoken without the LLM). Hindi and Hinglish avoid gendered first-person verbs.
PHRASES: dict[str, dict[str, str]] = {
    "en": {
        "wake_ack": "Yes?",
        "confirm": "I'm about to {intent}. Say yes to go ahead, or no to cancel.",
        "unclear": "Sorry, I only need a yes or a no.",
        "muted": "Muted. Say the wake phrase and unmute when you want me back.",
        "unmuted": "I'm listening again.",
        "neutral_on": "Using the neutral voice.",
        "cloned_on": "Using your cloned voice.",
        "cloned_blocked": "Guest mode is on, so I'll keep the neutral voice until you turn it off.",
        "guest_on": "Guest mode is on. {notice}",
        "guest_off": "Guest mode is off.",
        "notice": "Just so everyone knows, I'm an AI assistant, and this is a computer voice.",
        "reminder": "Reminder: {text}",
        "error": "Sorry, something went wrong on my side.",
        "switched": "Okay, I'll speak English.",
        "switched_auto": "Okay, I'll answer in whichever language you use.",
        "declined": "Okay, I won't do that.",
        "gave_up": "I could not finish that; I kept running into the same problem.",
    },
    "hi": {
        "wake_ack": "जी?",
        "confirm": "{intent} — करूँ? हाँ बोलिए तो आगे बढ़ें, ना बोलिए तो रद्द।",
        "unclear": "माफ़ कीजिए, बस हाँ या ना बोलिए।",
        "muted": "आवाज़ बंद। वापस बुलाने के लिए wake phrase के बाद unmute बोलिए।",
        "unmuted": "फिर से सुन रहे हैं।",
        "neutral_on": "अब सामान्य आवाज़ में बात होगी।",
        "cloned_on": "अब आपकी आवाज़ में बात होगी।",
        "cloned_blocked": "Guest mode चालू है, इसलिए सामान्य आवाज़ ही रहेगी।",
        "guest_on": "Guest mode चालू। {notice}",
        "guest_off": "Guest mode बंद।",
        "notice": "सबकी जानकारी के लिए: यह एक AI assistant है, और यह कंप्यूटर की आवाज़ है।",
        "reminder": "याद दिलाना था: {text}",
        "error": "माफ़ कीजिए, कुछ गड़बड़ हो गई।",
        "switched": "ठीक है, अब हिंदी में बात करते हैं।",
        "switched_auto": "ठीक है, आप जिस भाषा में बोलेंगे, उसी में जवाब मिलेगा।",
        "declined": "ठीक है, यह नहीं करेंगे।",
        "gave_up": "माफ़ कीजिए, यह पूरा नहीं हो पाया, बार-बार एक ही दिक्कत आ रही है।",
    },
    "hinglish": {
        "wake_ack": "हाँ जी?",
        "confirm": "{intent} — करूँ? Yes के लिए हाँ बोलिए, cancel के लिए ना।",
        "unclear": "Sorry, बस हाँ या ना बोलिए।",
        "muted": "Mute हो गया। वापस लाने के लिए wake phrase के बाद unmute बोलिए।",
        "unmuted": "फिर से sun रहे हैं।",
        "neutral_on": "अब neutral voice में बात होगी।",
        "cloned_on": "अब आपकी cloned voice में बात होगी।",
        "cloned_blocked": "Guest mode on है, इसलिए neutral voice ही रहेगी।",
        "guest_on": "Guest mode on। {notice}",
        "guest_off": "Guest mode off।",
        "notice": "Just so everyone knows, यह एक AI assistant है और यह computer की आवाज़ है।",
        "reminder": "Reminder: {text}",
        "error": "Sorry, कुछ गड़बड़ हो गई।",
        "switched": "Okay, अब Hinglish में बात करते हैं।",
        "switched_auto": "Okay, आप जिस language में बोलेंगे, उसी में जवाब मिलेगा।",
        "declined": "Okay, यह नहीं करेंगे।",
        "gave_up": "Sorry, यह पूरा नहीं हो पाया, बार-बार same problem आ रही है।",
    },
}
ACKS = {"hi": ["एक पल रुकिए।", "अभी देखते हैं।"], "hinglish": ["एक second।", "अभी check करते हैं।"]}


def phrase(lang: str, key: str, **kw) -> str:
    table = PHRASES.get(lang if lang in PHRASES else "en")
    return table[key].format(**kw) if kw else table[key]


# Spoken mode switches (handled without the LLM): a short utterance that names exactly one language plus a word that
# makes it a switch ("Hindi mein baat karo", "speak English", "हिंदी में बोलो"). Matched loosely on purpose: spoken by
# an English voice, "Hindi mein baat karo" came back as "हिंदी में बेत करो" / "Hindi main bait caro".
_LANG_NAMES = {
    "hinglish": r"hinglish|हिंग्लिश",
    "hi": r"hindi|हिंदी|हिन्दी|हंदी",
    "en": r"english|इंग्लिश|अंग्रेज़ी|अंग्रेजी",
}
_SWITCH_WORD = re.compile(r"\b(mein|main|mai|me|speak|talk|switch|use|answer|reply|bolo|baat)\b|में|मे\b|बोलो|बात")
_QUESTION = re.compile(r"\b(what|how|why|which|kya|kaise|kyun|meaning|word|translate)\b|क्या|कैसे|मतलब|अनुवाद")
_AUTO = re.compile(r"(please )?((auto|automatic) language|language auto|(match|follow) my language|jo bolun us mein bolo)")


def switch_command(normalized: str) -> str | None:
    if _AUTO.fullmatch(normalized):
        return "auto"
    words = normalized.split()
    if not 2 <= len(words) <= 6 or _QUESTION.search(normalized) or not _SWITCH_WORD.search(normalized):
        return None
    named = [mode for mode, pattern in _LANG_NAMES.items() if re.search(pattern, normalized)]
    return named[0] if len(named) == 1 else None
