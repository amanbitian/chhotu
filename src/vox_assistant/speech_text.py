"""Text as the voice model should read it, per language. Applied by the voice service to everything it speaks.

Measured with VoxCPM2 (synthesize, then transcribe back): English text is read correctly as written, digits included,
and VoxVortex's English normalizer made it worse ("R9700" -> "Rnine thousand seven hundred" -> heard as "Raman 1700").
In Hindi text digits are read badly ("52 डिग्री" -> heard "पास्ट दू") and spelled-out numbers well, so Hindi gets Hindi
number words (बावन), and Hinglish gets English number words (people say numbers in English when they mix).
Digits inside names (R9700, 27B, 3.8 in "Qwen 3.8") are left alone.
"""

from __future__ import annotations

import re

from . import language

_HI_0_99 = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस इक्कीस बाईस "
    "तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस "
    "उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास इक्यावन बावन तिरेपन "
    "चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरेसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर "
    "बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी "
    "सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे"
).split()
assert len(_HI_0_99) == 100

_EN_0_19 = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
            "seventeen eighteen nineteen").split()
_EN_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def hindi_number(n: int) -> str:
    """Indian grouping: करोड़ (10^7), लाख (10^5), हज़ार, सौ."""
    if n < 100:
        return _HI_0_99[n]
    parts = []
    for value, word in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n >= value:
            parts.append(f"{hindi_number(n // value)} {word}")
            n %= value
    if n:
        parts.append(_HI_0_99[n])
    return " ".join(parts)


def english_number(n: int) -> str:
    if n < 20:
        return _EN_0_19[n]
    if n < 100:
        return _EN_TENS[n // 10 - 2] + (f"-{_EN_0_19[n % 10]}" if n % 10 else "")
    for value, word in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")):
        if n >= value:
            rest = n % value
            return f"{english_number(n // value)} {word}" + (f" {english_number(rest)}" if rest else "")
    return str(n)


# A number that stands alone: not glued to letters (R9700, 27B, H100) and not part of a version (Qwen 3.8 stays).
_NUMBER = re.compile(r"(?<![\w.])(\d{1,12})(?:\.(\d{1,3}))?(?![\w.]|\.\d)")
_TIME = re.compile(r"(?<![\w:])([01]?\d|2[0-3]):([0-5]\d)(?![\w:])")


def _spell(match: re.Match, lang: str) -> str:
    whole, frac = int(match.group(1)), match.group(2)
    if lang == "hi":
        words = hindi_number(whole) + ("" if not frac else " दशमलव " + " ".join(_HI_0_99[int(d)] for d in frac))
    else:
        words = english_number(whole) + ("" if not frac else " point " + " ".join(_EN_0_19[int(d)] for d in frac))
    return words


def normalize(text: str, lang: str | None = None) -> str:
    """`lang` None: decided from the text itself ('en', 'hi' or 'hinglish')."""
    lang = lang or language.detect(text)
    if lang == "en":
        return text
    if lang == "hi":
        text = _TIME.sub(lambda m: f"{hindi_number(int(m.group(1)))} बजकर {hindi_number(int(m.group(2)))} मिनट", text)
        text = re.sub(r"\s*°\s*C\b", " डिग्री", text)
        text = re.sub(r"\s*%", " प्रतिशत", text)
        return _NUMBER.sub(lambda m: _spell(m, "hi"), text)
    # Hinglish: numbers the way they are said when mixing, in English
    text = _TIME.sub(lambda m: f"{english_number(int(m.group(1)))} {english_number(int(m.group(2))) if int(m.group(2)) else 'o clock'}", text)
    text = re.sub(r"\s*°\s*C\b", " degree", text)
    text = re.sub(r"\s*%", " percent", text)
    return _NUMBER.sub(lambda m: _spell(m, "en"), text)
