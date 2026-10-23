"""Text as the voice model should read it, per language. Applied by the voice service to everything it speaks.

Measured with VoxCPM2 (synthesize, then transcribe back): English text is read correctly as written, digits included,
and VoxVortex's English normalizer made it worse ("R9700" -> "Rnine thousand seven hundred" -> heard as "Raman 1700").
In Hindi text digits are read badly ("52 डिग्री" -> heard "पास्ट दू") and spelled-out numbers well, so Hindi gets Hindi
number words (बावन), and Hinglish gets English number words (people say numbers in English when they mix).
Digits inside names (R9700, 27B, 3.8 in "Qwen 3.8") are left alone. Clock times are said the Hindi way (शाम साढ़े सात
बजे). Hindi words written in English letters ("aapke dono GPU ka") are spoken from Devanagari: in Latin letters the
voice gives them English pronunciation.
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
# A clock time, with the part of the day and "बजे" the text may already say around it (said once, not twice).
_TIME = re.compile(r"(?:(सुबह|दोपहर|शाम|रात)\s+)?(?<![\w:])([01]?\d|2[0-3]):([0-5]\d)(?![\w:])(\s*बजे)?")

# Hindi words written in English letters, as the LLM often writes Hinglish when the user typed it that way. The voice
# reads Latin letters with English pronunciation ("hai" as "high"), so these are spoken from Devanagari. Only words
# that are not also English words: "to", "me", "main", "par", "hi", "use", "log", "mile", "tab", "banana" stay as written.
ROMAN_TO_DEVANAGARI = {
    # verbs: to be, auxiliaries
    "hai": "है", "hain": "हैं", "hoon": "हूँ", "hun": "हूँ", "hu": "हूँ", "ho": "हो", "tha": "था", "thi": "थी",
    "hoga": "होगा", "hogi": "होगी", "honge": "होंगे", "hota": "होता", "hoti": "होती", "hote": "होते", "hua": "हुआ",
    "hui": "हुई", "hue": "हुए", "huye": "हुए", "raha": "रहा", "rahi": "रही", "rahe": "रहे", "gaya": "गया", "gayi": "गई",
    "gaye": "गए", "diya": "दिया", "diye": "दिए", "liya": "लिया", "kiya": "किया", "kiye": "किए", "sakte": "सकते",
    "sakta": "सकता", "sakti": "सकती", "chahiye": "चाहिए", "chahie": "चाहिए", "padega": "पड़ेगा", "padegi": "पड़ेगी",
    # verbs: common stems and forms
    "karo": "करो", "karna": "करना", "karne": "करने", "karke": "करके", "kardo": "कर दो", "kar": "कर", "karta": "करता",
    "karti": "करती", "karte": "करते", "karenge": "करेंगे", "karega": "करेगा", "karegi": "करेगी", "karein": "करें",
    "karen": "करें", "kijiye": "कीजिए", "kariye": "करिए", "dijiye": "दीजिए", "dena": "देना", "dekho": "देखो",
    "dekhna": "देखना", "dekhte": "देखते", "dekh": "देख", "batao": "बताओ", "bataiye": "बताइए", "bata": "बता",
    "bataya": "बताया", "bolo": "बोलो", "bol": "बोल", "kholo": "खोलो", "khol": "खोल", "khulne": "खुलने", "khula": "खुला",
    "khuli": "खुली", "chalao": "चलाओ", "chal": "चल", "chalo": "चलो", "chalta": "चलता", "chalti": "चलती",
    "chalte": "चलते", "jaana": "जाना", "jana": "जाना", "jaata": "जाता", "jaati": "जाती", "jaate": "जाते", "ja": "जा",
    "jayega": "जाएगा", "jaayega": "जाएगा", "jayegi": "जाएगी", "aana": "आना", "aata": "आता", "aati": "आती",
    "aate": "आते", "aaya": "आया", "aayi": "आई", "aaye": "आए", "raho": "रहो", "rehna": "रहना", "rakho": "रखो",
    "lena": "लेना", "lagta": "लगता", "lagti": "लगती", "lagte": "लगते", "laga": "लगा", "lagi": "लगी", "milega": "मिलेगा",
    "milegi": "मिलेगी", "mila": "मिला", "mili": "मिली", "samjho": "समझो", "samajh": "समझ", "suno": "सुनो",
    "socho": "सोचो", "soch": "सोच", "badal": "बदल", "badalna": "बदलना", "banata": "बनाता", "banati": "बनाती",
    "banate": "बनाते", "bana": "बना", "bani": "बनी", "bane": "बने", "ban": "बन", "band": "बंद", "shuru": "शुरू",
    "dekhenge": "देखेंगे", "karun": "करूँ", "karoon": "करूँ", "ruko": "रुको", "chahte": "चाहते",
    "chahta": "चाहता", "chahti": "चाहती", "pata": "पता", "likho": "लिखो", "bhejo": "भेजो", "dhundo": "ढूँढो",
    # postpositions
    "ka": "का", "ki": "की", "ke": "के", "ko": "को", "se": "से", "mein": "में", "mei": "में", "pe": "पे", "tak": "तक",
    "wala": "वाला", "wali": "वाली", "wale": "वाले", "liye": "लिए", "saath": "साथ", "sath": "साथ", "bina": "बिना",
    "baad": "बाद", "pehle": "पहले", "pahle": "पहले", "upar": "ऊपर", "neeche": "नीचे", "niche": "नीचे",
    "andar": "अंदर", "bahar": "बाहर", "paas": "पास", "par": "पर",
    # pronouns
    "mujhe": "मुझे", "mujhse": "मुझसे", "mera": "मेरा", "meri": "मेरी", "mere": "मेरे", "hum": "हम", "hume": "हमें",
    "humein": "हमें", "hamara": "हमारा", "humara": "हमारा", "hamari": "हमारी", "humari": "हमारी", "hamare": "हमारे",
    "aap": "आप", "aapka": "आपका", "aapki": "आपकी", "aapke": "आपके", "aapko": "आपको", "aapse": "आपसे", "apka": "आपका",
    "apki": "आपकी", "apke": "आपके", "apko": "आपको", "tum": "तुम", "tumhara": "तुम्हारा", "tumhari": "तुम्हारी",
    "tumhare": "तुम्हारे", "tumhe": "तुम्हें", "yeh": "यह", "ye": "ये", "yah": "यह", "woh": "वह", "wo": "वो",
    "vo": "वो", "ise": "इसे", "isse": "इससे", "iska": "इसका", "iski": "इसकी", "iske": "इसके", "isme": "इसमें",
    "ismein": "इसमें", "usse": "उससे", "usme": "उसमें", "usmein": "उसमें", "uska": "उसका", "uski": "उसकी", "uske": "उसके",
    "unka": "उनका", "unki": "उनकी", "unke": "उनके", "sab": "सब", "sabhi": "सभी", "koi": "कोई", "kuch": "कुछ",
    "kisi": "किसी", "apna": "अपना", "apni": "अपनी", "apne": "अपने", "khud": "खुद",
    # questions and relatives
    "kya": "क्या", "kyu": "क्यों", "kyun": "क्यों", "kyon": "क्यों", "kaise": "कैसे", "kaisa": "कैसा", "kaisi": "कैसी",
    "kaun": "कौन", "kaunsa": "कौनसा", "kaunse": "कौनसे", "kaunsi": "कौनसी", "kab": "कब", "kahan": "कहाँ",
    "kitna": "कितना", "kitni": "कितनी", "kitne": "कितने", "jab": "जब", "jaise": "जैसे", "jo": "जो", "jis": "जिस",
    # conjunctions and adverbs
    "aur": "और", "ya": "या", "lekin": "लेकिन", "magar": "मगर", "toh": "तो", "bhi": "भी", "nahi": "नहीं",
    "nahin": "नहीं", "nhi": "नहीं", "haan": "हाँ", "abhi": "अभी", "ab": "अब", "phir": "फिर", "fir": "फिर", "bas": "बस",
    "bilkul": "बिल्कुल", "zaroor": "ज़रूर", "jaldi": "जल्दी", "thoda": "थोड़ा", "thodi": "थोड़ी", "bahut": "बहुत",
    "bohot": "बहुत", "bahot": "बहुत", "zyada": "ज़्यादा", "jyada": "ज़्यादा", "kam": "कम", "lagbhag": "लगभग",
    "shayad": "शायद", "sirf": "सिर्फ़", "matlab": "मतलब", "yaani": "यानी", "waise": "वैसे", "vaise": "वैसे",
    "acha": "अच्छा", "accha": "अच्छा", "achha": "अच्छा", "theek": "ठीक", "thik": "ठीक", "sahi": "सही",
    "galat": "ग़लत", "dono": "दोनों", "donon": "दोनों", "ek": "एक", "aaj": "आज", "kal": "कल", "yahan": "यहाँ",
    "wahan": "वहाँ", "hamesha": "हमेशा", "kabhi": "कभी", "agar": "अगर", "isliye": "इसलिए", "kyunki": "क्योंकि",
    "kyonki": "क्योंकि", "aisa": "ऐसा", "aise": "ऐसे", "itna": "इतना", "itni": "इतनी",
    # nouns and adjectives
    "baje": "बजे", "bajkar": "बजकर", "minat": "मिनट", "ghanta": "घंटा", "ghante": "घंटे",
    "din": "दिन", "raat": "रात", "subah": "सुबह", "shaam": "शाम", "samay": "समय", "waqt": "वक़्त", "baat": "बात",
    "kaam": "काम", "cheez": "चीज़", "jagah": "जगह", "naya": "नया", "nayi": "नई", "naye": "नए", "purana": "पुराना",
    "bada": "बड़ा", "badi": "बड़ी", "bade": "बड़े", "chota": "छोटा", "choti": "छोटी", "sabse": "सबसे", "pura": "पूरा",
    "poora": "पूरा", "puri": "पूरी", "alag": "अलग", "zaroorat": "ज़रूरत", "madad": "मदद", "sawal": "सवाल",
    "jawab": "जवाब", "dost": "दोस्त", "paani": "पानी", "khana": "खाना", "yaar": "यार", "ji": "जी",
    "shukriya": "शुक्रिया", "dhanyavaad": "धन्यवाद", "namaste": "नमस्ते", "maaf": "माफ़",
}
_LATIN_WORD = re.compile(r"(?<![\w-])[A-Za-z]+(?![\w-])")
# Also English words, so they are Hindi only straight after a Hindi word: "कर do" is कर दो, "यह to" is यह तो.
_AFTER_HINDI = {"do": "दो", "to": "तो", "hi": "ही"}
_AFTER_HINDI_WORD = re.compile(r"(?<=[ऀ-ॿ]) (do|to|hi)(?![\w-])", re.I)


def devanagari_hindi(text: str) -> str:
    """Hindi words in English letters, in Devanagari; English words and names stay as they are."""
    text = _LATIN_WORD.sub(lambda m: ROMAN_TO_DEVANAGARI.get(m.group(0).lower(), m.group(0)), text)
    return _AFTER_HINDI_WORD.sub(lambda m: " " + _AFTER_HINDI[m.group(1).lower()], text)


# Hindi clock times as people say them: "साढ़े सात", "सवा आठ", "पौने नौ", with the part of the day on a 12-hour clock.
def _day_part(h: int) -> str:
    return "रात" if h < 4 or h >= 20 else "सुबह" if h < 12 else "दोपहर" if h < 16 else "शाम"


def hindi_time(h: int, m: int, part: str | None = None) -> str:
    """The part of the day is added only when the 24-hour time makes it certain (19:30, 00:15): 12:00 or 9:05 could be
    either, and a wrong "दोपहर" is worse than none."""
    part = part or (_day_part(h) if h >= 13 or h == 0 else None)
    h12 = h % 12 or 12
    nxt = h12 % 12 + 1
    if m == 0:
        said = f"{hindi_number(h12)} बजे"
    elif m == 30:
        said = {1: "डेढ़ बजे", 2: "ढाई बजे"}.get(h12, f"साढ़े {hindi_number(h12)} बजे")
    elif m == 15:
        said = f"सवा {hindi_number(h12)} बजे"
    elif m == 45:
        said = "पौने एक बजे" if nxt == 1 else f"पौने {hindi_number(nxt)} बजे"
    else:
        said = f"{hindi_number(h12)} बजकर {hindi_number(m)} मिनट"
    return f"{part} {said}" if part else said


def english_time(h: int, m: int) -> str:
    """For Hinglish: "seven thirty", "seven oh five", "seven o'clock"; PM / AM only when the 24-hour time says so."""
    h12 = h % 12 or 12
    said = f"{english_number(h12)} o'clock" if m == 0 else \
        f"{english_number(h12)} {'oh ' + english_number(m) if m < 10 else english_number(m)}"
    return said + (" PM" if h >= 13 else " AM" if h == 0 else "")


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
    text = devanagari_hindi(text)
    if lang == "hi":
        text = _TIME.sub(lambda m: hindi_time(int(m.group(2)), int(m.group(3)), m.group(1)), text)
        text = re.sub(r"\s*°\s*C\b", " डिग्री", text)
        text = re.sub(r"\s*%", " प्रतिशत", text)
        return _NUMBER.sub(lambda m: _spell(m, "hi"), text)
    # Hinglish: numbers the way they are said when mixing, in English
    text = _TIME.sub(lambda m: (f"{m.group(1)} " if m.group(1) else "") + english_time(int(m.group(2)), int(m.group(3))),
                     text)  # a following "बजे" is dropped: "seven o'clock बजे" says it twice
    text = re.sub(r"\s*°\s*C\b", " degree", text)
    text = re.sub(r"\s*%", " percent", text)
    return _NUMBER.sub(lambda m: _spell(m, "en"), text)
