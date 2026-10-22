"""Speech recognition on the GPU (Whisper large-v3-turbo, PyTorch ROCm), for English, Hindi and Hinglish.

Runs inside the voice service process (Audio Clone environment). Measured on the 9070 XT: 0.12-0.24 s per 3 s
utterance, against ~0.8 s for small.en on the CPU; 1.7 GB of VRAM.

Every utterance is decoded twice in one batch, once as Hindi and once as English, because neither works alone and
Whisper's language identification is unreliable on code-mixed speech that starts with the English wake phrase:
forced Hindi sometimes loops on English speech ("वाट वाट वाट"), forced English turns Hindi into English-sounding
nonsense. `combine()` decides what the assistant sees (see there). Measured on 54 spoken requests (synthetic speakers):
right tool for English 18/18, Hindi 14/18, Hinglish 15/18 with both readings; 18/11/12 with English only.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any

import numpy as np

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def looping(text: str) -> bool:
    """Whisper's failure mode on the wrong language: the same word over and over."""
    words = re.findall(r"\S+", text)
    return len(words) >= 6 and len(set(words)) / len(words) < 0.5


_SECOND = "\n(the same speech heard as English: "


def split_readings(text: str) -> list[str]:
    """['hindi reading', 'english reading'] for a combined transcript, [text] otherwise."""
    if _SECOND in text and text.endswith(")"):
        first, second = text.split(_SECOND, 1)
        return [first, second[:-1]]
    return [text]


def join_readings(readings: list[str]) -> str:
    readings = [r.strip() for r in readings if r and r.strip()]
    if len(readings) >= 2 and readings[0] != readings[1]:
        return f"{readings[0]}{_SECOND}{readings[1]})"
    return readings[0] if readings else ""


# English only when Whisper is confident in the English reading and clearly more confident than in the Hindi one.
# Mean token log-probabilities measured on 68 utterances: English speech -0.02..-0.18 for the English reading, and at
# least 0.18 above a Devanagari Hindi reading; Hindi / Hinglish speech mostly -0.21..-1.07 for the English reading.
EN_CONFIDENT, EN_MARGIN = -0.2, 0.15


def combine(hindi: str, english: str, scores: dict[str, float] | None = None) -> tuple[str, str]:
    """(text for the assistant, reading used). English only when the speech was clearly English (Hindi reading looping,
    in Latin letters, or much less confident); otherwise both readings and the LLM resolves which makes sense.
    Without this, the Devanagari gibberish of an English sentence decoded as Hindi made auto-language answer in Hindi."""
    hindi, english = hindi.strip(), english.strip()
    scores = scores or {}
    if not hindi or looping(hindi):
        return english, "en"
    if not english or looping(english):
        return hindi, "hi"
    if not DEVANAGARI.search(hindi):
        return english, "en"
    s_en, s_hi = scores.get("en"), scores.get("hi")
    if s_en is not None and s_hi is not None and not DEVANAGARI.search(english) \
            and s_en >= EN_CONFIDENT and s_en - s_hi >= EN_MARGIN:
        return english, "en"
    return join_readings([hindi, english]), "both"


class GpuWhisper:
    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", device: str = "cuda:0") -> None:
        self.model_id, self.device = model_id, device
        self.lock = threading.Lock()
        self._model: Any = None
        self._proc: Any = None
        self.load_seconds: float | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        t0 = time.perf_counter()
        self._proc = AutoProcessor.from_pretrained(self.model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(self.model_id, torch_dtype=torch.float16)
        self._model = self._model.to(self.device).eval()
        self.load_seconds = time.perf_counter() - t0

    def readings(self, audio: np.ndarray, languages: tuple[str, ...] = ("hi", "en"),
                 prompts: dict[str, str] | None = None) -> dict[str, str]:
        """One transcript per language.

        `prompts`: per language, words to bias spelling towards (wake phrase, app names, technical terms), Whisper's
        usual way of handling a domain vocabulary. Each in its own script: an English prompt pushed the Hindi decode
        into Latin letters (Hindi requests understood 13/18 -> 10/18). Without prompts both languages go in one batch.
        The length cap (~15 tokens per second of audio) ends a looping decode early."""
        import torch

        prompts = prompts or {}
        max_tokens = int(min(128, 20 + 15 * len(audio) / 16000))
        with self.lock:
            self.load()
            feats = self._proc([audio], sampling_rate=16000, return_tensors="pt").input_features
            feats = feats.to(self.device, torch.float16)
            out: dict[str, str] = {}
            self.last_scores: dict[str, float] = {}
            groups = [list(languages)] if not any(prompts.get(l) for l in languages) else [[l] for l in languages]
            for group in groups:
                prompt = prompts.get(group[0], "") if len(group) == 1 else ""
                kw = {"prompt_ids": self._proc.get_prompt_ids(prompt, return_tensors="pt").to(self.device)} if prompt else {}
                with torch.inference_mode():
                    gen = self._model.generate(feats.expand(len(group), -1, -1), language=group, task="transcribe",
                                               max_new_tokens=max_tokens, output_scores=True,
                                               return_dict_in_generate=True, **kw)
                    trans = self._model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)
                ids = gen.sequences
                for i, (lang, text) in enumerate(zip(group, self._proc.batch_decode(ids, skip_special_tokens=True))):
                    if prompt and text.strip().startswith(prompt.strip()):  # some versions echo the prompt
                        text = text.strip()[len(prompt.strip()):]
                    out[lang] = " ".join(text.split())
                    lp = trans[i][torch.isfinite(trans[i])]
                    # Mean log-probability of the generated tokens: Whisper's own confidence in this reading.
                    self.last_scores[lang] = float(lp.mean()) if lp.numel() else -9.0
        return out
