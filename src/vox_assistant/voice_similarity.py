"""How much does the clone sound like the reference? Speaker-verification embeddings, compared by cosine similarity.

Model: microsoft/wavlm-base-plus-sv (x-vector head), on the CPU so it takes no VRAM. Its model card suggests 0.86 as
the same-speaker threshold. A score is a check, not a substitute for listening (brief: blind listening test).
Imported only inside the Audio Clone environment (it needs PyTorch + transformers).
"""

from __future__ import annotations

import threading

import numpy as np

MODEL_ID = "microsoft/wavlm-base-plus-sv"
SAME_SPEAKER_THRESHOLD = 0.86
TEST_SENTENCE = "I checked both graphics cards, and the temperatures are fine. Shall I open the project now?"


class SpeakerEmbedder:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = self._fx = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from transformers import AutoFeatureExtractor, WavLMForXVector

            self._fx = AutoFeatureExtractor.from_pretrained(MODEL_ID)
            self._model = WavLMForXVector.from_pretrained(MODEL_ID).to(self.device).eval()
        return self._model, self._fx

    def embed(self, audio: np.ndarray, rate: int) -> np.ndarray:
        import torch
        from scipy.signal import resample_poly

        if rate != 16000:
            from math import gcd

            g = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // g, rate // g)
        with self._lock:
            model, fx = self._load()
            inputs = fx(audio.astype(np.float32), sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.inference_mode():
                emb = model(**{k: v.to(self.device) for k, v in inputs.items()}).embeddings[0]
        emb = emb.float().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
