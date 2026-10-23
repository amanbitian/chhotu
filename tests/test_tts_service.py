import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vox_assistant.tts_service import create_app, split_for_tts
from vox_assistant.voices import VoiceProfiles

from conftest import tone


class FakeBackend:
    sample_rate = 24000
    description = {"gpu": "fake"}

    def __init__(self):
        self.prompts_built = 0
        self.generated: list[str] = []

    def build_prompt(self, voice, mode):
        self.prompts_built += 1
        return {"voice": voice["path"], "mode": mode}

    def generate(self, text, cache, preset, seed):
        self.generated.append(text)
        return np.full(self.sample_rate // 2, 0.1, dtype=np.float32)  # 0.5 s per chunk

    def synchronize(self):
        pass


@pytest.fixture
def client(settings):
    backend = FakeBackend()
    app = create_app(settings, backend_factory=lambda: backend)
    c = TestClient(app, base_url="http://127.0.0.1")
    c.backend = backend
    c.profiles = VoiceProfiles(settings.voices_dir)
    return c


def enroll(client, name="me"):
    return client.profiles.create(name, tone(), 16000, "hello there", permission_confirmed=True)


def test_split_sentences_and_long_words():
    assert split_for_tts("One. Two! Three?") == ["One.", "Two!", "Three?"]
    long = " ".join(["word"] * 100)
    chunks = split_for_tts(long, max_chars=80)
    assert all(len(c) <= 80 for c in chunks) and " ".join(chunks) == long


def test_requires_a_voice_profile(client):
    r = client.post("/audio/speech", json={"input": "hi"})
    assert r.status_code == 409 and "voice" in r.json()["detail"].lower()


def test_wav_response_carries_timing(client):
    enroll(client)
    r = client.post("/audio/speech", json={"input": "Hello. How are you?", "response_format": "wav"})
    assert r.status_code == 200
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1
        assert abs(w.getnframes() / 24000 - (0.5 * 2 + 0.14)) < 0.01
    assert r.headers["x-voice"] == "me_v1" and r.headers["x-chunks"] == "2"
    assert client.backend.generated == ["Hello.", "How are you?"]


def test_pcm_streams_and_prompt_is_cached(client):
    enroll(client)
    for _ in range(2):
        r = client.post("/audio/speech", json={"input": "One. Two."})
        assert r.status_code == 200 and r.headers["x-sample-rate"] == "24000"
        assert len(r.content) == 2 * 24000 // 2 * 2
    assert client.backend.prompts_built == 1


def test_switching_and_revoking_voices_take_effect(client):
    enroll(client, "a")
    b = enroll(client, "b")
    assert client.post("/audio/speech", json={"input": "hi"}).headers["x-voice"] == "b_v1"
    assert client.post("/voices/a_v1/activate", json={}).status_code == 200
    assert client.post("/audio/speech", json={"input": "hi"}).headers["x-voice"] == "a_v1"
    assert client.post("/audio/speech", json={"input": "hi", "voice": b["id"]}).headers["x-voice"] == "b_v1"
    r = client.delete("/voices/a_v1")  # a plain DELETE, as httpx and browsers send it: no body, no content type
    assert r.status_code == 200 and r.json()["active"] == "b_v1"
    assert client.post("/audio/speech", json={"input": "hi", "voice": "a_v1"}).status_code == 409
    assert client.post("/forget").status_code == 415  # a body-less POST still needs the JSON content type


def test_warm_touches_the_model_only_when_loaded_and_wav_is_labelled(client):
    from vox_assistant.disclosure import read_label

    enroll(client)
    assert client.post("/warm", json={}).json() == {"warmed": False}  # nothing loaded yet: do not load just to warm
    r = client.post("/audio/speech", json={"input": "Hi.", "response_format": "wav"})
    assert r.headers["x-ai-generated"] == "true" and "me_v1" in read_label(r.content)
    assert client.post("/warm", json={}).json()["warmed"] is True and client.backend.generated[-1] == "Okay."


class _T:
    """Just enough of a torch tensor for stream_generate."""

    def __init__(self, a):
        self.a = a

    def squeeze(self, _):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Torch:
    import contextlib

    seeds: list = []

    @staticmethod
    def manual_seed(s):
        _Torch.seeds.append(s)

    inference_mode = staticmethod(contextlib.nullcontext)


class StreamingBackend(FakeBackend):
    """Like VoxCPM2: the model yields one ~0.16 s piece per decoded latent patch."""

    torch = _Torch

    def __init__(self):
        super().__init__()
        outer = self

        class Model:
            def generate_with_prompt_cache_streaming(self, target_text, prompt_cache, inference_timesteps, cfg_value, max_len):
                outer.generated.append(target_text)
                for _ in range(3):
                    yield _T(np.full(outer.sample_rate * 16 // 100, 0.1, dtype=np.float32)), None, None

        self.model = Model()


def test_pcm_is_streamed_piece_by_piece_and_wav_stays_whole(settings):
    backend = StreamingBackend()
    c = TestClient(create_app(settings, backend_factory=lambda: backend), base_url="http://127.0.0.1")
    VoiceProfiles(settings.voices_dir).create("me", tone(), 16000, "hello there", permission_confirmed=True)
    with c.stream("POST", "/audio/speech", json={"input": "One. Two."}) as r:
        pieces = [p for p in r.iter_bytes() if p]
        assert r.headers["x-sample-rate"] == "24000" and float(r.headers["x-lock-wait-seconds"]) >= 0
    assert sum(len(p) for p in pieces) == 2 * 3 * (24000 * 16 // 100) * 2  # two sentences x three pieces
    assert backend.generated == ["One.", "Two."]
    w = c.post("/audio/speech", json={"input": "One.", "response_format": "wav"})
    assert w.status_code == 200 and backend.generated[-1] == "One."  # whole-sentence path (FakeBackend.generate)
    settings.tts_stream = False
    c2 = TestClient(create_app(settings, backend_factory=lambda: backend), base_url="http://127.0.0.1")
    assert c2.post("/audio/speech", json={"input": "One."}).status_code == 200  # the old whole-sentence pcm path


def test_a_failing_model_is_an_http_error_not_an_empty_stream(settings):
    class Broken(StreamingBackend):
        def __init__(self):
            super().__init__()

            class Model:
                def generate_with_prompt_cache_streaming(self, **kw):
                    raise RuntimeError("GPU fell over")
                    yield

            self.model = Model()

    c = TestClient(create_app(settings, backend_factory=Broken), base_url="http://127.0.0.1")
    VoiceProfiles(settings.voices_dir).create("me", tone(), 16000, "hello there", permission_confirmed=True)
    r = c.post("/audio/speech", json={"input": "Hello."})
    assert r.status_code == 500 and "GPU fell over" in r.json()["detail"]


def test_validation(client):
    enroll(client)
    assert client.post("/audio/speech", json={"input": "  "}).status_code == 400
    assert client.post("/audio/speech", json={"input": "x", "preset": "Nope"}).status_code == 400
    assert client.post("/audio/speech", json={"input": "x", "response_format": "mp3"}).status_code == 400
    assert client.get("/health").json()["model_loaded"] is False
