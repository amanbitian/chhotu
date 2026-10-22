"""Saying that speech is AI-generated (brief section 8): a spoken notice in guest mode, and a label in every WAV we write.

The label is a RIFF LIST/INFO chunk (ICMT comment, ISFT software): players ignore it, and file properties and audio
tools show it, so an exported clip of the cloned voice keeps saying what it is.
"""

from __future__ import annotations

import io
import struct
import wave

GUEST_NOTICE = "Just so everyone knows, I'm an AI assistant, and this is a computer voice."
SOFTWARE = "VoxVortex Desktop"


def label_text(voice: str) -> str:
    if voice == "neutral":
        return "AI-generated speech (synthetic neutral voice)."
    return f"AI-generated speech in a cloned voice (profile {voice}). Not a recording of a real person speaking."


def _info_chunk(comment: str) -> bytes:
    def sub(tag: bytes, text: str) -> bytes:
        data = text.encode("utf-8") + b"\x00"
        if len(data) % 2:
            data += b"\x00"
        return tag + struct.pack("<I", len(data)) + data

    body = b"INFO" + sub(b"ICMT", comment) + sub(b"ISFT", SOFTWARE)
    return b"LIST" + struct.pack("<I", len(body)) + body


def label_wav(wav: bytes, comment: str) -> bytes:
    """Append an INFO chunk to a RIFF/WAVE file and fix the RIFF size."""
    if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    out = wav + (b"\x00" if len(wav) % 2 else b"") + _info_chunk(comment)
    return out[:4] + struct.pack("<I", len(out) - 8) + out[8:]


def read_label(wav: bytes) -> str | None:
    pos = 12
    while pos + 8 <= len(wav):
        tag, size = wav[pos:pos + 4], struct.unpack("<I", wav[pos + 4:pos + 8])[0]
        body = wav[pos + 8:pos + 8 + size]
        if tag == b"LIST" and body[:4] == b"INFO":
            p = 4
            while p + 8 <= len(body):
                t, n = body[p:p + 4], struct.unpack("<I", body[p + 4:p + 8])[0]
                if t == b"ICMT":
                    return body[p + 8:p + 8 + n].rstrip(b"\x00").decode("utf-8")
                p += 8 + n + (n % 2)
        pos += 8 + size + (size % 2)
    return None


def labelled_wav(pcm: bytes, rate: int, voice: str) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return label_wav(buf.getvalue(), label_text(voice))
