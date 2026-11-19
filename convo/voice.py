"""The agent's voice. Two engines:
  edge  - Microsoft neural Saudi voices (natural, needs internet, ~1-3 s to the first sound)
  say   - macOS «Majed» (offline, fast, more robotic) - automatic fallback
Short fillers («طيب»، «إيه نعم») are synthesised once and cached on disk so they play instantly while the
brain is still thinking, the way a person says «طيب…» before answering.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

from turn.config import EXT_DIR

from . import http_pool

CACHE = EXT_DIR / "voice-cache"
VOICES = {"f": "ar-SA-ZariyahNeural", "m": "ar-SA-HamedNeural"}
FILLERS = ["طيب", "إيه نعم", "أبشر", "طيب، لحظة"]
RATE = 24000                 # playback quality: 16 kHz sounds like an old phone line
CACHE_CHARS = 120            # fixed lines are kept on disk so they play at once next time
ELEVEN_URL = "https://api.elevenlabs.io/v1"
ELEVEN_MODEL = "eleven_flash_v2_5"
GEMINI_TTS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"
GEMINI_VOICES = {"m": "Orus", "f": "Aoede"}     # the same voices as the live replies
GEMINI_STYLE = "موظف خدمة عملاء سعودي في مكالمة هاتفية، لهجة سعودية طبيعية، نبرة دافئة وهادئة وسرعة كلام عادية"


def _key(engine, voice, text):
    return hashlib.sha1(f"{engine}|{voice}|{RATE}|{text}".encode()).hexdigest()


async def _edge(text: str, voice: str, timeout: float = 8.0) -> np.ndarray:
    import edge_tts
    import soundfile as sf
    from scipy import signal

    async def go():
        b = b""
        async for ch in edge_tts.Communicate(text, voice).stream():
            if ch["type"] == "audio":
                b += ch["data"]
        return b
    data = await asyncio.wait_for(go(), timeout)
    x, sr = sf.read(io.BytesIO(data), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return (np.clip(signal.resample_poly(x, RATE, sr), -1, 1) * 32767 * 0.9).astype(np.int16)


_best = None


def best_say_voice() -> str:
    """The best Arabic macOS voice installed: Premium > Enhanced > the default compact «Majed»."""
    global _best
    if _best is None:
        lines = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout.splitlines()
        import re
        ar = [m.group(1).strip() for l in lines if (m := re.match(r"^(.*?)\s+ar_\w+", l))]
        _best = next((v for tag in ("Premium", "Enhanced") for v in ar if tag in v), ar[0] if ar else "Majed")
    return _best


def _say(text: str) -> np.ndarray:
    import soundfile as sf
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.wav"
        subprocess.run(["say", "-v", best_say_voice(), "-r", "185", "-o", str(out), "--file-format=WAVE",
                        f"--data-format=LEI16@{RATE}", text], check=True, capture_output=True)
        x, _ = sf.read(out, dtype="int16")
    return x


def _gemini_tts(text: str, gender: str) -> np.ndarray:
    import soundfile as sf
    from scipy import signal
    content = [{"type": "text", "text": text, "annotations": [{"type": "speech_metadata", "style": GEMINI_STYLE}]}]
    body = {"model": GEMINI_TTS_MODEL, "input": [{"type": "user_input", "content": content}], "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": GEMINI_VOICES.get(gender, GEMINI_VOICES["f"])}]}}
    headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"}
    with urllib.request.urlopen(urllib.request.Request(GEMINI_TTS_URL, json.dumps(body).encode(), headers), timeout=30) as r:
        steps = json.load(r)["steps"]
    data = next(c["data"] for s in steps if s["type"] == "model_output" for c in s["content"] if c["type"] == "audio")
    x, sr = sf.read(io.BytesIO(base64.b64decode(data)), dtype="int16")
    return x if sr == RATE else signal.resample_poly(x, RATE, sr).astype(np.int16)


_eleven_voices: dict = {}


def eleven_ready() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY"))


def _eleven_voice(gender: str) -> str:
    """ELEVENLABS_VOICE_ID when set, otherwise the first voice of the right gender in the account."""
    chosen = os.environ.get("ELEVENLABS_VOICE_ID")
    if chosen:
        return chosen
    if gender not in _eleven_voices:
        req = urllib.request.Request(f"{ELEVEN_URL}/voices", headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]})
        with urllib.request.urlopen(req, timeout=10) as r:
            voices = json.load(r)["voices"]
        want = "male" if gender == "m" else "female"
        _eleven_voices[gender] = next((v["voice_id"] for v in voices if v.get("labels", {}).get("gender") == want), voices[0]["voice_id"])
    return _eleven_voices[gender]


def _eleven(text: str, gender: str) -> np.ndarray:
    body = json.dumps({"text": text, "model_id": ELEVEN_MODEL, "language_code": "ar"}).encode()
    headers = {"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"}
    url = f"{ELEVEN_URL}/text-to-speech/{_eleven_voice(gender)}?output_format=pcm_{RATE}"
    return np.frombuffer(http_pool.post(url, body, headers, 15), dtype="<i2").copy()   # kept alive: no TLS per sentence


def warm_connection(engine: str) -> bool:
    """Open the TTS connection before the first reply. Never raises."""
    return engine == "eleven" and eleven_ready() and http_pool.warm(ELEVEN_URL)


def _live_line(text: str, gender: str) -> Path:
    return CACHE / f"{_key('gemini-tts', GEMINI_VOICES.get(gender, GEMINI_VOICES['f']), text)}.npy"


def has_live_line(text: str, gender: str) -> bool:
    return _live_line(text, gender).exists()


async def tts(text: str, gender: str = "f", engine: str = "edge") -> tuple[np.ndarray, str]:
    """Returns (int16 audio at RATE, engine actually used). Never raises."""
    import sys
    mac = sys.platform == "darwin"
    if engine == "live":                             # a fixed line in the live voice, generated once
        f = _live_line(text, gender)
        if f.exists():
            return np.load(f), "gemini"
        try:
            x = await asyncio.to_thread(_gemini_tts, text, gender)
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(f, x)
            return x, "gemini"
        except Exception:
            engine = "edge"
    if engine == "eleven":
        f = CACHE / f"{_key('eleven', gender, text)}.npy"
        if f.exists():
            return np.load(f), "eleven"
        try:
            x = await asyncio.to_thread(_eleven, text, gender)
            if len(text) < CACHE_CHARS:
                CACHE.mkdir(parents=True, exist_ok=True)
                np.save(f, x)
            return x, "eleven"
        except Exception:
            engine = "edge"
    if engine == "say" and not mac:                  # no macOS «say» on Windows / Linux: use Edge
        engine = "edge"
    voice = VOICES.get(gender, VOICES["f"]) if engine == "edge" else best_say_voice()
    k = _key(engine, voice, text)
    f = CACHE / f"{k}.npy"
    if f.exists():
        return np.load(f), engine
    if engine == "edge":
        try:
            x = await _edge(text, voice)
            CACHE.mkdir(parents=True, exist_ok=True)
            if len(text) < CACHE_CHARS:
                np.save(f, x)
            return x, "edge"
        except Exception:
            if not mac:
                return np.zeros(RATE // 10, np.int16), "none"
    x = await asyncio.to_thread(_say, text)
    if len(text) < CACHE_CHARS:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(f, x)
    return x, "say"


async def warm_fillers(gender: str = "f", engine: str = "edge"):
    for t in FILLERS:
        await tts(t, gender, engine)
