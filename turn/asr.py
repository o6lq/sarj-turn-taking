"""Speech recognition of the caller's words, on-device (Apple MLX).

Two engines (TT_ASR=cohere|whisper):
  * cohere  : Cohere Transcribe Arabic 07-2026 (2B, Apache-2.0, 8-bit MLX build) — trained for Arabic dialects;
              on Sarj's calls it hears «إيه نعم / إي هلا / اممم» where Whisper heard «إنا / بيهلا» or nothing,
              and on noise it says «فراغ» (empty) instead of hallucinating «ترجمة نانسي قنقر». Same speed (~97 ms).
  * whisper : Whisper large-v3-turbo q4 with a Gulf prompt and a short 10 s encoder window.

The prompt primes Whisper with the short Gulf words that matter here, so «إيه / هيه / مم / لحظة» come out as
words instead of being dropped. A disk cache (keyed by the exact audio) makes repeated simulator runs fast
while still charging the latency that was MEASURED the first time.
"""
from __future__ import annotations

import hashlib
import threading
import json
import os
import time
from pathlib import Path

import numpy as np

from .config import EXT_DIR

KIT = Path(__file__).resolve().parents[1]
EXT = EXT_DIR
COHERE = Path(os.environ.get("TT_COHERE_MODEL", str(EXT / "cohere-ar-mlx8")))
def _default_backend() -> str:
    import sys
    if sys.platform == "darwin":                                   # Apple MLX
        return "cohere" if (COHERE / "model.safetensors").exists() else "whisper"
    return "faster"                                                # Windows / Linux: faster-whisper (CPU or NVIDIA)


BACKEND = os.environ.get("TT_ASR", _default_backend())
FW_MODEL = os.environ.get("TT_FW_MODEL", "large-v3-turbo")       # faster-whisper model name or local folder
MODEL = os.environ.get("TT_ASR_MODEL", str(EXT / "whisper-large-v3-turbo-q4"))
PROMPT = os.environ.get("TT_ASR_PROMPT", "إيه، أها، مم، هيه، زين، طيب، لا لا، لحظة، اسمع، خلاص.")
CACHE_DIR = KIT / ".cache" / "asr"
class _PriorityLock:
    """One decode at a time (MLX cannot run two), but a waiting interruption read goes before any waiting turn read."""

    def __init__(self):
        self._cv = threading.Condition()
        self._busy = False
        self._urgent_waiting = 0

    def acquire(self, urgent: bool) -> None:
        with self._cv:
            self._urgent_waiting += urgent
            try:
                while self._busy or (not urgent and self._urgent_waiting):
                    self._cv.wait()
            finally:
                self._urgent_waiting -= urgent
            self._busy = True

    def release(self) -> None:
        with self._cv:
            self._busy = False
            self._cv.notify_all()


_LOCK = _PriorityLock()       # live demo: the turn-taker and the dialogue share the model
DECODER = "v3"                # part of the cache key: a change to decoding must not reuse old reads
URGENT_STEPS, NORMAL_STEPS = 24, 48
# Speed-up: Whisper always encodes 30 s of audio even for a 1 s «لحظة». With CTX_S > 0 we encode only a short
# window (the encoder's positional embedding is sliced), which cuts the latency. 0 = standard Whisper.
CTX_S = float(os.environ.get("TT_ASR_CTX", "10"))
_cache_on = os.environ.get("TT_ASR_CACHE", "1" if os.environ.get("TT_SIM") == "1" else "0") == "1"


def _key(audio: np.ndarray) -> str:
    h = hashlib.sha1(audio.tobytes())
    if BACKEND == "cohere":
        h.update(COHERE.name.encode() + b"cohere-v1" + DECODER.encode())
    elif BACKEND == "faster":
        h.update(FW_MODEL.encode() + PROMPT.encode() + b"fw-v1")
    else:
        h.update(Path(MODEL).name.encode() + PROMPT.encode() + b"v2" + (f"ctx{CTX_S}".encode() if CTX_S else b""))
    return h.hexdigest()


def warmup():
    transcribe(np.zeros(16000, np.int16), use_cache=False)


def transcribe(audio_i16: np.ndarray, use_cache: bool = True, priority: str = "normal") -> dict:
    """Returns {"text": str, "ms": compute time, "wait_ms": time waiting for the model}. Never raises. Thread-safe.
    priority="urgent" for reads that decide whether to stop; they go before waiting turn reads."""
    urgent = priority == "urgent"
    t0 = time.perf_counter()
    _LOCK.acquire(urgent)
    wait_ms = round((time.perf_counter() - t0) * 1000, 1)
    try:
        out = _transcribe(audio_i16, use_cache, urgent)
    finally:
        _LOCK.release()
    return {**out, "wait_ms": wait_ms}


def _transcribe(audio_i16: np.ndarray, use_cache: bool = True, urgent: bool = False) -> dict:
    if MODEL == "none":                       # acoustic-only mode (no words)
        return {"text": "", "no_speech": 1.0, "ms": 0.0}
    key = _key(audio_i16) + ("u" if urgent and BACKEND == "cohere" else "")
    path = CACHE_DIR / key[:2] / f"{key}.json"
    if use_cache and _cache_on and path.exists():
        return json.loads(path.read_text(encoding="utf-8")) | {"cached": True}
    if BACKEND in ("cohere", "faster"):
        out = _cohere(audio_i16, URGENT_STEPS if urgent else NORMAL_STEPS) if BACKEND == "cohere" else _faster(audio_i16)
        if use_cache and _cache_on and "error" not in out:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out
    import mlx_whisper
    t0 = time.perf_counter()
    audio = audio_i16.astype(np.float32) / 32768.0
    if CTX_S and len(audio) < (CTX_S - 0.5) * 16000:
        out = _short(audio, t0)
        if out is not None:
            if use_cache and _cache_on:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            return out
    try:
        r = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, language="ar", temperature=0.0,
                                   condition_on_previous_text=False, initial_prompt=PROMPT or None,
                                   no_speech_threshold=0.6, logprob_threshold=-1.0,
                                   compression_ratio_threshold=2.4, verbose=None,
                                   sample_len=40)
        segs = [s for s in r.get("segments", []) if s.get("no_speech_prob", 0) < 0.8]
        text = " ".join(s["text"].strip() for s in segs).strip()
        nsp = min([s.get("no_speech_prob", 1.0) for s in r.get("segments", [])] or [1.0])
        lp = [s.get("avg_logprob", -5.0) for s in segs]
        logprob = float(np.mean(lp)) if lp else -5.0
    except Exception as e:  # pragma: no cover
        print("ASR error", e)
        return {"text": "", "no_speech": 1.0, "ms": round((time.perf_counter() - t0) * 1000, 1), "error": repr(e)}
    out = {"text": text, "no_speech": round(float(nsp), 3), "logprob": round(logprob, 3),
           "ms": round((time.perf_counter() - t0) * 1000, 1)}
    if use_cache and _cache_on:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


_patched = False


def _short(audio: np.ndarray, t0: float) -> dict | None:
    """Decode with a shortened audio context (whisper.cpp's audio_ctx idea, in MLX)."""
    global _patched
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_whisper import whisper as W
        from mlx_whisper.audio import log_mel_spectrogram
        from mlx_whisper.decoding import DecodingOptions, decode
        from mlx_whisper.transcribe import ModelHolder
        if not _patched:
            def enc_call(self, x):
                x = nn.gelu(self.conv1(x))
                x = nn.gelu(self.conv2(x))
                x = x + self._positional_embedding[: x.shape[1]]
                for block in self.blocks:
                    x, _, _ = block(x)
                return self.ln_post(x)
            W.AudioEncoder.__call__ = enc_call
            _patched = True
        model = ModelHolder.get_model(MODEL, mx.float16)
        n_frames = int(CTX_S * 100)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
        if mel.shape[0] < n_frames:
            mel = mx.pad(mel, [(0, n_frames - mel.shape[0]), (0, 0)])
        mel = mel[:n_frames].astype(mx.float16)
        opts = DecodingOptions(language="ar", temperature=0.0, sample_len=40, prompt=PROMPT or None,
                               without_timestamps=True, fp16=True)
        r = decode(model, mel, opts)
        text = r.text.strip() if r.no_speech_prob < 0.8 else ""
        return {"text": text, "no_speech": round(float(r.no_speech_prob), 3), "logprob": round(float(r.avg_logprob), 3),
                "ms": round((time.perf_counter() - t0) * 1000, 1), "ctx": CTX_S}
    except Exception as e:  # fall back to standard Whisper
        print("short-context ASR failed:", e)
        return None


_coh = None
_EMPTY = "فراغ"          # Cohere's own "no speech here" output (after a run of @ tokens)


def _looping(ids: list[int]) -> bool:
    """A pattern of 1-3 tokens repeated at least 3 times in a row: the decoder is stuck («هههه», «!!!!», «آآآآ»).
    Two repeats are left alone, because people really say «لا لا» or «خمسة خمسة»."""
    for p in (1, 2, 3):
        n = max(6, 3 * p)
        if len(ids) >= n and all(ids[-n + i] == ids[-n + i % p] for i in range(n)):
            return True
    return False


def _cohere(audio_i16: np.ndarray, max_steps: int = NORMAL_STEPS) -> dict:
    """Greedy decode of Cohere Transcribe (same steps as mlx_audio's, batch 1), keeping each chosen token's
    log-probability so we get a confidence comparable to Whisper's avg_logprob."""
    global _coh
    t0 = time.perf_counter()
    try:
        import mlx.core as mx
        if _coh is None:
            from mlx_audio.stt import load
            _coh = load(str(COHERE))
        m = _coh
        x = audio_i16.astype(np.float32) / 32768.0
        if len(x) > 30 * 16000:
            x = x[-30 * 16000:]
        prompt = m._tokenizer.build_prompt_tokens("ar", True)
        enc, _, enc_mask = m._encode_waveforms([mx.array(x)])
        cache = m.make_cache()
        hidden, cache = m.transf_decoder(mx.array([prompt], dtype=mx.int32), enc, encoder_mask=enc_mask,
                                         cache=cache, start_pos=0)
        head = m.log_softmax.mlp.layer0
        eos = m._tokenizer.eos_token_id
        ids, lps = [], []
        logits = head(hidden[:, -1:, :])[0, -1]
        for step in range(max_steps):
            lp = logits - mx.logsumexp(logits)
            tok = int(mx.argmax(lp).item())
            if tok == eos:
                break
            ids.append(tok)
            lps.append(float(lp[tok].item()))
            if _looping(ids):
                break
            hidden, cache = m.transf_decoder(mx.array([[tok]], dtype=mx.int32), enc, encoder_mask=enc_mask,
                                             cache=cache, start_pos=len(prompt) + step)
            logits = head(hidden)[0, -1]
        text = m._tokenizer.decode(ids, skip_special_tokens=True).strip()
    except Exception as e:  # pragma: no cover
        print("Cohere ASR error", e)
        return {"text": "", "no_speech": 1.0, "ms": round((time.perf_counter() - t0) * 1000, 1), "error": repr(e)}
    empty = "@@" in text or text.strip("@ .") == _EMPTY or not text.strip("@ .")
    words = "" if empty else text
    logprob = float(np.mean(lps)) if lps and not empty else -5.0
    return {"text": words, "no_speech": 1.0 if empty else 0.0, "logprob": round(logprob, 3),
            "ms": round((time.perf_counter() - t0) * 1000, 1), "engine": "cohere"}


_fw = None


def _load_fw():
    """GPU only if it really runs: an NVIDIA card without the CUDA 12 libraries (cublas64_12.dll) loads the model
    fine and then fails on the first transcribe, so we run one tiny transcribe before trusting it.
    TT_FW_DEVICE=cpu|cuda forces a device (default auto)."""
    from faster_whisper import WhisperModel
    dev = os.environ.get("TT_FW_DEVICE", "auto")
    if dev in ("auto", "cuda"):
        try:
            m = WhisperModel(FW_MODEL, device="cuda", compute_type="float16")
            list(m.transcribe(np.zeros(16000, np.float32), language="ar", beam_size=1, without_timestamps=True)[0])
            return m
        except Exception as e:
            if dev == "cuda":
                raise
            print(f"faster-whisper: GPU not usable ({e}); using CPU", flush=True)
    return WhisperModel(FW_MODEL, device="cpu", compute_type="int8")


def _faster(audio_i16: np.ndarray) -> dict:
    """Windows / Linux: Whisper through CTranslate2 (faster-whisper). NVIDIA GPU if present, else CPU int8."""
    global _fw
    t0 = time.perf_counter()
    try:
        if _fw is None:
            _fw = _load_fw()
        x = audio_i16.astype(np.float32) / 32768.0
        segs, _ = _fw.transcribe(x, language="ar", beam_size=1, initial_prompt=PROMPT or None,
                                 condition_on_previous_text=False, without_timestamps=True, vad_filter=False)
        segs = [sg for sg in segs if sg.no_speech_prob < 0.8]
        text = " ".join(sg.text.strip() for sg in segs).strip()
        lp = float(np.mean([sg.avg_logprob for sg in segs])) if segs else -5.0
    except Exception as e:  # pragma: no cover
        print("faster-whisper error", e)
        return {"text": "", "no_speech": 1.0, "ms": round((time.perf_counter() - t0) * 1000, 1), "error": repr(e)}
    return {"text": text, "no_speech": 0.0 if text else 1.0, "logprob": round(lp, 3),
            "ms": round((time.perf_counter() - t0) * 1000, 1), "engine": "faster-whisper"}
