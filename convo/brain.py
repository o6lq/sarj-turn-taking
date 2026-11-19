"""The agent's brain: Gemini in the cloud by default, or a local language model behind an OpenAI-style HTTP server
(mlx_lm.server), in its own process so it never competes with Whisper inside our process. Streams the reply sentence
by sentence so the voice can start on the first sentence while the rest is still being written.

  gemini                   - the Gemini API (needs GEMINI_API_KEY in .env); Claude Haiku takes over if it fails

Models (tested on this Mac, Apple M5 24 GB):
  gemma-3-12b-it-qat-4bit  - best Arabic of the ones that fit (default once downloaded)
  Qwen3-14B-MLX-4bit       - already in LM Studio; correct facts but stiffer, more formal Arabic
  claude-opus / claude-haiku - Claude in the cloud, the LAST resort (needs internet; slower to the first word):
                             through the Anthropic API when ANTHROPIC_API_KEY is set, otherwise through the
                             Claude Code CLI (`claude -p`) on the user's own subscription.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from turn.config import EXT_DIR

from . import http_pool

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_MODEL = "gemini-3.5-flash-lite"
CLOUD_TIMEOUT_S = 8                                  # a live caller is waiting: give up early and use the fallback
EXT = EXT_DIR
MODELS = {
    "gemma4-26b": EXT / "gemma-4-26b-a4b-4bit",          # Gemma 4 26B-A4B (MoE: 26B knowledge, 4B active per word)
    "gemma4-e4b": EXT / "gemma-4-e4b-4bit",              # Gemma 4 E4B (small and fast)
    "gemma-12b": EXT / "gemma-3-12b-it-qat-4bit",
    "qwen3-14b": Path.home() / ".lmstudio" / "models" / "lmstudio-community" / "Qwen3-14B-MLX-4bit",
}
CLAUDE = {"claude-opus": "claude-opus-5-5", "claude-haiku": "claude-haiku-5-5"}
PORT = 8780
_proc = None


def _claude_ready() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or shutil.which("claude"))


def _available() -> dict:
    ok = {"gemini": bool(os.environ.get("GEMINI_API_KEY"))}
    for k, p in MODELS.items():
        # count the shards by name («model-00002-of-00002»): some repos ship a stale index.json (Gemma QAT does)
        shards = sorted(p.glob("model*.safetensors"))
        m = re.search(r"of-(\d+)", shards[0].name) if shards else None
        ok[k] = bool(shards) and (len(shards) == int(m.group(1)) if m else True)
    for k in CLAUDE:
        ok[k] = _claude_ready()
    if os.environ.get("TT_BRAIN_URL"):              # LM Studio / Ollama (Windows): an OpenAI-compatible server
        ok["local-url"] = True
    return ok


def available() -> dict:
    """Every brain and whether it can run, the default first so the demo page selects it."""
    ok, first = _available(), default_model()
    return {first: ok.get(first, False), **ok}


def default_model() -> str:
    """TT_BRAIN when it can run; otherwise cloud brains come first, so the big local model is only the last resort."""
    av, wanted = _available(), os.environ.get("TT_BRAIN")
    if wanted and av.get(wanted):
        return wanted
    return next((k for k in ("gemini", "local-url", "claude-haiku", "claude-opus", "gemma4-26b", "gemma-12b", "gemma4-e4b",
                             "qwen3-14b") if av.get(k)), wanted or "gemini")


def _up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=0.3):
            return True
    except OSError:
        return False


def start(model: str | None = None) -> str:
    """Start (or restart with another model) the brain server. Returns the model name."""
    global _proc, _current
    model = model or default_model()
    if _proc and _proc.poll() is None and globals().get("_current") == model:
        return model
    stop()
    if model in CLAUDE or model in ("local-url", "gemini"):       # nothing to start here
        _current = model
        return model
    extra = ["--chat-template-args", '{"enable_thinking": false}'] if model.startswith("gemma4") else []   # no thinking pause
    _proc = subprocess.Popen([sys.executable, "-m", "mlx_lm.server", "--model", str(MODELS[model]), "--port", str(PORT),
                              "--max-tokens", "160", *extra], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _current = model
    t0 = time.time()
    while time.time() - t0 < 180:                      # ready = it answers, not just that the port is open
        try:
            if _up():
                warm()
                return model
        except Exception:
            pass
        if _proc.poll() is not None:
            raise RuntimeError("brain server exited")
        time.sleep(0.5)
    raise RuntimeError("brain server did not start")


def stop():
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


def warm():
    for _ in stream([{"role": "user", "content": "هلا"}], max_tokens=4):
        pass


def warm_connection() -> bool:
    """Open the network connection to the brain now (TCP + TLS), so the first reply of the call does not pay for
    it. Never raises; returns False when there is nothing to warm or the network is down."""
    if globals().get("_current") in CLAUDE:
        return False
    return http_pool.warm(_endpoint()[0])


def normalize(messages: list[dict]) -> list[dict]:
    """Strict chat templates (Gemma) need: one system message first, then user/assistant alternating, user first."""
    sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
    out: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + " " + m["content"]}
        else:
            out.append(dict(m))
    if not out or out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(بدأت المكالمة ورد العميل على الخط)"})
    if out[-1]["role"] != "user":
        out.append({"role": "user", "content": "(كمّل)"})
    if sys_txt:
        if _current_is_gemma():                 # Gemma has no system role: put it in front of the first user turn
            out[0] = {"role": "user", "content": sys_txt + "\n\n" + out[0]["content"]}
        else:
            out.insert(0, {"role": "system", "content": sys_txt})
    return out


def _current_is_gemma() -> bool:
    return str(globals().get("_current", "")).startswith("gemma")


def _endpoint() -> tuple[str, dict, float]:
    """The URL, headers and timeout of the OpenAI-style server behind the current brain."""
    headers = {"Content-Type": "application/json"}
    if globals().get("_current") == "gemini":
        headers["Authorization"] = f"Bearer {os.environ.get('GEMINI_API_KEY', '')}"
        return GEMINI_URL, headers, CLOUD_TIMEOUT_S
    if os.environ.get("TT_BRAIN_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['TT_BRAIN_KEY']}"
    base = os.environ.get("TT_BRAIN_URL", f"http://127.0.0.1:{PORT}").rstrip("/")
    return f"{base}/v1/chat/completions", headers, 60


def stream(messages: list[dict], max_tokens: int = 120, temperature: float = 0.6):
    """Yield text pieces as the model writes them."""
    messages = normalize(messages)
    current = globals().get("_current", "")
    if current in CLAUDE:
        yield from _claude_stream(messages, max_tokens)
        return
    if current.startswith("qwen3") and messages and messages[0]["role"] == "system":
        messages = [{"role": "system", "content": messages[0]["content"] + "\n/no_think"}] + messages[1:]
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True}
    if current == "local-url":
        payload["model"] = os.environ.get("TT_BRAIN_MODEL", "local-model")
    elif current == "gemini":
        payload["model"] = os.environ.get("GEMINI_MODEL", GEMINI_MODEL)
        if os.environ.get("GEMINI_REASONING_EFFORT"):         # optional: e.g. "low"/"none" if the model allows it
            payload["reasoning_effort"] = os.environ["GEMINI_REASONING_EFFORT"]
    url, headers, timeout = _endpoint()
    in_think = started = False
    try:
        for raw in http_pool.post_lines(url, json.dumps(payload).encode(), headers, timeout):   # kept alive: no new TLS handshake per reply
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[5:])["choices"][0].get("delta", {}).get("content") or ""
            except (ValueError, KeyError, IndexError):
                continue
            if "<think>" in d:
                in_think = True
            if in_think:
                if "</think>" in d:
                    in_think = False
                    d = d.split("</think>", 1)[1]
                else:
                    continue
            if d:
                started = True
                yield d
    except (OSError, http.client.HTTPException) as e:
        if current != "gemini" or started or not _claude_ready():   # half a sentence was already spoken: do not restart it
            raise
        print(f"Gemini failed ({type(e).__name__} {getattr(e, 'code', '')}), using Claude Haiku", flush=True)
        yield from _claude_stream(messages, max_tokens, "claude-haiku")


_SENT = re.compile(r"(.+?[\.\!\؟\?…]+|.{70,}?[،,])(\s+|$)", re.S)     # split at commas only in long sentences
_FIRST = re.compile(r"(.{%d,}?[،,])(\s+|$)" % int(os.environ.get("TT_FIRST_CLAUSE_CHARS", "24")), re.S)
# The first piece of a reply may end at a comma once it has a few words: the voice starts on that clause while the
# rest of the sentence is still being written. A comma is where a speaker pauses anyway, so it does not sound cut.


def sentences(messages: list[dict]):
    """Yield complete, speakable sentences as soon as each one is finished."""
    buf, first = "", True
    for piece in stream(messages):
        buf += piece
        while True:
            m = _SENT.match(buf) or (_FIRST.match(buf) if first else None)
            if not m:
                break
            first = False
            s = clean(m.group(1))
            buf = buf[m.end():]
            if s:
                yield s
    if clean(buf):
        yield clean(buf)


def clean(s: str) -> str:
    s = re.sub(r"[\*\#\_\(\)\[\]\"«»“”]|<[^>]+>", " ", s)
    s = re.sub(r"[\U0001F300-\U0001FAFF☀-➿]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _claude_stream(messages: list[dict], max_tokens: int, name: str | None = None):
    """Claude as the brain. Thinking cannot be switched off on Opus 5.5, so effort is set to low for speed."""
    model_id = CLAUDE[name or _current]
    sys_txt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    conv = [m for m in messages if m["role"] != "system"]
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if model_id.startswith("claude-opus") else {}
        with client.beta.messages.stream(model=model_id, max_tokens=4000, system=sys_txt, messages=conv,
                                         output_config={"effort": "low"}, **extra) as st:
            for piece in st.text_stream:
                yield piece
        return
    # no API key: the Claude Code CLI, headless, no tools, no project settings
    who = {"assistant": "أنت", "user": "العميل"}
    prompt = ("سجل المكالمة حتى الآن:\n" + "\n".join(f"{who[m['role']]}: {m['content']}" for m in conv)
              + "\n\nاكتب ردك الآن فقط: الكلام اللي بتقوله بصوتك للعميل، بدون أي شرح أو تعليق.")
    cmd = ["claude", "-p", "--model", model_id, "--effort", "low", "--output-format", "stream-json",
           "--include-partial-messages", "--verbose", "--tools", "", "--system-prompt", sys_txt,
           "--no-session-persistence", "--strict-mcp-config", "--setting-sources", "", prompt]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True)
    try:
        for line in p.stdout:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "stream_event":
                ev = e["event"]
                if ev.get("type") == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                    yield ev["delta"]["text"]
    finally:
        p.wait()
