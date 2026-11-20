"""Live microphone test: the agent talks through your speakers, you interrupt (or just say «إيه»), and you see
what the system decided, what it heard and how fast. You can label each try; labelled tries are saved as new
calls in data/live (same format as the kit) so the next `tools.loop` scores the model on YOUR voice too.

    .venv/bin/python -m tools.live              # opens the demo; the try-out page is http://127.0.0.1:8790/
    .venv/bin/python -m tools.live --manifest   # list every recording in data/live (after copying in teammates' files)
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import sys
import threading
import time
import wave
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from ttbench.dataset import read_wav_stereo, write_wav_stereo  # noqa: E402
from turn import config  # noqa: E402,F401  (loads .env)

HTTP_PORT, WS_PORT = 8790, 8771   # 8770 is taken by macOS sharing (sharingd) on some Macs
LIVE = KIT / "data" / "live"
RECORDER = re.sub(r"[^A-Za-z0-9]", "", os.environ.get("TT_RECORDER", ""))[:10]   # who recorded: keeps ids unique across laptops
RECORD_DIALECT = os.environ.get("TT_DIALECT", "live-user")
RECORD_PLACE = os.environ.get("TT_PLACE", "live-mic")        # quiet, cafe, car, street, speakerphone ...
LIVE_THRESHOLDS = {"T_full_ms": 300, "T_dead_ms": 1500, "E_full_ms": 600, "E_dead_ms": 2000, "E_miss_ms": 3000}


def wav_bytes(x: np.ndarray) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(x.astype("<i2").tobytes())
    return b.getvalue()


def dev_calls():
    out = []
    for f in sorted((KIT / "data" / "dev" / "labels").glob("*.json")):
        lab = json.loads(f.read_text())
        u = max(lab["agent_utterances"], key=lambda u: u["end_ms"] - u["start_ms"])
        out.append({"sid": lab["scenario_id"], "text": u["text"], "start": u["start_ms"], "end": u["end_ms"]})
    return out


def replay(sid: str) -> dict:
    run = KIT / "runs" / "sim-dev-latest"
    lab = json.loads((KIT / "data" / "dev" / "labels" / f"{sid}.json").read_text())
    out = {"label": lab, "decisions": [], "trace": [], "events": []}
    if not (run / "score.json").exists():
        return out
    meta = json.loads((run / "run.json").read_text())
    s0 = next(s for s in meta["scenarios"] if s["scenario_id"] == sid)
    for line in (run / "log.jsonl").read_text().splitlines():
        l = json.loads(line)
        if s0["start_ms"] <= l["stream_ms"] < s0["start_ms"] + s0["duration_ms"] + 1500:
            out["decisions"].append({"t": l["stream_ms"] - s0["start_ms"], "d": l["msg"]["decision"]})
    tr = json.loads((run / "trace.json").read_text()).get(sid, [])
    out["trace"] = [{k: v for k, v in r.items() if k != "f"} for r in tr if r["kind"] in ("asr", "STOP", "candidate", "TAKE_TURN")]
    out["events"] = [e for e in json.loads((run / "score.json").read_text())["events"] if e["scenario_id"] == sid]
    return out


def results() -> dict:
    """Everything the demo shows as numbers, read from the real run files (nothing typed by hand)."""
    import csv
    rows = list(csv.DictReader((KIT / "runs" / "sim-history.csv").open(encoding="utf-8")))
    def pick(prefix, data):
        r = [x for x in rows if x["name"].startswith(prefix) and x["data"] == data]
        return {k: r[-1][k] for k in ("core", "R", "S", "false_stops", "misses")} if r else None
    sets = ["dev", "dev-aug", "synth-test", "synth-heldvoice"]
    out = {"sets": sets,
           "systems": [{"name": "البدائي (حزمة سرج)", "runs": {d: pick("baseline", d) for d in sets}},
                       {"name": "Whisper + قواعد + نموذج متعلّم", "runs": {d: pick("v12", d) or pick("resume-", d) for d in sets}},
                       {"name": "Cohere + طبقة المعنى + RESUME (الحالي)", "runs": {d: pick("sem5-", d) for d in sets}}]}
    for name in ("llm_judge", "brain_eval"):
        f = KIT / "analysis" / f"{name}.json"
        if f.exists():
            d = json.loads(f.read_text())
            out[name] = {k: {kk: vv for kk, vv in v.items() if kk not in ("pred", "rows", "batch_s")} | (
                {"median_first_s": v.get("median_first_s"), "facts_ok": v.get("facts_ok")} if name == "brain_eval" else {})
                for k, v in d.items() if not k.startswith("_")}
    m = KIT / "models" / "student" / "meta.json"
    out["student"] = json.loads(m.read_text()) if m.exists() else None
    return out


def tts(text: str) -> np.ndarray:
    import edge_tts
    import soundfile as sf
    from scipy import signal

    async def go():
        c = edge_tts.Communicate(text, "ar-SA-ZariyahNeural")
        b = b""
        async for ch in c.stream():
            if ch["type"] == "audio":
                b += ch["data"]
        return b
    x, sr = sf.read(io.BytesIO(asyncio.run(go())), dtype="float32")
    x = signal.resample_poly(x, 16000, sr)
    return (np.clip(x, -1, 1) * 32767 * 0.8).astype(np.int16)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self._send((KIT / "tools" / "live.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/demo":
            return self._send((KIT / "demo" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/call":                     # the call-style demo page (design by the team)
            return self._send((KIT / "demo" / "call.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/preflight":
            return self._send(json.dumps(preflight(), ensure_ascii=False).encode(), "application/json")
        if u.path == "/results":
            return self._send(json.dumps(results(), ensure_ascii=False).encode(), "application/json")
        if u.path == "/scenarios":
            from convo.scenarios import SCENARIOS
            from convo import brain, voice
            return self._send(json.dumps({"scenarios": {k: v["title"] for k, v in SCENARIOS.items()}, "voices": {"eleven": voice.eleven_ready()},
                                          "models": brain.available()}, ensure_ascii=False).encode(), "application/json")
        if u.path == "/calls":
            return self._send(json.dumps(dev_calls(), ensure_ascii=False).encode(), "application/json")
        if u.path == "/agent":
            sid = q["sid"][0]
            c = next(c for c in dev_calls() if c["sid"] == sid)
            st = read_wav_stereo(KIT / "data" / "dev" / "audio" / f"{sid}.wav")
            return self._send(wav_bytes(st[c["start"] * 16:c["end"] * 16, 0]), "audio/wav")
        if u.path == "/mix":                       # the whole call, both sides mixed, to listen along
            sid = q["sid"][0]
            st = read_wav_stereo(KIT / "data" / "dev" / "audio" / f"{sid}.wav").astype(np.int32)
            return self._send(wav_bytes(np.clip((st[:, 0] * 0.8 + st[:, 1]), -32768, 32767)), "audio/wav")
        if u.path == "/replay":                    # what the model decided on that call (latest tools.loop run)
            return self._send(json.dumps(replay(q["sid"][0]), ensure_ascii=False).encode(), "application/json")
        if u.path == "/tts":
            text = q["text"][0][:400]
            return self._send(wav_bytes(tts(text)), "audio/wav")
        self.send_error(404)


_active = {"ws": None, "convo": None}      # one call at a time: two calls would talk over each other


async def ws_handler(ws):
    try:
        await _serve(ws)
    finally:
        if _active["ws"] is ws:
            _end_call()


def _end_call():
    convo = _active["convo"]
    if convo is not None:
        convo.state = "ended"
        if convo.task and not convo.task.done():
            convo.task.cancel()
        convo.shutdown()
    _active.update(ws=None, convo=None)


async def _serve(ws):
    from turn import policy
    tt = None
    frames_a, frames_c = [], []
    agent_text = ""
    t_start = None
    decisions = []
    sent_trace = 0
    test_buf = None                      # mic test: collect a few seconds, transcribe, report clarity
    convo = None                         # full conversation mode (convo/engine.py)
    async def send(m):
        try:
            await ws.send(json.dumps(m, ensure_ascii=False))
        except Exception:
            pass
    async for raw in ws:
        if isinstance(raw, bytes) and convo is not None:
            fr = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
            await convo.on_frame(fr[:, 0], fr[:, 1])
            if convo.ms % 5000 == 0:              # autosave every 5 s so any odd moment can be inspected
                from convo.engine import save_session
                convo.save_dir = save_session(convo, KIT / "runs" / "live-calls", getattr(convo, "save_dir", None))
            if convo.ms % 100 == 0:
                lvl = 20 * np.log10(np.sqrt(np.mean(fr[:, 1].astype(np.float64) ** 2)) / 32768 + 1e-9)
                t = convo.tt
                c = t.cand or {}
                f = c.get("feats") or {}
                last = next((r for r in reversed(t.trace) if r["kind"] == "asr"), {})
                await send({"type": "meter", "vad": round(t.vad.prob, 2), "db": round(float(lvl), 1),
                            "ref": round(t.ref_db, 1), "floor": round(t._floor_db(), 1), "knows": len(t.ref_levels) >= 8,
                            "cand": bool(t.cand), "spk": None if c.get("spk") is None else round(c["spk"], 2),
                            "rel": None if not f else round(f.get("rel_db", 0), 1),
                            "sem": None if c.get("sem") is None else round(c["sem"], 3),
                            "words": c.get("text", ""), "p": last.get("p"), "why": last.get("why", ""),
                            "stopped": t.stopped, "state": convo.state, "echo": bool(convo.echo_flags[-1]) if convo.echo_flags else False})
            continue
        if isinstance(raw, bytes):
            if test_buf is not None:
                test_buf.append(np.frombuffer(raw, dtype="<i2").reshape(-1, 2)[:, 1].copy())
                continue
            if tt is None:
                continue
            fr = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
            frames_a.append(fr[:, 0].copy())
            frames_c.append(fr[:, 1].copy())
            d = tt.on_frame(fr[:, 0], fr[:, 1])
            ms = len(frames_a) * 20
            if d:
                dec = d["decision"] if isinstance(d, dict) else d
                decisions.append((ms, dec))
                await ws.send(json.dumps({"type": "decision", "decision": dec, "t": ms}))
            if len(frames_a) % 5 == 0:
                lvl = 20 * np.log10(np.sqrt(np.mean(fr[:, 1].astype(np.float64) ** 2)) / 32768 + 1e-9)
                await ws.send(json.dumps({"type": "meter", "vad": round(tt.vad.prob, 2), "db": round(float(lvl), 1),
                                          "ref": round(tt.ref_db, 1), "t": ms}))
            while sent_trace < len(tt.trace):
                await ws.send(json.dumps({"type": "trace", **{k: v for k, v in tt.trace[sent_trace].items() if k != "f"}}, ensure_ascii=False))
                sent_trace += 1
            continue
        msg = json.loads(raw)
        if msg["type"] == "convo_start":
            from convo import brain, voice
            from convo.engine import Conversation
            previous = _active["ws"]
            if previous is not None and previous is not ws:     # a second tab or a new call replaces the running one
                _end_call()
                await previous.close()
            await send({"type": "status", "state": "loading", "text": "يجهز العقل (النموذج اللغوي)…"})
            name = await asyncio.to_thread(brain.start, msg.get("model") or None)
            if msg.get("engine") != "live":
                await voice.warm_fillers(SCEN_GENDER.get(msg.get("scenario", "payment"), "f"), msg.get("engine", "edge"))
            convo = Conversation(send, msg.get("scenario", "payment"), msg.get("engine", "edge"))
            convo.mic = msg.get("mic")                    # the real microphone settings, kept with the session
            convo.preflight = dict(LAST_PREFLIGHT) or None
            _active.update(ws=ws, convo=convo)
            await send({"type": "status", "state": "ready", "model": name})
            asyncio.create_task(convo.start())
            continue
        if convo is not None:
            if msg["type"] == "played":
                await convo.on_played(msg["id"])
            elif msg["type"] == "idle":
                convo.browser_idle = True
                await convo.on_idle()
            elif msg["type"] == "relabel":
                convo.score.relabel(msg.get("id"), msg.get("expected"))
                await convo.send_score()
            elif msg["type"] == "stopped":
                await convo.on_stopped(msg.get("id", -1), msg.get("played", 0))
            elif msg["type"] == "convo_end":
                from convo.engine import save_session
                save_session(convo, KIT / "runs" / "live-calls")
                convo.state = "ended"
                if convo.task and not convo.task.done():
                    convo.task.cancel()
                convo.shutdown()
                await send({"type": "summary", "stats": convo.stats})
                convo = None
            continue
        if msg["type"] == "start":
            tt = policy.TurnTaker()
            tt.on_scenario_start()
            frames_a, frames_c, decisions, sent_trace = [], [], [], 0
            agent_text = msg.get("text", "")
            if msg.get("calibrate"):
                tt.calibrating = True
                await ws.send(json.dumps({"type": "calibrating"}))   # caller says «هلا، معك» first (like U0)
            else:
                tt.on_agent_start(agent_text)
                await ws.send(json.dumps({"type": "started"}))
            t_start = time.time()
        elif msg["type"] == "agent_start" and tt:
            tt.calibrating = False
            tt.on_agent_start(agent_text)
            await ws.send(json.dumps({"type": "started", "ref_db": round(tt.ref_db, 1),
                                      "voiceprint": tt.ref_emb is not None}))
        elif msg["type"] == "agent_end" and tt:
            tt.on_agent_end()
            agent_end_ms = len(frames_a) * 20
            await ws.send(json.dumps({"type": "agent_end", "t": agent_end_ms}))
        elif msg["type"] == "test_start":
            test_buf = []
        elif msg["type"] == "test_end" and test_buf is not None:
            x = np.concatenate(test_buf) if test_buf else np.zeros(1600, np.int16)
            test_buf = None
            from turn.asr import transcribe
            r = await asyncio.to_thread(transcribe, x, False)
            await ws.send(json.dumps({"type": "test_result", "text": r.get("text", ""), **clarity(x)}, ensure_ascii=False))
        elif msg["type"] == "save" and frames_a:
            sid = save_try(frames_a, frames_c, agent_text, msg["expected"], msg.get("agent_end_ms"), tt.trace)
            await ws.send(json.dumps({"type": "saved", "sid": sid}))

    if convo is not None:                  # page closed or refreshed mid-call: keep the recording
        from convo.engine import save_session
        save_session(convo, KIT / "runs" / "live-calls")


def _gender():
    from convo.scenarios import SCENARIOS
    return {k: v["gender"] for k, v in SCENARIOS.items()}


SCEN_GENDER = {}


def clarity(x: np.ndarray) -> dict:
    """How much of the voice's detail (2-4 kHz, where consonants live) reaches us. Normal speech: about -20 dB
    relative to 0.3-1 kHz; a muffled microphone: -25 dB or worse."""
    from scipy import signal
    y = x.astype(np.float64)
    if np.sqrt(np.mean(y ** 2)) < 100:
        return {"clarity_db": None, "level_db": round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) / 32768 + 1e-9)), 1)}
    f, P = signal.welch(y, 16000, nperseg=1024)
    hi = P[(f >= 2000) & (f < 4000)].sum()
    lo = P[(f >= 300) & (f < 1000)].sum()
    return {"clarity_db": round(float(10 * np.log10(hi / lo)), 1),
            "level_db": round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) / 32768)), 1)}


LAST_PREFLIGHT: dict = {}
ASR_RED_MS, VOICE_RED_S, FREE_RED_GB = 250, 1.5, 3.0


def _free_memory_gb() -> float | None:
    """Memory that is free or quickly reclaimable, without extra packages."""
    try:
        import subprocess
        if sys.platform == "darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+)", out).group(1))
            pages = sum(int(re.search(rf"{k}:\s+(\d+)", out).group(1)) for k in ("Pages free", "Pages inactive", "Pages speculative"))
            return round(pages * page / 2 ** 30, 1)
        if sys.platform.startswith("linux"):
            return round(int(re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)) / 2 ** 20, 1)
    except Exception:
        return None
    return None


def _thermal() -> str | None:
    if sys.platform != "darwin":
        return None
    import subprocess
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
    limit = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    if limit:
        return "ok" if int(limit.group(1)) >= 100 else f"slowed to {limit.group(1)}%"
    return "warning" if re.search(r"(?i)warning level", out) and not re.search(r"(?i)no thermal warning", out) else "ok"


def preflight() -> dict:
    """Is this laptop ready for a live demo right now?"""
    from turn import asr, semantic
    x = read_wav_stereo(KIT / "data" / "dev" / "audio" / "GI-D01.wav")[:, 1]
    speech = x[16000:16000 * 4]
    asr.transcribe(speech, use_cache=False)          # warm: the first read also loads the model
    reads = sorted(asr.transcribe(speech, use_cache=False, priority="urgent")["ms"] for _ in range(5))
    semantic.p_stop("هلا", "هلا")
    t0 = time.perf_counter()
    semantic.p_stop("حبيت أذكرك بالدفعة", "كم المبلغ")
    meaning_ms = round((time.perf_counter() - t0) * 1000, 1)
    voice_s = None
    try:
        from convo.live_voice import LiveVoice

        async def first_sound():
            v = LiveVoice("أنت موظف خدمة عملاء سعودي. رد بكلمة وحدة.", "f")
            await v.connect()
            t = time.perf_counter()
            async for _ in v.speak("هلا"):
                break
            await v.close()
            return round(time.perf_counter() - t, 2)
        voice_s = asyncio.run(first_sound())
    except Exception as e:
        voice_s = f"failed: {type(e).__name__}"
    free = _free_memory_gb()
    result = {"time": time.strftime("%H:%M:%S"),
              "asr_p90_ms": reads[-1], "asr_ok": reads[-1] <= ASR_RED_MS,
              "meaning_ms": meaning_ms,
              "voice_first_sound_s": voice_s, "voice_ok": isinstance(voice_s, float) and voice_s <= VOICE_RED_S,
              "thermal": _thermal(), "free_memory_gb": free, "memory_ok": free is None or free >= FREE_RED_GB}
    result["thermal_ok"] = result["thermal"] in (None, "ok")
    result["all_ok"] = all(result[k] for k in ("asr_ok", "voice_ok", "memory_ok", "thermal_ok"))
    LAST_PREFLIGHT.clear()
    LAST_PREFLIGHT.update(result)
    return result


def save_try(fa, fc, agent_text, expected, agent_end_ms, trace):
    """Store a labelled try as a kit-format call. Onset/offset from the system's own voice detection;
    evident = onset + 500 ms (we do not know the exact moment, the label says so)."""
    LIVE.mkdir(parents=True, exist_ok=True)
    (LIVE / "audio").mkdir(exist_ok=True)
    (LIVE / "labels").mkdir(exist_ok=True)
    prefix = f"LV-{RECORDER}-" if RECORDER else "LV-"
    n = 1
    while (LIVE / "labels" / f"{prefix}{n:03d}.json").exists():
        n += 1
    sid = f"{prefix}{n:03d}"
    pre = np.zeros(16000, np.int16)       # 1 s of silence before, so the call starts like the others
    a = np.concatenate([pre, np.concatenate(fa)])
    c = np.concatenate([pre, np.concatenate(fc)])
    write_wav_stereo(LIVE / "audio" / f"{sid}.wav", np.stack([a, c], 1))
    cands = [t for t in trace if t["kind"] == "candidate"]
    onset = (cands[0]["t"] if cands else 1000) + 1000
    ends = [t["t"] for t in trace if t["kind"] in ("candidate_end", "STOP")]
    offset = max(onset + 300, (ends[-1] if ends else onset + 1200) + 1000)
    dur = len(a) // 16
    end_agent = (agent_end_ms or dur - 1000) + 1000
    ev = {"event_id": f"{sid}.e1", "kind": "interruption", "expected": expected, "onset_ms": onset, "offset_ms": min(offset, dur - 10)}
    if expected == "STOP":
        ev["evident_ms"] = onset + 500
        ev["deadline_ms"] = onset + 2000
    else:
        ev["window_end_ms"] = ev["offset_ms"] + 500
    lab = {"schema_version": "1.0", "scenario_id": sid, "split": "live", "category": "GI" if expected == "STOP" else "BC",
           "subtype": "live", "difficulty": 2, "dialect": RECORD_DIALECT, "elicitation": "naturalistic",
           "speakers": [{"speaker_id": "U", "role": "caller", "source": "live microphone"}],
           "acoustic": {"conditions": [RECORD_PLACE]},
           "audio": {"file": f"{sid}.wav", "sample_rate": 16000, "channels": ["agent", "user"], "duration_ms": dur},
           "context": {"use_case": "live test", "language": "ar"},
           "agent_utterances": [{"utt_id": "A1", "start_ms": 1000, "end_ms": min(end_agent, dur - 10), "text": agent_text}],
           "user_segments": [], "events": [ev], "provenance": {"label_status": "user_labelled_live"}}
    (LIVE / "labels" / f"{sid}.json").write_text(json.dumps(lab, ensure_ascii=False, indent=1), encoding="utf-8")
    write_live_manifest()
    return sid


def write_live_manifest() -> int:
    """List every labelled recording in data/live, for example after copying in a teammate's files."""
    ids = sorted(p.stem for p in (LIVE / "labels").glob("*.json") if (LIVE / "audio" / f"{p.stem}.wav").exists())
    (LIVE / "manifest.json").write_text(json.dumps({"dataset": "live", "split": "live", "scenarios": [{"scenario_id": i} for i in ids],
                                                    "thresholds": LIVE_THRESHOLDS}, ensure_ascii=False, indent=1))
    return len(ids)


def main():
    if "--manifest" in sys.argv:                       # rebuild data/live/manifest.json and exit
        print(f"data/live: {write_live_manifest()} recordings listed")
        return
    from websockets.asyncio.server import serve
    SCEN_GENDER.update(_gender())
    from turn.asr import warmup
    print("loading speech recognition ...", flush=True)
    warmup()
    srv = ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    async def run():
        async with serve(ws_handler, "127.0.0.1", WS_PORT, max_size=2 ** 20):
            print(f"open http://127.0.0.1:{HTTP_PORT}", flush=True)
            url = f"http://127.0.0.1:{HTTP_PORT}/demo"
            if Path("/Applications/Google Chrome.app").exists():   # Chrome handles mic + 16 kHz audio best
                import subprocess
                subprocess.run(["open", "-a", "Google Chrome", url])
            else:
                webbrowser.open(url)
            await asyncio.Future()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        from convo import brain
        brain.stop()


if __name__ == "__main__":
    main()
