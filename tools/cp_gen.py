"""Render the checkpoint-mirror and stress banks (tools/cp_bank.py) into calls in the exact kit format, so
ttbench.play, ttbench.score and tools/sim run on them unchanged (PLAN_V3 §3.2).

    python -m tools.cp_gen --bank data/cp_bank.json --which mirror      --out data/cp-mirror
    python -m tools.cp_gen --bank data/cp_bank.json --which stress-dev  --out data/cp-stress-dev
    python -m tools.cp_gen --bank data/cp_bank.json --which stress-test --out data/cp-stress-test

Layout follows SCRIPTS.md: A0, U0 (a clean sample of the caller), A1 (one or two utterances), the caller's event
anchored to an agent word (start(«w»)+x or end(«w»)+x), and a tail. `evident` is the end of the word the writer
marked *like this*; the caller line is split there so the time is exact for every voice engine.

Voices, so the mirror does not sound like our training data:
  * female Saudi callers: Lahja SA (a Saudi VITS voice we never trained on), speed-perturbed per call;
  * other callers: edge-tts Gulf voices with rate and pitch OUTSIDE the range used for training (±13-22 %, ±26-40 Hz);
  * agent: the edge Saudi voices (Sarj's agent is a fixed synthetic Saudi voice too).
Measurement only: never copy these sentences into the lexicon, tests, teacher data or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from tools import synth_calls as sc  # noqa: E402
from ttbench.dataset import write_wav_stereo  # noqa: E402
from turn.config import EXT_DIR  # noqa: E402

SR = 16000
LAHJA_DIR = EXT_DIR / "tts-lahja"
LAHJA_CACHE = EXT_DIR / "tts-cache-lahja"
AGENT = {"m": "ar-SA-HamedNeural", "f": "ar-SA-ZariyahNeural"}
EDGE_BY_DIALECT = {"gulf-kuwaiti": "KW", "gulf-emirati": "AE", "gulf-bahraini": "BH", "gulf-qatari": "QA"}
EDGE = {("SA", "m"): "ar-SA-HamedNeural", ("SA", "f"): "ar-SA-ZariyahNeural", ("KW", "m"): "ar-KW-FahedNeural",
        ("KW", "f"): "ar-KW-NouraNeural", ("AE", "m"): "ar-AE-HamdanNeural", ("AE", "f"): "ar-AE-FatimaNeural",
        ("BH", "m"): "ar-BH-AliNeural", ("BH", "f"): "ar-BH-LailaNeural", ("QA", "m"): "ar-QA-MoazNeural",
        ("QA", "f"): "ar-QA-AmalNeural", ("OM", "m"): "ar-OM-AbdullahNeural", ("OM", "f"): "ar-OM-AyshaNeural"}
ESC_FOR = {"cough": "coughing", "laugh": "laughing", "sneeze": "sneezing", "sigh": "breathing", "throat": "coughing",
           "horn": "car_horn", "engine": "engine", "dog": "dog", "baby": "crying_baby"}
S3_STOP_OFFSETS = [-300, -200, -100, 0, 100, 200, 300]
S3_KEEP_OFFSETS = [-200, 0, 200]
PAUSE = re.compile(r"\[pause\s*(\d+)\s*(required|optional)?\]")
DIAC = re.compile(r"[ً-ْـ]")


def norm(w: str) -> str:
    w = DIAC.sub("", w)
    w = re.sub(r"[^\w]", "", w)
    return w.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")


# ---------------------------------------------------------------- voices
class Voice:
    """One caller (or side speaker) voice: an edge voice with a rate/pitch, or Lahja with a speed factor."""

    def __init__(self, engine: str, name: str, rate: str = "+0%", pitch: str = "+0Hz", speed: float = 1.0):
        self.engine, self.name, self.rate, self.pitch, self.speed = engine, name, rate, pitch, speed

    def source(self) -> str:
        return f"lahja-sa x{self.speed:.2f}" if self.engine == "lahja" else f"edge-tts:{self.name} {self.rate} {self.pitch}"

    def items(self, text: str):
        return [(self.name, text, self.rate, self.pitch)] if self.engine == "edge" else []

    def say(self, text: str):
        """(audio float32 at 16 kHz, word timings or None)."""
        if self.engine == "edge":
            return sc.tts_get(self.name, text, self.rate, self.pitch)
        x = lahja(text)
        x, _ = sc.perturb(x, [], self.speed)
        return x, None


_LAHJA = {}


def lahja(text: str) -> np.ndarray:
    key = hashlib.sha1(text.encode()).hexdigest()
    f = LAHJA_CACHE / f"{key}.wav"
    if f.exists():
        return sf.read(f, dtype="float32")[0]
    if not _LAHJA:
        import torch
        from transformers import AutoTokenizer, VitsModel
        _LAHJA["m"] = VitsModel.from_pretrained(LAHJA_DIR)
        _LAHJA["tok"] = AutoTokenizer.from_pretrained(LAHJA_DIR)
        _LAHJA["torch"] = torch
        _LAHJA["torch"].manual_seed(0)
    with _LAHJA["torch"].no_grad():
        x = _LAHJA["m"](**_LAHJA["tok"](text, return_tensors="pt")).waveform[0].numpy().astype(np.float32)
    LAHJA_CACHE.mkdir(parents=True, exist_ok=True)
    sf.write(f, x, SR)
    return x


def new_edge(locale: str, gender: str, rng) -> Voice:
    sign = lambda: 1 if rng.random() < 0.5 else -1  # noqa: E731
    rate = f"{sign() * int(rng.integers(13, 23)):+d}%"
    pitch = f"{sign() * int(rng.integers(26, 41)):+d}Hz"
    return Voice("edge", EDGE[(locale, gender)], rate, pitch)


def caller_voice(call: dict, rng) -> Voice:
    g = "f" if call.get("gender") == "f" else "m"
    loc = EDGE_BY_DIALECT.get(call.get("dialect", ""), "SA")
    if loc == "SA" and g == "f" and rng.random() < 0.8:
        return Voice("lahja", "lahja", speed=float(rng.uniform(0.9, 1.12)))
    return new_edge(loc, g, rng)


def side_voice(call: dict, caller: Voice, rng) -> Voice:
    who = call.get("side_who") or "woman"
    if who == "child":
        return Voice("edge", EDGE[("SA", "f") if rng.random() < 0.5 else ("KW", "f")], "+12%", "+55Hz")
    g = "f" if who == "woman" else "m"
    if g == "f" and caller.engine != "lahja" and rng.random() < 0.6:
        return Voice("lahja", "lahja", speed=float(rng.uniform(0.92, 1.08)))
    loc = rng.choice(["SA", "KW", "BH", "AE"])
    v = new_edge(loc, g, rng)
    return v if v.name != caller.name else new_edge("QA", g, rng)


# ---------------------------------------------------------------- caller line -> pieces
def pieces(u1: str) -> list[dict]:
    """Split a caller line into spoken pieces: chunks between [pause N], and inside a chunk a cut right after the
    *marked* word (so its end is exactly known). Returns [{text, pause_after, hold, evident_after}]."""
    out = []
    parts = PAUSE.split(u1)
    # parts = [text, N, hold, text, N, hold, ..., text]
    chunks = []
    for i in range(0, len(parts), 3):
        txt = parts[i].strip()
        pause = int(parts[i + 1]) if i + 1 < len(parts) else 0
        hold = parts[i + 2] if i + 2 < len(parts) else None
        if txt:
            chunks.append((txt, pause, hold))
        elif chunks and pause:
            t, p, h = chunks[-1]
            chunks[-1] = (t, p + pause, hold or h)
    for txt, pause, hold in chunks:
        m = re.search(r"\*([^*]+)\*", txt)
        if m:
            head, tail = txt[:m.end()], txt[m.end():]
            tail_clean = tail.strip(" ،,.")
            if tail_clean:
                out.append({"text": head, "pause_after": 40, "hold": None, "evident_after": True})
                out.append({"text": tail.strip(), "pause_after": pause, "hold": hold, "evident_after": False})
                continue
            out.append({"text": txt, "pause_after": pause, "hold": hold, "evident_after": True})
        else:
            out.append({"text": txt, "pause_after": pause, "hold": hold, "evident_after": False})
    return out


def trim_off(x, thr_db=-45):
    """synth_calls.trim, also returning how many ms were cut from the start."""
    e = 20 * np.log10(np.abs(x) + 1e-9)
    idx = np.where(e > thr_db)[0]
    if not len(idx):
        return x, 0.0
    a = max(0, idx[0] - 160)
    return x[a: idx[-1] + 160], a / 16


def tight(x, rel_db=32.0):
    """Cut a caller clip to its voiced part: 10 ms frames within rel_db of the clip's loudest frame (+10/20 ms)."""
    n = len(x) // 160
    if n < 3:
        return x
    db = 20 * np.log10(np.sqrt((x[: n * 160].reshape(n, 160) ** 2).mean(1)) + 1e-9)
    on = np.where(db > db.max() - rel_db)[0]
    return x[max(0, on[0] - 1) * 160: min(n, on[-1] + 3) * 160]


def clean(t: str) -> str:
    return PAUSE.sub(" ", t).replace("*", "").strip()


# ---------------------------------------------------------------- one call
def lvl_for(delivery: str, rng) -> float:
    return {"soft": rng.uniform(-9, -5), "loud": rng.uniform(2, 5)}.get(delivery or "normal", rng.uniform(-2, 1))


def conditions(call: dict, rng) -> list[str]:
    snd = call.get("sound") or "none"
    if snd == "echo":
        return ["speakerphone"]
    base = {"tv": "tv", "horn": "street", "engine": "street", "babble": "cafe", "baby": "household"}.get(snd)
    if call.get("side"):
        base = "household"
    base = base or rng.choice(["quiet", "quiet", "ac-hum", "car", "street", "household", "cafe"])
    return [base] + (["phone-nb"] if rng.random() < 0.42 else [])


def plan_items(call: dict, rng) -> tuple[dict, list]:
    """Choose voices and list every edge utterance this call needs (so they can be fetched in one batch)."""
    ag = "f" if rng.random() < 0.5 else "m"
    p = {"agent": Voice("edge", AGENT[ag]), "caller": caller_voice(call, rng)}
    if p["caller"].name == p["agent"].name:
        p["agent"] = Voice("edge", AGENT["m" if ag == "f" else "f"])
    if call.get("side"):
        p["side"] = side_voice(call, p["caller"], rng)
    if call.get("sound") == "tv" and call.get("tv"):
        p["tv"] = Voice("edge", rng.choice(sc.TV_VOICES))
    items = [(p["agent"].name, t, "+0%", "+0Hz") for t in [call["a0"], *call["a1"]]]
    items += p["caller"].items(call["u0"])
    for pc in pieces(call.get("u1") or ""):
        items += p["caller"].items(clean(pc["text"]))
    if "side" in p:
        items += p["side"].items(call["side"])
    if "tv" in p:
        items += p["tv"].items(call["tv"])
    return p, items


def find_anchor(words, anchor: str):
    a = norm(anchor or "")
    if not a:
        return None
    for w in words:
        if norm(w["w"]) == a:
            return w
    for w in words:
        nw = norm(w["w"])
        if a in nw or (len(nw) >= 3 and nw in a):
            return w
    return None


def render(i: int, call: dict, p: dict, rng, sid: str, split: str) -> tuple[np.ndarray, dict, list[str]]:
    notes = []
    agent, caller = p["agent"], p["caller"]
    cat, exp = call["category"], call["expected"]
    probe = call.get("probe")
    if probe == "S4" and exp == "STOP":
        cat = "GI"                                   # the control: the caller says it to the agent
    a0, _ = agent.say(call["a0"])
    a0 = sc.set_level(sc.trim(a0), -20)
    u0, _ = caller.say(call["u0"])
    caller_ref = -20 + rng.uniform(-4, 3)
    u0 = sc.set_level(tight(u0), caller_ref)
    t_a0 = 300
    a0_end = t_a0 + len(a0) / 16
    t_u0 = a0_end + rng.uniform(250, 500)
    u0_end = t_u0 + len(u0) / 16
    utts, t = [], u0_end + rng.uniform(400, 1100)
    for j, txt in enumerate(call["a1"]):
        x, w = agent.say(txt)
        x, cut_ms = trim_off(x, -50)
        x = sc.set_level(x, -20)
        utts.append({"text": txt, "x": x, "words": w, "start": t, "end": t + len(x) / 16, "cut": cut_ms})
        t = t + len(x) / 16 + (900 if j < len(call["a1"]) - 1 else 0)
    last = utts[-1]
    # agent word times in call ms (edge word offsets count from the untrimmed clip)
    words = [{"w": w["w"], "s": last["start"] + w["s"] - last["cut"], "e": last["start"] + w["s"] + w["d"] - last["cut"]}
             for w in last["words"]]

    # ---- the caller's event
    u1 = call.get("u1")
    segs = [{"seg_id": "U0", "speaker_id": "C", "start_ms": round(t_u0), "end_ms": round(u0_end), "kind": "speech",
             "text": call["u0"]}]
    events = []
    ev_id = lambda: f"{sid}.e{len(events) + 1}"  # noqa: E731
    n_tail = 2500
    agent_buf_extra = []
    caller_parts = []                               # (audio, start_ms)
    onset = offset = evident = None
    pauses = []
    if exp == "TAKE_TURN" and u1:
        tt = last["end"] + rng.uniform(400, 800)
        lvl = caller_ref
        for pc in pieces(u1):
            x, _ = caller.say(clean(pc["text"]))
            x = sc.set_level(tight(x), lvl)
            caller_parts.append((x, tt))
            onset = tt if onset is None else onset
            end = tt + len(x) / 16
            if pc["pause_after"]:
                pauses.append({"start_ms": round(end), "end_ms": round(end + pc["pause_after"]), "hold": pc["hold"] or "required"})
            offset = end
            tt = end + pc["pause_after"]
        n_tail = 3500
        events.append({"event_id": ev_id(), "kind": "end_of_turn", "expected": "TAKE_TURN", "onset_ms": round(onset),
                       "turn_end_ms": round(offset), "pauses": pauses, "rationale": call.get("why", "")})
        segs.append({"seg_id": "U1", "speaker_id": "C", "start_ms": round(onset), "end_ms": round(offset), "kind": "speech",
                     "text": clean(u1)})
    elif u1:
        aw = find_anchor(words, call.get("anchor"))
        if probe == "S3":
            offs = S3_STOP_OFFSETS if exp == "STOP" else S3_KEEP_OFFSETS
            off = offs[i % len(offs)]
            t_on = last["start"] + off
            call["_onset_rel_agent_ms"] = off
        elif aw is None:
            notes.append(f"anchor «{call.get('anchor')}» not found; at 40 % of the utterance")
            t_on = last["start"] + 0.4 * (last["end"] - last["start"])
        else:
            t_on = (aw["s"] if call.get("anchor_mode") == "start" else aw["e"]) + float(call.get("offset_ms") or 0)
        lvl = caller_ref + lvl_for(call.get("delivery"), rng)
        if cat in ("BC", "HT") and call.get("delivery") not in ("soft", "loud"):
            lvl = caller_ref + rng.uniform(-7, -2)
        tt, ev_t = t_on, None
        for pc in pieces(u1):
            x, w = caller.say(clean(pc["text"]))
            x = sc.set_level(tight(x), lvl)
            caller_parts.append((x, tt))
            end = tt + len(x) / 16
            if pc["evident_after"] and ev_t is None:
                ev_t = end
            tt = end + pc["pause_after"]
            offset = end
        onset = t_on
        if exp in ("STOP", "EITHER"):
            if ev_t is None and exp == "STOP":
                ev_t = onset + 400 if call.get("delivery") == "loud" else min(offset, onset + 700)
                notes.append("no marked word: evident by rule")
            if ev_t is not None and probe == "S3":
                ev_t = max(ev_t, last["start"])      # nobody can yield before the agent has started
            evident = ev_t
        # production constraints (SCRIPTS §3): agent talks >= 2 s after evident (STOP), >= 1 s after offset (KEEP)
        need = (evident + 2000) if exp == "STOP" and evident else offset + 1000
        if last["end"] < need and probe != "S3":
            shift = min(need - last["end"], t_on - (last["start"] + 200))
            if shift > 0:
                caller_parts = [(x, s - shift) for x, s in caller_parts]
                onset, offset = onset - shift, offset - shift
                evident = evident - shift if evident else None
                notes.append(f"moved {shift:.0f} ms earlier to keep the agent talking after the event")
        if last["end"] < ((evident + 2000) if exp == "STOP" and evident else offset + 1000):
            notes.append("constraint: agent ends too soon after the event")
        kind = "backchannel" if cat == "BC" else "filler" if cat == "HT" else "speech"
        segs.append({"seg_id": "U1", "speaker_id": "C", "start_ms": round(onset), "end_ms": round(offset), "kind": kind,
                     "text": clean(u1)})
        e = {"event_id": ev_id(), "kind": "interruption", "expected": exp, "onset_ms": round(onset), "offset_ms": round(offset),
             "rationale": call.get("why", "")}
        if exp in ("KEEP", "EITHER"):
            e["window_end_ms"] = round(offset + 500)
        if evident is not None:
            e["evident_ms"] = round(evident)
            if exp == "STOP":
                e["deadline_ms"] = round(evident + 1500)
        if exp == "EITHER":
            e["preferred"] = call.get("preferred") or "KEEP"
        events.append(e)

    dur = max(last["end"], offset or 0) + n_tail
    n = int(dur * 16)
    agent_ch = np.zeros(n, np.float32)
    caller_ch = np.zeros(n, np.float32)
    sc.place(agent_ch, a0, t_a0)
    for u in utts:
        sc.place(agent_ch, u["x"], u["start"])
    sc.place(caller_ch, u0, t_u0)
    for x, s in caller_parts:
        sc.place(caller_ch, x, s)

    snd = call.get("sound") or "none"
    if exp != "TAKE_TURN" and not u1:
        # caller silent: the event is a sound or another voice
        mid = last["start"] + rng.uniform(0.2, 0.45) * (last["end"] - last["start"])
        if call.get("side"):
            sv = p["side"]
            x, _ = sv.say(call["side"])
            rel = rng.uniform(0, 4) if probe == "S4" else (rng.uniform(-4, 2) if (call.get("side_who") == "child") else rng.uniform(-10, -3))
            x = sc.set_level(tight(x), caller_ref + rel)
            x = sc.reverb(x, rng, 0.3 if probe == "S4" else 0.45)
            sc.place(caller_ch, x, mid)
            segs.append({"seg_id": "X1", "speaker_id": "S2", "start_ms": round(mid), "end_ms": round(mid + len(x) / 16),
                         "kind": "side_speech", "text": call["side"]})
            events.append({"event_id": ev_id(), "kind": "interruption", "expected": "KEEP", "onset_ms": round(mid),
                           "offset_ms": round(mid + len(x) / 16), "window_end_ms": round(mid + len(x) / 16 + 500),
                           "rationale": call.get("why", "someone else in the room")})
            if mid + len(x) / 16 + 1000 > last["end"]:
                notes.append("constraint: side speech ends too close to the agent's end")
        elif snd == "tv" and "tv" in p:
            x, _ = p["tv"].say(call["tv"])
            x = sc.reverb(sc.set_level(x, caller_ref + rng.uniform(-18, -10)), rng, 0.6)
            x = np.tile(x, 4)[: int((dur - 300) * 16)]
            sc.place(caller_ch, x, 300)
            events.append({"event_id": ev_id(), "kind": "interruption", "expected": "KEEP", "onset_ms": round(last["start"]),
                           "offset_ms": round(last["end"]), "window_end_ms": round(last["end"] + 500), "rationale": "TV in the room"})
        elif snd == "echo":
            pass                                         # added below with the speakerphone condition
        elif snd == "babble":
            # the babble itself is the cafe bed added below
            events.append({"event_id": ev_id(), "kind": "interruption", "expected": "KEEP", "onset_ms": round(last["start"]),
                           "offset_ms": round(last["end"]), "window_end_ms": round(last["end"] + 500), "rationale": "cafe babble"})
        else:
            catname = ESC_FOR.get(snd, "coughing" if cat == "NV" else "car_horn")
            x = sc.esc_clip(catname, rng)[: int(2.5 * SR)]
            x = sc.set_level(x, caller_ref + (rng.uniform(-6, 1) if cat == "NV" else rng.uniform(-8, 3)))
            sc.place(caller_ch, x, mid)
            segs.append({"seg_id": "X1", "speaker_id": "C", "start_ms": round(mid), "end_ms": round(mid + len(x) / 16),
                         "kind": "nonverbal", "text": "", "delivery": catname})
            events.append({"event_id": ev_id(), "kind": "interruption", "expected": "KEEP", "onset_ms": round(mid),
                           "offset_ms": round(mid + len(x) / 16), "window_end_ms": round(mid + len(x) / 16 + 500),
                           "rationale": f"{catname}; nothing said"})

    conds = conditions(call, rng)
    if "speakerphone" in conds:
        lag = rng.uniform(150, 250)
        e = np.concatenate([np.zeros(int(lag * 16), np.float32), agent_ch])[:n]
        from scipy import signal
        e = signal.sosfilt(signal.butter(2, [250, 3500], "band", fs=SR, output="sos"), e)
        e = sc.reverb(e, rng, 0.3)[:n] * 10 ** (rng.uniform(-25, -15) / 20)
        caller_ch[: len(e)] += e[:n]
        if not events:
            events.append({"event_id": ev_id(), "kind": "interruption", "expected": "KEEP", "onset_ms": round(last["start"] + 300),
                           "offset_ms": round(last["end"]), "window_end_ms": round(last["end"] + 500),
                           "rationale": "speakerphone: the agent's own voice comes back"})
    base = conds[0] if conds[0] not in ("speakerphone", "tv") else "quiet"
    caller_ch += sc.bed(base, n, rng).astype(np.float32)
    if base in ("street", "cafe") and rng.random() < 0.6:
        sc.place(caller_ch, sc.set_level(sc.esc_clip(rng.choice(["car_horn", "engine", "footsteps"]), rng), caller_ref - 14),
                 rng.uniform(1000, max(1500, dur - 3000)))
    if "phone-nb" in conds:
        caller_ch = sc.phone(caller_ch).astype(np.float32)
    stereo = (np.stack([np.clip(agent_ch, -1, 1), np.clip(caller_ch, -1, 1)], 1) * 32767).astype(np.int16)

    if not events:
        notes.append("no event: skipped")
    a_utts = [{"utt_id": "A0", "start_ms": t_a0, "end_ms": round(a0_end), "text": call["a0"], "is_question": True}]
    for j, u in enumerate(utts):
        uid = "A1" if len(utts) == 1 else f"A1{chr(97 + j)}"
        a_utts.append({"utt_id": uid, "start_ms": round(u["start"]), "end_ms": round(u["end"]), "text": u["text"],
                       "is_question": u["text"].strip().endswith(("؟", "?"))})
    for e in events:
        e["category"] = cat
    label = {
        "schema_version": "1.0", "scenario_id": sid, "split": split, "category": cat,
        "subtype": f"{probe}-{exp}" if probe else exp, "difficulty": 3 if probe else 2,
        "dialect": call.get("dialect", "sa-najdi"), "elicitation": "scripted",
        "speakers": [{"speaker_id": "C", "role": "caller", "source": caller.source(), "gender": call.get("gender", "m"),
                      "dialect": call.get("dialect", "sa-najdi")}],
        "agent_voice": agent.name, "acoustic": {"conditions": conds, "noise_source_ids": ["ESC-50", "procedural"]},
        "audio": {"file": f"{sid}.wav", "sample_rate": SR, "channels": ["agent", "user"], "duration_ms": int(len(stereo) / 16)},
        "context": {"use_case": call.get("use_case", ""), "language": "ar"},
        "agent_utterances": a_utts, "user_segments": segs, "events": events,
        "provenance": {"production_mode": "synthetic", "label_status": "generated", "generator": "tools/cp_gen.py",
                       "bank": split, "probe": probe, "why": call.get("why", ""), "notes": notes,
                       **({"onset_rel_agent_ms": call["_onset_rel_agent_ms"]} if "_onset_rel_agent_ms" in call else {})},
    }
    return stereo, label, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", default=str(KIT / "data" / "cp_bank.json"))
    ap.add_argument("--which", required=True, help="mirror, stress-dev or stress-test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=31)
    a = ap.parse_args(argv)
    bank = json.loads(Path(a.bank).read_text(encoding="utf-8"))[a.which]
    calls = [c for key in bank for c in bank[key]]
    rng = np.random.default_rng(a.seed)
    plans = [plan_items(c, rng) for c in calls]
    items = [it for _, its in plans for it in its]
    print(f"{a.which}: {len(calls)} calls, {len(set(items))} edge utterances to fetch", flush=True)
    sc.tts_batch(items)
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    tag = {"mirror": "M", "stress-dev": "d", "stress-test": "t"}.get(a.which, "X")
    scen, counts, problems = [], {}, []
    for i, (c, (p, _)) in enumerate(zip(calls, plans)):
        key = c.get("probe") or c["category"]
        counts[key] = counts.get(key, 0) + 1
        cat = "GI" if (c.get("probe") == "S4" and c["expected"] == "STOP") else c["category"]
        sid = f"{cat}-{c['probe'] + tag if c.get('probe') else tag}{counts[key]:02d}"
        try:
            stereo, label, notes = render(counts[key] - 1, c, p, rng, sid, a.which)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{sid}: {type(e).__name__} {e}")
            continue
        if not label["events"]:
            problems.append(f"{sid}: no event")
            continue
        write_wav_stereo(out / "audio" / f"{sid}.wav", stereo)
        (out / "labels" / f"{sid}.json").write_text(json.dumps(label, ensure_ascii=False, indent=1), encoding="utf-8")
        scen.append({"scenario_id": sid})
        if notes:
            problems += [f"{sid}: {n}" for n in notes]
    manifest = {"dataset": "sarj-cp-" + a.which, "split": out.name, "label_status": "generated",
                "audio_source": "edge-tts Gulf voices (out-of-training rate/pitch) + Lahja SA + ESC-50 + procedural effects",
                "thresholds": {"T_full_ms": 300, "T_dead_ms": 1500, "E_full_ms": 600, "E_dead_ms": 2000, "E_miss_ms": 3000},
                "scenarios": scen}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "gen_notes.txt").write_text("\n".join(problems) + "\n", encoding="utf-8")
    print(f"wrote {len(scen)} calls to {out}; {len(problems)} notes in gen_notes.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
