"""Live-demo consistency benchmark: replay the saved demo calls through the real TurnTaker and the demo's echo guard.

    .venv/bin/python -m tools.replay_live --repeat 3 --note baseline

Speech recognition runs for real on every read (no cache), and the stream clock is charged each read's measured time,
so latency is honest without replaying hours of audio in real time. The same call is replayed several times to measure
how often the system makes the same decision. Read-only on runs/live-calls; reports go to runs/replay-live-<note>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["TT_SIM"] = "1"                 # charge measured compute on the stream clock
os.environ["TT_ASR_CACHE"] = "0"           # ...and really recompute every read
KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))

import numpy as np  # noqa: E402

from turn import clock, lexicon, policy  # noqa: E402

CALLS = KIT / "runs" / "live-calls"
FRAME = 320
AGENT_ON = 1e5                             # agent frame power that counts as playing (same as the demo echo gate)
AGENT_HANGOVER_MS = 300
SINGLE_ACK = lexicon.ACK_TOKENS


def unique_calls(min_s: float = 8.0) -> list[Path]:
    """Each session is saved more than once while it runs: keep the longest copy of each."""
    import soundfile as sf
    best: dict[str, tuple[int, Path]] = {}
    for d in sorted(CALLS.iterdir()):
        f = d / "call.wav"
        if not f.exists() or not (d / "events.json").exists():
            continue
        info = sf.info(f)
        if info.duration < min_s:
            continue
        head, _ = sf.read(f, frames=16000 * 6, dtype="int16")
        key = hashlib.sha1(head.tobytes()).hexdigest()
        if key not in best or info.frames > best[key][0]:
            best[key] = (info.frames, d)
    return [d for _, d in sorted(best.values(), key=lambda x: x[1].name)]


def agent_lines(events: list[dict]) -> list[tuple[int, str]]:
    return [(e["t"], e.get("text", "")) for e in events if e.get("type") in ("say", "pcm") and e.get("text")]


def replay(call: Path) -> dict:
    import soundfile as sf
    from convo.engine import Conversation
    x, _ = sf.read(call / "call.wav", dtype="int16")
    lines = agent_lines(json.loads((call / "events.json").read_text(encoding="utf-8")))
    tt = policy.TurnTaker()
    tt.on_scenario_start()
    guard = Conversation.__new__(Conversation)           # only the demo's echo gate and word guard
    guard.agent_pow, guard.coupling, guard.echo_flags, guard.gated, guard.tt = [], [], [], 0, tt
    stops, suppressed, resumes, cands = [], [], [], set()
    last_agent, active, ms = -10 ** 9, False, 0
    for i in range(len(x) // FRAME):
        ms = (i + 1) * 20
        clock.set_now(ms)
        a, c = x[i * FRAME:(i + 1) * FRAME, 0], x[i * FRAME:(i + 1) * FRAME, 1]
        if float(np.mean(a.astype(np.float64) ** 2)) > AGENT_ON:
            last_agent = ms
        playing = ms - last_agent <= AGENT_HANGOVER_MS
        if playing and not active:
            text = next((t for s, t in lines if s >= ms - 2000), "")
            tt.on_agent_start(text)
            active = True
        elif not playing and active:
            tt.on_agent_end()
            active = False
        guard.ms = ms
        c = guard.echo_gate(a, c)
        d = tt.on_frame(a, c)
        dec = (d["decision"] if isinstance(d, dict) else d) if d else None
        if tt.cand is not None:
            cands.add(tt.cand["start"] // 200)
        if dec == "STOP":
            cand = tt.cand or {}
            text = cand.get("text", "")
            row = {"t": ms, "cand_start": cand.get("start", ms), "latency_ms": ms - cand.get("start", ms), "text": text,
                   "why": next((r.get("why", "") for r in reversed(tt.trace) if r["kind"] == "STOP"), "")}
            if guard._real_words():
                stops.append(row)
            else:                                        # the demo keeps talking: no clear caller words
                suppressed.append(row)
                tt.stopped, tt.cand = False, None
        elif dec == "RESUME":
            resumes.append(ms)
    reads = [r for r in tt.trace if r["kind"] == "asr" and r.get("ms")]
    agent_text = " ".join(t for _, t in lines)
    for s in stops:
        a = lexicon.analyse(s["text"], agent_text)
        s["single_ack"] = a["n"] == 1 and a["words"][0] in SINGLE_ACK
        s["echo"] = a["n"] >= 1 and a["echoed"] / a["n"] >= 0.5
    return {"call": call.name, "seconds": round(len(x) / 16000, 1), "candidates": sorted(cands), "stops": stops,
            "suppressed": suppressed, "resumes": resumes, "read_ms": [r["ms"] for r in reads]}


def pct(v, q):
    return None if not v else round(float(np.percentile(v, q)))


def summarise(runs: list[list[dict]]) -> dict:
    first = runs[0]
    lat = [s["latency_ms"] for r in first for s in r["stops"]]
    reads = [m for run in runs for r in run for m in r["read_ms"]]
    agree = total = 0
    for k in range(len(first)):                         # same call, same candidate: did every repeat decide alike?
        buckets = set().union(*(set(run[k]["candidates"]) for run in runs))
        for b in buckets:
            outcomes = {any(s["cand_start"] // 200 == b for s in run[k]["stops"]) for run in runs}
            total += 1
            agree += len(outcomes) == 1
    stops = [s for r in first for s in r["stops"]]
    return {"calls": len(first), "audio_min": round(sum(r["seconds"] for r in first) / 60, 1), "repeats": len(runs),
            "stops": len(stops), "suppressed_by_demo_guard": sum(len(r["suppressed"]) for r in first),
            "resumes": sum(len(r["resumes"]) for r in first),
            "stop_latency_ms": {"p50": pct(lat, 50), "p90": pct(lat, 90), "p99": pct(lat, 99), "max": max(lat) if lat else None},
            "asr_read_ms": {"p50": pct(reads, 50), "p90": pct(reads, 90), "p99": pct(reads, 99), "max": max(reads) if reads else None},
            "single_ack_stops": sum(s["single_ack"] for s in stops), "echo_stops": sum(s["echo"] for s in stops),
            "agreement": round(agree / total, 3) if total else None, "candidates": total}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--note", default="run")
    ap.add_argument("--limit", type=int, default=0, help="only the first N calls")
    a = ap.parse_args(argv)
    from turn import asr
    asr.warmup()
    calls = unique_calls()[: a.limit or None]
    t0 = time.time()
    runs = [[replay(c) for c in calls] for _ in range(a.repeat)]
    summary = summarise(runs)
    out = KIT / "runs" / f"replay-live-{a.note}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps({"summary": summary, "calls": runs[0]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    for s in [s for r in runs[0] for s in r["stops"] if s["single_ack"] or s["echo"]][:12]:
        print(f"  {'single word' if s['single_ack'] else 'echo':11s} {s['latency_ms']:5d} ms  «{s['text'][:30]}»  {s['why'][:60]}")
    print(f"{time.time() - t0:.0f}s · report: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
