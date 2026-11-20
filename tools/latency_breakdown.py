"""PLAN_V4 S-1: where the time to each STOP goes, from a run's decision log (or sim trace) and its labels.

    python -m tools.latency_breakdown runs/v2.5-cp-mirror-rt
For every STOP event that was caught: caller onset -> candidate opened, -> first read back, which read fired, the
recogniser's compute time per read (p50/p90/p99), and stop time minus evident. Reported per category.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from tools import cp_report as R  # noqa: E402


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p / 100 * len(v)))] if v else None


def main(argv=None) -> int:
    run = Path((argv or sys.argv[1:])[0])
    sc = R.load_run(run)
    data = R.find_data(run, sc)
    dlog = run / "decisions.jsonl"
    traces = R.traces_for(run, sc, dlog if dlog.exists() else None)
    rows = defaultdict(lambda: defaultdict(list))
    asr_ms = []
    for sid, tr in traces.items():
        asr_ms += [x["ms"] for x in tr if x.get("kind") == "asr" and x.get("ms") is not None]
    for e in sc["events"]:
        if e["expected"] != "STOP" or e.get("outcome") != "yield":
            continue
        lb = json.loads((data / "labels" / f"{e['scenario_id']}.json").read_text(encoding="utf-8"))
        ev = next(x for x in lb["events"] if x["event_id"] == e["event_id"])
        tr = traces.get(e["scenario_id"], [])
        on = ev["onset_ms"]
        cands = [x["t"] for x in tr if x.get("kind") == "candidate" and x["t"] >= on - 400]
        reads = [x for x in tr if x.get("kind") == "asr" and x["t"] >= on - 200]
        stop = next((x for x in tr if x.get("kind") == "STOP" and x["t"] >= on - 300), None)
        r = rows[e["category"]]
        if cands:
            r["onset→candidate"].append(cands[0] - on)
        if reads:
            r["onset→first read"].append(reads[0]["t"] - on)
        if stop:
            r["read # that fired"].append(sum(1 for x in reads if x["t"] <= stop["t"]))
            r["stop − evident"].append(stop["t"] - ev.get("evident_ms", on))
    print(f"# Latency breakdown: {run.name}\n")
    print(f"Recogniser compute per read: p50 {pct(asr_ms, 50)} · p90 {pct(asr_ms, 90)} · p99 {pct(asr_ms, 99)} · max {max(asr_ms) if asr_ms else None} ms ({len(asr_ms)} reads)\n")
    keys = ["onset→candidate", "onset→first read", "read # that fired", "stop − evident"]
    print("| Category | n | " + " | ".join(f"{k} p50 / p90" for k in keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for c, r in sorted(rows.items()):
        n = len(r["stop − evident"])
        print(f"| {c} | {n} | " + " | ".join(f"{pct(r[k], 50)} / {pct(r[k], 90)}" if r[k] else "–" for k in keys) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
