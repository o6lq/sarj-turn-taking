"""Which weakness (PLAN_V3 W1–W5) explains each failure of a run? Evidence comes from the run's trace (tools.sim) or
its decision log (real time): the rule that fired, the word-layer features, timings.

    python -m tools.cp_diagnose runs/sim-v2.0-cp-stress-dev
    python -m tools.cp_diagnose runs/v2.0-cp-mirror-rt --dlog runs/v2.0-cp-mirror-rt/decisions.jsonl

Tags (a failure can carry several):
  W1 answer-as-echo   a STOP/EITHER event where a read had echoed words, no bid word and < 2 content words, and the
                      rules called it an acknowledgement (the caller's answer repeated the agent's words)
  W2 resume-over      a STOP inside the event, then RESUME before the caller finished (scored as a miss)
  W3 late-onset       the caller started < 300 ms after (or before) the agent's utterance start, and the first
                      candidate opened > 200 ms after the caller's onset, or never
  W4 bid-from-other   a false stop fired by "floor-bid word" (records spk_sim / rel_db / sem_p: would a speaker or
                      meaning check have vetoed it?)
  W5 ambiguous        AM failures, split into late (stopped after 300 ms), vetoed (the model or meaning veto
                      overruled a rule stop), not-read (no read showed the continuation in time), kept (read but ruled
                      an acknowledgement)
  other               anything else (the rule that fired is shown)
Writes <run>/diagnosis.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from tools import cp_report as R  # noqa: E402
from tools import errors as E  # noqa: E402


def reads(trace, a, b):
    return [x for x in trace if x.get("kind") == "asr" and a <= x.get("t", 0) <= b]


def kinds(trace, kind, a, b):
    return [x for x in trace if x.get("kind") == kind and a <= x.get("t", 0) <= b]


def feat(x, k, default=None):
    return (x.get("f") or {}).get(k, default)


def last_agent_start(label, t):
    starts = [u["start_ms"] for u in label["agent_utterances"] if u["start_ms"] <= t + 400]
    return max(starts) if starts else None


def diagnose(score, labels, traces):
    out = []
    events = {e["event_id"]: e for e in score["events"]}
    for err in E.classify(score, labels, traces):
        sid = err["call"]
        lb = labels.get(sid, {})
        tr = traces.get(sid, [])
        ev = events.get(err["event"], {})
        lab_ev = next((x for x in lb.get("events", []) if x["event_id"] == err["event"]), {})
        onset = lab_ev.get("onset_ms", err.get("t") or 0)
        offset = lab_ev.get("offset_ms", onset)
        evident = lab_ev.get("evident_ms", onset)
        a, b = onset - 300, max(offset, evident) + 1500
        rs = reads(tr, a, b)
        stops, resumes = kinds(tr, "STOP", a, b), kinds(tr, "RESUME", a, b)
        cands = kinds(tr, "candidate", onset - 400, b)
        tags, notes = [], []
        exp = lab_ev.get("expected", ev.get("expected", "KEEP"))
        if err["type"] in ("missed", "late") and exp in ("STOP", "EITHER"):
            if any(feat(x, "echoed", 0) >= 1 and feat(x, "bid", 0) == 0 and feat(x, "content", 0) < 2
                   and "acknowledgement" in str(x.get("why", "")) for x in rs):
                tags.append("W1")
            if stops and resumes and any(r["t"] > stops[0]["t"] and r["t"] < offset for r in resumes):
                tags.append("W2")
            ag = last_agent_start(lb, onset)
            if ag is not None and onset - ag < 300:
                first = min((c["t"] for c in cands), default=None)
                if first is None or first - onset > 200:
                    tags.append("W3")
                notes.append(f"onset {onset - ag:+.0f} ms vs agent start; first candidate "
                             f"{'none' if first is None else f'{first - onset:+.0f} ms'} after onset")
            if lb.get("category") == "AM" or ev.get("category") == "AM":
                if any("doubts" in str(x.get("why", "")) or "not for the agent" in str(x.get("why", "")) for x in rs):
                    tags.append("W5-vetoed")
                elif err["type"] == "late":
                    tags.append("W5-late")
                elif not rs:
                    tags.append("W5-not-read")
                else:
                    tags.append("W5-kept")
        if err["type"] == "false_stop":
            hit = stops[0] if stops else {}
            why = str(hit.get("why", err.get("rule", "")))
            if why.startswith("floor-bid word"):
                tags.append("W4")
                before = [x for x in rs if x.get("t", 0) <= hit.get("t", 1e12)]
                if before:
                    x = before[-1]
                    notes.append(f"bid={feat(x, 'bid')} spk_sim={feat(x, 'spk_sim')} rel_db={feat(x, 'rel_db')} sem_p={feat(x, 'sem_p')}")
        if not tags:
            tags.append("other")
        last = rs[-1] if rs else {}
        out.append({**err, "tags": tags, "notes": notes, "probe": lb.get("provenance", {}).get("probe"),
                    "subtype": lb.get("subtype", ""), "reads": [(x.get("t"), x.get("text", ""), x.get("why", ""), x.get("p"),
                                                                 feat(x, "bid"), feat(x, "content"), feat(x, "echoed"))
                                                                for x in rs][:6],
                    "last_rule": last.get("why", err.get("rule", "")), "n_stops": len(stops), "n_resumes": len(resumes)})
    return out


def report(name, diag) -> str:
    tagc = Counter(t for d in diag for t in d["tags"])
    lines = [f"# Diagnosis: {name}", "", f"{len(diag)} failures.", "", "| Tag | Failures |", "|---|---|",
             *[f"| {t} | {n} |" for t, n in sorted(tagc.items())], ""]
    by = defaultdict(list)
    for d in diag:
        for t in d["tags"]:
            by[t].append(d)
    for t in sorted(by):
        lines += [f"## {t}", "", "| Event | Type | Probe/subtype | Dialect · noise | Rule (last read / stop) | Notes | Reads (t, text, rule, p, bid, content, echoed) |",
                  "|---|---|---|---|---|---|---|"]
        for d in by[t]:
            rd = " ; ".join(f"{r[0]} «{r[1]}» {r[2]} p={r[3]} b{r[4]}/c{r[5]}/e{r[6]}" for r in d["reads"]).replace("|", " ")
            lines.append(f"| {d['event']} | {E.TYPES[d['type']]} | {d['subtype']} | {d['dialect']} · {d['noise']} | "
                         f"{str(d['last_rule']).replace('|', ' ')} | {'; '.join(d['notes'])} | {rd} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--data", default=None)
    ap.add_argument("--dlog", default=None)
    a = ap.parse_args(argv)
    run = Path(a.run)
    score = R.load_run(run)
    data = Path(a.data) if a.data else R.find_data(run, score)
    labels = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (data / "labels").glob("*.json")}
    traces = R.traces_for(run, score, Path(a.dlog) if a.dlog else None)
    diag = diagnose(score, labels, traces)
    text = report(run.name, diag)
    (run / "diagnosis.md").write_text(text, encoding="utf-8")
    (run / "diagnosis.json").write_text(json.dumps(diag, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
