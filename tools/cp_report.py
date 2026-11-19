"""Checkpoint-style report for a run (PLAN_V3 §3.2): Sarj's tables plus the uncertainty a small set hides.

    python -m tools.cp_report runs/v2.0-cp-mirror-rt
    python -m tools.cp_report runs/sim-v2.0-cp-mirror --vs runs/sim-v2.0-cp-mirror-old
    python -m tools.cp_report runs/v2.0-cp-mirror-rt --dlog runs/v2.0-cp-mirror-rt/decisions.jsonl
    python -m tools.cp_report runs/a --repeats runs/b,runs/c          # decision consistency (gate G11)

Prints and writes <run>/cp_report.md (+ cp_report.json):
  1. Sarj's headline: Core, R, S, interruptions caught, median/p90 stop latency, false stops, EOT.
  2. The by-category table, same columns as Sarj's T01 report (ttbench.report).
  3. Bootstrap 95 % CI of Core: whole calls resampled within each category (the hidden set has a fixed
     8-per-category design), 2000 times.
  4. "16-call draw": what a checkpoint-sized sample (2 calls per category) would show.
  5. 64-call projection: 8 calls per category drawn from this run's calls, with its interval.
  6. --vs: per-category change against a baseline run, and every event whose credit changed.
  7. Stress probes (labels with provenance.probe): pass rate per probe and variant.
  8. Every failure with the rule that fired: from trace.json (tools.sim runs) or a decision log (--dlog, real time).
All numbers come from the official ttbench.score functions.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from ttbench.score import aggregate  # noqa: E402
from tools import errors as E  # noqa: E402
from turn import decision_log  # noqa: E402

CAT_NAMES = {"AM": "Ambiguous", "BC": "Backchannel", "BG": "Background", "DV": "Dialect variation",
             "GI": "Genuine interruption", "HT": "Hesitation", "NV": "Caller non-speech / echo", "OV": "Overlap"}


def load_run(run: Path):
    score = json.loads((run / "score.json").read_text(encoding="utf-8"))
    return score


def find_data(run: Path, score: dict) -> Path:
    sid = score["events"][0]["scenario_id"]
    for labels in sorted((KIT / "data").glob("*/labels")):
        if (labels / f"{sid}.json").exists():
            return labels.parent
    raise SystemExit(f"No labels found for {run}. Pass --data.")


def scen_results(score: dict) -> dict[str, dict]:
    """Rebuild ttbench.score's per-scenario results from score.json (events + protected item)."""
    out = {s["scenario_id"]: {"scenario_id": s["scenario_id"], "events": [],
                              "protected": {"category": s["category"], "credit": s["credit"]}} for s in score["scenarios"]}
    for e in score["events"]:
        out[e["scenario_id"]]["events"].append(e)
    return out


def core_of(results: list[dict]) -> float:
    return aggregate(results, {})["core"]["core_score"]


def by_category(results: dict[str, dict], labels: dict) -> dict[str, list[str]]:
    cats = defaultdict(list)
    for sid in results:
        cats[labels[sid]["category"]].append(sid)
    return cats


def draws(results, cats, per_cat: int, n: int, replace: bool, rng) -> np.ndarray:
    vals = []
    for _ in range(n):
        pick = []
        for c, ids in cats.items():
            k = per_cat if per_cat else len(ids)
            rep = replace or len(ids) < k
            pick += [results[ids[j]] for j in rng.choice(len(ids), size=k, replace=rep)]
        vals.append(core_of(pick))
    return np.array(vals)


def fmt(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def headline(score: dict) -> list[str]:
    c, lat, r = score["core"], score["latency_ms"], score["rates"]
    caught = sum(v["stop_yielded"] for v in score["categories"].values())
    n_stop = sum(v["n_stop"] for v in score["categories"].values())
    return ["| Core | R | S | F | A | Interruptions caught | Stop latency median / p90 | False stops | Hiccups | Misses | EOT |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
            f"| **{c['core_score']:.1f}** | {c['R']:.2f} | {c['S']:.2f} | {c['F']:.2f} | {fmt(c['A'])} | {caught} of {n_stop} | "
            f"{fmt(lat['median'], 0)} / {fmt(lat['p90'], 0)} ms | {r['false_stops']} | {r['hiccups']} | {r['misses']} | "
            f"{score['eot']['eot_score']:.1f} |"]


def category_table(score: dict) -> list[str]:
    rows = ["| Category | R | S | genuine stops caught | backchannels/noise kept | scenarios with no stray stop |",
            "|---|---|---|---|---|---|"]
    for c, v in sorted(score["categories"].items()):
        rows.append(f"| {c} {CAT_NAMES.get(c, '')} | {fmt(v['R'])} | {fmt(v['S'])} | {v['stop_yielded']}/{v['n_stop']} | "
                    f"{v['keep_passed']}/{v['n_keep']} | {v['protected_clean']}/{v['n_scenarios']} |")
    return rows


def probe_table(score: dict, labels: dict) -> list[str]:
    rows = defaultdict(list)
    for e in score["events"]:
        lb = labels[e["scenario_id"]]
        probe = lb.get("provenance", {}).get("probe")
        if not probe:
            continue
        rows[(probe, e["expected"])].append(e)
    if not rows:
        return []
    out = ["| Probe | Expected | Events | Passed (credit 1.0) | Mean credit | Misses | False stops | Failures |", "|---|---|---|---|---|---|---|---|"]
    for (probe, exp), evs in sorted(rows.items()):
        fails = [e["scenario_id"] for e in evs if e["credit"] < 0.99]
        out.append(f"| {probe} | {exp} | {len(evs)} | {sum(e['credit'] >= 0.99 for e in evs)} | "
                   f"{statistics.fmean(e['credit'] for e in evs):.2f} | {sum(e.get('outcome') == 'miss' for e in evs)} | "
                   f"{sum(e.get('outcome') == 'false_stop' for e in evs)} | {', '.join(fails) or '—'} |")
    return out


def traces_for(run: Path, score: dict, dlog: Path | None) -> dict[str, list[dict]]:
    tj = run / "trace.json"
    if tj.exists() and not dlog:
        return json.loads(tj.read_text(encoding="utf-8"))
    if dlog is None and (run / "decisions.jsonl").exists():
        dlog = run / "decisions.jsonl"
    if dlog is None:
        return {}
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    order = [s["scenario_id"] for s in sorted(meta["scenarios"], key=lambda s: s["start_ms"])]
    by_call = decision_log.traces_by_call(decision_log.read(str(dlog)))
    calls = sorted(by_call)
    # the k-th call_start of the session is the k-th scenario of the stream
    return {sid: by_call[c] for sid, c in zip(order, calls[-len(order):])} if calls else {}


def failure_rows(score: dict, labels: dict, traces: dict) -> list[str]:
    errs = E.classify(score, labels, traces)
    if not errs:
        return ["No failures."]
    out = ["| Type | Event | Cat | Probe/subtype | Dialect | Noise | Latency | t (ms) | Heard | Rule | sem_p |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in sorted(errs, key=lambda e: (e["type"], e["call"])):
        lb = labels.get(e["call"], {})
        sub = lb.get("subtype", "")
        out.append(f"| {E.TYPES[e['type']]} | {e['event']} | {e['category']} | {sub} | {e['dialect']} | {e['noise']} | "
                   f"{fmt(e['latency_ms'], 0)} | {e['t']} | {str(e['text']).replace('|', ' ')} | "
                   f"{str(e['rule']).replace('|', ' ')} | {e['sem_p']} |")
    return out


def diff_rows(score: dict, base: dict) -> list[str]:
    out = ["| Category | R before → after | S before → after |", "|---|---|---|"]
    for c in sorted(set(score["categories"]) | set(base["categories"])):
        a, b = base["categories"].get(c, {}), score["categories"].get(c, {})
        out.append(f"| {c} | {fmt(a.get('R'))} → {fmt(b.get('R'))} | {fmt(a.get('S'))} → {fmt(b.get('S'))} |")
    old = {e["event_id"]: e for e in base["events"]}
    changed = [(e, old[e["event_id"]]) for e in score["events"] if e["event_id"] in old
               and abs(e["credit"] - old[e["event_id"]]["credit"]) > 0.01]
    out += ["", f"Core {base['core']['core_score']:.1f} → {score['core']['core_score']:.1f}. Events whose credit changed: {len(changed)}"]
    out += [f"- {e['event_id']} ({e['expected']}): {o['credit']:.2f} {o.get('outcome')} → {e['credit']:.2f} {e.get('outcome')}"
            for e, o in changed]
    return out


def consistency(score: dict, others: list[dict]) -> tuple[float, list[str]]:
    def key(e):
        return (e.get("outcome"), round(e["credit"], 1))
    base = {e["event_id"]: key(e) for e in score["events"]}
    same, total, diffs = 0, 0, []
    for o in others:
        for e in o["events"]:
            if e["event_id"] in base:
                total += 1
                if key(e) == base[e["event_id"]]:
                    same += 1
                else:
                    diffs.append(f"{e['event_id']}: {base[e['event_id']]} vs {key(e)}")
    return (same / total if total else 1.0), diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--data", default=None)
    ap.add_argument("--vs", default=None, help="baseline run folder")
    ap.add_argument("--dlog", default=None, help="decision log of a real-time run")
    ap.add_argument("--repeats", default=None, help="comma list of repeated runs of the same set (gate G11)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    run = Path(a.run)
    score = load_run(run)
    data = Path(a.data) if a.data else find_data(run, score)
    labels = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (data / "labels").glob("*.json")}
    res = scen_results(score)
    cats = by_category(res, labels)
    rng = np.random.default_rng(a.seed)
    boot = draws(res, cats, 0, a.boot, True, rng)
    d16 = draws(res, cats, 2, a.boot, False, rng)
    p64 = draws(res, cats, 8, a.boot, True, rng)
    q = lambda x, p: float(np.percentile(x, p))  # noqa: E731
    summary = {"run": str(run), "data": str(data), "system_version": score.get("system_version"), "speed": score.get("speed"),
               "core": score["core"], "latency_ms": score["latency_ms"], "rates": score["rates"], "eot": score["eot"]["eot_score"],
               "categories": score["categories"], "n_calls": len(res),
               "ci95": [q(boot, 2.5), q(boot, 97.5)], "draw16": {"mean": float(d16.mean()), "p5": q(d16, 5), "p50": q(d16, 50),
                                                                 "p95": q(d16, 95), "p_below_92": float((d16 < 92).mean())},
               "proj64": {"median": q(p64, 50), "lo": q(p64, 2.5), "hi": q(p64, 97.5), "p_ge_95": float((p64 >= 95).mean())},
               "rtt_ms": score.get("rtt_ms")}
    lines = [f"# Checkpoint-style report: {run.name}", "",
             f"Data `{data.relative_to(KIT) if data.is_relative_to(KIT) else data}` · {len(res)} calls · system {score.get('system_version')} · "
             f"{'real time' if score.get('official_speed') else 'speed ' + str(score.get('speed'))}"
             + (f" · network RTT median {score['rtt_ms'].get('median')} ms, max {score['rtt_ms'].get('max')} ms" if score.get("rtt_ms") else ""),
             "", "## 1. Headline (Sarj's table)", "", *headline(score),
             "", "## 2. By category (same columns as T01)", "", *category_table(score),
             "", "## 3–5. How sure is this number?", "",
             "| | Core |", "|---|---|",
             f"| This run | **{score['core']['core_score']:.1f}** |",
             f"| Bootstrap 95 % CI (calls resampled within category, {a.boot}×) | {q(boot, 2.5):.1f} – {q(boot, 97.5):.1f} |",
             f"| A 16-call checkpoint (2 per category): median [5th–95th pct] | {q(d16, 50):.1f} [{q(d16, 5):.1f} – {q(d16, 95):.1f}], P(< 92) = {(d16 < 92).mean():.0%} |",
             f"| 64-call final projection (8 per category): median [95 %] | {q(p64, 50):.1f} [{q(p64, 2.5):.1f} – {q(p64, 97.5):.1f}], P(≥ 95) = {(p64 >= 95).mean():.0%} |"]
    pt = probe_table(score, labels)
    if pt:
        lines += ["", "## 7. Stress probes", "", *pt]
    if a.vs:
        base = load_run(Path(a.vs))
        lines += ["", f"## 6. Against {Path(a.vs).name}", "", *diff_rows(score, base)]
    if a.repeats:
        others = [load_run(Path(p)) for p in a.repeats.split(",")]
        share, diffs = consistency(score, others)
        summary["consistency"] = share
        lines += ["", "## Consistency over repeated runs (G11)", "", f"Same outcome on **{share:.1%}** of events.",
                  *[f"- {d}" for d in diffs[:30]]]
    traces = traces_for(run, score, Path(a.dlog) if a.dlog else None)
    lines += ["", "## 8. Failures, with the rule that fired", "",
              *(failure_rows(score, labels, traces) if traces else ["(no trace or decision log for this run)"])]
    text = "\n".join(lines) + "\n"
    (run / "cp_report.md").write_text(text, encoding="utf-8")
    (run / "cp_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if not a.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
