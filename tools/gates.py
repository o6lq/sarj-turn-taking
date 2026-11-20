"""PLAN_V3 §4 gates in one command: run the system (tools.sim) on every set and compare with the v2.0 baseline.

    python -m tools.gates --name f1                          # all sets, gates vs runs/gates-v2.0.json
    python -m tools.gates --name f4b --set policy.CUE_MODE=2  # a variant, without editing code
    python -m tools.gates --baseline-from-runs               # (once) write runs/gates-v2.0.json from runs/sim-v2.0-*

Checks G1 novel-test ≥ base−0.3 · G2 dev-aug ≥ base−0.3 · G3 false stops summed ≤ base · G4 dev ≥ 99.0 and
0 misses · G5 median stop latency ≤ base+50 ms · G8 cp-mirror (3 renders combined) ≥ base and no category
drops > 0.05 · G9 per-probe table on cp-stress-test (printed; the targeted probe must improve, controls not drop) ·
G10 novel-test-2 ≥ 85.0. G6 (pytest, ttbench.ready), G7 (EXPERIMENTS entry) and G11 (3 real-time runs) are
separate. novel-test-2 is an alarm only: never read its sentences.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT))
from tools import cp_report as R  # noqa: E402
from tools.sim import simulate  # noqa: E402
from ttbench.score import aggregate  # noqa: E402

SETS = ["dev", "dev-aug", "novel-test", "synth-test", "novel-test-2", "cp-mirror", "cp-mirror-s32", "cp-mirror-s33",
        "cp-stress-dev", "cp-stress-test", "cp-ht-dev", "cp-ht-test"]
MIRRORS = ["cp-mirror", "cp-mirror-s32", "cp-mirror-s33"]


def summarize(set_name: str, score: dict) -> dict:
    return {"core": score["core"]["core_score"], "R": score["core"]["R"], "S": score["core"]["S"],
            "fs": score["rates"]["false_stops"], "miss": score["rates"]["misses"], "lat": score["latency_ms"]["median"] or 0.0,
            "cats": {c: {"R": v["R"], "S": v["S"]} for c, v in score["categories"].items()}}


def combined_mirror(run_of: dict) -> dict:
    res = []
    for s in MIRRORS:
        sc = json.loads((run_of[s] / "score.json").read_text(encoding="utf-8"))
        res += list(R.scen_results(sc).values())
    agg = aggregate(res, {})
    return {"core": agg["core"]["core_score"], "cats": {c: {"R": v["R"], "S": v["S"]} for c, v in agg["categories"].items()}}


def probes(run: Path, data: Path) -> dict:
    sc = json.loads((run / "score.json").read_text(encoding="utf-8"))
    labels = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (data / "labels").glob("*.json")}
    out: dict = {}
    for e in sc["events"]:
        pr = labels[e["scenario_id"]].get("provenance", {}).get("probe")
        if pr:
            k = f"{pr}-{e['expected']}"
            out.setdefault(k, []).append(e["credit"])
    return {k: [sum(v) / len(v), len(v)] for k, v in out.items()}


def run_all(name: str, overrides: dict, sets: list[str], reuse: set[str] | None = None) -> dict:
    out = {"name": name, "overrides": overrides, "sets": {}, "probes": {}}
    runs = {}
    for s in sets:
        t0 = time.time()
        d = KIT / "data" / s
        rd = KIT / "runs" / f"sim-{name}-{s}"
        if reuse is not None and s in reuse and (rd / "score.json").exists():
            sc = json.loads((rd / "score.json").read_text(encoding="utf-8"))      # kept from the last run of this name
        else:
            sc = simulate(str(KIT / "my_system.py"), d, overrides=overrides, out_dir=rd)
        runs[s] = rd
        out["sets"][s] = summarize(s, sc)
        x = out["sets"][s]
        print(f"  {s:15s} core {x['core']:5.1f}  R {x['R']:.2f}  S {x['S']:.2f}  FS {x['fs']:3d}  miss {x['miss']}  ({time.time() - t0:.0f}s)", flush=True)
        if s.startswith(("cp-stress", "cp-ht")):
            out["probes"][s] = probes(rd, d)
    if all(m in runs for m in MIRRORS):
        out["mirror3"] = combined_mirror(runs)
    return out


def gates(new: dict, base: dict) -> list[tuple[str, bool, str]]:
    S, B = new["sets"], base["sets"]
    g = []

    def have(*k):
        return all(x in S and x in B for x in k)
    if have("novel-test"):
        g.append(("G1 novel-test", S["novel-test"]["core"] >= B["novel-test"]["core"] - 0.3,
                  f"{B['novel-test']['core']:.2f} → {S['novel-test']['core']:.2f}"))
    if have("dev-aug"):
        g.append(("G2 dev-aug", S["dev-aug"]["core"] >= B["dev-aug"]["core"] - 0.3, f"{B['dev-aug']['core']:.2f} → {S['dev-aug']['core']:.2f}"))
    common = [s for s in S if s in B]
    fs_b, fs_n = sum(B[s]["fs"] for s in common), sum(S[s]["fs"] for s in common)
    g.append(("G3 false stops (sum)", fs_n <= fs_b, f"{fs_b} → {fs_n}"))
    if have("dev"):
        g.append(("G4 dev ≥ 99.0, 0 misses", S["dev"]["core"] >= 99.0 and S["dev"]["miss"] == 0,
                  f"{S['dev']['core']:.2f}, misses {S['dev']['miss']}"))
    worst = max((S[s]["lat"] - B[s]["lat"] for s in common), default=0)
    g.append(("G5 median latency ≤ +50 ms", worst <= 50, f"worst +{worst:.0f} ms"))
    if "mirror3" in new and "mirror3" in base:
        drops = [f"{c} {k} {base['mirror3']['cats'][c][k]:.2f}→{new['mirror3']['cats'][c][k]:.2f}"
                 for c in base["mirror3"]["cats"] for k in ("R", "S")
                 if base["mirror3"]["cats"][c][k] is not None and new["mirror3"]["cats"].get(c, {}).get(k) is not None
                 and new["mirror3"]["cats"][c][k] < base["mirror3"]["cats"][c][k] - 0.05]
        g.append(("G8 cp-mirror ×3", new["mirror3"]["core"] >= base["mirror3"]["core"] and not drops,
                  f"{base['mirror3']['core']:.2f} → {new['mirror3']['core']:.2f}" + (f"; drops: {', '.join(drops)}" if drops else "")))
    if have("novel-test-2"):
        g.append(("G10 novel-test-2 ≥ 85.0", S["novel-test-2"]["core"] >= 85.0, f"{S['novel-test-2']['core']:.2f}"))
    return g


def baseline_from_runs():
    out = {"name": "v2.0", "overrides": {}, "sets": {}, "probes": {}}
    runs = {}
    for s in SETS:
        rd = KIT / "runs" / f"sim-v2.0-{s}"
        sc = json.loads((rd / "score.json").read_text(encoding="utf-8"))
        out["sets"][s] = summarize(s, sc)
        runs[s] = rd
        if s.startswith(("cp-stress", "cp-ht")):
            out["probes"][s] = probes(rd, KIT / "data" / s)
    out["mirror3"] = combined_mirror(runs)
    (KIT / "runs" / "gates-v2.0.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote runs/gates-v2.0.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--set", action="append", default=[], help="NAME=VALUE override (module.NAME for turn modules)")
    ap.add_argument("--baseline", default=str(KIT / "runs" / "gates-v2.0.json"))
    ap.add_argument("--baseline-from-runs", action="store_true")
    ap.add_argument("--rerun", default=None, help="comma list: re-run only these sets, reuse the others from runs/sim-<name>-*")
    a = ap.parse_args(argv)
    if a.baseline_from_runs:
        baseline_from_runs()
        return 0
    base = json.loads(Path(a.baseline).read_text())
    sets = a.sets.split(",")
    reuse = None if not a.rerun else set(sets) - set(a.rerun.split(","))
    new = run_all(a.name, dict(s.split("=", 1) for s in a.set), sets, reuse)
    (KIT / "runs" / f"gates-{a.name}.json").write_text(json.dumps(new, indent=1, ensure_ascii=False))
    if "mirror3" in new:
        print(f"  cp-mirror ×3 combined: {base['mirror3']['core']:.2f} → {new['mirror3']['core']:.2f}   " +
              " ".join(f"{c}:{'' if v['R'] is None else 'R%.2f ' % v['R']}S{v['S']:.2f}" for c, v in sorted(new['mirror3']['cats'].items())))
    for s, pr in new["probes"].items():
        bp = base["probes"].get(s, {})
        print(f"  {s} probes: " + "  ".join(f"{k} {bp.get(k, [0])[0]:.2f}→{v[0]:.2f}" for k, v in sorted(pr.items())))
    ok = True
    for gname, passed, info in gates(new, base):
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {gname:28s} {info}")
    print("ALL GATES PASS" if ok else "SOME GATES FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
