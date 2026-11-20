"""Are two runs' decisions identical? Compares (decision, stream time) of every decision message, ignoring
label/confidence/transcript (metadata that never changes the score). The simulator stamps each decision with the
real compute time of that frame, so the same code differs by a fraction of a millisecond run to run: a decision
counts as the same when its type matches and its time is within TOL_MS. Run both sides in the same tree.

    python -m tools.compare_decisions runs/sim-a runs/sim-b      # prints IDENTICAL or DIFFERENT (+ the first diffs)
"""
import json
import sys
from pathlib import Path

TOL_MS = 5.0


def decisions(run: str) -> list[tuple[str, float]]:
    out = []
    for line in (Path(run) / "log.jsonl").read_text().splitlines():
        m = json.loads(line).get("msg", {})
        if m.get("type") == "decision":
            out.append((m["decision"], round(json.loads(line)["stream_ms"], 1)))
    return out


def main(a: str, b: str) -> int:
    da, db = decisions(a), decisions(b)
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(da, db)) if x[0] != y[0] or abs(x[1] - y[1]) > TOL_MS]
    if len(da) == len(db) and not diffs:
        print(f"IDENTICAL  {len(da)} decisions")
        return 0
    print(f"DIFFERENT  {len(da)} vs {len(db)} decisions, {len(diffs)} mismatches")
    for i, x, y in diffs[:5]:
        print(f"  #{i}: {x}  vs  {y}")
    return 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
