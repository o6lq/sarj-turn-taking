"""PLAN_V4 Task A.3 / gate G12: every STOP/RESUME the TurnTaker takes reaches the harness, in order, within 1 frame.

Runs 3 dev calls in real time through the real ttbench.server path (about 75 s, loads the models), so it is opt-in:
    TT_WIRE_TEST=1 .venv/bin/python -m pytest -q tests/test_wire_integrity.py
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[1]
CALLS = ["GI-D01", "BC-D02", "AM-D02"]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(os.environ.get("TT_WIRE_TEST") != "1", reason="real-time run; set TT_WIRE_TEST=1")
def test_trace_decisions_reach_the_wire(tmp_path):
    sys.path.insert(0, str(KIT))
    from ttbench import harness
    from turn import decision_log
    port, dlog = _free_port(), tmp_path / "decisions.jsonl"
    env = {**os.environ, "TT_DECISION_LOG": str(dlog), "PYTHONPATH": str(KIT)}
    proc = subprocess.Popen([sys.executable, str(KIT / "my_system.py"), "--port", str(port), "--quiet"], cwd=KIT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.5)
        out = tmp_path / "run"
        assert harness.main(["--data", str(KIT / "data" / "dev"), "--url", f"ws://127.0.0.1:{port}", "--out", str(out),
                             "--scenarios", ",".join(CALLS), "--no-score"]) == 0
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    meta = json.loads((out / "run.json").read_text())
    layout = sorted(meta["scenarios"], key=lambda s: s["start_ms"])
    wire = {s["scenario_id"]: [] for s in layout}
    for line in (out / "log.jsonl").read_text().splitlines():
        m = json.loads(line)
        d = m.get("msg", {}).get("decision")
        if m.get("dir") == "rx" and d in ("STOP", "RESUME"):
            owner = next(s for s in layout if s["start_ms"] <= m["stream_ms"] < s["start_ms"] + s["duration_ms"] + 1500)
            wire[owner["scenario_id"]].append((d, m["stream_ms"] - owner["start_ms"]))
    traces = decision_log.traces_by_call(decision_log.read(str(dlog)))
    calls = sorted(traces)[-len(layout):]
    for s, c in zip(layout, calls):
        mine = [(x["kind"], x["t"]) for x in traces[c] if x["kind"] in ("STOP", "RESUME")]
        sent = wire[s["scenario_id"]]
        assert [k for k, _ in mine] == [k for k, _ in sent], (s["scenario_id"], mine, sent)
        for (_, a), (_, b) in zip(mine, sent):
            assert abs(a - b) <= 25, (s["scenario_id"], a, b)       # 1 frame (+ a few ms of local delivery)
