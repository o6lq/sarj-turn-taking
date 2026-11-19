"""tools/cp_report rebuilds the official Core from score.json, and its resampling keeps the 8-category design."""
from pathlib import Path

import numpy as np

from tools import cp_report as R
from ttbench.score import load_labels, score

KIT = Path(__file__).resolve().parents[1]


def _score(decisions_at):
    labels, manifest = load_labels(KIT / "data" / "dev")
    layout, t, log = [], 0, []
    for sid, lb in labels.items():
        dur = lb["audio"]["duration_ms"]
        layout.append({"scenario_id": sid, "opaque_id": sid, "start_ms": t, "duration_ms": dur})
        for ms, d in decisions_at(lb):
            log.append({"dir": "rx", "stream_ms": t + ms, "msg": {"type": "decision", "decision": d}})
        t += dur + 1500
    meta = {"scenarios": layout, "hello": {}, "official": False, "speed": "test", "errors": []}
    return score(labels, meta, log, manifest.get("thresholds")), labels


def _stop_at_evident(lb):
    return [(e["evident_ms"] + 100, "STOP") for e in lb["events"] if e.get("expected") == "STOP"]


def test_core_matches_official():
    for policy in (lambda lb: [], _stop_at_evident):
        s, _ = _score(policy)
        assert abs(R.core_of(list(R.scen_results(s).values())) - s["core"]["core_score"]) < 1e-9


def test_draws_keep_every_category():
    s, labels = _score(_stop_at_evident)
    res = R.scen_results(s)
    cats = R.by_category(res, labels)
    assert len(cats) == 8
    d = R.draws(res, cats, 2, 50, False, np.random.default_rng(0))
    assert d.shape == (50,) and np.all((d >= 0) & (d <= 100))
