"""The decision log (W6): off by default, no text unless TT_LOG_TEXT=1, cheap for the caller's thread."""
import time

from turn import decision_log


def _features():
    return {"sem_p": 0.12345, "spk_sim": 0.8, "rel_db": -1.5, "n": 3, "bid": 0, "content": 1, "echoed": 2,
            "since_agent_start_s": 4.2, "not_logged": 7}


def test_off_without_env(monkeypatch):
    monkeypatch.delenv("TT_DECISION_LOG", raising=False)
    assert decision_log.get() is None


def test_no_text_by_default(tmp_path):
    path = tmp_path / "d.jsonl"
    log = decision_log.DecisionLog(str(path), with_text=False)
    log.record(1, 5200, "asr", {"text": "عبارة تجريبية", "why": "rule x", "p": 0.9, "f": _features()},
               {"agent_active": True, "stopped": False})
    log.event({"call": 1, "kind": "agent_start", "agent_text": "نص الوكيل"})
    log.close()
    lines = decision_log.read(str(path))
    asr = next(r for r in lines if r["kind"] == "asr")
    assert "text" not in asr and asr["why"] == "rule x" and asr["sem_p"] == 0.123 and "not_logged" not in asr
    assert asr["agent_active"] is True and asr["t"] == 5200
    assert all("agent_text" not in r for r in lines)


def test_text_when_allowed(tmp_path):
    path = tmp_path / "d.jsonl"
    log = decision_log.DecisionLog(str(path), with_text=True)
    log.record(2, 100, "STOP", {"text": "كلمة", "why": "floor-bid word"}, {})
    log.close()
    stop = next(r for r in decision_log.read(str(path)) if r["kind"] == "STOP")
    assert stop["text"] == "كلمة" and stop["call"] == 2


def test_cheap_on_the_callers_thread(tmp_path):
    log = decision_log.DecisionLog(str(tmp_path / "d.jsonl"))
    f = _features()
    t0 = time.perf_counter()
    for i in range(2000):
        log.record(1, i * 20, "asr", {"why": "w", "p": 0.5, "f": f}, {"agent_active": True})
    per_event_ms = (time.perf_counter() - t0) * 1000 / 2000
    log.close()
    assert per_event_ms < 0.2, per_event_ms


def test_traces_by_call_matches_errors_format(tmp_path):
    path = tmp_path / "d.jsonl"
    log = decision_log.DecisionLog(str(path), with_text=True)
    log.record(1, 300, "asr", {"text": "أ", "why": "r", "p": 0.1, "f": _features()}, {})
    log.record(2, 900, "STOP", {"text": "ب", "why": "s"}, {})
    log.close()
    tr = decision_log.traces_by_call(decision_log.read(str(path)))
    assert set(tr) == {1, 2}
    assert tr[1][0]["kind"] == "asr" and tr[1][0]["f"]["sem_p"] == 0.123
    assert tr[2][0]["why"] == "s" and tr[2][0]["t"] == 900
