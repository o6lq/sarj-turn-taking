"""Decision log without audio (PLAN_V3 §3.1, weakness W6). Off unless TT_DECISION_LOG is set.

    TT_DECISION_LOG=runs/x/decisions.jsonl python my_system.py     # timings, rules and features only
    TT_DECISION_LOG=... TT_LOG_TEXT=1 python my_system.py           # + recognised text (our own data only)

One JSON line per event the policy traces: a candidate opening, each recogniser read with the rule that fired,
STOP, RESUME, TAKE_TURN, agent start/end. Never audio. Text (recogniser output and the agent's script) is written
only when TT_LOG_TEXT=1, because official runs may not allow it.

Lines go through a queue to a background thread, so on_frame never waits for the disk.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time

# feature keys copied from a recogniser read (policy._features) into the log line
FEATURE_KEYS = ("sem_p", "spk_sim", "rel_db", "n", "bid", "content", "echoed", "bc", "filler", "repeated_ambig",
                "ambig_alone", "question", "only_bc", "agent_q_prev", "since_agent_start_s", "first_word_bid",
                "voiced_s", "cand_age_s", "logprob", "echo_ratio")
TEXT_KEYS = ("text", "agent_text")


def _round(v):
    if isinstance(v, float):
        return round(v, 3)
    return v


class DecisionLog:
    def __init__(self, path: str, with_text: bool = False):
        self.path = path
        self.with_text = with_text
        self.calls = 0                       # call counter for the whole process (one server session)
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")
        self._t = threading.Thread(target=self._run, name="decision-log", daemon=True)
        self._t.start()
        atexit.register(self.close)
        self.event({"kind": "log_open", "pid": os.getpid(), "with_text": with_text})

    def next_call(self) -> int:
        self.calls += 1
        return self.calls

    def event(self, rec: dict) -> None:
        """Queue one line (cheap: no JSON or disk work on the caller's thread)."""
        rec = {k: v for k, v in rec.items() if self.with_text or k not in TEXT_KEYS}
        rec["wall"] = time.time()
        self._q.put(rec)

    def record(self, call: int, t_ms: float, kind: str, kw: dict, state: dict) -> None:
        """A policy trace event -> one flat log line. `kw` is what policy._log received."""
        rec = {"call": call, "t": t_ms, "kind": kind, **state}
        for k in ("why", "p", "ms", "cand_start", "level_rel", "waited"):
            if k in kw:
                rec[k] = kw[k]
        if "text" in kw:
            rec["text"] = kw["text"]
        f = kw.get("f")
        if f:
            rec.update({k: _round(f[k]) for k in FEATURE_KEYS if k in f})
        self.event(rec)

    def _run(self):
        while True:
            rec = self._q.get()
            if rec is None:
                break
            batch = [rec]
            while True:                       # write everything already queued in one go
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._write(batch)
                    return
                batch.append(nxt)
            self._write(batch)

    def _write(self, batch):
        try:
            self._f.write("".join(json.dumps(r, ensure_ascii=False, default=_round) + "\n" for r in batch))
            self._f.flush()
        except (OSError, ValueError):
            pass

    def close(self):
        if self._t.is_alive():
            self._q.put(None)
            self._t.join(timeout=2)
        try:
            self._f.close()
        except OSError:
            pass


_LOG: DecisionLog | None = None
_LOCK = threading.Lock()


def get() -> DecisionLog | None:
    """The process-wide log, or None when TT_DECISION_LOG is not set."""
    global _LOG
    path = os.environ.get("TT_DECISION_LOG", "").strip()
    if not path:
        return None
    with _LOCK:
        if _LOG is None or _LOG.path != path:
            _LOG = DecisionLog(path, os.environ.get("TT_LOG_TEXT") == "1")
        return _LOG


def read(path: str) -> list[dict]:
    """All lines of a log file (for tools/cp_report.py)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def traces_by_call(lines: list[dict]) -> dict[int, list[dict]]:
    """Group log lines per call into the trace format tools/errors.py reads ({t, kind, why, text, f: {sem_p}})."""
    out: dict[int, list[dict]] = {}
    for r in lines:
        c = r.get("call")
        if not c:
            continue
        tr = {"t": r.get("t"), "kind": r.get("kind"), "why": r.get("why", ""), "text": r.get("text", ""),
              "f": {k: r[k] for k in FEATURE_KEYS if k in r}}
        for k in ("p", "ms", "agent_active", "stopped", "agent_utt", "cand_start", "level_rel", "cand"):
            if k in r:
                tr[k] = r[k]
        out.setdefault(c, []).append(tr)
    return out
