"""System logs: every step records what it took in, what it decided and why.

Two copies of each event: a row in the events table (for queries) and a line
in data/logs/YYYY-MM-DD.jsonl (readable even if the database is locked or
broken). Model calls and injections have their own full records in the
database (model_calls, injections).
"""
import json
import os
import time

from . import config
from .db import now


def _file_line(rec):
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        day = rec["at"][:10]
        with open(os.path.join(config.LOG_DIR, day + ".jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def event(conn, kind, summary, detail=None, ok=True, ms=None, input_id=None, context_id=None):
    rec = {"at": now(), "kind": kind, "ok": bool(ok), "ms": ms, "input_id": input_id,
           "context_id": context_id, "summary": summary, "detail": detail, "pid": os.getpid()}
    _file_line(rec)
    if conn is None:
        return
    try:
        conn.execute(
            "INSERT INTO events(at, kind, ok, ms, input_id, context_id, summary, detail) VALUES (?,?,?,?,?,?,?,?)",
            (rec["at"], kind, 1 if ok else 0, ms, input_id, context_id, summary,
             json.dumps(detail, ensure_ascii=False, default=str) if detail is not None else None))
    except Exception as e:  # never let logging break the step it records
        _file_line({"at": now(), "kind": "log_error", "ok": False, "summary": str(e), "detail": {"kind": kind}})


class timer:
    def __init__(self):
        self.t = time.perf_counter()

    def ms(self):
        return int((time.perf_counter() - self.t) * 1000)

    def s(self):
        return time.perf_counter() - self.t
