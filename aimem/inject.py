"""Inject: put what recall found in front of the model, and record it.

Order: states that say how Ryan wants things, then other states that hold
now, then perceptions (strongest first), then raw excerpts of history not
yet perceived. Each line carries an id the model can look up with
`python C:/ai-mem/mem.py source <id>`. Text stops at the length cap; what the
cap cut is logged.
"""
import json

from . import config, extract
from .db import now
from .perceive import local_time

HEADER = (extract.MEMORY_MARKER + " Memory recalled for this moment. Each line starts with its id."
          " To see where a memory came from, run: python C:/ai-mem/mem.py source <id>"
          " (add --raw for the original text).")


def _session_title(conn, sid, cache={}):
    if not sid:
        return None
    if sid not in cache:
        r = conn.execute("SELECT title FROM sessions WHERE session_id=?", (sid,)).fetchone()
        cache[sid] = r[0] if r else None
    return cache[sid]


def format_pool(conn, pool, passages):
    ids = list(pool)
    rows = {}
    if ids:
        for r in conn.execute("SELECT * FROM nodes WHERE id IN (%s)" % ",".join("?" * len(ids)), ids):
            rows[r["id"]] = r
    prefs, states, percs = [], [], []
    for nid, e in sorted(pool.items(), key=lambda kv: -kv[1]["strength"]):
        r = rows.get(nid)
        if r is None:
            continue
        if r["kind"] == "state":
            kind = json.loads(r["extra"] or "{}").get("kind")
            line = "[s%d] %s (as of %s; on the word of %s)" % (nid, r["text"], (r["said_at"] or "?")[:10], r["said_by"] or "?")
            (prefs if kind == "preference" else states).append((nid, line))
        else:
            title = _session_title(conn, r["session_id"])
            where = ' in "%s"' % title if title else ""
            tag = "situation" if r["kind"] == "situation" else (r["said_by"] or "?")
            percs.append((nid, "[p%d] %s, %s%s: %s" % (nid, local_time(r["said_at"]), tag, where, r["text"])))
    raw = []
    for p in passages:
        title = _session_title(conn, p.get("session"))
        where = ' in "%s"' % title if title else ""
        txt = " ".join((p.get("text") or "").split())
        if len(txt) > 450:
            txt = txt[:450] + "..."
        raw.append((p["passage"], '[r%d] %s, %s%s: "%s"' % (p["passage"], local_time(p.get("at")), p.get("said_by") or "?", where, txt)))
    sections = [("How Ryan wants things (holds now):", prefs, "node"), ("What holds now:", states, "node"),
                ("Memories:", percs, "node"), ("Raw excerpts from history not yet perceived:", raw, "passage")]
    out = [HEADER]
    used = len(HEADER)
    kept_nodes, kept_passages, cut = [], [], []
    for title, items, kind in sections:
        if not items:
            continue
        head_added = False
        for nid, line in items:
            if used + len(line) + len(title) + 4 > config.INJECT_MAX_CHARS:
                cut.append({"id": nid, "kind": kind, "why": "length cap %d characters" % config.INJECT_MAX_CHARS})
                continue
            if not head_added:
                out.append("")
                out.append(title)
                used += len(title) + 2
                head_added = True
            out.append(line)
            used += len(line) + 1
            (kept_nodes if kind == "node" else kept_passages).append(nid)
    if not kept_nodes and not kept_passages:
        return None, [], [], cut
    return "\n".join(out), kept_nodes, kept_passages, cut


SPOOL_FILE = "injections.jsonl"


def record(conn, hook, session_id, context_id, cwd, cue_text, text, node_ids, passage_ids, ms, trace):
    row = (now(), hook, session_id, context_id, cwd, (cue_text or "")[:4000], json.dumps(node_ids), json.dumps(passage_ids),
           text, ms, json.dumps(trace, default=str))
    try:
        cur = conn.execute(
            "INSERT INTO injections(at, hook, session_id, context_id, cwd, cue_text, node_ids, passage_ids, text, ms, trace)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)", row)
        return cur.lastrowid
    except Exception:
        # The database is busy: keep the record in the spool; the worker loads it within seconds.
        import os
        os.makedirs(config.SPOOL, exist_ok=True)
        with open(os.path.join(config.SPOOL, SPOOL_FILE), "a", encoding="utf-8") as f:
            f.write(json.dumps(list(row), ensure_ascii=False) + "\n")
        return "spooled"


def spooled_node_ids(context_id):
    """Node ids already injected into this context but still in the spool."""
    import os
    ids = set()
    p = os.path.join(config.SPOOL, SPOOL_FILE)
    if not context_id or not os.path.exists(p):
        return ids
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row[3] == context_id:
                    ids.update(json.loads(row[6] or "[]"))
    except OSError:
        pass
    return ids


def load_spool(conn):
    """Worker: move spooled injection records into the database."""
    import os
    p = os.path.join(config.SPOOL, SPOOL_FILE)
    if not os.path.exists(p):
        return 0
    tmp = p + ".loading"
    try:
        if not os.path.exists(tmp):
            os.replace(p, tmp)
    except OSError:
        return 0          # a hook is appending right now; next time
    n = 0
    with open(tmp, encoding="utf-8") as f:
        rows = []
        for line in f:
            try:
                rows.append(tuple(json.loads(line)))
            except Exception:
                continue
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.executemany(
            "INSERT INTO injections(at, hook, session_id, context_id, cwd, cue_text, node_ids, passage_ids, text, ms, trace)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.execute("COMMIT")
        n = len(rows)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    os.remove(tmp)
    return n
