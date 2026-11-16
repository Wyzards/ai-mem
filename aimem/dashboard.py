"""Read-only dashboard: every input, its source, the memories it became, every injection
with its search results, states with their versions, and what memory's model calls cost.

Run: python C:/ai-mem/mem.py dashboard   (then open http://localhost:8765)
It opens the database read-only, so it can never change memory.
"""
import json
import sqlite3
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config

PORT = 8765


def ro():
    c = sqlite3.connect("file:%s?mode=ro" % config.DB_PATH.replace("\\", "/"), uri=True, timeout=5,
                        check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def rows(c, q, args=()):
    return [dict(r) for r in c.execute(q, args).fetchall()]


def memory_cost_by_session(c, session_id=None):
    """Memory's model-call spend attributed to sessions: perceive calls by the inputs they read,
    state calls by the context their event names."""
    out = {}
    for r in c.execute("SELECT id, step, usage, input_ids FROM model_calls WHERE ok=1"):
        cost = (json.loads(r["usage"] or "{}").get("cost_usd") or 0)
        sess = None
        ids = json.loads(r["input_ids"] or "[]")
        if ids:
            x = c.execute("SELECT session_id FROM input_info WHERE input_id=?", (ids[0],)).fetchone()
            sess = x[0] if x else None
        out.setdefault(sess, {"perceive": 0.0, "state": 0.0})
        if r["step"] == "perceive":
            out[sess]["perceive"] += cost
    for e in c.execute("SELECT context_id, detail FROM events WHERE kind='state' AND ok=1"):
        d = json.loads(e["detail"] or "{}")
        call = d.get("model_call")
        if not call:
            continue
        u = c.execute("SELECT usage FROM model_calls WHERE id=?", (call,)).fetchone()
        cost = (json.loads(u[0] or "{}").get("cost_usd") or 0) if u else 0
        sess = (e["context_id"] or "").split("#")[0].split("/")[0] or None
        out.setdefault(sess, {"perceive": 0.0, "state": 0.0})["state"] += cost
    return out if session_id is None else out.get(session_id, {"perceive": 0.0, "state": 0.0})


def api(path, qs):
    c = ro()
    try:
        g = lambda k, d=None: qs.get(k, [d])[0]
        if path == "/api/overview":
            st = rows(c, "SELECT perceive, COUNT(*) n FROM input_info GROUP BY perceive")
            kinds = rows(c, "SELECT kind, COALESCE(status,'') status, COUNT(*) n FROM nodes GROUP BY kind, status")
            spend = rows(c, "SELECT step, COUNT(*) calls, ROUND(SUM(json_extract(usage,'$.cost_usd')),2) usd FROM model_calls GROUP BY step")
            errs = rows(c, "SELECT at, kind, summary FROM events WHERE ok=0 ORDER BY id DESC LIMIT 10")
            sessions = rows(c, "SELECT s.session_id, s.title, s.cwd, s.last_at, (SELECT COUNT(*) FROM input_info i WHERE i.session_id=s.session_id) inputs FROM sessions s ORDER BY last_at DESC LIMIT 60")
            costs = memory_cost_by_session(c)
            for s in sessions:
                m = costs.get(s["session_id"], {"perceive": 0, "state": 0})
                s["memory_usd"] = round(m["perceive"] + m["state"], 2)
            return {"status": st, "memories": kinds, "spend": spend, "errors": errs, "sessions": sessions}
        if path == "/api/inputs":
            where, args = ["1=1"], []
            if g("session"):
                where.append("session_id=?"); args.append(g("session"))
            if g("type"):
                where.append("type=?"); args.append(g("type"))
            if g("status"):
                where.append("perceive=?"); args.append(g("status"))
            if g("q"):
                where.append("text LIKE ?"); args.append("%" + g("q") + "%")
            if not g("all"):
                where.append("perceive IN ('pending','done','failed','history')")
            q = ("SELECT input_id, happened_at, type, said_by, context_id, perceive, LENGTH(text) chars, SUBSTR(text,1,200) preview,"
                 " (SELECT COUNT(*) FROM nodes n WHERE n.input_id=input_info.input_id) memories FROM input_info WHERE "
                 + " AND ".join(where) + " ORDER BY input_id DESC LIMIT ? OFFSET ?")
            return rows(c, q, args + [int(g("limit", 100)), int(g("offset", 0))])
        if path == "/api/input":
            i = int(g("id"))
            info = rows(c, "SELECT i.*, s.path source_path, s.entry_point source_entry FROM input_info i JOIN inputs x ON x.id=i.input_id"
                           " JOIN sources s ON s.id=x.source_id WHERE i.input_id=?", (i,))
            nodes = rows(c, "SELECT * FROM nodes WHERE input_id=? ORDER BY id", (i,))
            for n in nodes:
                n["links"] = rows(c, 'SELECT l.kind, l."end" target, l.energy, n2.kind target_kind, SUBSTR(n2.text,1,160) target_text'
                                     ' FROM links l JOIN nodes n2 ON n2.id=l."end" WHERE l.start=?', (n["id"],))
            calls = rows(c, "SELECT id, at, step, model, ms, ok, error, usage FROM model_calls WHERE input_ids LIKE ?",
                         ("%" + str(i) + "%",))
            calls = [x for x in calls if i in json.loads(c.execute("SELECT input_ids FROM model_calls WHERE id=?", (x["id"],)).fetchone()[0] or "[]")]
            raw = c.execute("SELECT raw FROM inputs WHERE id=?", (i,)).fetchone()
            return {"info": info[0] if info else None, "memories": nodes, "model_calls": calls,
                    "raw": bytes(raw[0]).decode("utf-8", "replace")[:200000] if raw else None}
        if path == "/api/injections":
            return rows(c, "SELECT id, at, hook, context_id, ms, json_array_length(node_ids) memories, json_array_length(passage_ids) excerpts,"
                           " SUBSTR(cue_text,1,160) cue FROM injections ORDER BY id DESC LIMIT ?", (int(g("limit", 100)),))
        if path == "/api/injection":
            r = rows(c, "SELECT * FROM injections WHERE id=?", (int(g("id")),))
            if not r:
                return None
            r = r[0]
            r["trace"] = json.loads(r["trace"] or "{}")
            ids = set()
            for cue in r["trace"].get("cues", []):
                ids.update(x["id"] for x in cue.get("considered", []))
            for x in r["trace"].get("crossings", []):
                ids.update(v for v in (x.get("from"), x.get("to")) if v)
            texts = {}
            for nid in ids:
                n = c.execute("SELECT kind, SUBSTR(text,1,220) t FROM nodes WHERE id=?", (nid,)).fetchone()
                if n:
                    texts[nid] = {"kind": n[0], "text": n[1]}
            r["texts"] = texts
            return r
        if path == "/api/memory":
            i = int(g("id"))
            n = rows(c, "SELECT * FROM nodes WHERE id=?", (i,))
            if not n:
                return None
            n = n[0]
            n["versions"] = rows(c, "SELECT * FROM state_versions WHERE state_id=? ORDER BY n", (i,))
            n["links_out"] = rows(c, 'SELECT l.kind, l."end" id, n2.kind k, SUBSTR(n2.text,1,200) t FROM links l JOIN nodes n2 ON n2.id=l."end" WHERE l.start=?', (i,))
            n["links_in"] = rows(c, 'SELECT l.kind, l.start id, n2.kind k, SUBSTR(n2.text,1,200) t FROM links l JOIN nodes n2 ON n2.id=l.start WHERE l."end"=?', (i,))
            n["injected"] = rows(c, "SELECT id, at, hook, context_id FROM injections WHERE node_ids LIKE ? ORDER BY id DESC LIMIT 50",
                                 ("%" + str(i) + "%",))
            return n
        if path == "/api/states":
            return rows(c, "SELECT id, subject, status, said_by, said_at, text, json_extract(extra,'$.kind') kind,"
                           " (SELECT COUNT(*) FROM state_versions v WHERE v.state_id=nodes.id) versions FROM nodes WHERE kind='state' ORDER BY said_at DESC")
        if path == "/api/search":
            from . import recall, db  # recall reads only
            conn = ro()
            pool, passages, trace = recall.recall(conn, [{"label": "query", "text": g("q", ""),
                                                          "kinds": ["perception", "situation", "state"]}], None)
            res = []
            for nid, e in sorted(pool.items(), key=lambda kv: -kv[1]["strength"]):
                n = conn.execute("SELECT kind, text FROM nodes WHERE id=?", (nid,)).fetchone()
                res.append({"id": nid, "via": e["via"], "strength": e["strength"], "kind": n[0], "text": n[1]})
            return {"results": res, "passages": passages, "trace": trace}
        if path == "/api/log":
            return rows(c, "SELECT id, at, kind, ok, ms, summary FROM events ORDER BY id DESC LIMIT ?", (int(g("limit", 200)),))
        if path == "/api/calls":
            return rows(c, "SELECT id, at, step, model, ms, ok, error, json_extract(usage,'$.cost_usd') usd,"
                           " json_extract(usage,'$.output_tokens') out_tokens FROM model_calls ORDER BY id DESC LIMIT 200")
        if path == "/api/call":
            r = rows(c, "SELECT * FROM model_calls WHERE id=?", (int(g("id")),))
            return r[0] if r else None
        if path == "/api/system":
            return system(c)
        if path == "/api/graph":
            return graph(c, g("around"), int(g("n", 160)))
        if path == "/api/timeline":
            return rows(c, "SELECT SUBSTR(happened_at,1,10) day, SUM(type IN (%s)) said, SUM(type IN ('tool_call','tool_result')) tool,"
                           " SUM(type NOT IN (%s) AND type NOT IN ('tool_call','tool_result')) other FROM input_info"
                           " WHERE happened_at >= date('now','-75 days') GROUP BY day ORDER BY day" % (SAID_SQL, SAID_SQL))
        return None
    finally:
        c.close()


from .extract import SAID_TYPES  # noqa: E402
SAID_SQL = ",".join("'%s'" % x for x in sorted(SAID_TYPES))


def system(c):
    from . import worker
    one = lambda q, a=(): c.execute(q, a).fetchone()[0]
    mode, spent = worker.budget_mode(c)
    q = {}
    for when, cond in (("live", "COALESCE(perceive_note,'') != 'backfill'"), ("history", "perceive_note = 'backfill'")):
        for kind, kc in (("said", "type IN (%s)" % SAID_SQL), ("tool", "type NOT IN (%s)" % SAID_SQL)):
            q[when + "_" + kind] = one("SELECT COUNT(*) FROM input_info WHERE perceive='pending' AND %s AND %s" % (cond, kc))
    since = worker._ago(0.25)
    act = {r[0]: r[1] for r in c.execute("SELECT kind, COUNT(*) FROM events WHERE at >= ? GROUP BY kind", (since,))}
    stored_recent = sum(json.loads(r[0] or "{}").get("inputs", 0) for r in
                        c.execute("SELECT detail FROM events WHERE kind='sync' AND at >= ?", (since,)))
    hours = rows(c, "SELECT SUBSTR(at,1,13) h, SUM(kind='perceive') perceive, SUM(kind='state') state, SUM(kind='recall') recall"
                    " FROM events WHERE at >= ? GROUP BY h ORDER BY h", (worker._ago(24),))
    spend_h = {r[0]: r[1] for r in c.execute("SELECT SUBSTR(at,1,13), SUM(json_extract(usage,'$.cost_usd')) FROM model_calls"
                                               " WHERE at >= ? GROUP BY 1", (worker._ago(24),))}
    for h in hours:
        h["usd"] = round(spend_h.get(h["h"]) or 0, 2)
    last = rows(c, "SELECT at, kind, ok, summary FROM events WHERE kind IN ('perceive','state','recall','sync','worker_start','worker_stop',"
                   "'budget','recall_skipped','perceive_failed','state_failed','hook_error','model_unavailable') ORDER BY id DESC LIMIT 14")
    rec = rows(c, "SELECT id, at, hook, ms, json_array_length(node_ids) n, SUBSTR(cue_text,1,90) cue FROM injections ORDER BY id DESC LIMIT 12")
    return {
        "raw_inputs": one("SELECT COUNT(*) FROM inputs"), "raw_mb": round(__import__("os").path.getsize(config.DB_PATH) / 1e6),
        "sources": one("SELECT COUNT(*) FROM sources"), "sessions": one("SELECT COUNT(*) FROM sessions"),
        "perceived": one("SELECT COUNT(*) FROM input_info WHERE perceive='done'"),
        "skipped": one("SELECT COUNT(*) FROM input_info WHERE perceive IN ('skip','duplicate')"),
        "history": one("SELECT COUNT(*) FROM input_info WHERE perceive='history'"),
        "queue": q, "state_queue": one("SELECT COUNT(*) FROM state_queue WHERE tries < 2"),
        "perceptions": one("SELECT COUNT(*) FROM nodes WHERE kind='perception'"),
        "situations": one("SELECT COUNT(*) FROM nodes WHERE kind='situation'"),
        "states_holds": one("SELECT COUNT(*) FROM nodes WHERE kind='state' AND status='holds'"),
        "states_held": one("SELECT COUNT(*) FROM nodes WHERE kind='state' AND status='held'"),
        "links": one("SELECT COUNT(*) FROM links"), "recalls": one("SELECT COUNT(*) FROM injections"),
        "recall_ms": one("SELECT AVG(ms) FROM (SELECT ms FROM injections ORDER BY id DESC LIMIT 20)"),
        "worker": worker.running(), "mode": mode, "spend": spent,
        "activity": {"stored": stored_recent, **act}, "hours": hours, "last": last, "recent_recalls": rec,
    }


def graph(c, around=None, n=160):
    if around:
        ids = {int(around)}
        frontier = set(ids)
        for _ in range(2):
            nxt = set()
            for i in frontier:
                for r in c.execute('SELECT "end" FROM links WHERE start=? UNION SELECT start FROM links WHERE "end"=?', (i, i)):
                    nxt.add(r[0])
            nxt -= ids
            ids |= set(list(nxt)[:40])
            frontier = nxt
    else:
        ids = {r[0] for r in c.execute("SELECT id FROM nodes ORDER BY id DESC LIMIT ?", (n,))}
        ids |= {r[0] for r in c.execute("SELECT id FROM nodes WHERE kind='state' AND status IN ('holds','held')")}
    ids = list(ids)
    ph = ",".join("?" * len(ids))
    nodes = rows(c, "SELECT id, kind, status, said_by, said_at, SUBSTR(text,1,240) text, subject, input_id FROM nodes WHERE id IN (%s)" % ph, ids)
    links = rows(c, 'SELECT start s, "end" t, kind FROM links WHERE start IN (%s) AND "end" IN (%s) AND start < "end"' % (ph, ph), ids + ids)
    return {"nodes": nodes, "links": links}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path.startswith("/api/"):
            try:
                body = json.dumps(api(u.path, urllib.parse.parse_qs(u.query)), default=str).encode("utf-8")
                code = 200
            except Exception as e:
                body, code = json.dumps({"error": repr(e)}).encode("utf-8"), 500
            ctype = "application/json"
        else:
            with open(__file__.replace("dashboard.py", "dashboard.html"), "rb") as f:
                body = f.read()
            code, ctype = 200, "text/html; charset=utf-8"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(port=PORT):
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print("ai-mem dashboard on http://localhost:%d" % port, flush=True)
    srv.serve_forever()


def statusline():
    """Claude Code status line: memory's spend attributed to this session, and its share of the session's cost."""
    import sys
    try:
        h = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace") or "{}")
    except Exception:
        h = {}
    sid = h.get("session_id")
    sess_usd = float(((h.get("cost") or {}).get("total_cost_usd")) or 0)
    try:
        c = ro()
        m = memory_cost_by_session(c, sid)
        mem = m["perceive"] + m["state"]
        c.close()
    except Exception:
        print("mem: n/a")
        return
    total = sess_usd + mem
    share = (100.0 * mem / total) if total > 0 else 0.0
    print("mem $%.2f (perceive $%.2f, state $%.2f) = %.0f%% of this session" % (mem, m["perceive"], m["state"], share))
