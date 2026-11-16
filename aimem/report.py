"""Commands for looking into memory: status, search, source, raw, transcript, why, log, states, check,
plus work, backfill and retry. Everything prints plainly so it can be read or pasted."""
import json
import os

from . import config, extract
from .perceive import local_time


def _j(s, default=None):
    try:
        return json.loads(s) if s else default
    except Exception:
        return default


def cmd_status(conn, pos, opts, out):
    from . import worker
    from .db import get_meta
    out("ai-mem status")
    out("  live since: %s" % (get_meta(conn, "live_since") or "not live (nothing new is queued)"))
    n, b = conn.execute("SELECT COUNT(*), COALESCE(SUM(length),0) FROM inputs").fetchone()
    out("  raw store: %d inputs, %.1f MB" % (n, b / 1e6))
    rows = conn.execute("SELECT perceive, COUNT(*) FROM input_info GROUP BY perceive").fetchall()
    out("  inputs by status: " + ", ".join("%s %d" % (r[0], r[1]) for r in rows))
    live = conn.execute("SELECT COUNT(*) FROM input_info WHERE perceive='pending' AND COALESCE(perceive_note,'') != 'backfill'").fetchone()[0]
    back = conn.execute("SELECT COUNT(*) FROM input_info WHERE perceive='pending' AND perceive_note = 'backfill'").fetchone()[0]
    sq = conn.execute("SELECT COUNT(*) FROM state_queue WHERE tries < 2").fetchone()[0]
    out("  queue: %d live, %d history (backfill); %d perceptions waiting for the state step" % (live, back, sq))
    mode, spent = worker.budget_mode(conn)
    out("  budget: perceiving %s; spent $%.2f of $%.0f (24 h), $%.2f of $%.0f (5 h), $%.2f (7 days)" % (
        worker.MODE_TEXT[mode], spent["spend_24h"], spent["budget_24h"], spent["spend_5h"], spent["budget_5h"],
        worker.spend(conn, 24 * 7)))
    rows = conn.execute("SELECT kind, COALESCE(status,''), COUNT(*) FROM nodes GROUP BY kind, status").fetchall()
    out("  memories: " + (", ".join("%s%s %d" % (r[0], " (" + r[1] + ")" if r[1] else "", r[2]) for r in rows) or "none yet"))
    rows = conn.execute("SELECT kind, COUNT(*) FROM links GROUP BY kind").fetchall()
    out("  links: " + (", ".join("%s %d" % (r[0], r[1]) for r in rows) or "none yet"))
    out("  worker: %s" % ("running" if worker.running() else "not running"))
    mu = get_meta(conn, "model_unavailable")
    out("  model: %s" % ("UNAVAILABLE since " + mu if mu else "ok (last calls succeeded or none made)"))
    r = conn.execute("SELECT COUNT(*), SUM(ok), SUM(ms), MAX(at) FROM model_calls").fetchone()
    out("  model calls: %d (%d ok), %.0f s total, last %s" % (r[0] or 0, r[1] or 0, (r[2] or 0) / 1000, r[3] or "-"))
    r = conn.execute("SELECT COUNT(*), MAX(at) FROM injections").fetchone()
    out("  injections: %d, last %s" % (r[0], r[1] or "-"))
    errs = conn.execute("SELECT at, kind, summary FROM events WHERE ok=0 ORDER BY id DESC LIMIT 5").fetchall()
    if errs:
        out("  last errors:")
        for e in errs:
            out("    %s %s: %s" % (local_time(e[0]), e[1], e[2]))
    return 0


def cmd_work(conn, pos, opts, out):
    from . import worker
    st = worker.work(conn, limit=int(opts["limit"]) if opts.get("limit") else None, quiet=bool(opts.get("quiet")),
                     idle_wait_s=0 if opts.get("once") else 20, say=None if opts.get("quiet") else out)
    if not opts.get("quiet"):
        out(json.dumps(st, indent=1))
    return 0


def cmd_backfill(conn, pos, opts, out):
    """Queue stored history for perceiving. Options: --types a,b --since DATE --until DATE --session ID
    --project TEXT (part of the session's folder) --limit N --dry"""
    where = ["perceive='history'"]
    args = []
    if opts.get("types"):
        ts = opts["types"].split(",")
        where.append("type IN (%s)" % ",".join("?" * len(ts)))
        args += ts
    if opts.get("since"):
        where.append("happened_at >= ?")
        args.append(opts["since"])
    if opts.get("until"):
        where.append("happened_at < ?")
        args.append(opts["until"])
    if opts.get("session"):
        where.append("session_id = ?")
        args.append(opts["session"])
    if opts.get("project"):
        where.append("session_id IN (SELECT session_id FROM sessions WHERE cwd LIKE ?)")
        args.append("%" + opts["project"] + "%")
    if opts.get("main"):
        where.append("context_id NOT LIKE '%/%'")
    q = "SELECT input_id, LENGTH(text) FROM input_info WHERE " + " AND ".join(where) + " ORDER BY input_id"
    if opts.get("limit"):
        q += " LIMIT %d" % int(opts["limit"])
    rows = conn.execute(q, args).fetchall()
    chars = sum(r[1] or 0 for r in rows)
    out("%d inputs, %.1f k characters (about %.0f k tokens)" % (len(rows), chars / 1000, chars / 4000))
    if opts.get("dry"):
        return 0
    from .db import tx
    from . import log
    with tx(conn):
        conn.executemany("UPDATE input_info SET perceive='pending', perceive_note='backfill' WHERE input_id=?",
                         [(r[0],) for r in rows])
    log.event(conn, "backfill", "queued %d stored inputs for perceiving" % len(rows), {"options": opts, "chars": chars})
    out("queued")
    return 0


def cmd_retry(conn, pos, opts, out):
    from .db import tx
    with tx(conn):
        n = conn.execute("UPDATE input_info SET perceive='pending' WHERE perceive='failed'").rowcount
    out("requeued %d failed inputs" % n)
    return 0


def _parse_id(s):
    s = s.strip().lower().strip("[]")
    if s[:1] in "sprix" and s[1:].isdigit():
        return s[0], int(s[1:])
    if s.isdigit():
        return "p", int(s)
    raise SystemExit("ids look like s12 (state), p34 (perception), r56 (raw passage), i78 (input)")


def _input_line(conn, iid):
    r = conn.execute("SELECT i.*, s.path FROM input_info i JOIN inputs x ON x.id=i.input_id JOIN sources s ON s.id=x.source_id"
                     " WHERE i.input_id=?", (iid,)).fetchone()
    if not r:
        return "input i%d not found" % iid, None
    t = conn.execute("SELECT title FROM sessions WHERE session_id=?", (r["session_id"],)).fetchone() if r["session_id"] else None
    return ('input i%d: %s %s by %s, session "%s" (%s), context %s\n  stored from %s' % (
        iid, local_time(r["happened_at"]), r["type"], r["said_by"], t[0] if t and t[0] else "-", r["session_id"] or "-",
        r["context_id"], r["path"])), r


def cmd_source(conn, pos, opts, out):
    if not pos:
        out("usage: source s12|p34|r56|i78 [--raw]")
        return 1
    k, n = _parse_id(pos[0])
    if k == "s":
        r = conn.execute("SELECT * FROM nodes WHERE id=? AND kind='state'", (n,)).fetchone()
        if not r:
            out("no state s%d" % n)
            return 1
        out("state s%d: %s [%s]" % (n, r["subject"], r["status"]))
        out("now: " + r["text"])
        for v in conn.execute("SELECT * FROM state_versions WHERE state_id=? ORDER BY n", (n,)):
            out("\nversion %d (%s, %s, as of %s, on the word of %s, confidence %s%s): %s" % (
                v["n"], v["op"], v["status"], local_time(v["as_of"]), v["on_whose_word"], v["confidence"],
                ", memory's own conclusion" if v["own_conclusion"] else "", v["text"]))
            out("  why: %s" % v["reason"])
            for pid in _j(v["rests_on"], []):
                p = conn.execute("SELECT * FROM nodes WHERE id=?", (pid,)).fetchone()
                if p:
                    line, _ = _input_line(conn, p["input_id"]) if p["input_id"] else ("", None)
                    out("  rests on p%d: %s" % (pid, p["text"]))
                    out("    quote: \"%s\"" % (p["quote"] or ""))
                    out("    " + line.replace("\n", "\n    "))
        return 0
    if k in ("p", "x"):
        p = conn.execute("SELECT * FROM nodes WHERE id=?", (n,)).fetchone()
        if not p:
            out("no memory p%d" % n)
            return 1
        if p["kind"] == "state":
            return cmd_source(conn, ["s%d" % n], opts, out)
        out("%s p%d: %s" % (p["kind"], n, p["text"]))
        out("said by %s at %s; quote: \"%s\"" % (p["said_by"], local_time(p["said_at"]), p["quote"] or ""))
        line, r = _input_line(conn, p["input_id"])
        out(line)
        for l in conn.execute('SELECT l.kind, l."end", n.kind k2, n.text FROM links l JOIN nodes n ON n.id=l."end" WHERE l.start=?', (n,)):
            out("  link %s -> %s%d: %s" % (l[0], "s" if l[2] == "state" else "p", l[1], l[3][:160]))
        if r is not None:
            out("\n--- the input it came from%s ---" % ("" if opts.get("raw") else " (first 3000 characters; --raw for all)"))
            txt = r["text"] or ""
            out(txt if opts.get("raw") else txt[:3000])
        return 0
    if k == "r":
        r = conn.execute("SELECT * FROM passages WHERE id=?", (n,)).fetchone()
        if not r:
            out("no passage r%d" % n)
            return 1
        line, ir = _input_line(conn, r["input_id"])
        out("raw passage r%d, characters %d-%d of input i%d" % (n, r["start"], r["end"], r["input_id"]))
        out(line)
        out("\n" + ((ir["text"] or "") if opts.get("raw") and ir is not None else r["text"]))
        return 0
    if k == "i":
        line, r = _input_line(conn, n)
        out(line)
        if r is not None:
            out("\n" + (r["text"] or "(no readable text; raw bytes are stored)"))
            if opts.get("raw"):
                raw = conn.execute("SELECT raw FROM inputs WHERE id=?", (n,)).fetchone()[0]
                out("\n--- raw bytes as stored ---")
                out(bytes(raw).decode("utf-8", "replace"))
        return 0
    return 1


def cmd_raw(conn, pos, opts, out):
    if not pos:
        out("usage: raw i78|p34|r56")
        return 1
    k, n = _parse_id(pos[0])
    if k == "p":
        r = conn.execute("SELECT input_id FROM nodes WHERE id=?", (n,)).fetchone()
        n = r[0] if r else None
    elif k == "r":
        r = conn.execute("SELECT input_id FROM passages WHERE id=?", (n,)).fetchone()
        n = r[0] if r else None
    if not n:
        out("not found")
        return 1
    return cmd_source(conn, ["i%d" % n], {"raw": True}, out)


def cmd_transcript(conn, pos, opts, out):
    """A whole session, readable, rebuilt from the raw store. Arg: session id, or any id inside it."""
    if not pos:
        out("usage: transcript SESSION_ID | i78 | p34 [--all]")
        return 1
    sid = pos[0]
    if sid[:1] in "ipr" and sid[1:].isdigit():
        k, n = _parse_id(sid)
        if k == "p":
            n = conn.execute("SELECT input_id FROM nodes WHERE id=?", (n,)).fetchone()[0]
        elif k == "r":
            n = conn.execute("SELECT input_id FROM passages WHERE id=?", (n,)).fetchone()[0]
        sid = conn.execute("SELECT session_id FROM input_info WHERE input_id=?", (n,)).fetchone()[0]
    t = conn.execute("SELECT title, cwd FROM sessions WHERE session_id=?", (sid,)).fetchone()
    out('session %s "%s" in %s' % (sid, t[0] if t else "-", t[1] if t else "-"))
    types = None if opts.get("all") else ("ryan_message", "claude_text", "command", "interrupt", "tool_call")
    for r in conn.execute("SELECT * FROM input_info WHERE session_id=? AND context_id NOT LIKE '%/%' ORDER BY input_id", (sid,)):
        if types and r["type"] not in types:
            continue
        txt = r["text"] or ""
        if r["type"] == "tool_call" and not opts.get("all"):
            txt = txt[:200]
        out("\n[i%d] %s %s (%s):\n%s" % (r["input_id"], local_time(r["happened_at"]), r["said_by"], r["type"], txt))
    return 0


def cmd_search(conn, pos, opts, out):
    from . import recall
    text = " ".join(pos)
    kinds = (opts.get("kinds") or "perception,situation,state").split(",")
    cues = [{"label": "query", "text": text, "kinds": kinds}]
    pool, passages, trace = recall.recall(conn, cues, None, with_passages=not opts.get("nopassages"))
    _print_trace(conn, trace, out, pool, passages)
    return 0


def _node_text(conn, nid):
    r = conn.execute("SELECT kind, text FROM nodes WHERE id=?", (nid,)).fetchone()
    return ("s" if r and r[0] == "state" else "p") + str(nid), (r[1] if r else "?")


def _print_trace(conn, trace, out, pool=None, passages=None, injected=None):
    out("vectors searched: %s" % trace.get("vectors"))
    for c in trace.get("cues", []):
        out("\ncue [%s] over %s: %s" % (c["label"], ",".join(c["kinds"]), c["text"][:300]))
        for x in c.get("considered", [])[:25]:
            lab, txt = _node_text(conn, x["id"])
            hd = " heading %s" % x["subject_cos"] if x.get("subject_cos") is not None else ""
            out("  %-7s cos %-6s bm25 %-7s%s %s | %s" % (lab, x.get("cos"), x.get("bm25", "-"), hd, x.get("decision", ""), txt[:110]))
    if trace.get("crossings"):
        out("\nchaining:")
        for x in trace["crossings"][:30]:
            out("  %s -%s-> %s passed %.2f" % (x["from"], x["kind"], x["to"], x["passed"]))
    if trace.get("dropped"):
        out("\ndropped after recall:")
        for d in trace["dropped"]:
            out("  %s: %s" % (d["id"], d["why"]))
    for p in trace.get("passages", []):
        out("\nraw passages for cue [%s]:" % p["cue"])
        for x in p["considered"][:12]:
            out("  r%-8s bm25 %-7s rel %-5s words %s %s" % (x["passage"], x["bm25"], x["bm25_rel"], x["shared_terms"], x["decision"]))
    if trace.get("cut"):
        out("\ncut by the length cap: %s" % ", ".join(str(c["id"]) for c in trace["cut"]))
    if pool is not None:
        out("\nresult:")
        for nid, e in sorted(pool.items(), key=lambda kv: -kv[1]["strength"]):
            lab, txt = _node_text(conn, nid)
            out("  %s strength %.2f via %s: %s" % (lab, e["strength"], e["via"], txt[:200]))
        for p in passages or []:
            out("  r%d (raw, %s): %s" % (p["passage"], p.get("said_by"), " ".join((p.get("text") or "").split())[:200]))


def cmd_why(conn, pos, opts, out):
    """The full record of one injection: what was asked, every candidate and why it was kept or dropped."""
    if not pos or pos[0] == "last":
        r = conn.execute("SELECT * FROM injections ORDER BY id DESC LIMIT 1").fetchone()
    elif pos[0] == "list":
        for r in conn.execute("SELECT id, at, hook, context_id, substr(cue_text,1,80) c, json_array_length(node_ids) n FROM injections ORDER BY id DESC LIMIT 20"):
            out("%5d %s %-16s %s memories: %s" % (r["id"], local_time(r["at"]), r["hook"], r["n"], " ".join((r["c"] or "").split())))
        return 0
    else:
        r = conn.execute("SELECT * FROM injections WHERE id=?", (int(pos[0]),)).fetchone()
    if not r:
        out("no injection found")
        return 1
    out("injection %d at %s, %s, context %s, %d ms" % (r["id"], local_time(r["at"]), r["hook"], r["context_id"], r["ms"] or 0))
    out("cue: " + " ".join((r["cue_text"] or "").split())[:500])
    _print_trace(conn, _j(r["trace"], {}), out)
    out("\n--- what was injected ---\n" + (r["text"] or "(nothing)"))
    return 0


def cmd_log(conn, pos, opts, out):
    n = int(opts.get("n", 30))
    q = "SELECT * FROM events"
    args = []
    if opts.get("kind"):
        q += " WHERE kind=?"
        args.append(opts["kind"])
    q += " ORDER BY id DESC LIMIT ?"
    args.append(n)
    for e in reversed(conn.execute(q, args).fetchall()):
        out("%s %-18s %s%s %s" % (local_time(e["at"]), e["kind"], "" if e["ok"] else "FAILED ", e["summary"],
                                  "(%d ms)" % e["ms"] if e["ms"] else ""))
        if opts.get("detail") and e["detail"]:
            out("    " + e["detail"][:3000])
    return 0


def cmd_calls(conn, pos, opts, out):
    """Model calls: list, or show one whole (prompt and answer)."""
    if pos:
        r = conn.execute("SELECT * FROM model_calls WHERE id=?", (int(pos[0]),)).fetchone()
        if not r:
            out("no call")
            return 1
        out("call %d %s %s %s ms ok=%s error=%s usage=%s" % (r["id"], r["step"], r["model"], r["ms"], r["ok"], r["error"], r["usage"]))
        out("\n--- system ---\n" + (r["system"] or ""))
        out("\n--- prompt ---\n" + (r["prompt"] or ""))
        o = _j(r["response"], {})
        out("\n--- answer ---\n" + json.dumps(o.get("structured_output", o), indent=1, ensure_ascii=False)[:20000])
        return 0
    for r in conn.execute("SELECT id, at, step, model, ms, ok, error, usage FROM model_calls ORDER BY id DESC LIMIT ?",
                          (int(opts.get("n", 20)),)):
        u = _j(r["usage"], {})
        out("%5d %s %-8s %-7s %6s ms ok=%s in=%s out=%s %s" % (r["id"], local_time(r["at"]), r["step"], r["model"], r["ms"], r["ok"],
                                                              u.get("input_tokens"), u.get("output_tokens"), r["error"] or ""))
    return 0


def cmd_states(conn, pos, opts, out):
    q = "SELECT * FROM nodes WHERE kind='state'" + ("" if opts.get("all") else " AND status='holds'") + " ORDER BY id"
    for r in conn.execute(q):
        nv = conn.execute("SELECT COUNT(*) FROM state_versions WHERE state_id=?", (r["id"],)).fetchone()[0]
        out("\ns%d [%s] %s (%d versions, as of %s, on the word of %s)" % (r["id"], r["status"], r["subject"], nv,
                                                                       (r["said_at"] or "?")[:10], r["said_by"]))
        out("  " + r["text"].replace("\n", "\n  "))
    return 0


def cmd_check(conn, pos, opts, out):
    """Integrity: raw hashes, add-only rules, every memory traces to an input, every version rests on memories."""
    from .extract import sha
    bad = 0
    n = 0
    sample = conn.execute("SELECT id, hash, raw FROM inputs ORDER BY RANDOM() LIMIT ?", (int(opts.get("sample", 2000)),))
    for r in sample:
        n += 1
        if sha(bytes(r[2])) != r[1]:
            bad += 1
            out("hash mismatch on input i%d" % r[0])
    out("raw hashes: %d checked, %d bad" % (n, bad))
    trig = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    for t in ("inputs_add_only_u", "inputs_add_only_d", "nodes_no_delete", "nodes_perceptions_fixed", "sv_add_only_u",
              "sv_add_only_d", "links_no_delete"):
        out("rule %-24s %s" % (t, "in force" if t in trig else "MISSING"))
    x = conn.execute("SELECT COUNT(*) FROM nodes WHERE kind!='state' AND (input_id IS NULL OR input_id NOT IN (SELECT id FROM inputs))").fetchone()[0]
    out("perceptions without an input: %d" % x)
    y = 0
    for v in conn.execute("SELECT state_id, n, rests_on FROM state_versions"):
        ids = _j(v[2], [])
        if not ids or conn.execute("SELECT COUNT(*) FROM nodes WHERE id IN (%s)" % ",".join("?" * len(ids)), ids).fetchone()[0] != len(ids):
            y += 1
    out("state versions resting on missing memories: %d" % y)
    z = conn.execute('SELECT COUNT(*) FROM links WHERE start NOT IN (SELECT id FROM nodes) OR "end" NOT IN (SELECT id FROM nodes)').fetchone()[0]
    out("links to missing memories: %d" % z)
    return 0


def cmd_rederive(conn, pos, opts, out):
    from . import store
    out(json.dumps(store.rederive(conn), indent=1))
    return 0


COMMANDS = {"status": cmd_status, "work": cmd_work, "backfill": cmd_backfill, "retry": cmd_retry, "source": cmd_source,
            "raw": cmd_raw, "transcript": cmd_transcript, "search": cmd_search, "why": cmd_why, "log": cmd_log,
            "calls": cmd_calls, "states": cmd_states, "check": cmd_check, "rederive": cmd_rederive}


def run(conn, cmd, pos, opts, out):
    f = COMMANDS.get(cmd)
    if not f:
        out("unknown command %s; run: python C:/ai-mem/mem.py help" % cmd)
        return 1
    return f(conn, pos, opts, out) or 0
