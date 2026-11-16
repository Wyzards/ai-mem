"""The store, fed by the Claude Code and inbox entry points.

Job: keep every input exactly as it came, forever, with what the entry point
knows about it. A transcript (.jsonl) is read as it grows: each new line is
one input. Any other file is stored whole each time its bytes change.

Nothing here judges meaning. Classification only says what kind of thing an
input is and who it came from, so the queue knows what to perceive.
"""
import datetime
import json
import os
import re

from . import config, extract, log
from .db import now, tx, get_meta

BATCH_LINES = 2000


def _worker_project_slug():
    return re.sub(r"[^A-Za-z0-9]", "-", config.WORKER_DIR)


def discover():
    out = []
    data_abs = os.path.abspath(config.DATA).lower()
    skip_slug = _worker_project_slug().lower()
    for root, ep in config.SOURCE_DIRS:
        if not os.path.isdir(root):
            continue
        for dp, dn, fn in os.walk(root):
            a = os.path.abspath(dp).lower()
            if a.startswith(data_abs) or os.path.basename(a) == skip_slug:
                dn[:] = []
                continue
            for f in fn:
                out.append((os.path.join(dp, f), ep))
    for p, ep in config.SOURCE_FILES:
        if os.path.isfile(p):
            out.append((p, ep))
    return out


def _chunks(text, size):
    """Cut text into pieces of about `size` characters at line or word breaks."""
    i, n = 0, len(text)
    while i < n:
        j = min(n, i + size)
        if j < n:
            k = text.rfind("\n", i + size // 2, j)
            if k < 0:
                k = text.rfind(" ", i + size // 2, j)
            if k > i:
                j = k + 1
        yield i, j, text[i:j]
        i = j


PASSAGE_TYPES = extract.PERCEIVE_TYPES | {"tool_output_file"}
DEDUPE_MIN_CHARS = 300  # basis: shorter texts ("yes", "ok") mean different things in different contexts


def _perceive_status(conn, info, live_since, before_id=1 << 62):
    typ, text = info["type"], info.get("text")
    if typ not in extract.PERCEIVE_TYPES:
        return "skip", extract.SKIP_REASONS.get(typ, "type not perceived")
    if not text or not text.strip():
        return "skip", "no text"
    if info.get("uuid"):
        r = conn.execute("SELECT input_id FROM input_info WHERE uuid=? AND input_id < ? LIMIT 1",
                         (info["uuid"], before_id)).fetchone()
        if r:
            return "duplicate", "same message as input %d" % r[0]
    if len(text) >= DEDUPE_MIN_CHARS:
        r = conn.execute(
            "SELECT input_id FROM input_info WHERE text_hash=? AND input_id < ?"
            " AND perceive IN ('pending','history','done','failed') LIMIT 1",
            (info["text_hash"], before_id)).fetchone()
        if r:
            return "duplicate", "same text as input %d" % r[0]
    if not live_since or (info.get("happened_at") or "") < live_since:
        return "history", "arrived before memory went live; perceived only by backfill"
    return "pending", None


def _write_input(conn, source_id, offset, raw, info, live_since, ts):
    h = extract.sha(raw)
    cur = conn.execute(
        "INSERT OR IGNORE INTO inputs(source_id, offset, length, hash, raw, stored_at) VALUES (?,?,?,?,?,?)",
        (source_id, offset, len(raw), h, raw, ts))
    if cur.rowcount != 1:
        return None
    iid = cur.lastrowid
    text = info.get("text")
    info["text_hash"] = extract.sha(text) if text else None
    status, note = _perceive_status(conn, info, live_since)
    conn.execute(
        "INSERT INTO input_info(input_id, entry_point, session_id, context_id, type, said_by, happened_at, uuid, ref,"
        " fields, text, text_hash, perceive, perceive_note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (iid, info.get("entry_point"), info.get("session_id"), info.get("context_id"), info["type"],
         info.get("said_by"), info.get("happened_at"), info.get("uuid"), info.get("ref"),
         json.dumps(info.get("fields") or {}, ensure_ascii=False, default=str), text, info["text_hash"], status, note))
    if text and info["type"] in PASSAGE_TYPES:
        for s, e, piece in _chunks(text, config.PASSAGE_CHARS):
            if piece.strip():
                conn.execute("INSERT INTO passages(input_id, start, end, text) VALUES (?,?,?,?)", (iid, s, e, piece))
    return iid, status


def _session_update(conn, sid, o):
    if not sid:
        return
    ts = o.get("timestamp")
    t = o.get("type")
    title = None
    if t == "custom-title":
        title = o.get("customTitle")
    conn.execute("INSERT OR IGNORE INTO sessions(session_id, first_at) VALUES (?, ?)", (sid, ts))
    if title:
        conn.execute("UPDATE sessions SET title=? WHERE session_id=?", (title, sid))
    elif t == "ai-title" and o.get("aiTitle"):
        conn.execute("UPDATE sessions SET title=COALESCE(title, ?) WHERE session_id=?", (o.get("aiTitle"), sid))
    if o.get("cwd"):
        conn.execute("UPDATE sessions SET cwd=COALESCE(cwd, ?) WHERE session_id=?", (o.get("cwd"), sid))
    if ts:
        conn.execute("UPDATE sessions SET last_at=? WHERE session_id=? AND (last_at IS NULL OR last_at < ?)", (ts, sid, ts))
        conn.execute("UPDATE sessions SET first_at=? WHERE session_id=? AND (first_at IS NULL OR first_at > ?)", (ts, sid, ts))


def _path_ids(path):
    """Session id and subagent id from a transcript path."""
    norm = path.replace("\\", "/")
    stem = os.path.splitext(os.path.basename(norm))[0]
    if "/subagents/" in norm:
        sess = norm.split("/subagents/")[0].rsplit("/", 1)[-1]
        agent = stem[6:] if stem.startswith("agent-") else stem
        return sess, agent
    return stem, None


def _tail_hash(path, pos):
    with open(path, "rb") as f:
        a = max(0, pos - 4096)
        f.seek(a)
        return extract.sha(f.read(pos - a))


def sync_lines(conn, src, path, ep, live_since, stats):
    start = src["synced_bytes"]
    state = json.loads(src["state"] or "{}")
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if start > 0:
            rewritten = size < start or _tail_hash(path, start) != src["tail_hash"]
            if rewritten:
                log.event(conn, "source_rewritten", "file changed before the point already stored; re-reading it whole",
                          {"path": path, "stored_bytes": start, "size": size})
                start, state = 0, {}
                stats["rewritten"] = stats.get("rewritten", 0) + 1
        f.seek(start)
        data = f.read(size - start)
    end = data.rfind(b"\n")
    if end < 0:
        return
    data = data[:end + 1]
    sess_path, agent = _path_ids(path)
    state.setdefault("compactions", 0)
    tool_names = {}

    def tool_name_of(tid):
        if not tid:
            return None
        if tid in tool_names:
            return tool_names[tid]
        r = conn.execute("SELECT fields FROM input_info WHERE ref=? AND type='tool_call' LIMIT 1", (tid,)).fetchone()
        if r:
            try:
                return json.loads(r[0]).get("tool")
            except Exception:
                return None
        return None

    pos = start
    lines = data.split(b"\n")[:-1]
    for b0 in range(0, len(lines), BATCH_LINES):
        batch = lines[b0:b0 + BATCH_LINES]
        ts = now()
        with tx(conn):
            for raw in batch:
                off = pos
                pos += len(raw) + 1
                try:
                    o = json.loads(raw)
                    if not isinstance(o, dict):
                        raise ValueError("not an object")
                except Exception:
                    info = {"type": "unparsed", "text": None, "fields": {}, "entry_point": ep}
                    r = _write_input(conn, src["id"], off, raw, info, live_since, ts)
                    if r:
                        stats["inputs"] = stats.get("inputs", 0) + 1
                        stats["types"]["unparsed"] = stats["types"].get("unparsed", 0) + 1
                    continue
                sid = o.get("sessionId") or sess_path
                if o.get("type") == "system" and o.get("subtype") == "compact_boundary":
                    state["compactions"] += 1
                ctx = {"session_id": sid, "agent_id": agent, "sidechain": agent is not None}
                info = extract.classify_line(o, ctx, tool_name_of)
                if info["type"] == "tool_call" and info.get("ref"):
                    tool_names[info["ref"]] = info["fields"].get("tool")
                if info.get("happened_at"):
                    state["last_ts"] = info["happened_at"]
                else:
                    info["happened_at"] = state.get("last_ts")
                info["entry_point"] = ep
                info["session_id"] = sid
                info["context_id"] = "%s%s#%d" % (sid, "/" + agent if agent else "", state["compactions"])
                if o.get("type") in ("custom-title", "ai-title") or o.get("cwd"):
                    _session_update(conn, sid, o)
                r = _write_input(conn, src["id"], off, raw, info, live_since, ts)
                if r:
                    stats["inputs"] = stats.get("inputs", 0) + 1
                    stats["types"][info["type"]] = stats["types"].get(info["type"], 0) + 1
                    if r[1] == "pending":
                        stats["pending"] = stats.get("pending", 0) + 1
            # The tail hash moves with the stored position in the same transaction, so another
            # process never sees a new position with an old tail and mistakes it for a rewrite.
            conn.execute(
                "UPDATE sources SET synced_bytes=?, tail_hash=?, state=?, last_synced=? WHERE id=?",
                (pos, _tail_hash(path, pos), json.dumps(state), ts, src["id"]))
    st = os.stat(path)
    conn.execute("UPDATE sources SET tail_hash=?, last_mtime=?, last_size=? WHERE id=?",
                 (_tail_hash(path, pos), st.st_mtime, st.st_size, src["id"]))


def sync_file(conn, src, path, ep, live_since, stats):
    st = os.stat(path)
    if st.st_size > config.MAX_FILE_BYTES:
        log.event(conn, "file_too_large", "not stored: larger than the limit", {"path": path, "bytes": st.st_size}, ok=False)
        conn.execute("UPDATE sources SET last_mtime=?, last_size=? WHERE id=?", (st.st_mtime, st.st_size, src["id"]))
        return
    with open(path, "rb") as f:
        data = f.read()
    h = extract.sha(data)
    ts = now()
    with tx(conn):
        if h != src["last_hash"]:
            info = extract.classify_file(path, data, ep, config.PROJECTS, config.CLAUDE_DIR)
            info["entry_point"] = ep
            info["happened_at"] = datetime.datetime.fromtimestamp(
                st.st_mtime, datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            info["context_id"] = "file:%d" % src["id"]   # a file's versions form their own context
            info["session_id"] = None
            r = _write_input(conn, src["id"], -1, data, info, live_since, ts)
            if r:
                stats["inputs"] = stats.get("inputs", 0) + 1
                stats["types"][info["type"]] = stats["types"].get(info["type"], 0) + 1
                if r[1] == "pending":
                    stats["pending"] = stats.get("pending", 0) + 1
        conn.execute("UPDATE sources SET last_hash=?, last_mtime=?, last_size=?, last_synced=? WHERE id=?",
                     (h, st.st_mtime, st.st_size, ts, src["id"]))


def sync(conn, budget_s=None, paths=None):
    """Store everything new. Returns counts. budget_s: stop starting new files after this many seconds."""
    t = log.timer()
    live_since = get_meta(conn, "live_since")
    stats = {"files_seen": 0, "files_changed": 0, "inputs": 0, "types": {}}
    files = discover() if paths is None else [(p, ep) for p, ep in paths]
    known = {r["path"]: r for r in conn.execute("SELECT * FROM sources")}
    for path, ep in files:
        if budget_s is not None and t.s() > budget_s:
            stats["stopped_early"] = True
            break
        stats["files_seen"] += 1
        try:
            st = os.stat(path)
        except OSError:
            continue
        kind = "lines" if path.lower().endswith(".jsonl") else "file"
        src = known.get(path)
        if src is None:
            conn.execute("INSERT OR IGNORE INTO sources(path, entry_point, kind, first_seen) VALUES (?,?,?,?)",
                         (path, ep, kind, now()))
            src = conn.execute("SELECT * FROM sources WHERE path=?", (path,)).fetchone()
        if kind == "lines":
            if st.st_size == src["synced_bytes"] and src["last_mtime"] == st.st_mtime:
                continue
        elif src["last_size"] == st.st_size and src["last_mtime"] == st.st_mtime:
            continue
        stats["files_changed"] += 1
        try:
            if kind == "lines":
                sync_lines(conn, src, path, ep, live_since, stats)
            else:
                sync_file(conn, src, path, ep, live_since, stats)
        except Exception as e:
            log.event(conn, "sync_error", "could not store from %s" % path, {"error": repr(e)}, ok=False)
    stats["ms"] = t.ms()
    if stats["inputs"]:
        log.event(conn, "sync", "stored %d new inputs from %d changed files" % (stats["inputs"], stats["files_changed"]),
                  stats, ms=stats["ms"])
    return stats


def rederive(conn):
    """Recompute what the entry point says about every stored input, from the raw bytes.

    The raw store is the source of truth; input_info is derived from it. This
    replays the reading in arrival order (so compactions and tool names come out
    the same) and rewrites each input's type, speaker, context and text. An input
    already perceived keeps its place in the queue."""
    t = log.timer()
    live_since = get_meta(conn, "live_since")
    srcs = {r["id"]: r for r in conn.execute("SELECT * FROM sources")}
    states, tool_names = {}, {}
    changed, n = {}, 0

    def tool_name_of(tid):
        return tool_names.get(tid)

    last = 0
    while True:
        with tx(conn):
            rows = conn.execute("SELECT id, source_id, raw FROM inputs WHERE id > ? ORDER BY id LIMIT 1000",
                                (last,)).fetchall()
            if rows:
                last = rows[-1][0]
            for iid, sid, raw in rows:
                n += 1
                src = srcs[sid]
                old = conn.execute("SELECT * FROM input_info WHERE input_id=?", (iid,)).fetchone()
                if src["kind"] == "lines":
                    st = states.setdefault(sid, {"compactions": 0})
                    sess_path, agent = _path_ids(src["path"])
                    try:
                        o = json.loads(raw)
                        if not isinstance(o, dict):
                            raise ValueError
                    except Exception:
                        continue
                    sess = o.get("sessionId") or sess_path
                    if o.get("type") == "system" and o.get("subtype") == "compact_boundary":
                        st["compactions"] += 1
                    info = extract.classify_line(o, {"session_id": sess, "agent_id": agent, "sidechain": agent is not None},
                                                 tool_name_of)
                    if info["type"] == "tool_call" and info.get("ref"):
                        tool_names[info["ref"]] = info["fields"].get("tool")
                    if info.get("happened_at"):
                        st["last_ts"] = info["happened_at"]
                    else:
                        info["happened_at"] = st.get("last_ts")
                    info["session_id"] = sess
                    info["context_id"] = "%s%s#%d" % (sess, "/" + agent if agent else "", st["compactions"])
                else:
                    data = bytes(raw)
                    info = extract.classify_file(src["path"], data, src["entry_point"], config.PROJECTS, config.CLAUDE_DIR)
                    info["happened_at"] = old["happened_at"] if old else None
                    info["session_id"] = None
                    info["context_id"] = "file:%d" % sid
                info["entry_point"] = src["entry_point"]
                text = info.get("text")
                info["text_hash"] = extract.sha(text) if text else None
                if old is not None and old["perceive"] in ("done", "failed", "pending"):
                    status, note = old["perceive"], old["perceive_note"]
                else:
                    status, note = _perceive_status(conn, info, live_since, before_id=iid)
                if old is not None and (old["type"] != info["type"] or old["said_by"] != info.get("said_by")):
                    k = "%s -> %s" % (old["type"], info["type"])
                    changed[k] = changed.get(k, 0) + 1
                conn.execute(
                    "INSERT OR REPLACE INTO input_info(input_id, entry_point, session_id, context_id, type, said_by, happened_at,"
                    " uuid, ref, fields, text, text_hash, perceive, perceive_note, perceived_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (iid, info["entry_point"], info.get("session_id"), info.get("context_id"), info["type"], info.get("said_by"),
                     info.get("happened_at"), info.get("uuid"), info.get("ref"),
                     json.dumps(info.get("fields") or {}, ensure_ascii=False, default=str), text, info["text_hash"],
                     status, note, old["perceived_at"] if old is not None else None))
                if text and info["type"] in PASSAGE_TYPES and not conn.execute(
                        "SELECT 1 FROM passages WHERE input_id=? LIMIT 1", (iid,)).fetchone():
                    for s, e, piece in _chunks(text, config.PASSAGE_CHARS):
                        if piece.strip():
                            conn.execute("INSERT INTO passages(input_id, start, end, text) VALUES (?,?,?,?)",
                                         (iid, s, e, piece))
        if not rows:
            break
    res = {"inputs": n, "changed": changed, "ms": t.ms()}
    log.event(conn, "rederive", "recomputed the entry point's reading of %d stored inputs" % n, res, ms=res["ms"])
    return res
