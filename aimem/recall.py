"""Recall: find what is relevant to a cue. Code only; no model.

Two ways in, as settled: a search match, or a path over links.
- Search: every memory whose vector is close enough to the cue (cosine at or
  above the threshold), plus memories that share enough of the cue's words
  (keyword search, for names and rare terms the vectors blur).
- Chaining: from each match, strength spreads over links. Strength starts at
  the match's margin over the threshold, in units of the cue's own spread of
  scores. Each link costs its energy, and a memory with many links passes
  less along each one (fan). Paths add. A perception's link to its state
  costs nothing, so a perception never arrives without the state it bears on.

Every candidate, score and decision goes into the trace, so a missing memory
can be explained afterwards (`mem.py why`).
"""
import json
import math
import re

import numpy as np

from . import config, embed

STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been being it its this that
these those i you he she we they me him her us them my your his our their not no do does did done have has had will
would can could should may might must shall just so than then there here what which who whom whose when where why how
all any each few more most other some such only own same too very don now ok yes also into about out up get got like
one two make made want wants need needs think im it's i'm dont didn't doesn't that's thats u ur""".split())

WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'\-\.]*")


def terms(text, max_terms=24):
    seen = []
    for w in WORD.findall((text or "").lower()):
        w = w.strip("'-.")
        if len(w) < 2 or w in STOP or w in seen:
            continue
        seen.append(w)
    # keep the longest words when there are too many: longer words are usually rarer
    if len(seen) > max_terms:
        keep = set(sorted(seen, key=len, reverse=True)[:max_terms])
        seen = [w for w in seen if w in keep]
    return seen


def fts_query(ts):
    if not ts:
        return None
    return " OR ".join('"%s"' % w.replace('"', "") for w in ts)


def _shared_terms(ts, text):
    low = (text or "").lower()
    return sum(1 for w in ts if w[:6] in low)


_subjects = {}


def _subject_scores(conn, qvec, statuses):
    """Cosine between the cue and each state's heading (its subject)."""
    rows = conn.execute("SELECT id, subject FROM nodes WHERE kind='state' AND status IN (%s)"
                        % ",".join("?" * len(statuses)), statuses).fetchall()
    todo = [(r[0], r[1] or "") for r in rows if (r[0], r[1]) not in _subjects]
    if todo:
        vs = embed.encode([s for _, s in todo])
        for (i, s), v in zip(todo, vs):
            _subjects[(i, s)] = v
    return {r[0]: float(_subjects[(r[0], r[1])] @ qvec) for r in rows if r[1]}


def search_nodes(conn, idx, cue, qvec, kinds, statuses=("holds",), exclude_ctx=None, top=None, subjects=False):
    """One cue against the graph. Returns (matched, considered) lists of candidate dicts.
    subjects=True also compares the cue with each state's heading (for cues that describe a moment)."""
    cands = {}
    if subjects and "state" in kinds:
        for sid, sc in _subject_scores(conn, qvec, statuses).items():
            cands.setdefault(sid, {})["subject_cos"] = round(sc, 4)
    spread, mean = None, None
    if len(idx.ids):
        s = idx.scores(qvec)
        k = min(config.VEC_TOP, len(s))
        top_i = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
        vals = s[top_i]
        mean, spread = float(vals.mean()), max(float(vals.std()), 0.03)
        for i in top_i:
            cands[int(idx.ids[i])] = {"cos": round(float(s[i]), 4), "z": round((float(s[i]) - mean) / spread, 2)}
    ts = terms(cue)
    q = fts_query(ts)
    best = None
    if q:
        try:
            rows = conn.execute("SELECT rowid, bm25(nodes_fts) AS b FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY b LIMIT ?",
                                (q, config.BM25_TOP)).fetchall()
        except Exception:
            rows = []
        if rows:
            best = -rows[0][1]
            for r in rows:
                c = cands.setdefault(int(r[0]), {})
                c["bm25"] = round(-r[1], 3)
                c["bm25_rel"] = round((-r[1]) / best, 3) if best > 0 else 0
    if not cands:
        return [], []
    ids = list(cands)
    meta = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        for r in conn.execute("SELECT id, kind, status, context_id, text, said_by, said_at, subject FROM nodes WHERE id IN (%s)"
                              % ",".join("?" * len(chunk)), chunk):
            meta[r["id"]] = r
    considered, matched = [], []
    for nid, c in cands.items():
        r = meta.get(nid)
        if r is None:
            continue
        c.update(id=nid, kind=r["kind"])
        if "cos" not in c:
            v = idx.vector(nid)
            c["cos"] = round(float(v @ qvec), 4) if v is not None else None
            if c["cos"] is not None and spread:
                c["z"] = round((c["cos"] - mean) / spread, 2)
        if r["kind"] not in kinds:
            continue
        if r["kind"] == "state" and r["status"] not in statuses:
            c["decision"] = "dropped: state is %s" % r["status"]
            considered.append(c)
            continue
        if exclude_ctx and r["context_id"] == exclude_ctx:
            c["decision"] = "dropped: from this context (already in it)"
            considered.append(c)
            continue
        cos = c.get("cos")
        shared = _shared_terms(ts, r["text"])
        c["shared_terms"] = shared
        sc = c.get("subject_cos")
        z = c.get("z")
        if subjects == "also" and sc is not None and sc >= config.SUBJECT_THRESHOLD:
            c["margin"] = round((sc - config.SUBJECT_THRESHOLD) / 0.1 + 1.0, 3)
            c["decision"] = "match: heading %.3f >= %.2f" % (sc, config.SUBJECT_THRESHOLD)
            matched.append(c)
            considered.append(c)
            continue
        if subjects is True:
            # A moment cue is matched against state headings only: its words (Ryan, Claude,
            # chat, project) are shared by nearly every memory, so its text match means nothing.
            if sc is not None and sc >= config.SUBJECT_THRESHOLD:
                c["margin"] = round((sc - config.SUBJECT_THRESHOLD) / 0.1 + 1.0, 3)
                c["decision"] = "match: heading %.3f >= %.2f" % (sc, config.SUBJECT_THRESHOLD)
                matched.append(c)
            else:
                c["decision"] = "below heading threshold %.2f" % config.SUBJECT_THRESHOLD
            considered.append(c)
            continue
        if cos is not None and (cos >= config.COS_SURE or (z is not None and z >= config.Z_MIN and cos >= config.COS_FLOOR)):
            c["margin"] = round(max((z or 0) - config.Z_MIN, 0) + 0.5, 3)
            c["decision"] = "match: cosine %.3f, %.1f spreads above this cue's top 100" % (cos, z or 0)
            matched.append(c)
        elif c.get("bm25_rel", 0) >= config.BM25_KEEP_RATIO and shared >= config.BM25_MIN_TERMS:
            c["margin"] = 0.5
            c["decision"] = "match: keywords (%.2f of best, %d shared words)" % (c["bm25_rel"], shared)
            matched.append(c)
        else:
            c["decision"] = "below threshold (z %s)" % z
        considered.append(c)
    matched.sort(key=lambda c: -c["margin"])
    limit = top or config.MAX_MATCHES_PER_CUE
    for c in matched[limit:]:
        c["decision"] += "; cut: more than %d matches for this cue" % limit
    matched = matched[:limit]
    considered.sort(key=lambda c: -max(c.get("cos") or 0, c.get("subject_cos") or 0))
    return matched, considered[:40]


# Raw excerpts are a safety net for history not yet perceived. Offered: what people and sessions
# said, and files put into context. Not offered: tool output and app text, which are mostly noise
# until perceived (they stay searchable with `mem.py search`).
# Measured 2026-11-12: excerpts of file reads and subagent briefs were mostly noise.
PASSAGE_SAID = {"ryan_message", "claude_text", "interrupt", "command"}


def search_passages(conn, cue, exclude_ctx, limit=40):
    """Raw passages of inputs not yet perceived, by keywords."""
    ts = terms(cue, 16)
    q = fts_query(sorted(ts, key=len, reverse=True)[:6])
    if not q:
        return [], []
    try:
        rows = conn.execute(
            "SELECT p.id, p.input_id, p.text, bm25(passages_fts) AS b FROM passages_fts JOIN passages p ON p.id = passages_fts.rowid"
            " WHERE passages_fts MATCH ? ORDER BY b LIMIT ?", (q, limit)).fetchall()
    except Exception:
        return [], []
    if not rows:
        return [], []
    best = -rows[0]["b"]
    out, considered, seen_text = [], [], set()
    for r in rows:
        info = conn.execute("SELECT perceive, context_id, said_by, happened_at, session_id, type FROM input_info WHERE input_id=?",
                            (r["input_id"],)).fetchone()
        key = " ".join((r["text"] or "").split())[:300]
        if key in seen_text:
            continue
        seen_text.add(key)
        c = {"passage": r["id"], "input": r["input_id"], "bm25": round(-r["b"], 3),
             "bm25_rel": round((-r["b"]) / best, 3) if best > 0 else 0, "shared_terms": _shared_terms(ts, r["text"])}
        if info is None:
            continue
        if info["perceive"] == "done":
            c["decision"] = "dropped: input already perceived (its perceptions are searched instead)"
        elif info["type"] not in PASSAGE_SAID:
            c["decision"] = "dropped: %s (acts and plumbing are not offered as excerpts)" % info["type"]
        elif exclude_ctx and (info["context_id"] == exclude_ctx or
                              info["session_id"] == exclude_ctx.split("#")[0].split("/")[0]):
            c["decision"] = "dropped: from this context"
        elif c["bm25_rel"] >= 0.6 and c["shared_terms"] >= config.BM25_MIN_TERMS:
            c["decision"] = "match"
            c.update(text=r["text"], said_by=info["said_by"], at=info["happened_at"], session=info["session_id"], type=info["type"])
            out.append(c)
        else:
            c["decision"] = "below threshold"
        considered.append(c)
    return out[:config.PASSAGE_MAX], considered


def _in_context(conn, context_id):
    ids = set()
    if not context_id:
        return ids
    for r in conn.execute("SELECT node_ids FROM injections WHERE context_id=?", (context_id,)).fetchall():
        ids.update(json.loads(r[0] or "[]"))
    from .inject import spooled_node_ids
    ids.update(spooled_node_ids(context_id))
    return ids


def _state_rests_only_on(conn, state_id, context_id):
    r = conn.execute("SELECT rests_on FROM state_versions WHERE state_id=? ORDER BY n DESC LIMIT 1", (state_id,)).fetchone()
    if not r or not context_id:
        return False
    ids = json.loads(r[0] or "[]")
    if not ids:
        return False
    rows = conn.execute("SELECT DISTINCT context_id FROM nodes WHERE id IN (%s)" % ",".join("?" * len(ids)), ids).fetchall()
    return all(x[0] == context_id for x in rows)


def chain(conn, starts, pool, trace):
    """starts: {node_id: strength}. Adds chained-in nodes to pool. Returns list of crossings for the trace."""
    frontier = dict(starts)
    received = {}
    crossings = []
    seen_hops = 0
    while frontier and seen_hops < 4:
        nxt = {}
        for nid, s in frontier.items():
            outs = conn.execute('SELECT "end", kind, energy FROM links WHERE start=?', (nid,)).fetchall()
            fan = sum(1 for o in outs if o["kind"] != "state")
            fan_cost = 0.5 * math.log2(fan) if fan > 1 else 0.0
            for o in outs:
                if o["kind"] == "state":
                    passed = s
                else:
                    passed = s - o["energy"] - fan_cost
                if passed <= 0:
                    continue
                end = o["end"]
                crossings.append({"from": nid, "to": end, "kind": o["kind"], "passed": round(passed, 3)})
                received[end] = received.get(end, 0) + passed
                nxt[end] = nxt.get(end, 0) + passed
        frontier = {k: v for k, v in nxt.items() if k not in starts}
        seen_hops += 1
    added = 0
    for nid, s in sorted(received.items(), key=lambda kv: -kv[1]):
        if nid in pool:
            continue
        if added >= config.CHAIN_MAX_NODES:
            crossings.append({"from": None, "to": nid, "kind": "cut", "passed": round(s, 3)})
            continue
        pool[nid] = {"id": nid, "strength": round(s, 3), "via": "chain"}
        added += 1
    return crossings


def recall(conn, cues, context_id=None, timer=None, budget_s=None, with_passages=True):
    """cues: list of dicts {label, text, kinds}. Returns (pool, passages, trace)."""
    trace = {"cues": [], "context_id": context_id}
    idx = embed.Index(conn)
    trace["vectors"] = int(len(idx.ids))
    qv = embed.encode([c["text"] for c in cues]) if cues else []
    pool = {}
    for c, q in zip(cues, qv):
        matched, considered = search_nodes(conn, idx, c["text"], q, c["kinds"], exclude_ctx=context_id,
                                           subjects=c.get("subjects", False))
        for m in matched:
            e = pool.get(m["id"])
            if e is None or m["margin"] > e["strength"]:
                pool[m["id"]] = {"id": m["id"], "strength": m["margin"], "via": "match", "cue": c["label"],
                                 "cos": m.get("cos")}
        trace["cues"].append({"label": c["label"], "text": c["text"][:500], "kinds": c["kinds"],
                              "matched": [m["id"] for m in matched], "considered": considered})
    starts = {k: v["strength"] for k, v in pool.items() if v["via"] == "match" and v["strength"] > 0}
    trace["crossings"] = chain(conn, starts, pool, trace)
    # what the context already holds
    held = _in_context(conn, context_id)
    final, dropped = {}, []
    if pool:
        ids = list(pool)
        rows = {r["id"]: r for r in conn.execute(
            "SELECT id, kind, status, context_id FROM nodes WHERE id IN (%s)" % ",".join("?" * len(ids)), ids)}
        for nid, e in pool.items():
            r = rows.get(nid)
            if r is None:
                continue
            e["kind"] = r["kind"]
            if nid in held:
                dropped.append({"id": nid, "why": "already put into this context"})
            elif context_id and r["context_id"] == context_id:
                dropped.append({"id": nid, "why": "from this context"})
            elif r["kind"] == "state" and r["status"] != "holds":
                dropped.append({"id": nid, "why": "state is %s" % r["status"]})
            elif r["kind"] == "state" and _state_rests_only_on(conn, nid, context_id):
                dropped.append({"id": nid, "why": "state rests only on what this context already holds"})
            else:
                final[nid] = e
    trace["dropped"] = dropped
    passages = []
    if with_passages and not (budget_s and timer and timer.s() > budget_s):
        seen = set()
        for c in cues:
            if "perception" not in c["kinds"]:
                continue
            got, considered = search_passages(conn, c["text"], context_id)
            trace.setdefault("passages", []).append({"cue": c["label"], "considered": considered})
            for g in got:
                if g["passage"] not in seen and g["input"] not in seen:
                    seen.add(g["passage"])
                    passages.append(g)
        passages = passages[:config.PASSAGE_MAX]
    elif with_passages:
        trace["passages_skipped"] = "time budget used up"
    return final, passages, trace
