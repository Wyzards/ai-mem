"""State: the model says what holds now. Code checks the rules and keeps every version.

In: the perceptions one perceive batch wrote, and the states that might be
affected: states search finds for each perception (and for the situation of
the moment), and states linked to the memories those perceptions rely on.
Out: new states, new versions, support, or endings. Each version records its
text, what it rests on, how sure the model is, and whether it is memory's own
conclusion. Code derives whose word it is on and as of when, writes the links
from each perception to its state (they cost nothing to cross), and holds
back a state that breaks a rule instead of letting it be recalled.

Rules code checks (each check is logged):
1. every version rests on at least one perception that exists;
2. nothing is overwritten: a change is a new version;
3. a rule about how Ryan wants things needs his own word, or, when memory
   infers it from his reactions, evidence from at least two different
   sessions (his rule: no broad rule from a single remark);
4. what Claude said is not evidence about what Ryan wants;
5. the current text is the version with the latest "as of".
"""
import json

from . import config, embed, log, model, recall
from .db import now, tx

SYSTEM = """You are the state step of Ryan's personal memory system. Ryan is the human user; Claude is the AI model working with him. Perceptions are what memory perceived, and they never change. States are what memory holds to be true now; they change only by new versions. States are what future contexts receive first, so they must be accurate, current and useful.

Given NEW PERCEPTIONS and the EXISTING STATES that may be affected, decide what memory should now hold:
- "update": an existing state's text must change. Give the whole new text.
- "support": a perception confirms an existing state without changing it.
- "new": something now holds that no existing state covers.
- "end": an existing state no longer holds.
Return no operations when nothing changes. Most routine steps of work (a file was read, a command ran) change no state unless they establish something that holds: a decision, a result that matters later, a fact about Ryan, a person, a project or its status, a plan.

The most important states say how Ryan wants Claude to work. Ryan often says this indirectly: by asking "why?", objecting ("I didn't read that"), correcting, praising, or restating what he meant. Read what each reaction says about how he wants things done and write it as a rule.
- One state per kind of moment, its text starting with that moment: "When Claude replies to Ryan in chat: ...", "When Claude proposes a design to Ryan: ...", "When Claude writes code for Ryan: ...", "When Claude writes a document for Ryan: ...". Put every rule for the same kind of moment in that one state, as a short list.
- Word each rule as broadly as the evidence supports. A rule he states about one case but clearly means generally is general; a rule that depends on the case keeps its condition.
- Keep his words in quotes where they carry the rule.
- If Ryan states the rule himself, own_conclusion is false. If you infer it from his reactions, own_conclusion is true.
- What Claude said is never evidence of what Ryan wants.

Other states: one per subject (a person, a project and its status, a decision, a plan, a fact about Ryan or his world). Prefer updating an existing state over making a new one; make a new state only when no shown state covers the subject. A state's text says what holds now, plainly, in full sentences that stand alone.

Name a project by its name and its folder, as the perceptions' sessions give it (for example "the ai-mem project (C:\\ai-mem)"), so a session opened in that folder finds it.

Keep every state short, because states are put into future contexts: at most about 100 words, or one short line per rule for a list of rules. A status says where things stand now, not the history of how they got there (the history stays in the perceptions). When a subject grows past that, split it into narrower states.

For each operation give: op; state (the [S..] label, for update, support and end); subject (short, for new); kind (preference: how Ryan wants things done; fact; status; decision; plan; person; other); text (for new and update); rests_on (the [P..] labels of the new perceptions it rests on); confidence (high, medium, low); own_conclusion; reason (one sentence on why).

Answer only with the JSON asked for."""

SCHEMA = {
    "type": "object",
    "properties": {"ops": {"type": "array", "items": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["new", "update", "support", "end"]},
        "state": {"type": "string"},
        "subject": {"type": "string"},
        "kind": {"type": "string", "enum": ["preference", "fact", "status", "decision", "plan", "person", "other"]},
        "text": {"type": "string"},
        "rests_on": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "own_conclusion": {"type": "boolean"},
        "reason": {"type": "string"}},
        "required": ["op", "rests_on", "reason"]}}},
    "required": ["ops"],
}

MAX_STATES_SHOWN = 30


def candidate_states(conn, perceptions):
    """States that might be affected: search for each perception, plus states linked from what they rely on."""
    idx = embed.Index(conn)
    texts = [p["text"] for p in perceptions]
    qv = embed.encode(texts) if texts else []
    found, why = {}, {}
    for p, q in zip(perceptions, qv):
        matched, considered = recall.search_nodes(conn, idx, p["text"], q, ["state"], statuses=("holds", "held"),
                                                  top=config.STATE_CANDIDATES_PER_CUE)
        for m in matched:
            found.setdefault(m["id"], m.get("cos") or 0)
            why.setdefault(m["id"], "search from P%d" % p["id"])
        # also the closest states even below the threshold, so the model can see near duplicates
        for c in considered[:3]:
            if c.get("kind") == "state" and (c.get("cos") or 0) > config.COS_FLOOR - 0.05:
                found.setdefault(c["id"], c.get("cos") or 0)
                why.setdefault(c["id"], "near P%d (cosine %.2f)" % (p["id"], c.get("cos") or 0))
    ids = [p["id"] for p in perceptions]
    if ids:
        q = ("SELECT DISTINCT l2.\"end\" FROM links l1 JOIN links l2 ON l2.start = l1.\"end\" AND l2.kind = 'state'"
             " WHERE l1.start IN (%s) AND l1.kind = 'relies'" % ",".join("?" * len(ids)))
        for r in conn.execute(q, ids):
            found.setdefault(r[0], 1.0)
            why.setdefault(r[0], "state of a memory relied on")
    order = sorted(found, key=lambda k: -found[k])[:MAX_STATES_SHOWN]
    rows = []
    for sid in order:
        r = conn.execute("SELECT id, text, subject, status, said_by, said_at, extra FROM nodes WHERE id=? AND kind='state'",
                         (sid,)).fetchone()
        if r:
            rows.append((r, why[sid]))
    return rows


def build_prompt(conn, perceptions, states):
    lines = []
    sess = []
    for p in perceptions:
        if p["session_id"] and p["session_id"] not in sess:
            sess.append(p["session_id"])
    for s in sess:
        r = conn.execute("SELECT title, cwd FROM sessions WHERE session_id=?", (s,)).fetchone()
        if r:
            lines.append('From Claude Code session "%s" in folder %s.' % (r["title"] or "untitled", r["cwd"] or "?"))
    if lines:
        lines.append("")
    lines.append("NEW PERCEPTIONS (oldest first):")
    for p in perceptions:
        lines.append("[P%d] (%s, %s) %s" % (p["id"], p["said_by"] or "?", (p["said_at"] or "?")[:16].replace("T", " "), p["text"]))
    lines.append("")
    lines.append("EXISTING STATES THAT MAY BE AFFECTED:")
    if not states:
        lines.append("(none found)")
    for r, why in states:
        nver = conn.execute("SELECT COUNT(*) FROM state_versions WHERE state_id=?", (r["id"],)).fetchone()[0]
        lines.append("[S%d] subject: %s | %s | as of %s | on the word of %s | %d versions" % (
            r["id"], r["subject"] or "?", r["status"], (r["said_at"] or "?")[:10], r["said_by"] or "?", nver))
        lines.append(r["text"])
    return "\n".join(lines)


def _derive(conn, rest_ids):
    rows = conn.execute("SELECT id, said_by, said_at, session_id, context_id FROM nodes WHERE id IN (%s)"
                        % ",".join("?" * len(rest_ids)), rest_ids).fetchall()
    speakers = sorted({r["said_by"] or "?" for r in rows})
    as_of = max((r["said_at"] or "" for r in rows), default=None)
    return rows, ", ".join(speakers), as_of


def _check(op, kind, own, rows, issues):
    """Rules 3 and 4. Returns (status, notes)."""
    notes = []
    ryan = [r for r in rows if (r["said_by"] or "").startswith("Ryan")]
    not_claude = [r for r in rows if not (r["said_by"] or "").startswith("Claude")]
    status = "holds"
    if kind == "preference" and not ryan:
        status = "held"
        notes.append("rule 4: how Ryan wants things rests on no perception of Ryan's own")
    if own:
        sessions = {r["session_id"] or r["context_id"] for r in not_claude}
        if len(sessions) < 2:
            status = "held"
            notes.append("rule 3: memory's own conclusion has evidence from %d session(s); needs 2" % len(sessions))
        else:
            notes.append("rule 3 passed: evidence from %d sessions" % len(sessions))
    return status, notes


def run(conn, perception_ids, context_id=None):
    if not perception_ids:
        return None
    t = log.timer()
    perceptions = conn.execute("SELECT id, text, said_by, said_at, session_id, context_id, kind FROM nodes WHERE id IN (%s)"
                               " ORDER BY id" % ",".join("?" * len(perception_ids)), perception_ids).fetchall()
    states = candidate_states(conn, perceptions)
    prompt = build_prompt(conn, perceptions, states)
    shown_p = {"P%d" % p["id"]: p["id"] for p in perceptions}
    shown_s = {"S%d" % r["id"]: r["id"] for r, _ in states}
    data, call_id = model.call(conn, "state", config.STATE_MODEL, SYSTEM, prompt, SCHEMA, input_ids=[],
                               effort=config.STATE_EFFORT)
    ts = now()
    applied, rejected, vec_items = [], [], []
    with tx(conn):
        for op in data.get("ops") or []:
            kind = op.get("kind") or "other"
            rests = [shown_p[x.strip("[] ")] for x in op.get("rests_on") or [] if x.strip("[] ") in shown_p]
            unknown = [x for x in op.get("rests_on") or [] if x.strip("[] ") not in shown_p]
            rec = {"op": op.get("op"), "state": op.get("state"), "subject": op.get("subject"), "kind": kind,
                   "reason": op.get("reason"), "unknown_rests_on": unknown}
            if not rests:
                rec["rejected"] = "rule 1: rests on no perception that exists"
                rejected.append(rec)
                continue
            sid = shown_s.get((op.get("state") or "").strip("[] "))
            if op["op"] in ("update", "support", "end") and sid is None:
                rec["rejected"] = "names a state that was not shown"
                rejected.append(rec)
                continue
            if op["op"] in ("new", "update") and not (op.get("text") or "").strip():
                rec["rejected"] = "no text"
                rejected.append(rec)
                continue
            if op["op"] == "support":
                for pid in rests:
                    conn.execute('INSERT OR IGNORE INTO links(start, "end", kind, energy, created_at, note) VALUES (?,?,?,?,?,?)',
                                 (pid, sid, "state", 0.0, ts, "support"))
                cur_status = conn.execute("SELECT status FROM nodes WHERE id=?", (sid,)).fetchone()[0]
                if cur_status != "held":
                    rec["applied"] = "linked %d perceptions to S%d" % (len(rests), sid)
                    applied.append(rec)
                    continue
                # A held state gets new evidence: write a version with the same text and check the rules again.
                prev = conn.execute("SELECT * FROM state_versions WHERE state_id=? ORDER BY n DESC LIMIT 1", (sid,)).fetchone()
                op["text"] = prev["text"]
                op["own_conclusion"] = bool(prev["own_conclusion"])
                op["confidence"] = op.get("confidence") or prev["confidence"]
                op["op"] = "update"
                op["_support"] = True
            if op["op"] == "new":
                rows, whose, as_of = _derive(conn, rests)
                status, notes = _check("new", kind, bool(op.get("own_conclusion")), rows, None)
                cur = conn.execute(
                    "INSERT INTO nodes(kind, text, said_by, said_at, subject, status, created_at, extra)"
                    " VALUES ('state',?,?,?,?,?,?,?)",
                    (op["text"].strip(), whose, as_of, (op.get("subject") or "").strip(), status, ts,
                     json.dumps({"kind": kind})))
                sid = cur.lastrowid
                n, all_rests = 1, rests
                rec["notes"] = notes
            else:
                prev = conn.execute("SELECT n, rests_on, as_of, text FROM state_versions WHERE state_id=? ORDER BY n DESC LIMIT 1",
                                    (sid,)).fetchone()
                prev_rests = json.loads(prev["rests_on"]) if prev else []
                all_rests = list(dict.fromkeys(prev_rests + rests))
                rows, whose, as_of = _derive(conn, all_rests)
                new_rows, _, new_as_of = _derive(conn, rests)
                skind = kind
                ex = conn.execute("SELECT extra FROM nodes WHERE id=?", (sid,)).fetchone()
                try:
                    skind = json.loads(ex[0] or "{}").get("kind") or kind
                except Exception:
                    pass
                status, notes = _check(op["op"], skind, bool(op.get("own_conclusion")), rows, None)
                if op["op"] == "end":
                    status = "ended"
                n = (prev["n"] if prev else 0) + 1
                rec["notes"] = notes
                text = (op.get("text") or "").strip() or (prev["text"] if prev else "")
                current = (prev is None) or ((new_as_of or "") >= (prev["as_of"] or ""))
                if current:
                    conn.execute("UPDATE nodes SET text=?, said_by=?, said_at=?, status=? WHERE id=?",
                                 (text, whose, as_of, status, sid))
                else:
                    notes.append("rule 5: this version is older than the current one, kept but not made current")
                op["text"] = text
            conn.execute(
                "INSERT INTO state_versions(state_id, n, op, text, rests_on, confidence, own_conclusion, on_whose_word, as_of,"
                " status, reason, written_at, model_call_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, n, "support" if op.get("_support") else op["op"], op["text"].strip(), json.dumps(all_rests), op.get("confidence"),
                 1 if op.get("own_conclusion") else 0, whose, as_of, status, op.get("reason"), ts, call_id))
            for pid in rests:
                conn.execute('INSERT OR IGNORE INTO links(start, "end", kind, energy, created_at, note) VALUES (?,?,?,?,?,?)',
                             (pid, sid, "state", 0.0, ts, op["op"]))
            vec_items.append((sid, op["text"].strip()))
            rec["applied"] = "%s S%d version %d (%s)" % (op["op"], sid, n, status)
            applied.append(rec)
        embed.store_vectors(conn, vec_items)
    log.event(conn, "state", "%d state changes, %d rejected, from %d perceptions (%d states shown)" % (
        len(applied), len(rejected), len(perceptions), len(states)),
        {"perceptions": perception_ids, "states_shown": [{"id": r["id"], "why": w} for r, w in states],
         "applied": applied, "rejected": rejected, "model_call": call_id}, ms=t.ms(), context_id=context_id)
    return applied
