"""Perceive: the model says what the system became aware of.

In: a batch of queued inputs from one context, in arrival order, plus what
that context already held, to resolve references against: the memories
memory put into it, its earlier perceptions, and the last raw text before the
batch. Nothing after the batch is shown, so everything resolves backwards.

Out: perceptions (each one complete thing that makes sense alone, saying who
said or did it), a situation for each message from Ryan (what kind of moment
it is), stream links both ways to the perception before, and links from each
perception to the memories in context it relies on.
"""
import json
import re

from . import config, embed, extract, log, model
from .db import now, tx

SYSTEM = """You are the perceiving step of Ryan's personal memory system. Ryan is the human user. Claude is the AI model working with him in Claude Code. Memory keeps everything that passed through Claude's context; your job is to say what was perceived, as statements memory can find and use months later.

Write perceptions for the NEW items only. A perception:
- is one complete thing: a fact, request, instruction, preference, decision, question, answer, plan, result, error, or event;
- makes sense alone to a reader who never saw this conversation: name people, files, projects and things in full; replace "it", "that", "the file", "he", "this" with what they refer to, using the context given; if you cannot resolve a reference, keep the words as said;
- says who said or did it: "Ryan said/asked/wants...", "Claude replied that...", "Claude ran...", "The Read tool showed that the file C:\\x\\y.md says...";
- keeps Ryan's own words in quotes where wording matters (instructions, preferences, corrections, decisions), including where and when he says they apply;
- adds nothing that is not in the item: no evaluation, advice or guesses.

How much to write:
- Messages from Ryan: every distinct point, request and instruction, one perception each.
- Claude's replies: each distinct claim, answer, decision, proposal or question to Ryan.
- Tool calls: the act in one perception (what Claude did, to which file or command, and why if stated). For a file Claude wrote or edited, say what the content does or says.
- Tool results and files: what someone could need later: facts, names, numbers, structure, decisions, statuses, errors. For code: what each part does and where. For long documents: each distinct point or section. Skip pure boilerplate.
- Claude's thinking (private reasoning, never shown to Ryan): what Claude considered, concluded or planned, written as "Claude thought that...".
- Text inside <system-reminder> tags, app notices and hook text come from the app, not from Ryan: perceive them only if they carry information (for example a report from another session), and say they came from the app.
- An item may yield no perceptions if it carries nothing (an empty acknowledgement, a progress tick).

For each perception give: item (the [I..] label it comes from), text, quote (a short exact excerpt, at most 200 characters, copied from the item), and relies_on (labels of memories shown in the context, [P..] or [S..], that this perception depends on to make sense or that it answers, corrects or continues; empty if none).

For each item that is a message from Ryan, also write one situation: what kind of moment it is, in general terms first (what kind of request or exchange, what Claude is about to do: reply in chat, explain, design, write code, write a document, run a task, review), then the specifics (topic, project). Example: "Ryan asks Claude to explain a design decision in chat and wants the reasoning behind it; the topic is search thresholds in the ai-mem project."

Answer only with the JSON asked for."""

SCHEMA = {
    "type": "object",
    "properties": {
        "perceptions": {"type": "array", "items": {"type": "object", "properties": {
            "item": {"type": "string"},
            "text": {"type": "string"},
            "quote": {"type": "string"},
            "relies_on": {"type": "array", "items": {"type": "string"}}},
            "required": ["item", "text"]}},
        "situations": {"type": "array", "items": {"type": "object", "properties": {
            "item": {"type": "string"},
            "text": {"type": "string"}},
            "required": ["item", "text"]}},
    },
    "required": ["perceptions", "situations"],
}

TYPE_LABEL = {
    "ryan_message": "message", "command": "ran a command", "claude_text": "reply text",
    "tool_call": "tool call", "tool_result": "tool result", "notification": "notification",
    "peer_message": "message from another session", "agent_prompt": "task given to a subagent",
    "coordinator_message": "message from the coordinating session", "session_brief": "brief from another session",
    "system_message": "app message", "command_output": "command output", "instructions": "instruction files loaded into context",
    "file_attachment": "file put into context", "file_changed": "file changed on disk", "hook_context": "hook text",
    "memory_file": "auto-memory file", "inbox_file": "file in the memory inbox", "user_other": "message",
    "reasoning": "Claude's private thinking (not shown to Ryan)", "interrupt": "interrupted Claude",
}


def local_time(iso):
    if not iso:
        return "?"
    try:
        import datetime
        d = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone()
        return d.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso[:16]


def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


def next_batch(conn, mode="full"):
    """The next batch: inputs of one context, oldest first by when they happened.
    Live inputs come before history queued by backfill. mode 'said': only what people and
    sessions said (memory's budget is spent; tool output waits). mode 'stop': nothing."""
    # mode 'full': everything; mode 'live': history waits (its pace is capped), live goes on.
    # Order of work: what people and sessions said comes before tool output, because it is where
    # instructions, preferences and decisions are; live before history within each.
    said = "type IN (%s)" % ",".join("'%s'" % x for x in sorted(extract.SAID_TYPES))
    live, hist = "COALESCE(perceive_note,'') != 'backfill'", "perceive_note = 'backfill'"
    tiers = [(live, said)] + ([(hist, said)] if mode == "full" else []) + [(live, "NOT " + said)] + (
        [(hist, "NOT " + said)] if mode == "full" else [])
    first, cond = None, None
    import datetime
    settle = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=config.LIVE_SETTLE_S)
              ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    for when, kind in tiers:
        cond = " AND %s AND %s" % (when, kind)
        first = conn.execute("SELECT input_id, context_id, happened_at FROM input_info WHERE perceive='pending'" + cond +
                             " ORDER BY happened_at, input_id LIMIT 1").fetchone()
        if first and when == live and (first["happened_at"] or "") > settle:
            n = conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(text)),0) FROM input_info WHERE perceive='pending'" + cond +
                             " AND context_id IS ?", (first["context_id"],)).fetchone()
            if n[0] < config.PERCEIVE_MAX_INPUTS and n[1] < config.PERCEIVE_BATCH_CHARS:
                first = None          # still settling: let the reply and what follows join this batch
        if first:
            break
    if not first:
        return None, []
    rows = conn.execute(
        "SELECT * FROM input_info WHERE perceive='pending' AND context_id IS ? AND happened_at >= ?" + cond +
        " ORDER BY happened_at, input_id LIMIT ?",
        (first["context_id"], first["happened_at"], config.PERCEIVE_MAX_INPUTS)).fetchall()
    batch, total = [], 0
    for r in rows:
        n = len(r["text"] or "")
        if batch and (total + n > config.PERCEIVE_BATCH_CHARS or n > config.PERCEIVE_PART_CHARS):
            break
        batch.append(r)
        total += n
        if n > config.PERCEIVE_PART_CHARS:
            break
    return first["context_id"], batch


def _session_line(conn, ctx, batch):
    sid = batch[0]["session_id"]
    if sid:
        s = conn.execute("SELECT title, cwd FROM sessions WHERE session_id=?", (sid,)).fetchone()
        if s:
            return 'Claude Code session "%s" in %s (context %s).' % (s["title"] or "untitled", s["cwd"] or "?", ctx)
    f = json.loads(batch[0]["fields"] or "{}")
    if f.get("path"):
        return "File %s." % f["path"]
    return "Context %s." % ctx


def context_block(conn, ctx, before_id):
    shown, lines = {}, []
    inj = conn.execute("SELECT node_ids FROM injections WHERE context_id=? ORDER BY id DESC LIMIT 6", (ctx,)).fetchall()
    ids = []
    for r in inj:
        for n in json.loads(r[0] or "[]"):
            if n not in ids:
                ids.append(n)
    if ids:
        lines.append("Memories that memory put into this context:")
        for n in ids[:10]:
            row = conn.execute("SELECT id, kind, text FROM nodes WHERE id=?", (n,)).fetchone()
            if row:
                lab = ("S%d" if row["kind"] == "state" else "P%d") % row["id"]
                shown[lab] = row["id"]
                lines.append("[%s] %s" % (lab, row["text"]))
    ps = conn.execute(
        "SELECT id, kind, text FROM nodes WHERE context_id IS ? AND kind IN ('perception','situation') AND input_id < ?"
        " ORDER BY id DESC LIMIT ?", (ctx, before_id, config.PERCEIVE_CONTEXT_PERCEPTIONS)).fetchall()
    if ps:
        lines.append("Earlier perceptions in this context, oldest first:")
        for r in reversed(ps):
            lab = "P%d" % r["id"]
            shown[lab] = r["id"]
            lines.append("[%s] %s%s" % (lab, "(situation) " if r["kind"] == "situation" else "", r["text"]))
    tail = conn.execute(
        "SELECT said_by, type, text FROM input_info WHERE context_id IS ? AND input_id < ? AND text IS NOT NULL"
        " AND type IN ('ryan_message','claude_text','tool_call','tool_result','command') ORDER BY input_id DESC LIMIT 6",
        (ctx, before_id)).fetchall()
    if tail:
        buf, used = [], 0
        for r in tail:
            piece = "%s (%s): %s" % (r["said_by"], TYPE_LABEL.get(r["type"], r["type"]), r["text"])
            room = config.PERCEIVE_RAW_TAIL_CHARS - used
            if room <= 200:
                break
            if len(piece) > room:
                piece = "..." + piece[-room:]
            buf.append(piece)
            used += len(piece)
        lines.append("Raw text just before the new items, oldest first:")
        lines.extend(reversed(buf))
    return lines, shown


def _parts(text):
    n = config.PERCEIVE_PART_CHARS
    if len(text) <= n:
        return [text]
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + n)
        if j < len(text):
            k = text.rfind("\n", i + n // 2, j)
            j = k + 1 if k > i else j
        out.append(text[i:j])
        i = j
    return out


def build_prompt(conn, ctx, items):
    """items: list of (label, input_row, part_no, part_count, text)."""
    head = [_session_line(conn, ctx, [it[1] for it in items])]
    ctx_lines, shown = context_block(conn, ctx, items[0][1]["input_id"])
    body = head + [""]
    if ctx_lines:
        body += ["WHAT THIS CONTEXT ALREADY HELD (for resolving references only; do not perceive these again):"] + ctx_lines + [""]
    body.append("NEW ITEMS TO PERCEIVE, in order:")
    for lab, r, pn, pc, text in items:
        f = json.loads(r["fields"] or "{}")
        extra = ""
        if f.get("tool"):
            extra = " | tool %s" % f["tool"]
        if f.get("file"):
            extra += " | file %s" % f["file"]
        if f.get("path"):
            extra += " | file %s" % f["path"]
        part = " | part %d of %d" % (pn, pc) if pc > 1 else ""
        body.append("[%s] %s | %s | %s%s%s" % (lab, local_time(r["happened_at"]), r["said_by"] or "?",
                                               TYPE_LABEL.get(r["type"], r["type"]), extra, part))
        body.append(text)
        body.append("")
    return "\n".join(body), shown


def _write(conn, ctx, items, data, shown, call_id, issues):
    by_label = {it[0]: it for it in items}
    per_item = {it[0]: [] for it in items}
    for p in data.get("perceptions") or []:
        lab = str(p.get("item", "")).strip("[] ")
        text = (p.get("text") or "").strip()
        if lab not in by_label:
            issues.append({"issue": "perception names an unknown item", "item": lab, "text": text[:200]})
            continue
        if not text:
            continue
        quote = (p.get("quote") or "").strip()[:400]
        if quote and _norm(quote) not in _norm(by_label[lab][4]):
            issues.append({"issue": "quote not found in item", "item": lab, "quote": quote[:200]})
        rel = []
        for x in p.get("relies_on") or []:
            k = str(x).strip("[] ")
            if k in shown:
                rel.append(shown[k])
            else:
                issues.append({"issue": "relies_on names a memory not shown", "label": k})
        per_item[lab].append((text, quote, rel))
    situations = {}
    for s in data.get("situations") or []:
        lab = str(s.get("item", "")).strip("[] ")
        if lab in by_label and (s.get("text") or "").strip():
            if by_label[lab][1]["type"] in ("ryan_message", "command", "user_other"):
                situations[lab] = s["text"].strip()
            else:
                issues.append({"issue": "situation for an item that is not from Ryan", "item": lab})
    ts = now()
    new_nodes, links = [], 0
    # The perception just before these items in the context (by when the input came, not when
    # it was perceived: tool output perceived later still links to its place in the stream).
    i0, p0 = items[0][1]["input_id"], items[0][2]
    prev = conn.execute("SELECT id FROM nodes WHERE context_id IS ? AND kind='perception'"
                        " AND (input_id < ? OR (input_id = ? AND part < ?)) ORDER BY input_id DESC, part DESC, id DESC LIMIT 1",
                        (ctx, i0, i0, p0)).fetchone()
    prev = prev[0] if prev else None

    def link(a, b, kind, note=None):
        nonlocal links
        cur = conn.execute('INSERT OR IGNORE INTO links(start, "end", kind, energy, created_at, note) VALUES (?,?,?,?,?,?)',
                           (a, b, kind, config.LINK_ENERGY[kind], ts, note))
        links += cur.rowcount

    for lab, r, pn, pc, text in items:
        first_of_item = None
        for ptext, quote, rel in per_item[lab]:
            cur = conn.execute(
                "INSERT INTO nodes(kind, text, input_id, part, quote, said_by, said_at, context_id, session_id, created_at, extra)"
                " VALUES ('perception',?,?,?,?,?,?,?,?,?,?)",
                (ptext, r["input_id"], pn, quote, r["said_by"], r["happened_at"], ctx, r["session_id"], ts,
                 json.dumps({"model_call": call_id})))
            nid = cur.lastrowid
            new_nodes.append((nid, ptext))
            first_of_item = first_of_item or nid
            if prev:
                link(prev, nid, "stream")
                link(nid, prev, "stream")
            prev = nid
            for target in rel:
                link(nid, target, "relies")
                link(target, nid, "relies")
        if lab in situations:
            cur = conn.execute(
                "INSERT INTO nodes(kind, text, input_id, part, said_by, said_at, context_id, session_id, created_at, extra)"
                " VALUES ('situation',?,?,?,?,?,?,?,?,?)",
                (situations[lab], r["input_id"], pn, "memory (situation of the moment)", r["happened_at"], ctx,
                 r["session_id"], ts, json.dumps({"model_call": call_id})))
            sid = cur.lastrowid
            new_nodes.append((sid, situations[lab]))
            if first_of_item:
                link(sid, first_of_item, "situation")
                link(first_of_item, sid, "situation")
    embed.store_vectors(conn, new_nodes)
    return new_nodes, links


def run_batch(conn, ctx, batch):
    """Perceive one batch. Returns list of new perception node ids (for the state step)."""
    t = log.timer()
    items, labels = [], 0
    long_input = len(batch) == 1 and len(batch[0]["text"] or "") > config.PERCEIVE_PART_CHARS
    new_ids = []
    if long_input:
        r = batch[0]
        parts = _parts(r["text"])
        note = json.loads(r["perceive_note"] or "{}") if (r["perceive_note"] or "").startswith("{") else {}
        done = int(note.get("parts_done", 0))
        for pn in range(done + 1, len(parts) + 1):
            items = [("I1", r, pn, len(parts), parts[pn - 1])]
            ids = _call_and_write(conn, ctx, items, t)
            new_ids += ids
            with tx(conn):
                conn.execute("UPDATE input_info SET perceive_note=? WHERE input_id=?",
                             (json.dumps({"parts_done": pn, "parts": len(parts)}), r["input_id"]))
        with tx(conn):
            conn.execute("UPDATE input_info SET perceive='done', perceived_at=? WHERE input_id=?", (now(), r["input_id"]))
        return new_ids
    for r in batch:
        labels += 1
        items.append(("I%d" % labels, r, 1, 1, r["text"] or ""))
    new_ids = _call_and_write(conn, ctx, items, t, mark_done=[r["input_id"] for r in batch])
    return new_ids


def _call_and_write(conn, ctx, items, t, mark_done=None):
    prompt, shown = build_prompt(conn, ctx, items)
    input_ids = sorted({it[1]["input_id"] for it in items})
    data, call_id = model.call(conn, "perceive", config.PERCEIVE_MODEL, SYSTEM, prompt, SCHEMA, input_ids=input_ids,
                               effort=config.PERCEIVE_EFFORT)
    issues = []
    with tx(conn):
        new_nodes, links = _write(conn, ctx, items, data, shown, call_id, issues)
        if mark_done:
            conn.executemany("UPDATE input_info SET perceive='done', perceived_at=? WHERE input_id=?",
                             [(now(), i) for i in mark_done])
    counts = {}
    for it in items:
        counts[it[0]] = 0
    log.event(conn, "perceive",
              "perceived %d inputs into %d memories (%d links)" % (len(input_ids), len(new_nodes), links),
              {"inputs": input_ids, "model_call": call_id, "new_nodes": [n for n, _ in new_nodes],
               "situations": len(data.get("situations") or []), "issues": issues,
               "shown_context_memories": len(shown)},
              ms=t.ms(), input_id=input_ids[0], context_id=ctx)
    return [n for n, _ in new_nodes]
