"""Claude Code hooks: the entry point's live side.

SessionStart       store what is new in this session's transcript; recall states for the moment and the project; inject.
UserPromptSubmit   store; recall for Ryan's message and the moment; inject. Must finish within Ryan's ~3 s tolerance.
Stop, SubagentStop, PreCompact, SessionEnd
                   store what is new; start the worker so the reply, tool results and anything else get perceived.

A hook never blocks or breaks a session: every failure is logged and the hook
exits 0 with no output.
"""
import json
import os
import sys
import traceback

from . import config, log


def _context_id(conn, session_id, transcript_path):
    n = 0
    if transcript_path:
        r = conn.execute("SELECT state FROM sources WHERE path=?", (transcript_path,)).fetchone()
        if r and r[0]:
            n = json.loads(r[0]).get("compactions", 0)
    return "%s#%d" % (session_id, n) if session_id else None


def _project_titles(conn, cwd):
    rows = conn.execute("SELECT DISTINCT title FROM sessions WHERE cwd=? AND title IS NOT NULL ORDER BY last_at DESC LIMIT 5",
                        (cwd,)).fetchall()
    return ("Sessions there: " + "; ".join(r[0] for r in rows) + ".") if rows else ""


def _title(conn, sid):
    r = conn.execute("SELECT title FROM sessions WHERE session_id=?", (sid,)).fetchone() if sid else None
    return r[0] if r and r[0] else "untitled"


# Prompts the app submits on its own (background-task results, messages from other sessions,
# app events). They are stored and perceived like everything else, but they are not a moment
# where Claude replies to Ryan, so no recall runs for them.
SYSTEM_PROMPTS = ("<task-notification>", "<side-sessions-event>", "[SYSTEM NOTIFICATION", "<ci-monitor-event>",
                  "Another Claude session sent a message", "<agent-message", "<system-reminder>")


def _sync(conn, tp, t):
    """Store what is new in this transcript, unless the database is busy (the worker stores it soon)."""
    from . import store
    if not (tp and os.path.exists(tp)):
        return
    try:
        store.sync(conn, paths=[(tp, "claude-code")])
    except Exception as e:
        log.event(None, "hook_sync_skipped", "database busy; the worker will store this transcript", {"error": repr(e)},
                  ms=t.ms())


def on_prompt(conn, h, t):
    from . import inject, recall, store
    prompt = h.get("prompt") or ""
    sid, tp, cwd = h.get("session_id"), h.get("transcript_path"), h.get("cwd") or ""
    _sync(conn, tp, t)
    ctx = _context_id(conn, sid, tp)
    if prompt.lstrip().startswith(SYSTEM_PROMPTS):
        log.event(conn, "recall_skipped", "UserPromptSubmit from the app, not from Ryan: no recall",
                  {"starts": prompt.lstrip()[:120], "hook_input_keys": sorted(h)}, ms=t.ms(), context_id=ctx)
        return None
    project = os.path.basename(cwd.rstrip("\\/")) or cwd
    # The moment, written by code at once: every message from Ryan is a moment where Claude replies to him.
    moment = "Claude is about to reply to Ryan in chat, in a Claude Code session in project %s." % project
    cues = [{"label": "message", "text": prompt[:6000], "kinds": ["perception", "state"]},
            {"label": "moment", "text": moment, "kinds": ["state"], "subjects": True}]
    # The last moment the model perceived in this context: resolved backwards, it says what kind of
    # exchange this conversation is in (design, code, a document...).
    last = conn.execute("SELECT id, text FROM nodes WHERE kind='situation' AND context_id=? ORDER BY id DESC LIMIT 1",
                        (ctx,)).fetchone() if ctx else None
    if last:
        cues.append({"label": "last moment p%d" % last[0], "text": last[1], "kinds": ["state"], "subjects": True})
    t_before = t.ms()
    pool, passages, trace = recall.recall(conn, cues, ctx, timer=t, budget_s=config.HOOK_BUDGET_S - 0.8)
    t_recall = t.ms()
    text, nodes, pids, cut = inject.format_pool(conn, pool, passages)
    trace["cut"] = cut
    trace["timing_ms"] = {"start_to_recall": t_before, "recall": t_recall - t_before, "format": t.ms() - t_recall}
    inj = inject.record(conn, "UserPromptSubmit", sid, ctx, cwd, prompt, text, nodes, pids, t.ms(), trace)
    log.event(conn, "recall", "UserPromptSubmit: injected %d memories and %d raw excerpts (%d chars)" % (
        len(nodes), len(pids), len(text or "")), {"injection": inj, "ms": t.ms(), "vectors": trace.get("vectors")},
        ms=t.ms(), context_id=ctx)
    if text:
        return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": text}}
    return None


def on_session_start(conn, h, t):
    from . import inject, recall, store
    sid, tp, cwd = h.get("session_id"), h.get("transcript_path"), h.get("cwd") or ""
    _sync(conn, tp, t)
    ctx = _context_id(conn, sid, tp)
    project = os.path.basename(cwd.rstrip("\\/")) or cwd
    situation = ("A Claude Code session is starting (%s) in %s, project %s. Claude is about to start working with Ryan;"
                 " how Ryan wants Claude to work and where the project stands." % (h.get("source") or "startup", cwd, project))
    cues = [{"label": "moment", "text": "Claude starts a session with Ryan and replies to him in chat, in project %s." % project,
             "kinds": ["state"], "subjects": True},
            {"label": "situation", "text": situation, "kinds": ["state"]},
            {"label": "project", "text": "Status of the %s project (%s): its decisions, plans and where its work stands. %s" % (
                project, cwd, _project_titles(conn, cwd)), "kinds": ["state"], "subjects": "also"}]
    pool, passages, trace = recall.recall(conn, cues, ctx, timer=t, with_passages=False)
    text, nodes, pids, cut = inject.format_pool(conn, pool, passages)
    trace["cut"] = cut
    inj = inject.record(conn, "SessionStart", sid, ctx, cwd, situation, text, nodes, pids, t.ms(), trace)
    log.event(conn, "recall", "SessionStart (%s): injected %d memories" % (h.get("source"), len(nodes)),
              {"injection": inj, "ms": t.ms()}, ms=t.ms(), context_id=ctx)
    if text:
        return {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": text}}
    return None


def on_other(conn, h, event, t):
    from . import store
    _sync(conn, h.get("transcript_path"), t)
    try:
        store.sync(conn, budget_s=1.5)
    except Exception as e:
        log.event(None, "hook_sync_skipped", "database busy; the worker will store the rest", {"error": repr(e)}, ms=t.ms())


def main(argv):
    event = argv[0] if argv else "unknown"
    if os.environ.get("AI_MEM_CHILD"):
        return 0
    t = log.timer()
    try:
        raw = sys.stdin.buffer.read()
        h = json.loads(raw.decode("utf-8", "replace")) if raw.strip() else {}
    except Exception:
        h = {}
    conn, out = None, None
    try:
        from . import db
        conn = db.connect(busy_ms=config.HOOK_BUSY_MS)
        if event == "UserPromptSubmit":
            out = on_prompt(conn, h, t)
        elif event == "SessionStart":
            out = on_session_start(conn, h, t)
        else:
            on_other(conn, h, event, t)
    except Exception:
        log.event(conn, "hook_error", "%s hook failed; the session went on without memory" % event,
                  {"error": traceback.format_exc()[-3000:], "session": h.get("session_id")}, ok=False)
    if out:
        try:
            sys.stdout.write(json.dumps(out))
            sys.stdout.flush()
        except Exception:
            pass
    try:
        from . import worker
        worker.spawn(conn, event)
    except Exception:
        log.event(conn, "hook_error", "could not start the worker", {"error": traceback.format_exc()[-2000:]}, ok=False)
    return 0
