"""The Claude Code entry point's reading of each input: its type, who said it,
which context it belongs to, and the readable text the model saw.

Everything here is derived from the raw bytes, so it can be recomputed. Types
that are app plumbing (token counters, tool listings, request records) are
stored raw but not perceived; the list is explicit below so a log can show
why anything was skipped.
"""
import hashlib
import json
import os

# Types the perceive step reads. Everything else is stored raw only.
PERCEIVE_TYPES = {
    "ryan_message", "command", "claude_text", "tool_call", "tool_result",
    "notification", "peer_message", "agent_prompt", "coordinator_message",
    "session_brief", "system_message", "command_output", "instructions",
    "file_attachment", "file_changed", "hook_context", "memory_file",
    "inbox_file", "user_other", "interrupt", "reasoning",
}

# What people and sessions said, as opposed to tool output and app text. Perceived even
# when memory's budget is spent, and a batch holding any of these runs the state step.
SAID_TYPES = {
    "ryan_message", "command", "interrupt", "claude_text", "peer_message", "agent_prompt",
    "coordinator_message", "session_brief", "instructions", "memory_file", "inbox_file", "user_other",
}

# Why each skipped type is not perceived (shown in logs and `mem.py status`).
SKIP_REASONS = {
    "compact_summary": "Claude's summary of an earlier part of the same session; restates what is already perceived",
    "memory_injection": "memory's own injected text; recorded only",
    "meta": "app text marked as meta (skill bodies, caveats, notices)",
    "api_error": "API error notice",
    "app": "app plumbing (token counters, tool listings, request records, titles, modes)",
    "attachment_other": "attachment type not yet handled; stored raw",
    "compact_boundary": "marks where a compaction starts a new context",
    "tool_output_file": "full output saved beside the transcript; the model saw a preview, which is perceived from the transcript",
    "file_backup": "Claude Code's backup of a file before an edit",
    "app_file": "app metadata file",
    "instructions_file": "perceived from the transcript, where it entered context",
    "prompt_history": "copy of prompts already in the transcripts",
    "binary": "not text; stored raw",
    "unparsed": "line is not valid JSON; stored raw",
}

MEMORY_MARKER = "[ai-mem]"

NOISE_ATTACHMENTS = {
    "total_tokens_reminder", "deferred_tools_record", "deferred_tools_delta", "silent_turn_reminder",
    "prompt_snapshot", "environment", "remote_session_change", "date", "mcp_instructions_delta", "model",
    "agent_listing_delta", "session_context", "auto_mode", "skill_listing", "credential_org",
    "batching_reminder_sent", "compact_file_reference", "task_status", "thinking_drop", "invoked_skills",
    "read_truncation_notice", "task_reminder", "command_permissions",
}


def sha(b):
    if isinstance(b, str):
        b = b.encode("utf-8", "surrogatepass")
    return hashlib.sha256(b).hexdigest()


def block_text(b):
    if not isinstance(b, dict):
        return ""
    t = b.get("type")
    if t == "text":
        return b.get("text", "") or ""
    if t == "image":
        return "[image]"
    if t == "document":
        return "[document]"
    if t == "tool_result":
        c = b.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "\n".join(block_text(x) for x in c)
        return ""
    if t == "tool_use":
        return "%s %s" % (b.get("name"), json.dumps(b.get("input"), ensure_ascii=False))
    return ""


def rendered_text(o):
    r = o.get("rendered")
    if isinstance(r, list):
        return "\n".join(x.get("content", "") for x in r if isinstance(x, dict) and isinstance(x.get("content"), str))
    return None


def _user_kind(o, text, sidechain):
    origin = (o.get("origin") or {}).get("kind") if isinstance(o.get("origin"), dict) else None
    s = text.lstrip()
    if o.get("isCompactSummary") or s.startswith("This session is being continued from a previous conversation"):
        return "compact_summary", "Claude (compaction summary)"
    if origin == "human":
        if s.startswith("<command-name>"):
            return "command", "Ryan"
        return "ryan_message", "Ryan"
    if origin == "task-notification":
        return "notification", "system (background task)"
    if origin == "peer":
        return "peer_message", "another Claude session"
    if origin == "coordinator":
        return "coordinator_message", "Claude (coordinating session)"
    if origin:
        return "user_other", "origin:" + str(origin)
    if o.get("isMeta") or s.startswith("<local-command-caveat>"):
        return "meta", "system"
    if sidechain:
        return "agent_prompt", "Claude (parent session)"
    if s.startswith("[Request interrupted by user"):
        return "interrupt", "Ryan"
    if s.startswith("<command-name>"):
        return "command", "Ryan"
    if s.startswith("<local-command-stdout>") or s.startswith("<local-command-stderr>"):
        return "command_output", "system"
    if s.startswith("Background (from the session"):
        return "session_brief", "Claude (another session)"
    if s.startswith("<system-reminder>"):
        return "system_message", "system"
    if s.startswith("<") and not s.startswith("<pasted_content"):
        return "user_other", "unknown"
    # Transcripts written before the app recorded an origin (versions up to about 2.1.2x)
    # mark Ryan's typed messages with nothing else; every other plain text has a marker above.
    return "ryan_message", "Ryan"


def norm_ts(v):
    """Transcript timestamps are ISO strings; a few line types carry epoch milliseconds."""
    if v is None:
        return None
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
        import datetime
        x = float(v)
        if x > 1e11:
            x /= 1000.0
        return datetime.datetime.fromtimestamp(x, datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return str(v)


def classify_line(o, ctx, tool_name_of):
    """o: parsed transcript line. ctx: dict with session_id, agent_id, compactions, sidechain.
    tool_name_of(tool_use_id) -> tool name or None.
    Returns dict(type, said_by, text, fields, uuid, ref, happened_at)."""
    t = o.get("type")
    sidechain = bool(o.get("isSidechain")) or ctx.get("sidechain", False)
    fields = {k: o.get(k) for k in ("cwd", "gitBranch", "entrypoint", "version", "agentId", "parentUuid",
                                     "promptSource", "userType") if o.get(k) is not None}
    if sidechain:
        fields["sidechain"] = True
    out = {"type": "app", "said_by": None, "text": None, "fields": fields, "uuid": o.get("uuid"),
           "ref": None, "happened_at": norm_ts(o.get("timestamp"))}
    if t == "user":
        m = o.get("message") or {}
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
            ids = [b.get("tool_use_id") for b in c if isinstance(b, dict) and b.get("type") == "tool_result"]
            name = tool_name_of(ids[0]) if ids else None
            parts = []
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    parts.append(block_text(b))
                elif isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b.get("text", ""))
            out.update(type="tool_result", said_by="tool:%s" % (name or "unknown"), text="\n".join(parts), ref=ids[0] if ids else None)
            fields["tool"] = name
            fields["tool_use_ids"] = ids
            if any(isinstance(b, dict) and b.get("is_error") for b in c):
                fields["is_error"] = True
            return out
        text = c if isinstance(c, str) else "\n".join(block_text(b) for b in (c or []))
        kind, who = _user_kind(o, text, sidechain)
        if isinstance(o.get("origin"), dict):
            fields["origin"] = o["origin"].get("kind")
        if o.get("isMeta"):
            fields["isMeta"] = True
        out.update(type=kind, said_by=who, text=text)
        return out
    if t == "assistant":
        m = o.get("message") or {}
        c = m.get("content") or []
        if isinstance(c, str):
            c = [{"type": "text", "text": c}]
        fields["model"] = m.get("model")
        fields["message_id"] = m.get("id")
        who = "Claude (subagent)" if sidechain else "Claude"
        if o.get("isApiErrorMessage"):
            out.update(type="api_error", said_by=who, text="\n".join(block_text(b) for b in c))
            return out
        uses = [b for b in c if isinstance(b, dict) and b.get("type") == "tool_use"]
        texts = [b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"]
        if uses:
            fields["tool"] = uses[0].get("name")
            out.update(type="tool_call", said_by=who, ref=uses[0].get("id"),
                       text="\n".join(texts + ["Called %s" % block_text(b) for b in uses]))
        elif any(x.strip() for x in texts):
            out.update(type="claude_text", said_by=who, text="\n".join(texts))
        elif any(isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking") for b in c):
            th = "\n".join(b.get("thinking", "") or "" for b in c if isinstance(b, dict) and b.get("type") == "thinking")
            out.update(type="reasoning", said_by=who + " (thinking)", text=th if th.strip() else None)
        else:
            out.update(type="app", said_by=who)
        return out
    if t == "attachment":
        a = o.get("attachment") or {}
        at = a.get("type")
        fields["attachment"] = at
        rt = rendered_text(o)
        if at == "queued_command":
            origin = (a.get("origin") or {}).get("kind")
            fields["origin"] = origin
            if origin == "human":
                p = a.get("prompt")
                out.update(type="ryan_message", said_by="Ryan", text=p if isinstance(p, str) else rt,
                           happened_at=norm_ts(a.get("timestamp") or o.get("timestamp")))
            else:
                out.update(type="notification", said_by="system (%s)" % (origin or "queued"), text=rt or a.get("prompt"))
            return out
        if at == "instructions":
            out.update(type="instructions", said_by="instruction files (CLAUDE.md)", text=rt)
            fields["files"] = [f.get("path") for f in a.get("files", []) if isinstance(f, dict)]
            return out
        if at == "file":
            out.update(type="file_attachment", said_by="file read into context", text=rt)
            fields["file"] = a.get("filename")
            return out
        if at == "edited_text_file":
            out.update(type="file_changed", said_by="system", text=rt)
            fields["file"] = a.get("filename")
            return out
        if at == "hook_additional_context":
            content = a.get("content")
            joined = "\n".join(content) if isinstance(content, list) else (content or "")
            fields["hook"] = a.get("hookName")
            if MEMORY_MARKER in joined:
                out.update(type="memory_injection", said_by="memory", text=joined)
            else:
                out.update(type="hook_context", said_by="hook:%s" % a.get("hookName"), text=rt or joined)
            return out
        if at in NOISE_ATTACHMENTS:
            out.update(type="app", text=None)
            return out
        out.update(type="attachment_other", text=rt)
        return out
    if t == "system":
        st = o.get("subtype")
        fields["subtype"] = st
        if st == "compact_boundary":
            out.update(type="compact_boundary")
            return out
        if st == "local_command":
            out.update(type="command", said_by="Ryan", text=o.get("content") if isinstance(o.get("content"), str) else None)
            return out
        out.update(type="app", text=o.get("content") if isinstance(o.get("content"), str) else None)
        return out
    return out


TEXT_EXT = {".md", ".txt", ".json", ".jsonl", ".py", ".js", ".ts", ".tsx", ".css", ".html", ".csv", ".yaml",
            ".yml", ".toml", ".ini", ".gd", ".cs", ".java", ".go", ".rs", ".sh", ".ps1", ".bat", ".xml", ".sql", ".log"}


def decode_text(data, path):
    ext = os.path.splitext(path)[1].lower()
    if b"\x00" in data[:8192] and ext not in TEXT_EXT:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        if ext in TEXT_EXT:
            return data.decode("utf-8", "replace")
        return None


def classify_file(path, data, entry_point, projects_dir, claude_dir):
    norm = path.replace("\\", "/")
    text = decode_text(data, path)
    pj = projects_dir.replace("\\", "/")
    out = {"type": "app_file", "said_by": None, "text": text, "fields": {"path": path}}
    if entry_point == "inbox":
        out.update(type="inbox_file" if text is not None else "binary", said_by="Ryan (inbox)")
        return out
    if norm.startswith(pj):
        if "/memory/" in norm and norm.endswith(".md"):
            out.update(type="memory_file", said_by="Claude's auto-memory file")
        elif "/tool-results/" in norm:
            out.update(type="tool_output_file", said_by="tool output file")
        else:
            out.update(type="app_file")
        return out
    if "/file-history/" in norm:
        out.update(type="file_backup", said_by="file backup")
        return out
    if norm.endswith("/CLAUDE.md"):
        out.update(type="instructions_file", said_by="instruction file")
        return out
    return out
