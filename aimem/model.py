"""Model calls: the only place memory asks a model to understand something.

Calls go through the Claude Code CLI (claude.exe) in print mode, with no
tools, no MCP servers, no hooks or CLAUDE.md (safe mode), no saved session,
and a JSON schema for the answer. Every call is recorded whole in the
model_calls table: system prompt, prompt, answer, time, tokens, error.
"""
import json
import os
import subprocess
import time

from . import config, log
from .db import now


class ModelUnavailable(Exception):
    """The model cannot be reached (logged out, usage limit, CLI missing). The queue waits."""


def _record(conn, step, model, system, prompt, response, ok, error, ms, usage, input_ids):
    try:
        cur = conn.execute(
            "INSERT INTO model_calls(at, step, model, system, prompt, response, ok, error, ms, usage, input_ids)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (now(), step, model, system, prompt, response, 1 if ok else 0, error, ms,
             json.dumps(usage) if usage is not None else None, json.dumps(input_ids or [])))
        return cur.lastrowid
    except Exception as e:
        log.event(None, "log_error", "could not record a model call", {"error": repr(e)}, ok=False)
        return None


def call(conn, step, model, system, prompt, schema, input_ids=None, timeout=None, effort=None):
    """Returns (answer dict, model_call id). Raises ModelUnavailable or RuntimeError."""
    exe = config.claude_exe()
    if not exe:
        raise ModelUnavailable("Claude Code CLI not found")
    os.makedirs(config.WORKER_DIR, exist_ok=True)
    cmd = [exe, "-p", "--safe-mode", "--no-session-persistence", "--model", model] + (
        ["--effort", effort] if effort else []) + ["--tools", "",
           "--strict-mcp-config", "--setting-sources", "project", "--system-prompt", system,
           "--output-format", "json", "--json-schema", json.dumps(schema)]
    env = dict(os.environ)
    env["AI_MEM_CHILD"] = "1"
    # Each call's prompt is new, so a cache write (billed at twice the input price for the
    # 1-hour cache) is never read back. Measured 2026-11-12: 6-9k tokens written per call.
    env["DISABLE_PROMPT_CACHING"] = "1"
    last_err = None
    for attempt in range(2):
        t = log.timer()
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout or config.MODEL_TIMEOUT_S,
                               cwd=config.WORKER_DIR, env=env,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            last_err = "timed out after %ss" % (timeout or config.MODEL_TIMEOUT_S)
            _record(conn, step, model, system, prompt, None, False, last_err, t.ms(), None, input_ids)
            continue
        ms = t.ms()
        try:
            o = json.loads(r.stdout)
        except Exception:
            last_err = "unreadable CLI output (rc %s): %s %s" % (r.returncode, r.stdout[:300], r.stderr[:300])
            _record(conn, step, model, system, prompt, r.stdout, False, last_err, ms, None, input_ids)
            continue
        usage = {k: v for k, v in (o.get("usage") or {}).items() if isinstance(v, (int, float))}
        usage["cost_usd"] = o.get("total_cost_usd")
        usage["api_ms"] = o.get("duration_api_ms")
        if o.get("is_error"):
            msg = str(o.get("result") or o.get("subtype"))
            cid = _record(conn, step, model, system, prompt, r.stdout, False, msg, ms, usage, input_ids)
            low = msg.lower()
            if o.get("api_error_status") == 401 or "authenticate" in low or "log in" in low or "login" in low:
                raise ModelUnavailable("CLI login needed: " + msg)
            if "usage limit" in low or "rate limit" in low or "limit reached" in low:
                raise ModelUnavailable("usage limit: " + msg)
            last_err = msg
            time.sleep(5)
            continue
        data = o.get("structured_output")
        if data is None:
            try:
                txt = (o.get("result") or "").strip()
                if txt.startswith("```"):
                    txt = txt.strip("`")
                    txt = txt[txt.find("{"):]
                data = json.loads(txt)
            except Exception:
                data = None
        cid = _record(conn, step, model, system, prompt, r.stdout, data is not None,
                      None if data is not None else "no structured answer", ms, usage, input_ids)
        if data is None:
            last_err = "no structured answer"
            continue
        return data, cid
    raise RuntimeError(last_err or "model call failed")
