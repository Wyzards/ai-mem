"""Commands. Each one prints what it did; every step also writes the logs.

  sync                     store everything new (raw, byte for byte)
  status                   counts, queue, worker, model, last errors
  work [--limit N]         run the queue now in this window (perceive, then state)
  backfill [opts]          put stored history into the queue (see below)
  search "text" [--raw]    run recall on a text and show every candidate and decision
  source ID [--raw]        where a memory came from: s12 state, p34 perception, r56 raw passage, i78 input
  raw ID                   the readable text of input i78 (or the input behind p34 / r56)
  transcript SESSION       a whole session, readable, from the raw store
  why [N|last]             the full trace of an injection: cues, candidates, scores, cuts
  log [-n N] [--kind K]    recent events
  states [--all]           states that hold now (or every state with history)
  check                    integrity checks
  live                     set the time from which new inputs are queued (now)
  hook EVENT               entry for Claude Code hooks (reads the hook's JSON on stdin)
  dashboard                local read-only dashboard at http://localhost:8765
  statusline               Claude Code status line: memory's spend for this session and its share
"""
import json
import sys


def _out(s=""):
    try:
        sys.stdout.write(s + "\n")
    except UnicodeEncodeError:
        sys.stdout.buffer.write((s + "\n").encode("utf-8", "replace"))


def _args(argv):
    pos, opts = [], {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            k = a[2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                opts[k] = argv[i + 1]
                i += 2
                continue
            opts[k] = True
        elif a == "-n" and i + 1 < len(argv):
            opts["n"] = argv[i + 1]
            i += 2
            continue
        else:
            pos.append(a)
        i += 1
    return pos, opts


def main(argv):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not argv or argv[0] in ("help", "-h", "--help"):
        _out(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "hook":
        from . import hooks
        return hooks.main(rest)
    if cmd == "statusline":
        from . import dashboard
        dashboard.statusline()
        return 0
    if cmd == "dashboard":
        from . import dashboard
        dashboard.serve()
        return 0
    from . import db
    conn = db.connect()
    pos, opts = _args(rest)
    if cmd == "sync":
        from . import store
        st = store.sync(conn)
        _out(json.dumps(st, indent=1))
        return 0
    if cmd == "live":
        from .db import now, set_meta, get_meta
        from . import log
        if get_meta(conn, "live_since") and not opts.get("reset"):
            _out("live since " + get_meta(conn, "live_since"))
            return 0
        t = now()
        set_meta(conn, "live_since", t)
        log.event(conn, "live", "new inputs from now on are queued for perceiving", {"live_since": t})
        _out("live since " + t)
        return 0
    from . import report
    return report.run(conn, cmd, pos, opts, _out)
