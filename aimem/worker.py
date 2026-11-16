"""The queue worker: perceive, then state, batch by batch, in arrival order.

One worker runs at a time (a lock file). Hooks start it after each event if
it is not running; it stores anything new, works through the queue, and exits
after a short idle spell. If the model cannot be reached (logged out, usage
limit) it stops and leaves the queue as it is; nothing is lost, the next
start picks up where it stopped.
"""
import json
import msvcrt
import os
import subprocess
import time
import traceback

from . import config, embed, extract, log, model, perceive, state, store
from .db import now, set_meta, tx


class Lock:
    def __init__(self, path):
        self.path = path
        self.f = None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.f = open(self.path, "a+b")
        try:
            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            self.f.close()
            self.f = None
            return False

    def release(self):
        if self.f:
            try:
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self.f.close()
            self.f = None


def running():
    lk = Lock(config.WORKER_LOCK)
    if lk.acquire():
        lk.release()
        return False
    return True


def spawn(conn, reason):
    """Start a detached worker unless one is running or one was started in the last few seconds."""
    if os.environ.get("AI_MEM_CHILD") or running():
        return False
    try:
        if time.time() - os.path.getmtime(config.SPAWN_STAMP) < 5:
            return False
    except OSError:
        pass
    with open(config.SPAWN_STAMP, "w") as f:
        f.write(now())
    cmd = [config.pythonw(), "-I", os.path.join(config.ROOT, "mem.py"), "work", "--quiet"]
    base = (getattr(subprocess, "DETACHED_PROCESS", 0x8) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
    kw = dict(cwd=config.ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    try:
        subprocess.Popen(cmd, creationflags=base | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0x01000000), **kw)
        how = "detached, outside the hook's job"
    except OSError:
        subprocess.Popen(cmd, creationflags=base, **kw)
        how = "detached"
    log.event(conn, "worker_spawn", "started the queue worker", {"reason": reason, "how": how})
    return True


MODE_TEXT = {"full": "everything, history included", "live": "live work only (history paused for today)"}


def _ago(hours):
    import datetime
    d = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)
    return d.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def spend(conn, hours):
    r = conn.execute("SELECT COALESCE(SUM(json_extract(usage, '$.cost_usd')), 0) FROM model_calls WHERE at >= ?",
                     (_ago(hours),)).fetchone()
    return float(r[0] or 0)


def budget_mode(conn):
    """Live work is never capped. Processing old history is paced: 'live' once memory's calls in
    the last 24 h / 5 h pass the pace below, so the bulk backfill can't eat Ryan's weekly limit."""
    s24, s5 = spend(conn, 24), spend(conn, 5)
    worst = max(s24 / config.MEMORY_BUDGET_24H_USD, s5 / config.MEMORY_BUDGET_5H_USD)
    mode = "full" if worst < 1 else "live"
    return mode, {"spend_24h": round(s24, 2), "budget_24h": config.MEMORY_BUDGET_24H_USD,
                  "spend_5h": round(s5, 2), "budget_5h": config.MEMORY_BUDGET_5H_USD}


def _queue_state(conn, ids, ctx):
    ts = now()
    with tx(conn):
        conn.executemany("INSERT OR IGNORE INTO state_queue(node_id, context_id, added_at) VALUES (?,?,?)",
                         [(i, ctx, ts) for i in ids])


def _flush_state(conn, ctx, stats, say):
    """Run the state step on the perceptions of this context that it has not seen yet."""
    while True:
        rows = conn.execute("SELECT node_id FROM state_queue WHERE context_id IS ? AND tries < 2 ORDER BY node_id LIMIT ?",
                            (ctx, config.STATE_CHUNK)).fetchall()
        if not rows:
            return
        ids = [r[0] for r in rows]
        say("state step on %d perceptions from %s" % (len(ids), ctx))
        try:
            state.run(conn, ids, ctx)
        except model.ModelUnavailable:
            raise
        except Exception:
            with tx(conn):
                conn.executemany("UPDATE state_queue SET tries = tries + 1 WHERE node_id=?", [(i,) for i in ids])
            log.event(conn, "state_failed", "state step failed; will try once more", {"perceptions": ids,
                      "error": traceback.format_exc()[-2000:]}, ok=False, context_id=ctx)
            return
        stats["state_runs"] += 1
        with tx(conn):
            conn.executemany("DELETE FROM state_queue WHERE node_id=?", [(i,) for i in ids])


def _flush_all(conn, stats, say):
    for r in conn.execute("SELECT DISTINCT context_id FROM state_queue WHERE tries < 2").fetchall():
        _flush_state(conn, r[0], stats, say)


def work(conn, limit=None, quiet=True, idle_wait_s=20, say=None):
    """Perceive the queue, then run the state step: right away for a batch that holds something
    said, otherwise once enough tool output has gathered or the queue is empty."""
    say = say or (lambda s: None)
    lock = Lock(config.WORKER_LOCK)
    if not lock.acquire():
        say("another worker is already running")
        return {"started": False}
    t = log.timer()
    stats = {"batches": 0, "perceptions": 0, "state_runs": 0, "failed": 0, "stopped": None}
    log.event(conn, "worker_start", "queue worker started", {"pid": os.getpid(), "limit": limit})
    last_mode = None
    code_dir = os.path.dirname(os.path.abspath(__file__))

    def code_time():
        return max(os.path.getmtime(os.path.join(code_dir, f)) for f in os.listdir(code_dir) if f.endswith(".py"))

    started_code, last_code_check = code_time(), time.time()
    try:
        idle_since, last_sync = None, 0.0
        while True:
            if time.time() - last_code_check > 15:
                last_code_check = time.time()
                if code_time() > started_code:
                    # The code was changed while this worker ran: stop and start a fresh worker on it.
                    stats["stopped"] = "code changed"
                    break
            if time.time() - last_sync > config.WORKER_SYNC_EVERY_S:
                store.sync(conn)
                try:
                    from .inject import load_spool
                    n = load_spool(conn)
                    if n:
                        log.event(conn, "spool", "loaded %d injection records written while the database was busy" % n)
                except Exception:
                    log.event(conn, "spool_error", "could not load the spool", {"error": traceback.format_exc()[-1500:]}, ok=False)
                last_sync = time.time()
            mode, spent = budget_mode(conn)
            if mode != last_mode:
                if last_mode is not None or mode != "full":
                    log.event(conn, "budget", "perceiving %s" % MODE_TEXT[mode], spent)
                    say("budget: perceiving %s %s" % (MODE_TEXT[mode], spent))
                last_mode = mode
            try:
                ctx, batch = perceive.next_batch(conn, mode)
                if not batch:
                    _flush_all(conn, stats, say)
                    if idle_since is None:
                        idle_since = time.time()
                    settling = conn.execute("SELECT COUNT(*) FROM input_info WHERE perceive='pending'"
                                            " AND COALESCE(perceive_note,'') != 'backfill'").fetchone()[0]
                    if time.time() - idle_since > idle_wait_s + (config.LIVE_SETTLE_S if settling else 0):
                        stats["stopped"] = "queue empty" if mode == "full" else "live queue empty; history paused for today"
                        break
                    time.sleep(2)
                    continue
                idle_since = None
                ids = [r["input_id"] for r in batch]
                say("perceiving %d inputs from %s" % (len(ids), ctx))
                try:
                    new_ids = perceive.run_batch(conn, ctx, batch)
                except model.ModelUnavailable:
                    raise
                except Exception as e:
                    stats["failed"] += len(ids)
                    with tx(conn):
                        conn.executemany("UPDATE input_info SET perceive='failed', perceive_note=? WHERE input_id=?",
                                         [(json.dumps({"error": repr(e)[:500]}), i) for i in ids])
                    log.event(conn, "perceive_failed", "perceiving failed; inputs marked failed", {"inputs": ids,
                              "error": traceback.format_exc()[-2000:]}, ok=False, context_id=ctx)
                    say("failed: %r" % e)
                    continue
                set_meta(conn, "model_unavailable", "")
                stats["batches"] += 1
                percs = [r[0] for r in conn.execute(
                    "SELECT id FROM nodes WHERE id IN (%s) AND kind='perception'" % ",".join("?" * len(new_ids)),
                    new_ids)] if new_ids else []
                stats["perceptions"] += len(percs)
                if percs:
                    _queue_state(conn, percs, ctx)
                said = any(r["type"] in extract.SAID_TYPES for r in batch)
                waiting = conn.execute("SELECT COUNT(*) FROM state_queue WHERE context_id IS ? AND tries < 2",
                                       (ctx,)).fetchone()[0]
                if waiting and (said or waiting >= config.STATE_FLUSH_AT):
                    _flush_state(conn, ctx, stats, say)
            except model.ModelUnavailable as e:
                set_meta(conn, "model_unavailable", "%s %s" % (now(), e))
                log.event(conn, "model_unavailable", "stopped: the model cannot be reached", {"error": str(e)}, ok=False)
                stats["stopped"] = "model unavailable: %s" % e
                say(stats["stopped"])
                break
            if stats["batches"] and stats["batches"] % 10 == 0:
                embed.rebuild_cache(conn)
            if limit and stats["batches"] >= limit:
                stats["stopped"] = "limit reached"
                break
        if stats["batches"]:
            embed.rebuild_cache(conn)
    finally:
        stats["ms"] = t.ms()
        log.event(conn, "worker_stop", "queue worker stopped: %s" % stats.get("stopped"), stats, ms=stats["ms"])
        lock.release()
    if stats.get("stopped") == "code changed":
        try:
            os.remove(config.SPAWN_STAMP)
        except OSError:
            pass
        spawn(conn, "code changed")
    return stats
