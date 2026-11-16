"""Every setting in one place. Each number says where it comes from.

Paths, the sources the store follows, the models, and the starting values
that search and injection use. Values marked "starting value" are choices to
be measured on real use; the logs record every decision they drive.
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DB_PATH = os.path.join(DATA, "memory.db")
LOG_DIR = os.path.join(DATA, "logs")
INBOX = os.path.join(ROOT, "inbox")
WORKER_DIR = os.path.join(DATA, "worker")        # empty folder model calls run in
WORKER_LOCK = os.path.join(DATA, "worker.lock")
SPAWN_STAMP = os.path.join(DATA, "worker.spawned")
VEC_CACHE = os.path.join(DATA, "vectors_cache.npz")
EMBED_NPY = os.path.join(DATA, "potion32m_fp16.npy")

HOME = os.path.expanduser("~")
CLAUDE_DIR = os.path.join(HOME, ".claude")
PROJECTS = os.path.join(CLAUDE_DIR, "projects")

# What the store follows. Everything found under these is kept raw, byte for
# byte. Left out on purpose: credentials, settings, shell snapshots (they can
# hold secrets), telemetry, caches and plugin code.
SOURCE_DIRS = [
    (PROJECTS, "claude-code"),
    (os.path.join(CLAUDE_DIR, "file-history"), "claude-code"),
    (INBOX, "inbox"),
]
SOURCE_FILES = [
    (os.path.join(CLAUDE_DIR, "CLAUDE.md"), "claude-code"),
    (os.path.join(CLAUDE_DIR, "history.jsonl"), "claude-code"),
]
# Basis: the baseline system's capture limit (docs/baseline-memory-graph.md).
MAX_FILE_BYTES = 50 * 1024 * 1024

# Model calls run through the Claude Code CLI on Ryan's subscription.
CLAUDE_EXE_CANDIDATES = [
    os.environ.get("AI_MEM_CLAUDE_EXE", ""),
]
# Full model ids: the installed CLI (2.1.187) maps the short names to older models
# (measured 2026-11-11: "haiku" ran Haiku 4.5 at 84 s a batch; claude-haiku-5-5 took 14 s).
PERCEIVE_MODEL = "claude-haiku-5-5"    # bulk reading: fast and cheap
STATE_MODEL = "claude-sonnet-5-5"      # judging what holds now and how broadly a rule applies
MODEL_TIMEOUT_S = 600
# Effort for perceiving. Measured 2026-11-12 on the same 20-input batch: default effort
# wrote 98 fragments with 23,885 output tokens (two thirds of them thinking); low effort
# wrote 24-40 complete perceptions with 4,500-7,800 tokens, covering the same points.
PERCEIVE_EFFORT = "low"
STATE_EFFORT = None          # state judges how broadly a rule applies: default effort

# Memory's own spending, in the CLI's API-equivalent dollars (total_cost_usd of each call).
# Basis: at low effort a perceive batch of ~16k characters costs about $0.18 and a state
# call about $0.13, so $20 a day covers about 1M characters of new content: Ryan's median
# active day (measured 2026-11-11: 270k tokens). Above budget, tool output waits in the
# queue and only what people and sessions said is perceived; above twice the budget,
# everything waits. The 5-hour budget keeps a burst from eating Ryan's 5-hour window.
# Ryan, 2026-11-11: "Please don't go past 80% of weekly usage". Measured 2026-11-12: $3.89
# of memory's calls, together with a heavy Claude session running at the same time, moved
# his 5-hour use 9% -> 13% and his weekly use 35% -> 36%. Taken as an upper bound, $15 a
# day is at most about 27% of a week, which with his own use (about 40% a week) stays
# under 80%. Remeasure when no other session runs (see STATE.md).
# Ryan, 2026-11-12: the 80% cap was "just for you building the memory system and processing ALL
# of my claude history". So live perceiving is never capped; only the history backfill is paced.
MEMORY_BUDGET_24H_USD = 15.0
MEMORY_BUDGET_5H_USD = 6.0
WORKER_SYNC_EVERY_S = 10
STATE_FLUSH_AT = 80          # tool-only perceptions waiting before a state step runs anyway
STATE_CHUNK = 150            # perceptions per state call

# Hooks. Basis: Ryan's stated tolerance of about 3 s for the reply to start
# (docs/baseline-memory-graph.md), minus Python start-up.
HOOK_BUDGET_S = 2.6
HOOK_BUSY_MS = 500           # a hook waits at most this long to write; reads never wait (WAL)
SPOOL = os.path.join(DATA, "spool")   # hook records that could not be written at once; the worker loads them
# Basis: Claude Code caps a hook's additionalContext at 10,000 characters
# (docs/baseline-memory-graph.md); kept under it with room for the header.
INJECT_MAX_CHARS = 7000

# Perceiving. Basis: Haiku's 200k-token window; a call stays far below it so
# the answer has room and one failure costs little.
PERCEIVE_BATCH_CHARS = 24000
PERCEIVE_PART_CHARS = 16000      # longer inputs are perceived in parts
# Context shown to perceive for resolving references. Measured 2026-11-12: with 30 earlier
# perceptions and 3,000 raw characters, a one-message batch still sent 17-21k characters.
PERCEIVE_CONTEXT_PERCEPTIONS = 12
PERCEIVE_RAW_TAIL_CHARS = 1500
# A live batch waits until its oldest input is this old (or the batch is full), so a message
# and the reply to it are perceived together in one call instead of one call each.
LIVE_SETTLE_S = 120
PERCEIVE_MAX_INPUTS = 40

# Search. Starting values, to be measured.
VEC_TOP = 100                # candidates per cue from the vector side
BM25_TOP = 50                # candidates per cue from the keyword side
# A match must stand out from the cue's own 100 best scores: at least Z_MIN spreads above
# their mean, and at least COS_FLOOR; a cosine of COS_SURE or more always matches.
# Measured 2026-11-12 on six real cues: this kept the judged-relevant memories and let in
# fewer unrelated ones than the fixed 0.45 cutoff, which missed both relevant memories for
# "should the store use postgres or sqlite" (cosines 0.41 and 0.40) and let unrelated ones
# in for long technical cues (unrelated texts reached 0.51-0.55).
Z_MIN = 2.5
COS_FLOOR = 0.35
COS_SURE = 0.60
COS_THRESHOLD = COS_FLOOR    # kept for older callers
# A state's heading (its subject, e.g. "When Claude replies to Ryan in chat") against
# the moment: the matching heading scored 0.92, every other heading about 0.4.
SUBJECT_THRESHOLD = 0.60
BM25_KEEP_RATIO = 0.80       # keyword-only match must reach this share of the cue's best keyword score
BM25_MIN_TERMS = 2           # and share at least this many distinct cue words
MAX_MATCHES_PER_CUE = 6
STATE_CANDIDATES_PER_CUE = 6

# Chaining. Starting values: a strong match reaches its neighbours, a weak one
# does not. Strength is a match's margin over the threshold, in units of the
# cue's spread of scores.
LINK_ENERGY = {"stream": 1.0, "relies": 0.5, "situation": 1.0, "state": 0.0, "direct": 0.5}
CHAIN_MAX_NODES = 8

# Raw passages: searched only for inputs not yet perceived.
PASSAGE_CHARS = 1200
PASSAGE_MAX = 2

EMBED_MODEL_DIRS = glob.glob(os.path.join(
    HOME, ".cache", "huggingface", "hub", "models--minishlab--potion-base-32M", "snapshots", "*"))


def claude_exe():
    for p in CLAUDE_EXE_CANDIDATES:
        if os.path.exists(p):
            return p
    import shutil
    return shutil.which("claude.exe") or shutil.which("claude")


def pythonw():
    d = os.path.dirname(sys.executable)
    p = os.path.join(d, "pythonw.exe")
    return p if os.path.exists(p) else sys.executable
