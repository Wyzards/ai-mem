"""The database: one SQLite file holding the raw store, the graph and the logs.

Rules the database itself enforces (triggers), so no code path can break them:
- inputs (the raw store) are add-only: no update, no delete;
- perceptions never change and no node is ever deleted;
- state versions are add-only.
"""
import datetime
import os
import sqlite3

from . import config

SCHEMA_VERSION = 2

SCHEMA = r"""
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS sources(
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  entry_point TEXT NOT NULL,
  kind TEXT NOT NULL,                 -- 'lines' (append-only, one input per line) or 'file' (one input per version)
  synced_bytes INTEGER NOT NULL DEFAULT 0,
  tail_hash TEXT,                     -- sha256 of the last <=4096 bytes taken, to notice a rewritten file
  last_hash TEXT,                     -- 'file': hash of the last version stored
  last_mtime REAL,
  last_size INTEGER,
  state TEXT,                         -- JSON kept while reading: compactions seen, session, agent
  first_seen TEXT NOT NULL,
  last_synced TEXT
);

-- The raw store. Bytes exactly as they were, never changed.
CREATE TABLE IF NOT EXISTS inputs(
  id INTEGER PRIMARY KEY,             -- arrival number
  source_id INTEGER NOT NULL REFERENCES sources(id),
  offset INTEGER NOT NULL,            -- byte offset in the source; -1 for a whole-file version
  length INTEGER NOT NULL,
  hash TEXT NOT NULL,
  raw BLOB NOT NULL,
  stored_at TEXT NOT NULL,
  UNIQUE(source_id, offset, hash)
);
CREATE TRIGGER IF NOT EXISTS inputs_add_only_u BEFORE UPDATE ON inputs
  BEGIN SELECT RAISE(ABORT, 'inputs are add-only'); END;
CREATE TRIGGER IF NOT EXISTS inputs_add_only_d BEFORE DELETE ON inputs
  BEGIN SELECT RAISE(ABORT, 'inputs are add-only'); END;

-- What the entry point says about each input. Derived from the raw bytes, so
-- it can be rebuilt; perceive is the input's place in the queue.
CREATE TABLE IF NOT EXISTS input_info(
  input_id INTEGER PRIMARY KEY REFERENCES inputs(id),
  entry_point TEXT,
  session_id TEXT,
  context_id TEXT,
  type TEXT NOT NULL,
  said_by TEXT,
  happened_at TEXT,
  uuid TEXT,
  ref TEXT,                           -- tool_use_id for tool calls and results
  fields TEXT,                        -- JSON
  text TEXT,                          -- readable text, as the model saw it
  text_hash TEXT,
  perceive TEXT NOT NULL,             -- pending / history / skip / duplicate / done / failed
  perceive_note TEXT,
  perceived_at TEXT
);
CREATE INDEX IF NOT EXISTS input_info_ctx ON input_info(context_id, input_id);
CREATE INDEX IF NOT EXISTS input_info_perceive ON input_info(perceive, input_id);
CREATE INDEX IF NOT EXISTS input_info_ref ON input_info(ref);
CREATE INDEX IF NOT EXISTS input_info_uuid ON input_info(uuid);
CREATE INDEX IF NOT EXISTS input_info_texthash ON input_info(text_hash);
CREATE INDEX IF NOT EXISTS input_info_session ON input_info(session_id, input_id);
CREATE INDEX IF NOT EXISTS input_info_perceive_time ON input_info(perceive, happened_at);

-- Work queue for the state step: perceptions not yet seen by it (rows leave when done).
CREATE TABLE IF NOT EXISTS state_queue(
  node_id INTEGER PRIMARY KEY,
  context_id TEXT,
  added_at TEXT NOT NULL,
  tries INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions(
  session_id TEXT PRIMARY KEY,
  title TEXT,
  cwd TEXT,
  first_at TEXT,
  last_at TEXT
);

-- Raw passages: the readable text cut into pieces for keyword search.
CREATE TABLE IF NOT EXISTS passages(
  id INTEGER PRIMARY KEY,
  input_id INTEGER NOT NULL REFERENCES inputs(id),
  start INTEGER NOT NULL,
  end INTEGER NOT NULL,
  text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS passages_input ON passages(input_id);
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
  text, content='passages', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS passages_ai AFTER INSERT ON passages
  BEGIN INSERT INTO passages_fts(rowid, text) VALUES (new.id, new.text); END;

-- The graph. A memory is a node: a perception, a situation (a perception of
-- the moment) or a state.
CREATE TABLE IF NOT EXISTS nodes(
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,                 -- perception / situation / state
  text TEXT NOT NULL,                 -- for a state: its current version
  input_id INTEGER REFERENCES inputs(id),
  part INTEGER,
  quote TEXT,
  said_by TEXT,
  said_at TEXT,
  context_id TEXT,
  session_id TEXT,
  subject TEXT,                       -- state: what it is about
  status TEXT,                        -- state: holds / held / ended
  created_at TEXT NOT NULL,
  extra TEXT
);
CREATE INDEX IF NOT EXISTS nodes_ctx ON nodes(context_id, id);
CREATE INDEX IF NOT EXISTS nodes_input ON nodes(input_id);
CREATE INDEX IF NOT EXISTS nodes_kind ON nodes(kind, status);
CREATE TRIGGER IF NOT EXISTS nodes_no_delete BEFORE DELETE ON nodes
  BEGIN SELECT RAISE(ABORT, 'memories are never deleted'); END;
CREATE TRIGGER IF NOT EXISTS nodes_perceptions_fixed BEFORE UPDATE OF text, input_id, said_by, said_at ON nodes
  WHEN OLD.kind != 'state'
  BEGIN SELECT RAISE(ABORT, 'perceptions never change'); END;
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
  text, content='nodes', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes
  BEGIN INSERT INTO nodes_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE OF text ON nodes
  BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO nodes_fts(rowid, text) VALUES (new.id, new.text);
  END;

CREATE TABLE IF NOT EXISTS state_versions(
  id INTEGER PRIMARY KEY,
  state_id INTEGER NOT NULL REFERENCES nodes(id),
  n INTEGER NOT NULL,
  op TEXT NOT NULL,                   -- new / update / end
  text TEXT NOT NULL,
  rests_on TEXT NOT NULL,             -- JSON list of node ids
  confidence TEXT,
  own_conclusion INTEGER NOT NULL DEFAULT 0,
  on_whose_word TEXT,                 -- derived by code from what it rests on
  as_of TEXT,                         -- when the newest thing it rests on was said
  status TEXT NOT NULL,
  reason TEXT,
  written_at TEXT NOT NULL,
  model_call_id INTEGER,
  UNIQUE(state_id, n)
);
CREATE TRIGGER IF NOT EXISTS sv_add_only_u BEFORE UPDATE ON state_versions
  BEGIN SELECT RAISE(ABORT, 'state versions are add-only'); END;
CREATE TRIGGER IF NOT EXISTS sv_add_only_d BEFORE DELETE ON state_versions
  BEGIN SELECT RAISE(ABORT, 'state versions are add-only'); END;

CREATE TABLE IF NOT EXISTS links(
  id INTEGER PRIMARY KEY,
  start INTEGER NOT NULL REFERENCES nodes(id),
  "end" INTEGER NOT NULL REFERENCES nodes(id),
  kind TEXT NOT NULL,                 -- stream / relies / situation / state / direct
  energy REAL NOT NULL,
  created_at TEXT NOT NULL,
  note TEXT,
  UNIQUE(start, "end", kind)
);
CREATE INDEX IF NOT EXISTS links_start ON links(start);
CREATE INDEX IF NOT EXISTS links_end ON links("end");
CREATE TRIGGER IF NOT EXISTS links_no_delete BEFORE DELETE ON links
  BEGIN SELECT RAISE(ABORT, 'links are never deleted'); END;

CREATE TABLE IF NOT EXISTS vectors(
  node_id INTEGER PRIMARY KEY REFERENCES nodes(id),
  seq INTEGER NOT NULL,
  v BLOB NOT NULL                     -- float16, 512 values
);
CREATE INDEX IF NOT EXISTS vectors_seq ON vectors(seq);

-- Records: what was put into which context, and why.
CREATE TABLE IF NOT EXISTS injections(
  id INTEGER PRIMARY KEY,
  at TEXT NOT NULL,
  hook TEXT,
  session_id TEXT,
  context_id TEXT,
  cwd TEXT,
  cue_text TEXT,
  node_ids TEXT,                      -- JSON
  passage_ids TEXT,                   -- JSON
  text TEXT,
  ms INTEGER,
  trace TEXT                          -- JSON: every candidate, score and decision
);
CREATE INDEX IF NOT EXISTS injections_ctx ON injections(context_id);

-- Logs.
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY,
  at TEXT NOT NULL,
  kind TEXT NOT NULL,
  ok INTEGER NOT NULL DEFAULT 1,
  ms INTEGER,
  input_id INTEGER,
  context_id TEXT,
  summary TEXT,
  detail TEXT
);
CREATE INDEX IF NOT EXISTS events_kind ON events(kind, id);
CREATE TABLE IF NOT EXISTS model_calls(
  id INTEGER PRIMARY KEY,
  at TEXT NOT NULL,
  step TEXT NOT NULL,
  model TEXT,
  system TEXT,
  prompt TEXT,
  response TEXT,
  ok INTEGER,
  error TEXT,
  ms INTEGER,
  usage TEXT,
  input_ids TEXT
);
"""


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def connect(busy_ms=60000):
    os.makedirs(config.DATA, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=busy_ms / 1000.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=%d" % int(busy_ms))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ver = None
    try:
        r = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        ver = int(r[0]) if r else None
    except sqlite3.OperationalError:
        ver = None
    if ver != SCHEMA_VERSION:
        conn.executescript(SCHEMA)
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    return conn


class tx:
    """BEGIN IMMEDIATE ... COMMIT, or ROLLBACK on error."""

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, et, ev, tb):
        if et is None:
            self.conn.execute("COMMIT")
        else:
            self.conn.execute("ROLLBACK")
        return False


def get_meta(conn, key, default=None):
    r = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return r[0] if r else default


def set_meta(conn, key, value):
    conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, str(value)))
