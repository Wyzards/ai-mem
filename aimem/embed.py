"""Vectors for similarity search, from the potion-base-32M static model.

Why this model: it is already on this PC (no download), runs on the CPU in
milliseconds, and loads fast enough for a hook. It averages word vectors, so
it finds paraphrases that keyword search misses but is weaker than a full
sentence model; keyword search (FTS5) runs beside it.

The encoding matches model2vec's StaticModel.encode exactly: tokens without
special tokens, unknown token dropped, at most 512 tokens, mean, then
normalised. The matrix is cached once as float16 in data/ for a fast load.
"""
import os

import numpy as np

from . import config

_state = {}


def _load():
    if _state:
        return _state
    from tokenizers import Tokenizer
    if not config.EMBED_MODEL_DIRS:
        raise RuntimeError("potion-base-32M is not in the Hugging Face cache")
    d = config.EMBED_MODEL_DIRS[0]
    if not os.path.exists(config.EMBED_NPY):
        from safetensors.numpy import load_file
        m = load_file(os.path.join(d, "model.safetensors"))["embeddings"]
        os.makedirs(config.DATA, exist_ok=True)
        tmp = config.EMBED_NPY + ".tmp.npy"
        np.save(tmp, m.astype(np.float16))
        os.replace(tmp, config.EMBED_NPY)
    emb = np.load(config.EMBED_NPY)
    tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    vocab = tok.get_vocab()
    unk = None
    try:
        u = tok.model.unk_token
        unk = vocab.get(u) if u else None
    except Exception:
        unk = None
    lens = sorted(len(t) for t in vocab)
    median = lens[len(lens) // 2]
    _state.update(emb=emb, tok=tok, unk=unk, median=median, dim=emb.shape[1])
    return _state


def dim():
    return _load()["dim"]


def encode(texts, max_tokens=512):
    s = _load()
    texts = [t[: max_tokens * s["median"]] for t in texts]
    encs = s["tok"].encode_batch(texts, add_special_tokens=False)
    out = np.zeros((len(texts), s["dim"]), dtype=np.float32)
    for i, e in enumerate(encs):
        ids = [x for x in e.ids if x != s["unk"]][:max_tokens]
        if ids:
            v = s["emb"][ids].astype(np.float32).mean(axis=0)
            n = np.linalg.norm(v)
            out[i] = v / n if n > 0 else v
    return out


def to_blob(v):
    return np.asarray(v, dtype=np.float16).tobytes()


def from_blob(b):
    return np.frombuffer(b, dtype=np.float16)


def store_vectors(conn, items):
    """items: list of (node_id, text). Writes or replaces each node's vector."""
    if not items:
        return
    vecs = encode([t for _, t in items])
    seq = conn.execute("SELECT COALESCE(MAX(seq), 0) FROM vectors").fetchone()[0]
    for (nid, _), v in zip(items, vecs):
        seq += 1
        conn.execute("INSERT OR REPLACE INTO vectors(node_id, seq, v) VALUES (?,?,?)", (nid, seq, to_blob(v)))


class Index:
    """All node vectors in memory: the cache file plus anything newer from the database."""

    def __init__(self, conn):
        ids, mats, seq0 = np.zeros(0, dtype=np.int64), [], 0
        if os.path.exists(config.VEC_CACHE):
            try:
                z = np.load(config.VEC_CACHE)
                ids, mats, seq0 = z["ids"], [z["m"]], int(z["seq"])
            except Exception:
                ids, mats, seq0 = np.zeros(0, dtype=np.int64), [], 0
        rows = conn.execute("SELECT node_id, v FROM vectors WHERE seq > ? ORDER BY seq", (seq0,)).fetchall()
        if rows:
            new_ids = np.array([r[0] for r in rows], dtype=np.int64)
            new_m = np.stack([from_blob(r[1]) for r in rows])
            if len(ids):
                keep = ~np.isin(ids, new_ids)
                ids = ids[keep]
                mats = [mats[0][keep]] if mats else []
            ids = np.concatenate([ids, new_ids])
            mats.append(new_m)
        self.ids = ids
        self.m = np.concatenate(mats).astype(np.float32) if mats else np.zeros((0, 512), dtype=np.float32)
        self.pos = {int(n): i for i, n in enumerate(self.ids)}
        self.tail = len(rows)

    def scores(self, q):
        if not len(self.ids):
            return np.zeros(0, dtype=np.float32)
        return self.m @ q

    def vector(self, node_id):
        i = self.pos.get(int(node_id))
        return None if i is None else self.m[i]


def rebuild_cache(conn):
    """Write all vectors to the cache file so hooks load them in one read."""
    rows = conn.execute("SELECT node_id, seq, v FROM vectors ORDER BY seq").fetchall()
    if not rows:
        return 0
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    m = np.stack([from_blob(r[2]) for r in rows])
    seq = max(r[1] for r in rows)
    tmp = config.VEC_CACHE + ".tmp.npz"
    np.savez(tmp, ids=ids, m=m, seq=np.int64(seq))
    try:
        os.replace(tmp, config.VEC_CACHE)
    except PermissionError:
        return -1
    return len(rows)
