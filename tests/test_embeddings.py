"""Unit tests for the embedding backends and hybrid fusion helpers.

These tests do NOT require Ollama to be running — they exercise the
pure-Python helpers (pack/unpack, cosine) and the NoOp fallback path, plus
an injected fake backend to validate store.py fusion logic deterministically.
"""

import sys
import math
import tempfile
from pathlib import Path

import pytest

# Allow running both as a module and standalone
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_memory.decay import DecayParams
from cognitive_memory import embeddings as emb
from cognitive_memory.store import MemoryStore


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def test_pack_unpack_roundtrip():
    vec = [0.1, -0.25, 0.3333333, 1.0, -1.0]
    blob = emb.pack_vector(vec)
    out = emb.unpack_vector(blob)
    assert len(out) == len(vec)
    for a, b in zip(vec, out):
        assert abs(a - b) < 1e-6


def test_cosine_identical():
    v = [0.5, 0.5, 0.5]
    assert abs(emb.cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal():
    assert abs(emb.cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-9


def test_cosine_degenerate():
    assert emb.cosine_similarity([], []) == 0.0
    assert emb.cosine_similarity([1.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# Backend factory / fallback
# ---------------------------------------------------------------------------
def test_noop_backend_always_unavailable():
    b = emb.NoOpBackend()
    assert b.available is False
    assert b.embed("anything") is None


def test_factory_disabled_returns_noop():
    b = emb.get_embedding_backend(enabled=False)
    assert isinstance(b, emb.NoOpBackend)


class _FakeBackend(emb.EmbeddingBackend):
    """Deterministic fake: embeds by hashing tokens to a fixed-dim vector."""

    model = "fake"
    dim = 4

    @property
    def available(self):
        return True

    def embed(self, text: str):
        # Deterministic pseudo-embedding from word set (Jaccard-ish semantic).
        words = set(text.lower().split())
        vec = [0.0] * self.dim
        for i, w in enumerate(sorted(words)):
            vec[i % self.dim] += (hash(w) % 100) / 100.0
        return vec


def test_factory_with_reachable_backend_returns_it(monkeypatch):
    # Force the Ollama backend to report available without a real server.
    b = _FakeBackend()
    monkeypatch.setattr(emb, "OllamaEmbeddingBackend", lambda *a, **k: b)
    out = emb.get_embedding_backend(enabled=True)
    assert out is b


# ---------------------------------------------------------------------------
# Store integration (NoOp -> lexical only, schema migration)
# ---------------------------------------------------------------------------
def test_store_creates_embeddings_table():
    s = MemoryStore(Path(tempfile.mkdtemp()) / "t.db", DecayParams())
    s.connect()
    with s._lock:
        cols = [r["name"] for r in s._conn.execute(
            "PRAGMA table_info(memory_embeddings)").fetchall()]
    assert cols == ["memory_id", "model", "dim", "vector"]
    s.close()


def test_store_embed_disabled_is_noop():
    s = MemoryStore(Path(tempfile.mkdtemp()) / "t.db", DecayParams())
    s.connect()
    s.configure_embeddings(enabled=False)
    mid = s.add("memory", "push code to main branch safely")
    assert s._get_embedding(mid) is None
    # search still works (lexical-only)
    res, crit = s.search("push code")
    assert isinstance(res, list)
    s.close()


def test_store_embed_and_fuse_with_fake_backend(monkeypatch):
    """End-to-end: fake backend -> embedding stored + semantic recall works."""
    s = MemoryStore(Path(tempfile.mkdtemp()) / "t.db", DecayParams())
    s.connect()
    s._embedding_backend = _FakeBackend()
    s._embedding_enabled = True

    # Two memories; one shares tokens with the query, one is semantic-only.
    a = s.add("memory", "git push deploy branch main")
    b = s.add("memory", "never assume the default branch is main inspect first")

    # Query phrased differently from B but meaningally close (shared tokens via fake)
    res, _ = s.search("check the current branch before pushing")
    ids = [m["id"] for m, _ in res]
    # Both should be retrievable; at minimum the lexical one (a) must appear.
    assert a in ids
    assert b in ids
    # Embedding should be stored for both
    assert s._get_embedding(a) is not None
    assert s._get_embedding(b) is not None
    s.close()


def test_backfill_is_idempotent(monkeypatch):
    s = MemoryStore(Path(tempfile.mkdtemp()) / "t.db", DecayParams())
    s.connect()
    # Add first with embeddings disabled, so backfill has real work to do.
    s.configure_embeddings(enabled=False)
    s.add("memory", "alpha beta")
    s.add("memory", "gamma delta")
    # Now enable a fake backend and backfill.
    s._embedding_backend = _FakeBackend()
    s._embedding_enabled = True
    n1 = s.backfill_embeddings()
    n2 = s.backfill_embeddings()  # second run should embed nothing new
    assert n1 == 2
    assert n2 == 0
    s.close()


def test_replace_reembeds_content():
    """replace() must update the stored embedding to match new content.

    Regression guard: without this, semantic search keeps matching the
    stale pre-replace text (embedding drift bug).
    """
    s = MemoryStore(Path(tempfile.mkdtemp()) / "t.db", DecayParams())
    s.connect()
    s._embedding_backend = _FakeBackend()
    s._embedding_enabled = True
    mid = s.add("memory", "original content about apples")
    old_vec = s._get_embedding(mid)
    assert old_vec is not None
    s.replace(mid, "completely different text about oranges")
    new_vec = s._get_embedding(mid)
    assert new_vec is not None
    # Fake backend embeddings depend on token set, so different content
    # must produce a different vector.
    assert new_vec != old_vec
    s.close()


def test_merge_reembeds_content():
    """Merging a duplicate must re-embed with the merged content."""
    s = MemoryStore(Path(tempfile.mkdtemp()) / "t.db", DecayParams())
    s.connect()
    s._embedding_backend = _FakeBackend()
    s._embedding_enabled = True
    mid = s.add("memory", "base fact about networking")
    old_vec = s._get_embedding(mid)
    assert old_vec is not None
    # Directly exercise the merge path (deterministic, bypasses dedup threshold)
    with s._lock:
        s._merge_locked(mid, "base fact about networking and also DNS specifics",
                        new_importance=0.8, new_confidence=0.8, new_reliability=1.0,
                        new_hard_to_find=False, new_pinned=False, new_temporal="stable")
    new_vec = s._get_embedding(mid)
    assert new_vec is not None
    assert new_vec != old_vec  # content changed -> embedding changed
    merged = s.get(mid)
    assert merged is not None
    assert "dns" in merged["content"].lower()
    s.close()


def test_remove_drops_embedding_row():
    """remove() must also delete the orphaned embedding row."""
    s = MemoryStore(Path(tempfile.mkdtemp()) / "t.db", DecayParams())
    s.connect()
    s._embedding_backend = _FakeBackend()
    s._embedding_enabled = True
    mid = s.add("memory", "something to delete")
    assert s._get_embedding(mid) is not None
    s.remove(mid)
    # Embedding row should be gone (query returns None)
    assert s._get_embedding(mid) is None
    s.close()
