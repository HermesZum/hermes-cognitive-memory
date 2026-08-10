"""Tests for the SQLite memory store."""

import os
import tempfile
import time
import pytest

from cognitive_memory.decay import DecayParams
from cognitive_memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    """Create a temporary MemoryStore for each test."""
    db_path = tmp_path / "test_memory.db"
    params = DecayParams(decay_rate=0.15, decay_floor=0.05, access_boost=0.3)
    s = MemoryStore(db_path, params)
    s.connect()
    yield s
    s.close()


class TestAdd:
    def test_add_returns_id(self, store):
        mem_id = store.add("memory", "test content", origin="user_preference")
        assert mem_id is not None
        assert len(mem_id) > 0

    def test_add_increases_count(self, store):
        store.add("memory", "content 1")
        store.add("memory", "content 2")
        assert store.count() == 2

    def test_add_with_target(self, store):
        store.add("user", "prefers dark mode", origin="user_preference")
        assert store.count("user") == 1
        assert store.count("memory") == 0

    def test_add_with_tags(self, store):
        mem_id = store.add("memory", "tagged content", tags=["python", "test"])
        mem = store.get(mem_id)
        assert mem is not None
        assert "python" in mem["tags"]


class TestGet:
    def test_get_existing(self, store):
        mem_id = store.add("memory", "get me")
        mem = store.get(mem_id)
        assert mem is not None
        assert mem["content"] == "get me"

    def test_get_nonexistent(self, store):
        mem = store.get("nonexistent-id")
        assert mem is None


class TestRemove:
    def test_remove_existing(self, store):
        mem_id = store.add("memory", "delete me")
        assert store.remove(mem_id) is True
        assert store.count() == 0

    def test_remove_nonexistent(self, store):
        assert store.remove("fake-id") is False

    def test_remove_by_content(self, store):
        store.add("memory", "find and delete this entry")
        store.add("memory", "keep this one")
        count = store.remove_by_content("find and delete")
        assert count == 1
        assert store.count() == 1

    def test_remove_by_content_underscore_is_literal(self, store):
        """Regression: _ in LIKE is a wildcard — must be escaped."""
        store.add("memory", "test_data one")
        store.add("memory", "testXdata two")
        count = store.remove_by_content("test_data")
        assert count == 1  # only "test_data one", not "testXdata two"
        remaining = store.get_all()
        assert len(remaining) == 1
        assert remaining[0]["content"] == "testXdata two"

    def test_remove_by_content_percent_is_literal(self, store):
        """Regression: % in LIKE is a wildcard — must be escaped."""
        store.add("memory", "Disk at 100% full")
        store.add("memory", "Disk at 200% full")
        count = store.remove_by_content("100%")
        assert count == 1
        remaining = store.get_all()
        assert len(remaining) == 1
        assert remaining[0]["content"] == "Disk at 200% full"


class TestSearch:
    def test_search_finds_matching(self, store):
        store.add("memory", "User prefers Python over JavaScript")
        store.add("memory", "Server runs on port 443")
        results = store.search("Python")
        assert len(results) >= 1
        assert "Python" in results[0][0]["content"]

    def test_search_returns_score(self, store):
        store.add("memory", "important fact about databases")
        results = store.search("databases")
        assert len(results) == 1
        mem, score = results[0]
        assert isinstance(score, float)
        assert score >= 0.0

    def test_search_empty_query_returns_importance_ranked(self, store):
        store.add("memory", "low importance", origin="agent_inference")
        store.add("memory", "high importance", origin="user_correction")
        results = store.search("")
        assert len(results) == 2
        # User correction should rank higher (higher initial importance)
        assert results[0][0]["importance"] >= results[1][0]["importance"]

    def test_search_with_target_filter(self, store):
        store.add("memory", "agent memory about Python")
        store.add("user", "user preference for Python")
        results = store.search("Python", target="user")
        assert len(results) == 1
        assert results[0][0]["target"] == "user"

    def test_search_limit(self, store):
        for i in range(20):
            store.add("memory", f"memory item number {i}")
        results = store.search("memory", limit=5)
        assert len(results) <= 5


class TestDecayApplication:
    def test_global_decay_reduces_importance(self, store):
        mem_id = store.add("memory", "will decay", origin="user_preference")
        # Simulate time passing by manually setting last_access to the past
        store._conn.execute(
            "UPDATE memories SET last_access = ? WHERE id = ?",
            (time.time() - 86400, mem_id),  # 1 day ago
        )
        store._conn.commit()

        before = store.get(mem_id)["importance"]
        store.apply_global_decay()
        after = store.get(mem_id)["importance"]
        assert after < before

    def test_prune_removes_low_importance(self, store):
        mem_id = store.add("memory", "will be pruned", origin="agent_inference")
        # Force importance below floor
        store._conn.execute(
            "UPDATE memories SET importance = 0.01 WHERE id = ?",
            (mem_id,),
        )
        store._conn.commit()

        pruned = store.prune()
        assert pruned == 1
        assert store.count() == 0

    def test_prune_keeps_high_importance(self, store):
        mem_id = store.add("memory", "will survive", origin="user_correction")
        store._conn.execute(
            "UPDATE memories SET importance = 0.8 WHERE id = ?",
            (mem_id,),
        )
        store._conn.commit()

        pruned = store.prune()
        assert pruned == 0
        assert store.count() == 1


class TestRetrievalEffects:
    def test_access_reinforcement_on_retrieval(self, store):
        mem_id = store.add("memory", "access me", origin="environment_fact")
        before = store.get(mem_id)["importance"]

        store.search("access")

        after = store.get(mem_id)["importance"]
        assert after > before  # boosted by access

    def test_access_count_increments(self, store):
        mem_id = store.add("memory", "count my accesses")
        assert store.get(mem_id)["access_count"] == 0

        store.search("count")

        assert store.get(mem_id)["access_count"] == 1


class TestStats:
    def test_count_by_target(self, store):
        store.add("memory", "mem1")
        store.add("memory", "mem2")
        store.add("user", "user1")
        assert store.count("memory") == 2
        assert store.count("user") == 1
        assert store.count() == 3

    def test_total_chars(self, store):
        store.add("memory", "hello")  # 5 chars
        store.add("memory", "world!")  # 6 chars
        assert store.total_chars() == 11