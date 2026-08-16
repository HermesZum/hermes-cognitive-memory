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
    params = DecayParams(decay_rate=0.02, decay_floor=0.05, access_boost=0.3)
    s = MemoryStore(db_path, params)
    s.connect()
    yield s
    s.close()


_OLD_SCHEMA_SQL = """
CREATE TABLE memories (
    id            TEXT PRIMARY KEY,
    target        TEXT NOT NULL,
    content       TEXT NOT NULL,
    importance    REAL NOT NULL,
    confidence    REAL NOT NULL,
    created_at    REAL NOT NULL,
    last_access   REAL NOT NULL,
    access_count  INTEGER DEFAULT 0,
    origin        TEXT NOT NULL DEFAULT 'unknown',
    tags          TEXT DEFAULT '[]'
);
"""


class TestMigration:
    """Databases created by earlier plugin versions must migrate on connect."""

    def _make_old_db(self, path, seed_content="old schema memory"):
        import sqlite3

        conn = sqlite3.connect(str(path))
        conn.executescript(_OLD_SCHEMA_SQL)
        conn.execute(
            "INSERT INTO memories (id, target, content, importance, confidence,"
            " created_at, last_access, access_count, origin) VALUES"
            " ('old-1', 'memory', ?, 0.8, 0.7, 0, 0, 0, 'user_preference')",
            (seed_content,),
        )
        conn.commit()
        conn.close()

    def test_connect_migrates_old_schema(self, tmp_path):
        """Connecting to a pre-existing old-schema DB adds the new columns."""
        db_path = tmp_path / "old.db"
        self._make_old_db(db_path)
        params = DecayParams(decay_rate=0.02, decay_floor=0.05)
        s = MemoryStore(db_path, params)
        s.connect()
        try:
            import sqlite3

            conn = sqlite3.connect(str(db_path))
            cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)")]
            conn.close()
            for col in ("reliability", "hard_to_find", "pinned", "temporal",
                        "superseded", "supersedes"):
                assert col in cols, f"column {col} missing after migration"
            # Existing row survived with defaults
            mem = s.get("old-1")
            assert mem is not None
            assert not mem["pinned"]
            assert mem["temporal"] == "stable"
            assert mem["reliability"] == 1.0
        finally:
            s.close()

    def test_pin_works_after_migration(self, tmp_path):
        """set_pinned must work on a migrated old-schema DB."""
        db_path = tmp_path / "old.db"
        self._make_old_db(db_path)
        params = DecayParams(decay_rate=0.02, decay_floor=0.05)
        s = MemoryStore(db_path, params)
        s.connect()
        try:
            assert s.set_pinned("old-1", True) is True
            assert bool(s.get("old-1")["pinned"]) is True
            assert s.set_pinned("old-1", False) is True
            assert not s.get("old-1")["pinned"]
        finally:
            s.close()

    def test_add_with_new_fields_after_migration(self, tmp_path):
        """add() with the new fields must work on a migrated old-schema DB."""
        db_path = tmp_path / "old.db"
        self._make_old_db(db_path)
        params = DecayParams(decay_rate=0.02, decay_floor=0.05)
        s = MemoryStore(db_path, params)
        s.connect()
        try:
            mem_id = s.add(
                "memory", "new-style entry", origin="research_finding",
                reliability=0.9, hard_to_find=True, pinned=True,
                temporal="timeless",
            )
            mem = s.get(mem_id)
            assert mem is not None
            assert bool(mem["pinned"]) is True
            assert bool(mem["hard_to_find"]) is True
            assert mem["temporal"] == "timeless"
            assert mem["reliability"] == 0.9
        finally:
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
        results, _ = store.search("Python")
        assert len(results) >= 1
        assert "Python" in results[0][0]["content"]

    def test_search_returns_score(self, store):
        store.add("memory", "important fact about databases")
        results, _ = store.search("databases")
        assert len(results) == 1
        mem, score = results[0]
        assert isinstance(score, float)
        assert score >= 0.0

    def test_search_empty_query_returns_importance_ranked(self, store):
        store.add("memory", "low importance", origin="agent_inference")
        store.add("memory", "high importance", origin="user_correction")
        results, _ = store.search("")
        assert len(results) == 2
        # User correction should rank higher (higher initial importance)
        assert results[0][0]["importance"] >= results[1][0]["importance"]

    def test_search_with_target_filter(self, store):
        store.add("memory", "agent memory about Python")
        store.add("user", "user preference for Python")
        results, _ = store.search("Python", target="user")
        assert len(results) == 1
        assert results[0][0]["target"] == "user"

    def test_search_limit(self, store):
        for i in range(20):
            store.add("memory", f"memory item number {i}")
        results, _ = store.search("memory", limit=5)
        assert len(results) <= 5


class TestDecayApplication:
    def test_global_decay_reports_prunable(self, store):
        """apply_global_decay now reports prunable count without modifying stored importance."""
        mem_id = store.add("memory", "will decay", origin="agent_inference")
        # Simulate time passing — 15 days is enough for agent_inference (0.35, floor 0.05)
        store._conn.execute(
            "UPDATE memories SET last_access = ? WHERE id = ?",
            (time.time() - 86400 * 15, mem_id),  # 15 days ago
        )
        store._conn.commit()

        # Stored importance should NOT change (decay is computed on the fly)
        before = store.get(mem_id)["importance"]
        prunable = store.apply_global_decay()
        after = store.get(mem_id)["importance"]

        assert after == before  # stored importance unchanged
        assert prunable == 1    # but it IS prunable (decayed to ~0.043, below 0.05)

    def test_decay_computed_on_the_fly(self, store):
        """Search ranking uses decayed importance computed at retrieval time."""
        mem_id = store.add("memory", "old memory", origin="user_preference")
        # Set last_access to 5 days ago
        store._conn.execute(
            "UPDATE memories SET last_access = ? WHERE id = ?",
            (time.time() - 86400 * 5, mem_id),
        )
        store._conn.commit()

        stored_importance = store.get(mem_id)["importance"]
        # Compute what decayed importance should be
        from cognitive_memory.decay import apply_decay
        import time as _time
        decayed = apply_decay(stored_importance, _time.time() - 86400 * 5, _time.time(), store._params)

        assert decayed < stored_importance  # decayed is lower
        assert decayed > 0.0                 # but not zero

    def test_prune_removes_decayed_memory(self, store):
        """Prune deletes memories whose computed decayed importance is below floor."""
        mem_id = store.add("memory", "will be pruned", origin="agent_inference")
        # Set last_access far in the past so decayed importance < floor
        store._conn.execute(
            "UPDATE memories SET last_access = ? WHERE id = ?",
            (time.time() - 86400 * 30, mem_id),  # 30 days ago
        )
        store._conn.commit()

        pruned = store.prune()
        assert pruned == 1
        assert store.count() == 0

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


    def test_prune_protects_user_corrections(self, store):
        """User corrections survive longer than agent inferences.

        With decay_rate=0.02:
          - user_correction (0.95, floor 0.02): 0.95/(1+0.02*h) < 0.02 → h > 2325h = 97 days
          - agent_inference (0.35, floor 0.05): 0.35/(1+0.02*h) < 0.05 → h > 300h = 12.5 days

        At 15 days: correction survives, inference is pruned.
        """
        correction_id = store.add("memory", "important correction", origin="user_correction")
        inference_id = store.add("memory", "low value inference", origin="agent_inference")

        import time
        fifteen_days_ago = time.time() - 86400 * 15
        store._conn.execute(
            "UPDATE memories SET last_access = ? WHERE id IN (?, ?)",
            (fifteen_days_ago, correction_id, inference_id),
        )
        store._conn.commit()

        pruned = store.prune()
        # agent_inference pruned (decayed below 0.05)
        # user_correction survives (decayed to ~0.24, well above 0.02)
        assert pruned == 1
        assert store.get(correction_id) is not None
        assert store.get(inference_id) is None

    def test_prune_protects_user_preferences(self, store):
        """User preferences survive longer than environment facts.

        With decay_rate=0.02:
          - user_preference (0.85, floor 0.03): 0.85/(1+0.02*h) < 0.03 → h > 1367h = 57 days
          - environment_fact (0.60, floor 0.05): 0.60/(1+0.02*h) < 0.05 → h > 550h = 23 days

        At 25 days: preference survives, environment_fact is pruned.
        """
        pref_id = store.add("memory", "my preference", origin="user_preference")
        fact_id = store.add("memory", "environment detail", origin="environment_fact")

        import time
        twenty_five_days_ago = time.time() - 86400 * 25
        store._conn.execute(
            "UPDATE memories SET last_access = ? WHERE id IN (?, ?)",
            (twenty_five_days_ago, pref_id, fact_id),
        )
        store._conn.commit()

        pruned = store.prune()
        # environment_fact pruned (decayed below 0.05)
        # user_preference survives (decayed to ~0.15, well above 0.03)
        assert pruned == 1
        assert store.get(pref_id) is not None
        assert store.get(fact_id) is None


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


class TestPinnedProtection:
    """Pinned memories are never pruned, regardless of decay."""

    def test_pinned_memory_not_pruned(self, store):
        """A pinned memory with very low importance should NOT be pruned."""
        mem_id = store.add(
            "memory", "critical fact", origin="agent_inference",
            pinned=True,
        )
        # Force importance to near-zero and last_access to very old
        store._conn.execute(
            "UPDATE memories SET importance = 0.001, last_access = 0 WHERE id = ?",
            (mem_id,),
        )
        store._conn.commit()

        pruned = store.prune()
        assert pruned == 0
        assert store.get(mem_id) is not None

    def test_pinned_not_counted_as_prunable(self, store):
        """Pinned memories should not be counted in apply_global_decay."""
        mem_id = store.add(
            "memory", "pinned fact", origin="agent_inference",
            pinned=True,
        )
        store._conn.execute(
            "UPDATE memories SET importance = 0.001, last_access = 0 WHERE id = ?",
            (mem_id,),
        )
        store._conn.commit()

        prunable = store.apply_global_decay()
        assert prunable == 0

    def test_non_pinned_with_same_importance_is_pruned(self, store):
        """Non-pinned memory with same low importance SHOULD be pruned."""
        pinned_id = store.add("memory", "pinned", origin="agent_inference", pinned=True)
        normal_id = store.add("memory", "normal", origin="agent_inference")

        # Set both to near-zero importance
        store._conn.execute(
            "UPDATE memories SET importance = 0.001, last_access = 0 WHERE id IN (?, ?)",
            (pinned_id, normal_id),
        )
        store._conn.commit()

        pruned = store.prune()
        assert pruned == 1
        assert store.get(pinned_id) is not None
        assert store.get(normal_id) is None


class TestHardToFindProtection:
    """Hard-to-find memories get a lower decay floor (0.01)."""

    def test_hard_to_find_survives_longer(self, store):
        """Hard-to-find memory should survive when a normal one is pruned."""
        htf_id = store.add(
            "memory", "rare finding", origin="agent_inference",
            hard_to_find=True,
        )
        normal_id = store.add("memory", "common fact", origin="agent_inference")

        # Set both to 25 days old — normal agent_inference decays to ~0.04 (below 0.05)
        # but hard-to-find has floor 0.01, so it survives
        old_time = time.time() - 86400 * 25
        store._conn.execute(
            "UPDATE memories SET last_access = ? WHERE id IN (?, ?)",
            (old_time, htf_id, normal_id),
        )
        store._conn.commit()

        pruned = store.prune()
        # Normal one is pruned, hard-to-find survives
        assert pruned == 1
        assert store.get(htf_id) is not None
        assert store.get(normal_id) is None

    def test_hard_to_find_gets_importance_boost(self, store):
        """Hard-to-find memories get +0.15 importance on creation."""
        htf_id = store.add(
            "memory", "rare", origin="environment_fact", hard_to_find=True,
        )
        normal_id = store.add("memory", "common", origin="environment_fact")

        htf = store.get(htf_id)
        normal = store.get(normal_id)
        assert htf["importance"] > normal["importance"]
        assert htf["importance"] == pytest.approx(normal["importance"] + 0.15, abs=0.01)


class TestReliabilityRanking:
    """Reliability score multiplies into search ranking."""

    def test_reliable_source_ranks_higher(self, store):
        """A memory with higher reliability should rank higher than one with lower."""
        reliable_id = store.add(
            "memory", "EURUSD shows strong correlation with DXY index over 2 years",
            origin="research_finding", reliability=0.95,
        )
        unreliable_id = store.add(
            "memory", "GBPUSD might correlate with DXY but needs more study",
            origin="research_finding", reliability=0.3,
        )

        results, _ = store.search("DXY")
        assert len(results) >= 2

        # The reliable one should rank higher
        top_mem = results[0][0]
        assert top_mem["id"] == reliable_id

    def test_default_reliability_is_1(self, store):
        """Memories created without reliability should default to 1.0."""
        mem_id = store.add("memory", "test content", origin="environment_fact")
        mem = store.get(mem_id)
        assert mem["reliability"] == 1.0


class TestResearchFindingProtection:
    """Research findings get a lower decay floor (0.03), like user preferences."""

    def test_research_finding_survives_longer_than_inference(self, store):
        """Research findings should survive when agent inferences are pruned."""
        research_id = store.add(
            "memory", "backtested study evidence finding",
            origin="research_finding",
        )
        inference_id = store.add(
            "memory", "maybe it works",
            origin="agent_inference",
        )

        # Set both to 20 days old
        # research_finding: starts 0.80, floor 0.03 → decays to ~0.31 (survives)
        # agent_inference: starts 0.35, floor 0.05 → decays to ~0.14 (survives)
        # Need more time for inference to hit 0.05:
        # 0.35 * 1/(1+0.02*hours) < 0.05 → hours > 300 → ~12.5 days
        # 0.80 * 1/(1+0.02*hours) < 0.03 → hours > 1283 → ~53 days
        # So at 15 days: research=~0.39, inference=~0.06 (pruned)
        old_time = time.time() - 86400 * 15
        store._conn.execute(
            "UPDATE memories SET last_access = ? WHERE id IN (?, ?)",
            (old_time, research_id, inference_id),
        )
        store._conn.commit()

        pruned = store.prune()
        # Inference is pruned, research survives
        assert pruned == 1
        assert store.get(research_id) is not None
        assert store.get(inference_id) is None


class TestSetPinned:
    """Manual pin/unpin override (WebUI management surface)."""

    def test_set_pinned_marks_memory(self, store):
        mem_id = store.add("memory", "pin me", origin="agent_inference")
        assert store.get(mem_id)["pinned"] == 0

        assert store.set_pinned(mem_id, True) is True
        assert store.get(mem_id)["pinned"] == 1

        assert store.set_pinned(mem_id, False) is True
        assert store.get(mem_id)["pinned"] == 0

    def test_set_pinned_missing_id_returns_false(self, store):
        assert store.set_pinned("no-such-id", True) is False

    def test_set_pinned_protects_from_prune(self, store):
        mem_id = store.add("memory", "pin-protected", origin="agent_inference")
        store.set_pinned(mem_id, True)
        # Force near-zero importance + ancient last_access
        store._conn.execute(
            "UPDATE memories SET importance = 0.001, last_access = 0 WHERE id = ?",
            (mem_id,),
        )
        store._conn.commit()

        assert store.prune() == 0
        assert store.get(mem_id) is not None

    def test_unpinned_then_prunable_again(self, store):
        mem_id = store.add("memory", "unpin me", origin="agent_inference")
        store.set_pinned(mem_id, True)
        store.set_pinned(mem_id, False)
        store._conn.execute(
            "UPDATE memories SET importance = 0.001, last_access = 0 WHERE id = ?",
            (mem_id,),
        )
        store._conn.commit()

        assert store.prune() == 1
        assert store.get(mem_id) is None