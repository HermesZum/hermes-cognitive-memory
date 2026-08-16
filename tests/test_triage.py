"""Tests for the 6 triage improvements:

1. Auto-pinning by access frequency
2. Prune logging
3. Access-count decay floor
4. Semantic deduplication
5. Conflict supersession
6. Temporal relevance classification
"""

import os
import tempfile
import time
from pathlib import Path

import pytest

from cognitive_memory.decay import (
    DecayParams,
    apply_decay,
    classify_temporal,
    detect_conflict,
    semantic_similarity,
    tokenize,
)
from cognitive_memory.store import MemoryStore


@pytest.fixture
def params():
    return DecayParams(
        decay_rate=0.02,
        decay_floor=0.05,
        auto_pin_threshold=3,  # low threshold for testing
    )


@pytest.fixture
def store(params, tmp_path):
    db_path = tmp_path / "test_cognitive.db"
    s = MemoryStore(db_path, params)
    s.connect()
    yield s
    s.close()


# -- 1. Auto-pinning by access frequency -----------------------------------

class TestAutoPinning:
    """Memories accessed N times should auto-pin."""

    def test_auto_pin_after_threshold(self, store):
        """After auto_pin_threshold accesses, memory is pinned."""
        mem_id = store.add("memory", "important fact about FX risk management",
                          origin="user_preference")

        # Access it threshold times
        for _ in range(3):
            results = store.search("FX risk")
            assert len(results) > 0

        mem = store.get(mem_id)
        assert mem["pinned"] == 1, f"Expected pinned after 3 accesses, got access_count={mem['access_count']}"
        assert mem["access_count"] >= 3

    def test_auto_pin_protects_from_pruning(self, store):
        """An auto-pinned memory should survive prune even with zero importance."""
        mem_id = store.add("memory", "critical rule never delete this",
                          origin="agent_inference", importance=0.01)

        # Access it enough to auto-pin
        for _ in range(3):
            store.search("critical rule")

        mem = store.get(mem_id)
        assert mem["pinned"] == 1

        # Prune should not delete it
        pruned = store.prune()
        assert store.get(mem_id) is not None, "Auto-pinned memory was pruned!"


# -- 2. Prune logging -----------------------------------------------------

class TestPruneLogging:
    """Pruned memories should be logged to prune_log.md."""

    def test_prune_log_created(self, store):
        """When memories are pruned, a log file is created."""
        store.add("memory", "disposable inference",
                  origin="agent_inference", importance=0.01)

        # Fast-forward last_access to simulate old memory
        with store._lock:
            store._conn.execute(
                "UPDATE memories SET last_access = ? WHERE content = ?",
                (time.time() - 3600 * 24 * 365, "disposable inference"),
            )
            store._conn.commit()

        store.prune()

        log_path = store._prune_log_path
        assert log_path.exists(), "Prune log was not created"

        log_content = log_path.read_text()
        assert "disposable inference" in log_content
        assert "pruned" in log_content

    def test_prune_log_not_created_when_nothing_pruned(self, store):
        """If nothing is pruned, no log entry is written."""
        store.add("memory", "important fact",
                  origin="user_correction", importance=0.95)

        store.prune()

        # Log file may not exist or be empty
        log_path = store._prune_log_path
        if log_path.exists():
            content = log_path.read_text()
            assert "important fact" not in content


# -- 3. Access-count decay floor -------------------------------------------

class TestAccessCountFloor:
    """Memories accessed more frequently should have a lower decay floor."""

    def test_accessed_memory_survives_longer(self, store):
        """A memory accessed 10 times should survive longer than one accessed 0 times."""
        accessed_id = store.add("memory", "frequently accessed research finding",
                                origin="agent_inference")
        fresh_id = store.add("memory", "never accessed inference",
                            origin="agent_inference")

        # Access the first one 10 times
        for _ in range(10):
            store.search("frequently accessed")

        # Fast-forward both to 100 days ago
        old_time = time.time() - 3600 * 24 * 100
        with store._lock:
            store._conn.execute(
                "UPDATE memories SET last_access = ?", (old_time,)
            )
            store._conn.commit()

        # The accessed one should have a lower floor
        floor_accessed = store._effective_floor("agent_inference", False, 10)
        floor_fresh = store._effective_floor("agent_inference", False, 0)

        assert floor_accessed < floor_fresh, \
            f"Accessed floor ({floor_accessed}) should be lower than fresh ({floor_fresh})"

    def test_zero_access_uses_base_floor(self, store):
        """Memories with 0 accesses should use the base floor."""
        floor = store._effective_floor("agent_inference", False, 0)
        assert floor == store._params.decay_floor


# -- 4. Semantic deduplication --------------------------------------------

class TestSemanticDeduplication:
    """Near-duplicate memories should merge instead of creating duplicates."""

    def test_exact_duplicate_merges(self, store):
        """Adding the same content twice should merge, not duplicate."""
        id1 = store.add("memory", "FX demo account starting balance is 1000 dollars",
                        origin="user_preference")
        id2 = store.add("memory", "FX demo account starting balance is 1000 dollars",
                        origin="user_preference")

        assert id1 == id2, "Exact duplicates should merge into the same ID"

        all_mems = store.get_all()
        assert len(all_mems) == 1

    def test_near_duplicate_merges(self, store):
        """Very similar content (above threshold) should merge."""
        id1 = store.add("memory",
                        "User prefers markdown formatting in all responses with code blocks",
                        origin="user_preference")
        id2 = store.add("memory",
                        "User prefers markdown formatting in all responses with code blocks",
                        origin="user_preference")

        assert id1 == id2, "Near-duplicates should merge"

    def test_different_content_does_not_merge(self, store):
        """Completely different content should not merge."""
        id1 = store.add("memory", "EURUSD trading strategy with breakout entries",
                        origin="research_finding")
        id2 = store.add("memory", "Messaging app configuration on port 1234",
                        origin="environment_fact")

        assert id1 != id2
        assert len(store.get_all()) == 2

    def test_merge_keeps_higher_importance(self, store):
        """When merging, the higher importance should be kept."""
        id1 = store.add("memory", "FX demo account starting balance is 1000 dollars",
                        origin="user_preference", importance=0.85)
        id2 = store.add("memory", "FX demo account starting balance is 1000 dollars",
                        origin="user_correction", importance=0.95)

        mem = store.get(id1)
        assert mem["importance"] == 0.95, "Merged memory should keep higher importance"

    def test_merge_ors_pinned_flag(self, store):
        """When merging, pinned flag should be ORed."""
        id1 = store.add("memory", "FX demo account starting balance is 1000 dollars",
                        origin="user_preference", pinned=False)
        id2 = store.add("memory", "FX demo account starting balance is 1000 dollars",
                        origin="user_preference", pinned=True)

        mem = store.get(id1)
        assert mem["pinned"] == 1, "Merged memory should be pinned if either was pinned"


# -- 5. Conflict supersession ---------------------------------------------

class TestConflictSupersession:
    """Conflicting new memories should supersede old ones."""

    def test_number_change_supersedes(self, store):
        """If a new memory has different numbers for the same topic, old is superseded."""
        old_id = store.add("memory",
                          "Risk cap is 1.5 percent per trade for FX demo",
                          origin="user_preference")
        new_id = store.add("memory",
                          "Risk cap is 2 percent per trade for FX demo",
                          origin="user_correction")

        old_mem = store.get(old_id)
        assert old_mem["superseded"] == 1, "Old memory should be superseded"

        new_mem = store.get(new_id)
        assert new_mem["superseded"] == 0, "New memory should be active"

    def test_superseded_excluded_from_search(self, store):
        """Superseded memories should not appear in search results."""
        old_id = store.add("memory",
                          "Risk cap is 1.5 percent per trade for FX demo",
                          origin="user_preference")
        new_id = store.add("memory",
                          "Risk cap is 2 percent per trade for FX demo",
                          origin="user_correction")

        results, _ = store.search("risk cap")
        ids = [r[0]["id"] for r in results]
        assert old_id not in ids, "Superseded memory appeared in search results"
        assert new_id in ids, "New memory should appear in search results"

    def test_different_topic_does_not_supersede(self, store):
        """Memories about different topics should not supersede each other."""
        id1 = store.add("memory",
                        "Risk cap is 1.5 percent per trade for FX demo",
                        origin="user_preference")
        id2 = store.add("memory",
                        "Messaging app is configured on port 1234 for the gateway",
                        origin="environment_fact")

        mem1 = store.get(id1)
        assert mem1["superseded"] == 0, "Different topic should not supersede"

    def test_negation_change_supersedes(self, store):
        """If negation changes, old memory should be superseded."""
        old_id = store.add("memory",
                          "Always make live system changes on the VM",
                          origin="agent_inference")
        new_id = store.add("memory",
                          "Never make live system changes on the VM",
                          origin="user_correction")

        old_mem = store.get(old_id)
        assert old_mem["superseded"] == 1, "Negation change should supersede"


# -- 6. Temporal relevance -------------------------------------------------

class TestTemporalRelevance:
    """Temporal classification adjusts decay rate."""

    def test_timeless_detected_from_content(self):
        """Content with 'always', 'never', 'rule' should be timeless."""
        assert classify_temporal("Always use markdown formatting") == "timeless"
        assert classify_temporal("Never modify working configs") == "timeless"
        assert classify_temporal("This is a mandatory rule") == "timeless"

    def test_ephemeral_detected_from_content(self):
        """Content with 'pending', 'TBD', 'temporary' should be ephemeral."""
        assert classify_temporal("Broker platform still to be confirmed") == "ephemeral"
        assert classify_temporal("This is a temporary config") == "ephemeral"
        assert classify_temporal("Subject to change pending review") == "ephemeral"

    def test_stable_for_neutral_content(self):
        """Content without temporal indicators should be stable."""
        assert classify_temporal("EURUSD shows correlation with DXY") == "stable"
        assert classify_temporal("The VM has 4 vCPU and 8GB RAM") == "stable"

    def test_timeless_decays_slower(self, store):
        """Timeless memories should decay slower than stable ones."""
        timeless_id = store.add("memory", "Always use markdown formatting rule",
                                origin="user_preference", temporal="timeless")
        stable_id = store.add("memory", "EURUSD shows correlation with DXY index",
                             origin="research_finding", temporal="stable")

        # Fast-forward 30 days
        old_time = time.time() - 3600 * 24 * 30
        with store._lock:
            store._conn.execute(
                "UPDATE memories SET last_access = ?", (old_time,)
            )
            store._conn.commit()

        now = time.time()
        timeless_mem = store.get(timeless_id)
        stable_mem = store.get(stable_id)

        timeless_decayed = apply_decay(
            timeless_mem["importance"], timeless_mem["last_access"], now,
            store._params, "timeless"
        )
        stable_decayed = apply_decay(
            stable_mem["importance"], stable_mem["last_access"], now,
            store._params, "stable"
        )

        assert timeless_decayed > stable_decayed, \
            f"Timeless ({timeless_decayed:.4f}) should decay slower than stable ({stable_decayed:.4f})"

    def test_ephemeral_decays_faster(self, store):
        """Ephemeral memories should decay faster than stable ones."""
        ephemeral_id = store.add("memory", "Broker platform still to be confirmed",
                                origin="environment_fact", temporal="ephemeral")
        stable_id = store.add("memory", "EURUSD shows correlation with DXY index",
                             origin="research_finding", temporal="stable")

        # Fast-forward 10 days
        old_time = time.time() - 3600 * 24 * 10
        with store._lock:
            store._conn.execute(
                "UPDATE memories SET last_access = ?", (old_time,)
            )
            store._conn.commit()

        now = time.time()
        ephemeral_mem = store.get(ephemeral_id)
        stable_mem = store.get(stable_id)

        ephemeral_decayed = apply_decay(
            ephemeral_mem["importance"], ephemeral_mem["last_access"], now,
            store._params, "ephemeral"
        )
        stable_decayed = apply_decay(
            stable_mem["importance"], stable_mem["last_access"], now,
            store._params, "stable"
        )

        assert ephemeral_decayed < stable_decayed, \
            f"Ephemeral ({ephemeral_decayed:.4f}) should decay faster than stable ({stable_decayed:.4f})"


# -- Helper function tests -------------------------------------------------

class TestSemanticSimilarity:
    """Test the semantic similarity and conflict detection helpers."""

    def test_identical_content_high_similarity(self):
        sim = semantic_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_completely_different_low_similarity(self):
        sim = semantic_similarity("hello world", "goodbye universe")
        assert sim < 0.2

    def test_partial_overlap_medium_similarity(self):
        sim = semantic_similarity(
            "FX demo account starting balance",
            "FX demo account risk cap"
        )
        assert 0.3 < sim < 0.8

    def test_conflict_different_numbers(self):
        assert detect_conflict(
            "Risk cap is 1.5 percent per trade",
            "Risk cap is 2 percent per trade"
        ) == True

    def test_conflict_negation_change(self):
        assert detect_conflict(
            "Always make live changes",
            "Never make live changes"
        ) == True

    def test_no_conflict_different_topics(self):
        assert detect_conflict(
            "Risk cap is 1.5 percent",
            "Messaging app on port 1234"
        ) == False

    def test_tokenize(self):
        tokens = tokenize("Hello, World! 123")
        assert tokens == {"hello", "world", "123"}