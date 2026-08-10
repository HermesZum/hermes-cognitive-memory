"""Tests for cognitive memory decay mechanisms."""

import time
import pytest

from cognitive_memory.decay import (
    DecayParams,
    apply_access_reinforcement,
    apply_confidence_decay,
    apply_decay,
    apply_reconsolidation,
    apply_rif_penalty,
    classify_origin,
    initial_confidence,
    initial_importance,
    relevance_score,
    should_prune,
)


class TestDecay:
    """Ebbinghaus decay tests."""

    def test_no_decay_when_just_accessed(self):
        now = time.time()
        params = DecayParams()
        result = apply_decay(0.8, now, now, params)
        assert result == pytest.approx(0.8, abs=0.01)

    def test_decay_after_one_hour(self):
        now = time.time()
        one_hour_ago = now - 3600
        params = DecayParams(decay_rate=0.15)
        result = apply_decay(0.8, one_hour_ago, now, params)
        # stability = 1 / (1 + 0.15 * 1) = 1/1.15 ≈ 0.87
        # 0.8 * 0.87 ≈ 0.696
        assert result < 0.8
        assert result == pytest.approx(0.696, abs=0.05)

    def test_decay_increases_over_time(self):
        now = time.time()
        params = DecayParams(decay_rate=0.15)
        one_hour = apply_decay(0.8, now - 3600, now, params)
        one_day = apply_decay(0.8, now - 86400, now, params)
        assert one_day < one_hour

    def test_decay_floor_is_zero(self):
        now = time.time()
        params = DecayParams(decay_rate=1.0)
        result = apply_decay(0.8, now - 99999999, now, params)
        assert result >= 0.0
        assert result < 0.01


class TestAccessReinforcement:
    """Access reinforcement (spaced repetition) tests."""

    def test_boost_increases_importance(self):
        params = DecayParams(access_boost=0.3)
        result = apply_access_reinforcement(0.5, params)
        assert result == pytest.approx(0.8)

    def test_boost_caps_at_1(self):
        params = DecayParams(access_boost=0.3)
        result = apply_access_reinforcement(0.9, params)
        assert result == 1.0


class TestReconsolidation:
    """Reconsolidation tests."""

    def test_weak_memory_gets_bigger_boost(self):
        params = DecayParams(reconsolidation_rate=0.1)
        weak = apply_reconsolidation(0.2, params)
        strong = apply_reconsolidation(0.8, params)
        weak_boost = weak - 0.2
        strong_boost = strong - 0.8
        assert weak_boost > strong_boost

    def test_capped_at_1(self):
        params = DecayParams(reconsolidation_rate=0.1)
        result = apply_reconsolidation(0.95, params)
        assert result <= 1.0


class TestRIF:
    """Retrieval-induced forgetting tests."""

    def test_penalty_decreases_importance(self):
        params = DecayParams(rif_penalty=0.05)
        result = apply_rif_penalty(0.5, params)
        assert result == pytest.approx(0.45)

    def test_penalty_floor_is_zero(self):
        params = DecayParams(rif_penalty=0.5)
        result = apply_rif_penalty(0.1, params)
        assert result >= 0.0


class TestConfidenceDecay:
    """Source-confidence decay tests."""

    def test_confidence_decays_slower_than_importance(self):
        now = time.time()
        params = DecayParams()
        old = now - 86400  # 1 day ago
        imp = apply_decay(0.8, old, now, params)
        conf = apply_confidence_decay(0.8, old, now, params)
        assert conf > imp  # confidence decays slower


class TestPruning:
    """Decay-floor pruning tests."""

    def test_above_floor_not_prunable(self):
        params = DecayParams(decay_floor=0.05)
        assert not should_prune(0.1, params)

    def test_below_floor_prunable(self):
        params = DecayParams(decay_floor=0.05)
        assert should_prune(0.03, params)

    def test_at_floor_not_prunable(self):
        params = DecayParams(decay_floor=0.05)
        assert not should_prune(0.05, params)


class TestOriginClassification:
    """Origin classification tests."""

    def test_replace_is_correction(self):
        result = classify_origin("replace", "memory", "new content")
        assert result == "user_correction"

    def test_remove_is_correction(self):
        result = classify_origin("remove", "memory", "old content")
        assert result == "user_correction"

    def test_preference_indicators(self):
        result = classify_origin("add", "user", "User prefers concise responses")
        assert result == "user_preference"

    def test_environment_indicators(self):
        result = classify_origin("add", "memory", "nginx runs on port 443 with systemd")
        assert result == "environment_fact"

    def test_metadata_override(self):
        result = classify_origin(
            "add", "memory", "some content",
            metadata={"write_origin": "user_correction"},
        )
        assert result == "user_correction"

    def test_unknown_defaults_to_inference(self):
        result = classify_origin("add", "memory", "some random content")
        assert result == "agent_inference"


class TestInitialImportance:
    """Initial importance scoring tests."""

    def test_user_correction_starts_high(self):
        result = initial_importance("user_correction", DecayParams())
        assert result >= 0.9

    def test_agent_inference_starts_low(self):
        result = initial_importance("agent_inference", DecayParams())
        assert result <= 0.4

    def test_ordering(self):
        params = DecayParams()
        assert (
            initial_importance("user_correction", params)
            > initial_importance("user_preference", params)
            > initial_importance("environment_fact", params)
            > initial_importance("agent_inference", params)
        )