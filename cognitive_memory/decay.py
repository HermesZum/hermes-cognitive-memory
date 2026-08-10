"""Cognitive memory decay mechanisms.

Eight neuroscience-inspired mechanisms that govern how memories strengthen
and fade over time:

1. Ebbinghaus decay — exponential forgetting curve
2. Reconsolidation — retrieval modifies the memory trace
3. Retrieval-induced forgetting — recalling one item suppresses competitors
4. Source-confidence decay — trust in the origin erodes
5. Access reinforcement — each retrieval strengthens the trace
6. Competition suppression — lower-rank competitors lose importance
7. Importance-weighted retrieval — final relevance combines all factors
8. Decay-floor pruning — memories below threshold are eligible for removal
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class DecayParams:
    """Tunable parameters for all eight mechanisms."""

    # 1. Ebbinghaus decay
    #   importance *= 1 / (1 + decay_rate * hours_elapsed)
    #   Higher = faster forgetting. 0.15 means ~15% per hour at baseline.
    decay_rate: float = 0.15

    # 5. Access reinforcement
    #   importance += access_boost on each retrieval
    access_boost: float = 0.3

    # 2. Reconsolidation
    #   retrieved memory importance += reconsolidation_rate * (1 - importance)
    #   Strengthens weak memories more than strong ones.
    reconsolidation_rate: float = 0.1

    # 3. Retrieval-induced forgetting
    #   competitors lose importance: importance -= rif_penalty
    rif_penalty: float = 0.05

    # 4. Source-confidence decay
    #   confidence *= 1 / (1 + confidence_decay_rate * hours_elapsed)
    #   Slower than importance decay — trust erodes gradually.
    confidence_decay_rate: float = 0.02

    # 8. Decay floor — memories below this are prunable
    decay_floor: float = 0.05

    # 7. Max memories to inject per turn
    max_context: int = 15

    # Default source confidence by origin type
    source_confidence_defaults: dict = None  # type: ignore

    def __post_init__(self):
        if self.source_confidence_defaults is None:
            self.source_confidence_defaults = {
                "user_correction": 1.0,
                "user_preference": 0.9,
                "environment_fact": 0.7,
                "agent_inference": 0.4,
                "unknown": 0.5,
            }


def initial_importance(origin: str, params: DecayParams) -> float:
    """Determine the starting importance for a new memory based on its origin.

    User corrections and preferences start high (they override prior beliefs).
    Environment facts start medium. Agent inferences start low (they may be
    wrong and should fade if never reinforced).
    """
    defaults = {
        "user_correction": 0.95,
        "user_preference": 0.85,
        "environment_fact": 0.6,
        "agent_inference": 0.35,
        "unknown": 0.5,
    }
    return defaults.get(origin, 0.5)


def initial_confidence(origin: str, params: DecayParams) -> float:
    """Determine the starting source-confidence for a new memory."""
    return params.source_confidence_defaults.get(origin, 0.5)


def apply_decay(
    importance: float,
    last_access: float,
    now: float,
    params: DecayParams,
) -> float:
    """Apply Ebbinghaus decay to a memory's importance.

    Uses a hyperbolic decay function (similar to the Ebbinghaus forgetting
    curve): importance decays faster early on, then levels out. The rate is
    scaled by elapsed hours so it's intuitive to tune.

    Formula:
        elapsed_hours = (now - last_access) / 3600
        stability = 1 / (1 + decay_rate * elapsed_hours)
        new_importance = importance * stability

    Returns the decayed importance, clamped to [0, 1].
    """
    elapsed_hours = max(0.0, (now - last_access) / 3600.0)
    stability = 1.0 / (1.0 + params.decay_rate * elapsed_hours)
    return max(0.0, min(1.0, importance * stability))


def apply_confidence_decay(
    confidence: float,
    created_at: float,
    now: float,
    params: DecayParams,
) -> float:
    """Apply source-confidence decay.

    Confidence in the source of a memory erodes over time, but more slowly
    than importance. A memory you heard from a reliable source a week ago
    is still somewhat trustworthy, just less than a fresh one.

    Formula:
        elapsed_hours = (now - created_at) / 3600
        stability = 1 / (1 + confidence_decay_rate * elapsed_hours)
        new_confidence = confidence * stability
    """
    elapsed_hours = max(0.0, (now - created_at) / 3600.0)
    stability = 1.0 / (1.0 + params.confidence_decay_rate * elapsed_hours)
    return max(0.0, min(1.0, confidence * stability))


def apply_access_reinforcement(
    importance: float,
    params: DecayParams,
) -> float:
    """Strengthen a memory that was just retrieved.

    Each access gives a flat boost, capped at 1.0. This is the mechanism
    behind 'spaced repetition' — memories that are recalled periodically
    stay strong.
    """
    return max(0.0, min(1.0, importance + params.access_boost))


def apply_reconsolidation(
    importance: float,
    params: DecayParams,
) -> float:
    """Apply reconsolidation — retrieval modifies the memory trace.

    Reconsolidation strengthens the memory proportionally to how much
    'room' it has to grow. A memory at importance 0.3 gets a bigger boost
    than one at 0.8. This models the finding that weak memories are more
    malleable after retrieval than strong ones.

    Formula:
        boost = reconsolidation_rate * (1 - importance)
        new_importance = importance + boost
    """
    boost = params.reconsolidation_rate * (1.0 - max(0.0, importance))
    return max(0.0, min(1.0, importance + boost))


def apply_rif_penalty(
    importance: float,
    params: DecayParams,
) -> float:
    """Apply retrieval-induced forgetting to a competing memory.

    When memory A is retrieved, memories B and C that also matched the
    query but ranked lower get a small importance penalty. This models
    the finding that retrieval is not passive recall — it actively
    suppresses competing traces.
    """
    return max(0.0, importance - params.rif_penalty)


def should_prune(importance: float, params: DecayParams) -> bool:
    """Check if a memory has decayed below the floor and should be pruned."""
    return importance < params.decay_floor


def classify_origin(
    action: str,
    target: str,
    content: str,
    metadata: Optional[dict] = None,
) -> str:
    """Classify the origin of a memory write.

    Returns one of: user_correction, user_preference, environment_fact,
    agent_inference, unknown.

    This is a heuristic classifier based on the action, target, and content.
    """
    if metadata:
        origin = metadata.get("write_origin", "")
        if origin == "user_correction":
            return "user_correction"
        if origin == "user_preference":
            return "user_preference"

    # Heuristic: if the action is 'replace' or 'remove', it's likely a
    # correction (the user is changing something).
    if action in ("replace", "remove"):
        return "user_correction"

    # If the content mentions preferences, instructions, or conventions
    content_lower = content.lower() if content else ""
    preference_indicators = [
        "prefers", "wants", "likes", "dislikes", "uses", "always",
        "never", "should", "must", "convention", "workflow",
    ]
    if any(word in content_lower for word in preference_indicators):
        return "user_preference"

    # If the content describes the environment (paths, config, setup)
    environment_indicators = [
        "path", "directory", "config", "service", "port", "url",
        "installed", "running", "systemd", "nginx", "python",
    ]
    if any(word in content_lower for word in environment_indicators):
        return "environment_fact"

    # If target is 'user', it's likely a preference
    if target == "user":
        return "user_preference"

    # Default: agent inference
    return "agent_inference"