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

Additional triage mechanisms:

9. Temporal relevance — timeless memories decay slower, ephemeral faster
10. Auto-pinning by access frequency — frequently accessed memories self-protect
11. Access-count decay floor — proven-valuable memories survive longer
12. Semantic deduplication — near-duplicates merge instead of competing
13. Conflict supersession — contradictory new memories supersede old ones
14. Prune logging — deleted memories are logged for audit before removal
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class DecayParams:
    """Tunable parameters for all eight mechanisms."""

    # 1. Ebbinghaus decay
    #   importance *= 1 / (1 + decay_rate * hours_elapsed)
    #   Higher = faster forgetting. 0.02 means ~2% per hour at baseline.
    #   With rate=0.02: user corrections survive ~97 days without access,
    #   agent inferences survive ~12.5 days. Accessed memories survive
    #   indefinitely (reinforcement boosts offset decay).
    decay_rate: float = 0.02

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
    #   Default is 10x slower than importance decay.
    confidence_decay_rate: float = 0.002

    # 8. Decay floor — memories below this are prunable
    decay_floor: float = 0.05

    # 7. Max memories to inject per turn
    max_context: int = 15

    # 7b. Critical safety-tier budget — always-injected critical memories
    #   are rendered in their OWN section and do NOT consume max_context slots.
    #   Mirrors Letta's core/archival split: the always-on core never
    #   competes with relevance-ranked archival retrieval for context.
    critical_budget: int = 5

    # Default source confidence by origin type
    source_confidence_defaults: dict = None  # type: ignore

    # 10. Auto-pinning — when access_count crosses this threshold,
    #   the memory is automatically pinned (never pruned).
    auto_pin_threshold: int = 5

    # 9. Temporal decay multipliers — timeless memories decay slower,
    #   ephemeral memories decay faster.
    temporal_decay_multiplier: dict = None  # type: ignore

    # 12. Semantic deduplication threshold — if Jaccard similarity
    #   between new and existing memory exceeds this, merge instead of add.
    dedup_similarity_threshold: float = 0.85

    # 13. Conflict detection threshold — if similarity is above this but
    #   below dedup threshold, and content conflicts, supersede the old.
    conflict_similarity_threshold: float = 0.60

    def __post_init__(self):
        if self.source_confidence_defaults is None:
            self.source_confidence_defaults = {
                "user_correction": 1.0,
                "user_preference": 0.9,
                "research_finding": 0.85,
                "environment_fact": 0.7,
                "agent_inference": 0.4,
                "unknown": 0.5,
            }
        if self.temporal_decay_multiplier is None:
            self.temporal_decay_multiplier = {
                "timeless": 0.3,   # survives ~3x longer
                "stable": 1.0,      # normal decay
                "ephemeral": 3.0,   # clears ~3x faster
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
        "research_finding": 0.80,
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
    temporal: str = "stable",
) -> float:
    """Apply Ebbinghaus decay to a memory's importance.

    Uses a hyperbolic decay function (similar to the Ebbinghaus forgetting
    curve): importance decays faster early on, then levels out. The rate is
    scaled by elapsed hours so it's intuitive to tune.

    Temporal relevance adjusts the decay rate:
    - timeless: rate × 0.3 (survives 3x longer)
    - stable: normal rate (default)
    - ephemeral: rate × 3.0 (clears 3x faster)

    Formula:
        elapsed_hours = (now - last_access) / 3600
        multiplier = temporal_decay_multiplier[temporal]
        effective_rate = decay_rate * multiplier
        stability = 1 / (1 + effective_rate * elapsed_hours)
        new_importance = importance * stability

    Returns the decayed importance, clamped to [0, 1].
    """
    elapsed_hours = max(0.0, (now - last_access) / 3600.0)
    multiplier = params.temporal_decay_multiplier.get(temporal, 1.0)
    effective_rate = params.decay_rate * multiplier
    stability = 1.0 / (1.0 + effective_rate * elapsed_hours)
    return max(0.0, min(1.0, importance * stability))


def apply_confidence_decay(
    confidence: float,
    last_access: float,
    now: float,
    params: DecayParams,
) -> float:
    """Apply source-confidence decay.

    Confidence in the source of a memory erodes over time, but more slowly
    than importance. A memory you heard from a reliable source a week ago
    is still somewhat trustworthy, just less than a fresh one.

    Computed on the fly at retrieval time from last_access (not stored),
    so there is no compounding.

    Formula:
        elapsed_hours = (now - last_access) / 3600
        stability = 1 / (1 + confidence_decay_rate * elapsed_hours)
        new_confidence = confidence * stability
    """
    elapsed_hours = max(0.0, (now - last_access) / 3600.0)
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


# -- Temporal classification (mechanism 9) ---------------------------------

_EPHEMERAL_INDICATORS = [
    "still to be", "pending", "tbd", "temporary", "waiting for",
    "not yet", "to be confirmed", "not confirmed", "wip", "draft",
    "subject to change", "might change", "tentative", "preliminary",
]

_TIMELESS_INDICATORS = [
    "always", "never", "rule", "principle", "convention", "workflow",
    "must", "require", "mandatory", "standard", "non-negotiable",
    "every time", "forever", "permanent",
]


def classify_temporal(content: str) -> str:
    """Classify temporal relevance: timeless, stable, or ephemeral.

    - timeless: content describes permanent rules, conventions, principles.
        These decay 3x slower.
    - ephemeral: content describes temporary, pending, or tentative state.
        These decay 3x faster.
    - stable: everything else. Normal decay rate.

    Auto-detected from content keywords. Can be overridden by passing
    temporal explicitly in metadata.
    """
    content_lower = content.lower() if content else ""
    if any(p in content_lower for p in _EPHEMERAL_INDICATORS):
        return "ephemeral"
    if any(p in content_lower for p in _TIMELESS_INDICATORS):
        return "timeless"
    return "stable"


# -- Semantic similarity (mechanism 12) -----------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> Set[str]:
    """Tokenize text into a set of lowercase alphanumeric tokens."""
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard similarity between two sets: |A ∩ B| / |A ∪ B|."""
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def semantic_similarity(content_a: str, content_b: str) -> float:
    """Compute semantic similarity between two text strings.

    Uses token-level Jaccard similarity. Fast, no external dependencies.
    Returns 0-1, where 1 = identical and 0 = no overlap.
    """
    tokens_a = tokenize(content_a)
    tokens_b = tokenize(content_b)
    return jaccard_similarity(tokens_a, tokens_b)


# -- Conflict detection (mechanism 13) -------------------------------------

_NUMBER_RE = re.compile(r"\b\d+\.?\d*\b")

_NEGATION_WORDS = {"not", "never", "no", "none", "without", "dont", "stop", "cancel"}


def _extract_numbers(text: str) -> Set[str]:
    """Extract all number-like tokens from text."""
    return set(_NUMBER_RE.findall(text))


def detect_conflict(old_content: str, new_content: str) -> bool:
    """Detect if new_content contradicts old_content.

    Heuristic: if the two contents share many keywords (similarity > 0.4)
    BUT have different numbers OR different negation patterns, it's likely
    a conflict (the new memory supersedes the old one).

    Returns True if the new content likely supersedes the old.
    """
    tokens_old = tokenize(old_content)
    tokens_new = tokenize(new_content)
    similarity = jaccard_similarity(tokens_old, tokens_new)
    if similarity < 0.4:
        return False  # Not similar enough to conflict

    # Check for different numbers
    nums_old = _extract_numbers(old_content)
    nums_new = _extract_numbers(new_content)
    if nums_old and nums_new and nums_old != nums_new:
        return True  # Numbers changed → likely an update

    # Check for negation flip (old says "X", new says "not X" or vice versa)
    neg_old = _NEGATION_WORDS & tokens_old
    neg_new = _NEGATION_WORDS & tokens_new
    if neg_old != neg_new:
        return True  # Negation changed → likely a correction

    return False


def classify_origin(
    action: str,
    target: str,
    content: str,
    metadata: Optional[dict] = None,
) -> str:
    """Classify the origin of a memory write.

    Returns one of: user_correction, user_preference, research_finding,
    environment_fact, agent_inference, unknown.

    Research findings are detected by keywords indicating sourced, verified,
    or hard-to-find information (e.g. "research", "study", "data shows",
    "according to", "found that", "evidence"). This can be overridden by
    passing origin="research_finding" in metadata.
    """
    if metadata:
        origin = metadata.get("write_origin", metadata.get("origin", ""))
        if origin in (
            "user_correction", "user_preference", "research_finding",
            "environment_fact", "agent_inference",
        ):
            return origin

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

    # Research finding indicators — content that references sources, data,
    # studies, or verified information from reliable fonts.
    research_indicators = [
        "research", "study", "studies", "data shows", "data show",
        "according to", "found that", "evidence", "analysis shows",
        "report", "benchmark", "measured", "tested", "verified",
        "source:", "reliable", "documented", "finding", "findings",
        "survey", "statistics", "correlation", "backtested",
        "win rate", "expectancy", "sharpe", "drawdown",
    ]
    research_count = sum(1 for w in research_indicators if w in content_lower)
    if research_count >= 2:
        return "research_finding"

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