"""Built-in memory lifecycle manager.

Reads the built-in Hermes memory files (MEMORY.md / USER.md), cross-references
the cognitive store, and produces a compaction plan so the built-in files stay
small while the cognitive store carries the long-form detail.

Actions per entry:
  - keep:    no cognitive mirror (would lose data), pinned, actively used,
             or mirror is dying (below safety floor — the built-in copy is
             the last surviving copy).
  - compact: mirrored + medium importance -> shorten entry to a pointer
             into the cognitive store ("details in cognitive memory").
  - remove:  mirrored + strong importance -> safe to drop from the built-in
             file; the cognitive store retains the full content.

Safety invariants (never violated, enforced by this module):
  1. Never remove/compact an entry with no cognitive mirror (data loss).
  2. Never remove/compact pinned entries.
  3. Never remove/compact entries whose mirror is below the decay floor
     (the mirror itself could be pruned next session).
  4. Never touch entries with access_count >= access_keep (actively used).
  5. Back up both files before any write.
  6. Every change is appended to the prune log.
  7. Dry-run by default — apply only with an explicit flag.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .decay import DecayParams, apply_decay, semantic_similarity
from .store import MemoryStore

logger = logging.getLogger(__name__)

# The exact delimiter Hermes uses for memory entries (memory_tool.py)
ENTRY_DELIMITER = "\n§\n"

# Default thresholds (overridable via memory.cognitive.sync_* config keys)
DEFAULT_MIRROR_THRESHOLD = 0.60   # min similarity to consider an entry mirrored
DEFAULT_KEEP_IMPORTANCE = 0.30    # mirror above this = strong, safe to remove built-in copy
DEFAULT_COMPACT_IMPORTANCE = 0.15 # mirror in [compact, keep) = shorten built-in to pointer
DEFAULT_ACCESS_KEEP = 3           # entries accessed >= this are actively used -> keep
DEFAULT_TRIGGER_PCT = 85          # propose sync when built-in usage exceeds this %


@dataclass
class EntryDecision:
    """Decision for one built-in memory entry."""

    entry: str
    action: str  # 'keep' | 'compact' | 'remove'
    reason: str
    mirror_id: Optional[str] = None
    mirror_similarity: float = 0.0
    mirror_importance: float = 0.0
    mirror_access: int = 0
    replacement: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "mirror_id": self.mirror_id,
            "mirror_similarity": round(self.mirror_similarity, 3),
            "mirror_importance": round(self.mirror_importance, 3),
            "mirror_access": self.mirror_access,
            "replacement": self.replacement,
            "entry_preview": self.entry[:120],
        }


@dataclass
class SyncPlan:
    """Full compaction plan for one target ('memory' or 'user')."""

    target: str
    decisions: List[EntryDecision] = field(default_factory=list)
    built_in_path: Optional[Path] = None
    usage_pct: float = 0.0
    limit: int = 0

    @property
    def keeps(self) -> List[EntryDecision]:
        return [d for d in self.decisions if d.action == "keep"]

    @property
    def compacts(self) -> List[EntryDecision]:
        return [d for d in self.decisions if d.action == "compact"]

    @property
    def removes(self) -> List[EntryDecision]:
        return [d for d in self.decisions if d.action == "remove"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "usage_pct": round(self.usage_pct, 1),
            "limit": self.limit,
            "counts": {
                "total": len(self.decisions),
                "keep": len(self.keeps),
                "compact": len(self.compacts),
                "remove": len(self.removes),
            },
            "decisions": [d.to_dict() for d in self.decisions],
        }


class BuiltinMemorySync:
    """Compaction planner + applier for the built-in memory files."""

    def __init__(
        self,
        hermes_home: Path,
        store: MemoryStore,
        params: DecayParams,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._hermes_home = Path(hermes_home)
        self._store = store
        self._params = params
        cfg = config or {}

        self._mirror_threshold = float(
            cfg.get("sync_mirror_threshold", DEFAULT_MIRROR_THRESHOLD)
        )
        self._keep_importance = float(
            cfg.get("sync_keep_importance", DEFAULT_KEEP_IMPORTANCE)
        )
        self._compact_importance = float(
            cfg.get("sync_compact_importance", DEFAULT_COMPACT_IMPORTANCE)
        )
        self._access_keep = int(cfg.get("sync_access_keep", DEFAULT_ACCESS_KEEP))
        self.trigger_pct = float(cfg.get("sync_trigger_pct", DEFAULT_TRIGGER_PCT))

        self._memories_dir = self._hermes_home / "memories"
        self._paths = {
            "memory": self._memories_dir / "MEMORY.md",
            "user": self._memories_dir / "USER.md",
        }
        self._backup_dir = self._memories_dir / "backups"

    # -- Reading ------------------------------------------------------------

    def _read_entries(self, target: str) -> List[str]:
        """Parse entries exactly like Hermes memory_tool does."""
        path = self._paths[target]
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return []
        return [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]

    def _char_count(self, target: str) -> int:
        entries = self._read_entries(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def usage_pct(self, target: str, limit: int) -> float:
        if limit <= 0:
            return 0.0
        return (self._char_count(target) / limit) * 100.0

    # -- Mirror lookup ------------------------------------------------------

    def _find_mirror(self, target: str, entry: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """Find the best cognitive-store mirror for a built-in entry.

        Returns (memory_dict, similarity). A mirror is only considered if
        similarity >= mirror_threshold.
        """
        best_mem: Optional[Dict[str, Any]] = None
        best_sim = 0.0
        try:
            candidates = self._store.get_by_target(target)
        except Exception:
            return None, 0.0
        for mem in candidates:
            sim = semantic_similarity(entry, mem.get("content", ""))
            if sim > best_sim:
                best_sim = sim
                best_mem = mem
        if best_sim >= self._mirror_threshold:
            return best_mem, best_sim
        return None, best_sim

    def _mirror_effective_importance(self, mem: Dict[str, Any]) -> float:
        """Decayed importance of a cognitive mirror at this moment."""
        importance = float(mem.get("importance", 0.0))
        last_access = float(mem.get("last_access", 0.0) or 0.0)
        if last_access <= 0:
            last_access = float(mem.get("created_at", 0.0) or 0.0)
        return apply_decay(
            importance, last_access, time.time(), self._params,
        )

    # -- Planning -----------------------------------------------------------

    def build_plan(self, target: str, limit: int) -> SyncPlan:
        """Classify every built-in entry -> keep / compact / remove."""
        entries = self._read_entries(target)
        plan = SyncPlan(
            target=target,
            built_in_path=self._paths[target],
            usage_pct=self.usage_pct(target, limit),
            limit=limit,
        )
        for entry in entries:
            decision = self._decide(target, entry)
            plan.decisions.append(decision)
        return plan

    def _decide(self, target: str, entry: str) -> EntryDecision:
        """Classify a single entry against its cognitive mirror."""
        mirror, sim = self._find_mirror(target, entry)

        # Invariant 1: no mirror -> keep (removing would lose data)
        if mirror is None:
            return EntryDecision(
                entry=entry, action="keep", reason="no cognitive mirror (data loss risk)",
                mirror_similarity=sim,
            )

        mirror_id = mirror.get("id")
        mirror_importance = float(mirror.get("importance", 0.0))
        mirror_access = int(mirror.get("access_count", 0) or 0)
        eff_importance = self._mirror_effective_importance(mirror)
        pinned = bool(mirror.get("pinned", 0))
        origin = mirror.get("origin", "unknown")

        # Invariant 2: pinned -> keep
        if pinned:
            return EntryDecision(
                entry=entry, action="keep", reason="mirror is pinned",
                mirror_id=mirror_id, mirror_similarity=sim,
                mirror_importance=eff_importance, mirror_access=mirror_access,
            )

        # Invariant 2b: user corrections/preferences are the always-injected
        # critical rules (FX graduation criteria, LESSONs, formatting rules).
        # They NEVER leave the built-in file — removing them would make the
        # agent's per-turn context lose them even if the cognitive store
        # retains a searchable copy.
        if origin in ("user_correction", "user_preference"):
            return EntryDecision(
                entry=entry, action="keep", reason=f"critical origin ({origin}) — always visible",
                mirror_id=mirror_id, mirror_similarity=sim,
                mirror_importance=eff_importance, mirror_access=mirror_access,
            )

        # Invariant 3: mirror below safety floor -> keep (mirror may be pruned
        # next session; the built-in copy is the last surviving copy)
        floor = getattr(self._params, "decay_floor", 0.05)
        if eff_importance < max(floor, self._compact_importance):
            return EntryDecision(
                entry=entry, action="keep",
                reason=f"mirror decaying below floor ({eff_importance:.3f} < {max(floor, self._compact_importance):.3f})",
                mirror_id=mirror_id, mirror_similarity=sim,
                mirror_importance=eff_importance, mirror_access=mirror_access,
            )

        # Invariant 4: actively used -> keep
        if mirror_access >= self._access_keep:
            return EntryDecision(
                entry=entry, action="keep", reason=f"actively used (access={mirror_access})",
                mirror_id=mirror_id, mirror_similarity=sim,
                mirror_importance=eff_importance, mirror_access=mirror_access,
            )

        # Short entries (< 80 chars) are already compact — keep them as-is;
        # the cost of a pointer exceeds the savings.
        if len(entry) < 80:
            return EntryDecision(
                entry=entry, action="keep", reason="already compact",
                mirror_id=mirror_id, mirror_similarity=sim,
                mirror_importance=eff_importance, mirror_access=mirror_access,
            )

        # Strong mirror -> safe to remove the built-in copy entirely.
        # Only non-critical origins reach this point: environment facts,
        # agent inferences, and research findings whose full detail lives
        # in the cognitive store.
        if eff_importance >= self._keep_importance:
            return EntryDecision(
                entry=entry, action="remove",
                reason=f"mirror strong ({eff_importance:.3f} >= {self._keep_importance:.3f}), origin={origin}",
                mirror_id=mirror_id, mirror_similarity=sim,
                mirror_importance=eff_importance, mirror_access=mirror_access,
            )

        # Medium mirror -> compact to a pointer
        replacement = (
            f"{entry[:60].rsplit(' ', 1)[0]}… "
            f"(full detail in cognitive memory, id={mirror_id[:8]})"
        )
        return EntryDecision(
            entry=entry, action="compact",
            reason=f"mirror medium ({eff_importance:.3f}) — shorten to pointer",
            mirror_id=mirror_id, mirror_similarity=sim,
            mirror_importance=eff_importance, mirror_access=mirror_access,
            replacement=replacement,
        )

    # -- Applying -----------------------------------------------------------

    def _backup(self) -> None:
        """Back up both built-in files before any write."""
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for target, path in self._paths.items():
            if path.exists():
                dest = self._backup_dir / f"{target}.{stamp}.md"
                shutil.copy2(path, dest)
                logger.info("cognitive-memory: backup %s -> %s", path, dest)

    def _log_changes(self, plan: SyncPlan) -> None:
        """Append every change to the prune log."""
        log_path = self._store.prune_log_path  # same log the store uses
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        for d in plan.removes:
            lines.append(
                f"{stamp} | SYNC-REMOVE [{plan.target}] mirror={d.mirror_id} "
                f"imp={d.mirror_importance:.3f} | {d.entry[:80]}"
            )
        for d in plan.compacts:
            lines.append(
                f"{stamp} | SYNC-COMPACT [{plan.target}] mirror={d.mirror_id} "
                f"imp={d.mirror_importance:.3f} | {d.entry[:80]}"
            )
        if lines:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    def apply_plan(self, plan: SyncPlan, dry_run: bool = True) -> Dict[str, Any]:
        """Apply a compaction plan to the built-in file.

        Returns a report dict. With dry_run=True (default) nothing is
        written — the report describes what WOULD happen.
        """
        if not plan.decisions:
            return {"applied": not dry_run, "changes": 0, "entries": []}

        if dry_run:
            return {
                "applied": False,
                "changes": len(plan.compacts) + len(plan.removes),
                "entries": [d.to_dict() for d in plan.decisions],
            }

        self._backup()

        kept = [d.entry for d in plan.decisions if d.action == "keep"]
        for d in plan.decisions:
            if d.action == "compact" and d.replacement:
                kept.append(d.replacement)
        # 'remove' entries are simply not carried over

        new_content = ENTRY_DELIMITER.join(kept) if kept else ""
        path = plan.built_in_path
        assert path is not None

        # Atomic write: temp file + rename (same pattern as Hermes)
        tmp = path.with_name(f".mem_{path.name}.tmp")
        tmp.write_text(new_content, encoding="utf-8")
        os.replace(tmp, path)

        self._log_changes(plan)

        logger.info(
            "cognitive-memory: SYNC %s applied — %d kept, %d compacted, %d removed (%d -> %d chars)",
            plan.target, len(plan.keeps), len(plan.compacts), len(plan.removes),
            self._char_count_before(plan), len(new_content),
        )
        return {
            "applied": True,
            "changes": len(plan.compacts) + len(plan.removes),
            "counts": {
                "keep": len(plan.keeps),
                "compact": len(plan.compacts),
                "remove": len(plan.removes),
            },
            "new_char_count": len(new_content),
        }

    def _char_count_before(self, plan: SyncPlan) -> int:
        return len(ENTRY_DELIMITER.join(d.entry for d in plan.decisions)) if plan.decisions else 0
