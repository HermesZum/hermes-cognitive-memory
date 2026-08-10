"""Tests for the built-in memory sync/compaction module."""

import shutil
import tempfile
import time
from pathlib import Path

import pytest

from cognitive_memory.decay import DecayParams, semantic_similarity
from cognitive_memory.store import MemoryStore
from cognitive_memory.sync import (
    BuiltinMemorySync,
    ENTRY_DELIMITER,
    SyncPlan,
)


@pytest.fixture
def home(tmp_path: Path):
    """A fake HERMES_HOME with memories dir."""
    mem_dir = tmp_path / "memories"
    mem_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def store(tmp_path: Path):
    db_path = tmp_path / "cognitive" / "memory.db"
    s = MemoryStore(db_path, DecayParams())
    s.connect()
    yield s
    s.close()


@pytest.fixture
def sync(home: Path, store: MemoryStore) -> BuiltinMemorySync:
    return BuiltinMemorySync(home, store, DecayParams(), {})


def _write_memory(home: Path, target: str, entries):
    path = home / "memories" / f"{'MEMORY' if target == 'memory' else 'USER'}.md"
    path.write_text(ENTRY_DELIMITER.join(entries), encoding="utf-8")


# -- Reading ----------------------------------------------------------------

def test_read_entries_parses_delimiter(sync, home):
    _write_memory(home, "memory", ["one", "two", "three"])
    assert sync._read_entries("memory") == ["one", "two", "three"]


def test_read_entries_empty_file(sync, home):
    (home / "memories" / "MEMORY.md").write_text("", encoding="utf-8")
    assert sync._read_entries("memory") == []


def test_read_entries_missing_file(sync, home):
    assert sync._read_entries("memory") == []


def test_usage_pct(sync, home):
    _write_memory(home, "memory", ["12345"])
    assert sync.usage_pct("memory", 100) == 5.0
    assert sync.usage_pct("memory", 0) == 0.0


# -- Mirror lookup ----------------------------------------------------------

def test_find_mirror_matches(store, sync, home):
    store.add("memory", "FX demo: starting $1k, micro lots, risk cap 1-1.5%", origin="user_preference")
    mirror, sim = sync._find_mirror("memory", "FX demo: starting $1k, micro lots, risk cap 1-1.5% per trade")
    assert mirror is not None
    assert sim >= 0.8


def test_find_mirror_no_match(store, sync):
    store.add("memory", "WhatsApp allowlist +000", origin="environment_fact")
    mirror, sim = sync._find_mirror("memory", "The weather in Lisbon is sunny")
    assert mirror is None


def test_find_mirror_respects_target(store, sync):
    store.add("user", "User prefers markdown formatting", origin="user_preference")
    mirror, sim = sync._find_mirror("memory", "User prefers markdown formatting")
    assert mirror is None  # different target


# -- Planning ---------------------------------------------------------------

def test_plan_keep_no_mirror(sync, home):
    """Entry with no mirror must be kept (data loss risk)."""
    _write_memory(home, "memory", ["This entry has no cognitive mirror at all"])
    plan = sync.build_plan("memory", 2200)
    assert len(plan.keeps) == 1
    assert plan.keeps[0].reason == "no cognitive mirror (data loss risk)"


def test_plan_keep_pinned(store, sync, home):
    """Pinned mirrors protect the built-in entry."""
    mid = store.add("memory", "LESSON: never modify working configs without approval", origin="user_correction")
    store.set_pinned(mid, True)
    _write_memory(home, "memory", ["LESSON: never modify working configs without approval"])
    plan = sync.build_plan("memory", 2200)
    assert len(plan.keeps) == 1
    assert plan.keeps[0].reason == "mirror is pinned"


def test_plan_keep_active(store, sync, home):
    """High-access mirrors protect the built-in entry."""
    mid = store.add("memory", "Model fallback chain glm big-pickle deepseek", origin="environment_fact")
    # Simulate access_count >= access_keep by direct update
    store._conn.execute("UPDATE memories SET access_count = 5 WHERE id = ?", (mid,))
    store._conn.commit()
    _write_memory(home, "memory", ["Model fallback chain glm big-pickle deepseek"])
    plan = sync.build_plan("memory", 2200)
    assert len(plan.keeps) == 1
    assert "actively used" in plan.keeps[0].reason


def test_plan_keep_short_entry(store, sync, home):
    """Short entries are kept even when mirrored (pointer costs more)."""
    store.add("memory", "Short fact", origin="agent_inference")
    _write_memory(home, "memory", ["Short fact"])
    plan = sync.build_plan("memory", 2200)
    assert len(plan.keeps) == 1
    assert plan.keeps[0].reason == "already compact"


def test_plan_remove_strong_mirror(store, sync, home):
    """Strong mirror (importance >= keep) -> safe to remove built-in copy."""
    store.add(
        "memory",
        "hermes-webui :8780 systemd fork github.com/MyOrg/hermes-webui upstream nesquena hermes-webui nginx subdomain hermes.lan.local Open WebUI 8082 chat only Memory Cognitive panels exist only in hermes-webui",
        origin="environment_fact", importance=0.95,
    )
    _write_memory(
        home, "memory",
        ["hermes-webui :8780 systemd fork github.com/MyOrg/hermes-webui upstream nesquena hermes-webui nginx subdomain hermes.lan.local Open WebUI 8082 chat only Memory Cognitive panels exist only in hermes-webui"],
    )
    plan = sync.build_plan("memory", 2200)
    assert len(plan.removes) == 1
    assert "mirror strong" in plan.removes[0].reason


def test_plan_compact_medium_mirror(store, sync, home):
    """Medium mirror -> compact to pointer."""
    content = (
        "The pipeline consists of a morning briefing before the session starts "
        "and then a thirty minute level watchdog that monitors the key price "
        "levels and then a trade journaling step after each session with "
        "strategy templates stored in the finances folder."
    )
    store.add("memory", content, origin="agent_inference", importance=0.20)
    _write_memory(home, "memory", [content])
    plan = sync.build_plan("memory", 2200)
    assert len(plan.compacts) == 1
    assert plan.compacts[0].replacement is not None
    assert "cognitive memory" in plan.compacts[0].replacement


def test_plan_keep_decaying_mirror(store, sync, home):
    """Mirror below safety floor -> keep (mirror may be pruned soon)."""
    store.add("memory", "Old stale inference about something long forgotten", origin="agent_inference", importance=0.01)
    _write_memory(home, "memory", ["Old stale inference about something long forgotten"])
    plan = sync.build_plan("memory", 2200)
    assert len(plan.keeps) == 1
    assert "floor" in plan.keeps[0].reason


def test_plan_keep_critical_origin(store, sync, home):
    """user_correction/user_preference entries NEVER leave the built-in file."""
    content = (
        "FX project (Trader): demo-first. Trade demo until a strategy "
        "graduates with twenty journaled trades and positive expectancy then "
        "scale into real money slowly never rush no strategy hopping"
    )
    store.add("memory", content, origin="user_correction", importance=0.95)
    _write_memory(home, "memory", [content])
    plan = sync.build_plan("memory", 2200)
    assert len(plan.keeps) == 1
    assert plan.keeps[0].reason == "critical origin (user_correction) — always visible"


def test_plan_remove_noncritical_strong(store, sync, home):
    """Environment facts with strong mirrors may be trimmed (detail in store)."""
    content = (
        "hermes-webui :8780 systemd fork github.com/MyOrg/hermes-webui "
        "upstream nesquena hermes-webui nginx subdomain hermes.lan.local "
        "Open WebUI 8082 chat only Memory Cognitive panels exist only in hermes-webui"
    )
    store.add("memory", content, origin="environment_fact", importance=0.95)
    _write_memory(home, "memory", [content])
    plan = sync.build_plan("memory", 2200)
    assert len(plan.removes) == 1
    assert "origin=environment_fact" in plan.removes[0].reason


# -- Applying ---------------------------------------------------------------

def test_apply_dry_run_writes_nothing(sync, home):
    _write_memory(home, "memory", ["entry one", "entry two"])
    before = (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    plan = sync.build_plan("memory", 2200)
    report = sync.apply_plan(plan, dry_run=True)
    assert report["applied"] is False
    after = (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    assert before == after


def test_apply_removes_and_compacts(store, sync, home):
    strong = (
        "hermes-webui :8780 systemd fork github.com/MyOrg/hermes-webui "
        "upstream nesquena hermes-webui nginx subdomain hermes.lan.local "
        "Open WebUI 8082 chat only Memory Cognitive panels exist only in hermes-webui"
    )
    medium = (
        "The pipeline consists of a morning briefing before the session starts "
        "and then a thirty minute level watchdog that monitors the key price "
        "levels and then a trade journaling step after each session."
    )
    store.add("memory", strong, origin="environment_fact", importance=0.95)
    store.add("memory", medium, origin="agent_inference", importance=0.20)
    store.add("memory", "Keep me, no mirror", origin="agent_inference")

    _write_memory(home, "memory", [strong, medium, "Keep me, no mirror"])
    plan = sync.build_plan("memory", 2200)
    assert len(plan.removes) == 1
    assert len(plan.compacts) == 1

    report = sync.apply_plan(plan, dry_run=False)
    assert report["applied"] is True
    assert report["counts"]["remove"] == 1
    assert report["counts"]["compact"] == 1

    # Backup created
    backups = list((home / "memories" / "backups").glob("memory.*.md"))
    assert len(backups) == 1

    # File has the kept + compacted entries, not the removed one
    remaining = (home / "memories" / "MEMORY.md").read_text(encoding="utf-8")
    assert "Keep me, no mirror" in remaining
    assert "full detail in cognitive memory" in remaining
    assert "hermes-webui :8780 systemd" not in remaining


def test_apply_empty_plan(sync, home):
    (home / "memories" / "MEMORY.md").write_text("", encoding="utf-8")
    plan = SyncPlan(target="memory", built_in_path=home / "memories" / "MEMORY.md")
    report = sync.apply_plan(plan, dry_run=False)
    assert report["changes"] == 0


def test_apply_backs_up_both_files(store, sync, home):
    store.add("memory", "dupe content that is long enough to be considered for removal yes indeed", origin="environment_fact", importance=0.9)
    _write_memory(home, "memory", ["dupe content that is long enough to be considered for removal yes indeed"])
    _write_memory(home, "user", ["user entry one", "user entry two"])
    plan = sync.build_plan("memory", 2200)
    sync.apply_plan(plan, dry_run=False)
    backups = {p.name for p in (home / "memories" / "backups").glob("*.md")}
    assert any(n.startswith("memory.") for n in backups)
    assert any(n.startswith("user.") for n in backups)


def test_apply_logs_changes(store, sync, home):
    content = (
        "log me removal candidate with enough length to qualify for the "
        "removal action because it exceeds eighty characters easily yes"
    )
    store.add("memory", content, origin="environment_fact", importance=0.9)
    _write_memory(home, "memory", [content])
    plan = sync.build_plan("memory", 2200)
    sync.apply_plan(plan, dry_run=False)
    log_text = store.prune_log_path.read_text(encoding="utf-8")
    assert "SYNC-REMOVE" in log_text


# -- Config ----------------------------------------------------------------

def test_config_overrides(sync, home):
    s2 = BuiltinMemorySync(home, sync._store, DecayParams(), {
        "sync_mirror_threshold": 0.5,
        "sync_keep_importance": 0.9,
        "sync_compact_importance": 0.4,
        "sync_access_keep": 1,
        "sync_trigger_pct": 50,
    })
    assert s2._mirror_threshold == 0.5
    assert s2._keep_importance == 0.9
    assert s2._compact_importance == 0.4
    assert s2._access_keep == 1
    assert s2.trigger_pct == 50


# -- Data-loss window regression (audit finding) ----------------------------

def test_plan_keep_hard_to_find_mirror(store, sync, home):
    """hard_to_find mirrors NEVER leave the built-in file, even with a strong
    mirror — the store floor (0.01) is a grace period, not a guarantee."""
    content = (
        "EURUSD shows 0.85 correlation with DXY index backtested over two "
        "years with a fifty eight percent win rate and this is a long entry "
        "that would otherwise qualify for removal by strength alone"
    )
    store.add(
        "memory", content,
        origin="research_finding", importance=0.95, hard_to_find=True,
    )
    _write_memory(home, "memory", [content])
    plan = sync.build_plan("memory", 2200)
    assert len(plan.keeps) == 1
    assert plan.keeps[0].reason == "hard-to-find research — must remain visible"
    assert len(plan.removes) == 0
    assert len(plan.compacts) == 0


def test_apply_pins_removed_mirror(store, sync, home):
    """When a built-in copy is removed, its mirror is pinned so the store copy
    is permanent — closing the decay-to-prune data-loss window."""
    content = (
        "hermes-webui :8780 systemd fork github.com/MyOrg/hermes-webui "
        "upstream nesquena hermes-webui nginx subdomain hermes.lan.local "
        "Open WebUI 8082 chat only Memory Cognitive panels exist only in hermes-webui"
    )
    mem_id = store.add("memory", content, origin="environment_fact", importance=0.95)
    _write_memory(home, "memory", [content])
    plan = sync.build_plan("memory", 2200)
    assert len(plan.removes) == 1

    sync.apply_plan(plan, dry_run=False)

    # The mirror is now pinned -> prune() will never delete it
    mem = store.get(mem_id)
    assert mem is not None
    assert bool(mem["pinned"]) is True

    # And prune() leaves it alone even if importance decays to the floor
    pruned = store.prune()
    mem_after = store.get(mem_id)
    assert mem_after is not None
    assert bool(mem_after["pinned"]) is True
