"""Cognitive memory bridge for the WebUI.

Serves the hermes-cognitive-memory store to the Memory panel under
``/api/memory/cognitive`` — list memories with cognitive metadata, pin /
unpin / delete them, add new ones, and view the prune log.

Design notes:

- The Hermes plugin lives at ``~/.hermes/plugins/cognitive/``. Its
  ``__init__.py`` imports Hermes Agent internals (``agent.memory_provider``),
  so this bridge loads only ``store.py`` and ``decay.py`` under a synthetic
  package name — the WebUI process never imports Hermes Agent code and the
  plugin package's side effects never run here.
- The active Hermes home is resolved profile-aware (``api.profiles``), so
  the bridge reads/writes the same SQLite DB the plugin uses.
- DecayParameters are read from the WebUI config snapshot (``memory.cognitive.*``)
  with the same defaults as ``DecayParams`` in the plugin.
- Concurrency: the plugin's MemoryStore is already thread-safe (RLock) and
  sets ``PRAGMA busy_timeout``, so the WebUI process and the Hermes agent
  process can share the DB safely.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

from api.helpers import bad, j

logger = logging.getLogger(__name__)

# Cache the loaded store module so we don't re-exec module code per request.
# Keyed by resolved Hermes home so profile switches get their own store.
_store_lock = threading.Lock()
_store_instances: Dict[str, Any] = {}
_store_params: Optional[Any] = None

_VALID_ORIGINS = {
    "user_correction", "user_preference", "research_finding",
    "environment_fact", "agent_inference", "unknown",
}
_VALID_TEMPORAL = {"timeless", "stable", "ephemeral"}
_VALID_TARGETS = {"memory", "user"}


def _resolve_hermes_home() -> Path:
    """Return the active profile's Hermes home (fall back to ~/.hermes)."""
    try:
        from api.profiles import get_active_hermes_home

        home = get_active_hermes_home()
        if home:
            return Path(home)
    except Exception:
        logger.debug("get_active_hermes_home failed; using ~/.hermes", exc_info=True)
    return Path.home() / ".hermes"


def _plugin_dir(home: Path) -> Path:
    return home / "plugins" / "cognitive"


def _db_path(home: Path) -> Path:
    return home / "cognitive_memory" / "memory.db"


def _prune_log_path(home: Path) -> Path:
    return home / "cognitive_memory" / "prune_log.md"


def _load_plugin_modules(home: Path):
    """Load ``decay``, ``store``, and ``sync`` from the plugin dir without
    running its package ``__init__.py`` (which imports Hermes Agent internals).

    Returns ``(store_module, decay_module, sync_module)``.
    """
    plugin_dir = _plugin_dir(home)
    pkg_name = "_cognitive_webui"
    store_mod = sys.modules.get(pkg_name + ".store")
    decay_mod = sys.modules.get(pkg_name + ".decay")
    sync_mod = sys.modules.get(pkg_name + ".sync")
    if store_mod is not None and decay_mod is not None and sync_mod is not None:
        return store_mod, decay_mod, sync_mod

    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(plugin_dir)]  # type: ignore[attr-defined]
    sys.modules[pkg_name] = pkg

    decay_mod = _load_single_module(pkg_name + ".decay", plugin_dir / "decay.py")
    store_mod = _load_single_module(pkg_name + ".store", plugin_dir / "store.py")
    sync_mod = _load_single_module(pkg_name + ".sync", plugin_dir / "sync.py")
    return store_mod, decay_mod, sync_mod


def _load_single_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_params(store_module, decay_module, home: Path):
    """Build DecayParams from the WebUI config snapshot, plugin defaults otherwise."""
    defaults: Dict[str, Any] = {}
    try:
        from api.config import get_config_snapshot

        cfg = get_config_snapshot()
        mem = cfg.get("memory") if isinstance(cfg, dict) else {}
        cog = mem.get("cognitive") if isinstance(mem, dict) else {}
        if isinstance(cog, dict):
            defaults = {k: v for k, v in cog.items()}
    except Exception:
        logger.debug("config snapshot unavailable; using plugin defaults", exc_info=True)

    DecayParams = decay_module.DecayParams
    try:
        params = DecayParams(**defaults)
    except Exception:
        logger.warning("Invalid cognitive config keys ignored; using defaults", exc_info=True)
        params = DecayParams()
    return params


def _get_store():
    """Return a connected MemoryStore for the active profile (lazy, per-home)."""
    global _store_params
    home = _resolve_hermes_home()
    key = str(home)
    with _store_lock:
        cached = _store_instances.get(key)
        if cached is not None:
            return cached
        db_path = _db_path(home)
        plugin_dir = _plugin_dir(home)
        if not plugin_dir.is_dir():
            raise RuntimeError(
                "Cognitive memory plugin not installed "
                f"(missing {plugin_dir})."
            )
        if not db_path.exists():
            raise RuntimeError(
                "Cognitive memory database not found at "
                f"{db_path}. Restart Hermes with memory.provider=cognitive "
                "to initialize it."
            )
        store_module, decay_module, sync_module = _load_plugin_modules(home)
        params = _build_params(store_module, decay_module, home)
        store = store_module.MemoryStore(db_path, params)
        store.connect()
        _store_instances[key] = store
        _store_params = params
        return store


def _effective_importance(decay_module, row: Dict[str, Any], params, now: float) -> float:
    """Importance after decay from last_access to now (computed on the fly)."""
    try:
        return decay_module.apply_decay(
            float(row.get("importance", 0.0)),
            float(row.get("last_access", now)),
            now,
            params,
            row.get("temporal") or "stable",
        )
    except Exception:
        return float(row.get("importance", 0.0))


def _row_to_payload(decay_module, row: Dict[str, Any], params, now: float) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "target": row.get("target"),
        "content": row.get("content"),
        "importance": round(float(row.get("importance", 0.0)), 4),
        "effective_importance": round(_effective_importance(decay_module, row, params, now), 4),
        "confidence": round(float(row.get("confidence", 0.0)), 4),
        "origin": row.get("origin", "unknown"),
        "reliability": round(float(row.get("reliability", 1.0)), 4),
        "hard_to_find": bool(row.get("hard_to_find")),
        "pinned": bool(row.get("pinned")),
        "temporal": row.get("temporal", "stable"),
        "superseded": bool(row.get("superseded")),
        "supersedes": row.get("supersedes"),
        "access_count": int(row.get("access_count", 0)),
        "created_at": row.get("created_at"),
        "last_access": row.get("last_access"),
    }


def _stats(store, decay_module, params, now: float) -> Dict[str, Any]:
    rows = store.get_all_raw()
    total = len(rows)
    pinned = sum(1 for r in rows if r.get("pinned"))
    hard = sum(1 for r in rows if r.get("hard_to_find"))
    superseded = sum(1 for r in rows if r.get("superseded"))
    by_origin: Dict[str, int] = {}
    by_temporal: Dict[str, int] = {}
    prunable = 0
    for r in rows:
        if r.get("pinned") or r.get("superseded"):
            continue
        eff = _effective_importance(decay_module, r, params, now)
        if decay_module.should_prune(eff, params):
            prunable += 1
        o = r.get("origin", "unknown")
        by_origin[o] = by_origin.get(o, 0) + 1
        t = r.get("temporal", "stable")
        by_temporal[t] = by_temporal.get(t, 0) + 1
    return {
        "total": total,
        "pinned": pinned,
        "hard_to_find": hard,
        "superseded": superseded,
        "prunable": prunable,
        "by_origin": by_origin,
        "by_temporal": by_temporal,
    }


def _read_prune_log(home: Path, limit: int = 30) -> List[str]:
    path = _prune_log_path(home)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return []
    return lines[-limit:]


# ── HTTP handlers ─────────────────────────────────────────────────────────────


def handle_cognitive_get(handler, parsed=None) -> None:
    """GET /api/memory/cognitive → memories + stats + prune log."""
    home = _resolve_hermes_home()
    try:
        store = _get_store()
    except Exception as exc:
        return j(handler, {"available": False, "reason": str(exc)})

    _, decay_module, _ = _load_plugin_modules(home)
    params = _store_params
    now = time.time()
    rows = store.get_all()
    payload = [_row_to_payload(decay_module, r, params, now) for r in rows]
    # Pinned first, then by effective importance descending.
    payload.sort(key=lambda m: (not m["pinned"], -m["effective_importance"]))
    return j(
        handler,
        {
            "available": True,
            "db_path": str(_db_path(home)),
            "memories": payload,
            "stats": _stats(store, decay_module, params, now),
            "prune_log": _read_prune_log(home),
        },
    )


def handle_cognitive_post(handler, body) -> None:
    """POST /api/memory/cognitive — actions: pin, unpin, delete, add, sync_plan, sync_apply."""
    if not isinstance(body, dict):
        return bad(handler, "JSON body required")
    action = body.get("action", "")
    if action not in ("pin", "unpin", "delete", "add", "sync_plan", "sync_apply"):
        return bad(handler, "action must be one of: pin, unpin, delete, add, sync_plan, sync_apply")

    home = _resolve_hermes_home()
    try:
        store = _get_store()
    except Exception as exc:
        return bad(handler, str(exc), 404)

    if action in ("pin", "unpin"):
        mem_id = body.get("id")
        if not isinstance(mem_id, str) or not mem_id:
            return bad(handler, "id is required")
        ok = store.set_pinned(mem_id, action == "pin")
        if not ok:
            return bad(handler, "Memory not found", 404)
        return j(handler, {"ok": True, "action": action, "id": mem_id})

    if action == "delete":
        mem_id = body.get("id")
        if not isinstance(mem_id, str) or not mem_id:
            return bad(handler, "id is required")
        ok = store.remove(mem_id)
        if not ok:
            return bad(handler, "Memory not found", 404)
        return j(handler, {"ok": True, "action": "delete", "id": mem_id})

    if action in ("sync_plan", "sync_apply"):
        return _handle_sync_action(handler, body, store, home, apply=action == "sync_apply")

    # action == "add"
    content = body.get("content")
    if not isinstance(content, str) or not content.strip():
        return bad(handler, "content is required")
    target = body.get("target", "memory")
    if target not in _VALID_TARGETS:
        return bad(handler, "target must be 'memory' or 'user'")
    origin = body.get("origin", "unknown")
    if origin not in _VALID_ORIGINS:
        return bad(handler, f"invalid origin: {origin}")
    temporal = body.get("temporal", "stable")
    if temporal not in _VALID_TEMPORAL:
        return bad(handler, "temporal must be timeless, stable, or ephemeral")
    try:
        reliability = float(body.get("reliability", 1.0))
    except (TypeError, ValueError):
        return bad(handler, "reliability must be a number")
    reliability = max(0.0, min(1.0, reliability))
    tags = body.get("tags")
    if isinstance(tags, list):
        tags = [str(t) for t in tags[:20]]
    else:
        tags = None
    mem_id = store.add(
        target=target,
        content=content.strip(),
        origin=origin,
        tags=tags,
        reliability=reliability,
        hard_to_find=bool(body.get("hard_to_find")),
        pinned=bool(body.get("pinned")),
        temporal=temporal,
    )
    return j(handler, {"ok": True, "action": "add", "id": mem_id})


def _handle_sync_action(handler, body, store, home: Path, apply: bool) -> None:
    """Build (and optionally apply) the built-in memory compaction plan.

    sync_plan  -> dry-run: returns the keep/compact/remove plan per target.
    sync_apply -> writes the plan (backup + prune log automatic).
    """
    target = body.get("target", "both")
    if target not in ("memory", "user", "both"):
        target = "both"

    try:
        _, decay_module, sync_module = _load_plugin_modules(home)
    except Exception as exc:
        return bad(handler, f"sync module unavailable: {exc}", 500)

    params = _store_params
    hermes_home = _resolve_hermes_home()
    sync = sync_module.BuiltinMemorySync(hermes_home, store, params, {})

    limits = {"memory": 2200, "user": 1375}
    try:
        from api.config import get_config_snapshot
        cfg = get_config_snapshot()
        mem_cfg = cfg.get("memory") if isinstance(cfg, dict) else {}
        if isinstance(mem_cfg, dict):
            limits["memory"] = int(mem_cfg.get("memory_char_limit", 2200))
            limits["user"] = int(mem_cfg.get("user_char_limit", 1375))
    except Exception:
        pass

    targets = ["memory", "user"] if target == "both" else [target]
    results = []
    for t in targets:
        plan = sync.build_plan(t, limits.get(t, 2200))
        report = sync.apply_plan(plan, dry_run=not apply) if apply else None
        results.append({"plan": plan.to_dict(), "report": report})

    return j(handler, {
        "ok": True,
        "action": "sync_apply" if apply else "sync_plan",
        "target": target,
        "dry_run": not apply,
        "results": results,
    })
