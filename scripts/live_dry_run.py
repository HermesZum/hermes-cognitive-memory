#!/usr/bin/env python3
"""Live dry-run of the built-in memory sync against the real Hermes data.

Reads ~/.hermes/memories/*.md and ~/.hermes/cognitive_memory/memory.db
and prints the compaction plan WITHOUT writing anything.

Loads the plugin modules directly by file path (like the webui bridge) so
this runs standalone, outside the Hermes Agent environment.
"""
import importlib.util
import sys
import types
from pathlib import Path

REPO = Path("/root/hermes-cognitive-memory")
PKG = REPO / "cognitive_memory"

# Stub package so relative imports inside store.py/sync.py resolve,
# WITHOUT executing the real __init__.py (which imports Hermes internals).
pkg = types.ModuleType("cognitive_memory")
pkg.__path__ = [str(PKG)]
sys.modules["cognitive_memory"] = pkg


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


decay = _load("cognitive_memory.decay", PKG / "decay.py")
store_mod = _load("cognitive_memory.store", PKG / "store.py")
sync_mod = _load("cognitive_memory.sync", PKG / "sync.py")

DecayParams = decay.DecayParams
MemoryStore = store_mod.MemoryStore
BuiltinMemorySync = sync_mod.BuiltinMemorySync

HOME = Path.home() / ".hermes"
db_path = HOME / "cognitive_memory" / "memory.db"
store = MemoryStore(db_path, DecayParams())
store.connect()

sync = BuiltinMemorySync(HOME, store, DecayParams(), {})
limits = {"memory": 2200, "user": 1375}

for target, limit in limits.items():
    pct = sync.usage_pct(target, limit)
    plan = sync.build_plan(target, limit)
    print(f"\n=== {target}.md: {pct:.0f}% of {limit}-char limit ===")
    print(f"counts: {plan.to_dict()['counts']}")
    for d in plan.decisions:
        flag = {"keep": "KEEP   ", "compact": "COMPACT", "remove": "REMOVE "}[d.action]
        print(f"  {flag} | {d.reason}")
        print(f"        | {d.entry[:90]}")
        if d.action == "compact" and d.replacement:
            print(f"        |   -> {d.replacement[:90]}")
    print()

store.close()
