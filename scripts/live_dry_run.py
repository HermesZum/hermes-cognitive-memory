#!/usr/bin/env python3
"""Live dry-run of the built-in memory sync against the real Hermes data.

Reads ~/.hermes/memories/*.md and ~/.hermes/cognitive_memory/memory.db
and prints the compaction plan WITHOUT writing anything.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/root/hermes-cognitive-memory")

from cognitive_memory.decay import DecayParams
from cognitive_memory.store import MemoryStore
from cognitive_memory.sync import BuiltinMemorySync

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
