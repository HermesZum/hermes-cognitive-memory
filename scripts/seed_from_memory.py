#!/usr/bin/env python3
"""Seed the cognitive memory store from built-in MEMORY.md and USER.md.

Usage:
    python scripts/seed_from_memory.py --hermes-home ~/.hermes
    python scripts/seed_from_memory.py --hermes-home ~/.hermes --dry-run
"""

import argparse
import json
import os
import re
import sys
import time

# Add the plugin to the path so we can import it
script_dir = os.path.dirname(os.path.abspath(__file__))
plugin_dir = os.path.join(os.path.dirname(script_dir), "cognitive_memory")
sys.path.insert(0, plugin_dir)

from cognitive_memory import CognitiveMemoryProvider
from cognitive_memory.decay import classify_origin, initial_importance, initial_confidence, DecayParams

# Matches memory entries separated by § delimiters in MEMORY.md / USER.md
ENTRY_DELIMITER = re.compile(r"^§\s*$", re.MULTILINE)


def parse_memory_file(filepath: str) -> list[dict]:
    """Parse a MEMORY.md or USER.md file into individual memory entries.

    These files use § as entry delimiter. Each entry is a plain text block.
    """
    if not os.path.exists(filepath):
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on § delimiter
    raw_entries = ENTRY_DELIMITER.split(content)

    entries = []
    for raw in raw_entries:
        # Strip whitespace, headers, and empty lines
        text = raw.strip()
        if not text:
            continue

        # Skip section headers like "MEMORY (your personal notes)"
        if text.startswith("MEMORY (") or text.startswith("USER PROFILE ("):
            # Extract the header line and continue with the rest
            lines = text.split("\n")
            # Find where the header block ends (after the ═══ line)
            in_header = False
            body_lines = []
            for line in lines:
                if "═══" in line:
                    in_header = not in_header
                    continue
                if not in_header:
                    body_lines.append(line)
            text = "\n".join(body_lines).strip()

        if not text:
            continue

        entries.append({"content": text})

    return entries


def seed_store(provider: CognitiveMemoryProvider, entries: list[dict], target: str, dry_run: bool = False) -> int:
    """Seed the cognitive store with parsed entries.

    Args:
        provider: The cognitive memory provider instance.
        entries: List of {"content": "..."} dicts.
        target: "memory" or "user".
        dry_run: If True, print what would be stored but don't write.

    Returns:
        Number of entries actually stored.
    """
    stored = 0

    for entry in entries:
        content = entry["content"]

        # Classify origin from content keywords
        origin = classify_origin("add", target, content, None)
        # Use default DecayParams if provider not fully initialized
        try:
            params = provider._decay_params
        except AttributeError:
            params = DecayParams()
        importance = initial_importance(origin, params)
        confidence = initial_confidence(origin, params)

        if dry_run:
            print(f"  [{target}] origin={origin} importance={importance:.2f} confidence={confidence:.2f}")
            preview = content[:80].replace("\n", " ")
            print(f"    {preview}...")
            stored += 1
            continue

        # Store via the provider's tool interface
        result = provider._handle_remember({
            "content": content,
            "target": target,
            "origin": origin,
            "tags": ["seeded"],
        })

        if "error" not in result:
            stored += 1
            print(f"  [{target}] ✓ {origin} (importance={importance:.2f}) — {content[:60].replace(chr(10), ' ')}...")
        else:
            print(f"  [{target}] ✗ {result['error']} — {content[:60].replace(chr(10), ' ')}...")

    return stored


def main():
    parser = argparse.ArgumentParser(
        description="Seed cognitive memory from built-in MEMORY.md and USER.md"
    )
    parser.add_argument(
        "--hermes-home",
        default=os.path.expanduser("~/.hermes"),
        help="Path to Hermes home directory (default: ~/.hermes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be stored without writing to the DB",
    )
    args = parser.parse_args()

    hermes_home = os.path.expanduser(args.hermes_home)

    # Parse memory files — check both ~/.hermes/ and ~/.hermes/memories/
    memory_file = os.path.join(hermes_home, "MEMORY.md")
    user_file = os.path.join(hermes_home, "USER.md")

    # Hermes stores memories in a memories/ subdirectory
    if not os.path.exists(memory_file):
        alt = os.path.join(hermes_home, "memories", "MEMORY.md")
        if os.path.exists(alt):
            memory_file = alt
    if not os.path.exists(user_file):
        alt = os.path.join(hermes_home, "memories", "USER.md")
        if os.path.exists(alt):
            user_file = alt

    print(f"Hermes home: {hermes_home}")
    print()

    memory_entries = parse_memory_file(memory_file)
    user_entries = parse_memory_file(user_file)

    print(f"Found {len(memory_entries)} memory entries in MEMORY.md")
    print(f"Found {len(user_entries)} user entries in USER.md")
    print()

    if not memory_entries and not user_entries:
        print("No entries found. Make sure MEMORY.md and USER.md exist and contain §-delimited entries.")
        return 1

    if args.dry_run:
        print("DRY RUN — no data will be written\n")

    # Initialize the provider
    provider = CognitiveMemoryProvider()
    provider._hermes_home = hermes_home
    provider._config = {
        "decay_rate": 0.15,
        "decay_floor": 0.05,
        "access_boost": 0.3,
        "max_context": 15,
        "reconsolidation_rate": 0.1,
        "rif_penalty": 0.05,
    }
    provider.initialize({})

    # Seed memory entries
    print("Seeding memory entries:")
    mem_count = seed_store(provider, memory_entries, "memory", args.dry_run)
    print()

    # Seed user entries
    print("Seeding user entries:")
    user_count = seed_store(provider, user_entries, "user", args.dry_run)
    print()

    print(f"Done. {mem_count} memory + {user_count} user entries {'would be ' if args.dry_run else ''}seeded.")

    if not args.dry_run:
        stats = provider._handle_stats({})
        print(f"\nCognitive store stats:")
        print(json.dumps(stats, indent=2))

    provider.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())