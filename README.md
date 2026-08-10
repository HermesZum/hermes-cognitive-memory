# hermes-cognitive-memory

A Hermes Agent memory provider plugin implementing neuroscience-inspired
cognitive memory mechanisms — Ebbinghaus decay, reconsolidation,
retrieval-induced forgetting, source-confidence erosion, importance-weighted
retrieval, semantic deduplication, conflict supersession, and temporal
relevance — with a management panel in [hermes-webui](https://github.com/nesquena/hermes-webui).

## Why

The built-in Hermes memory stores flat text entries that persist forever
until manually removed. Over time, the store fills with stale environment
facts, outdated preferences, and one-time lessons — all at equal weight,
none of them ever forgotten. The store hits its character limit, and the
agent spends context on irrelevant memories.

This plugin adds a **decay-weighted retrieval layer** on top of the existing
memory system. Old, never-accessed memories fade. Frequently-relevant
memories stay strong. Important research is preserved. The store
self-manages.

## What it does

| # | Mechanism | What it does |
|---|---|---|
| 1 | **Ebbinghaus decay** | Memories fade hyperbolically: `stability = 1/(1 + rate × hours)` |
| 2 | **Reconsolidation** | Retrieved weak memories get a bigger boost than strong ones |
| 3 | **Retrieval-induced forgetting** | Competing memories (same topic, lower rank) lose importance |
| 4 | **Source-confidence decay** | Trust in a memory's origin erodes 10x slower than importance |
| 5 | **Access reinforcement** | Each retrieval adds `access_boost` (spaced repetition) |
| 6 | **Importance-weighted retrieval** | `score = fts_rank × decayed_importance × decayed_confidence × reliability` |
| 7 | **Origin classification** | `user_correction(0.95) > preference(0.85) > research(0.80) > env(0.6) > inference(0.35)` |
| 8 | **Decay-floor pruning** | Memories below their effective floor are pruned at session end |
| 9 | **Temporal relevance** | Timeless rules decay 3x slower, ephemeral state 3x faster |
| 10 | **Research triage** | `research_finding` origin, `reliability` score, `hard_to_find` protection |
| 11 | **Pinning** | `pinned` memories are never pruned; auto-pin after 5 accesses |
| 12 | **Semantic dedup** | Near-duplicates (>85% similar) merge instead of piling up |
| 13 | **Conflict supersession** | Contradictory new info supersedes stale memories |
| 14 | **Prune log** | Every pruned memory is logged to `prune_log.md` for audit |

## Install

### Step 1 — copy the plugin

```bash
mkdir -p ~/.hermes/plugins/cognitive
cp -r cognitive_memory/* ~/.hermes/plugins/cognitive/
```

### Step 2 — activate in config

```bash
hermes config set memory.provider cognitive
hermes config set memory.cognitive.decay_rate 0.02
hermes config set memory.cognitive.decay_floor 0.05
```

The warnings about unrecognized config keys are expected — these are custom
plugin keys, not core Hermes config. They are saved and read correctly.

### Step 3 — restart Hermes

```bash
sudo systemctl restart hermes-webui
```

### Step 4 — verify

```bash
# Check the DB was created
ls -la ~/.hermes/cognitive_memory/memory.db

# Check the logs for plugin loading
journalctl -u hermes-webui --since "5 min ago" --no-pager | grep cognitive
```

You should see:
```
INFO agent.memory_manager: Memory provider 'cognitive' registered (4 tools)
INFO cognitive: cognitive-memory: initialized (db=.../cognitive_memory/memory.db)
INFO run_agent: Memory provider 'cognitive' activated
```

### Step 5 — seed existing memories (first-time only)

The plugin only mirrors **new** `memory` tool writes. Existing memories
written before installation need to be seeded manually. See the
[Seeding](#seeding-existing-memories) section below.

## Config

All config keys live under `memory.cognitive` in `config.yaml`:

```yaml
memory:
  provider: cognitive
  cognitive:
    # Decay rate — higher = faster forgetting (default: 0.02)
    decay_rate: 0.02
    # Confidence decay rate — 10x slower than importance (default: 0.002)
    confidence_decay_rate: 0.002
    # Minimum importance to keep a memory (default: 0.05)
    decay_floor: 0.05
    # Boost on each access (default: 0.3)
    access_boost: 0.3
    # Reconsolidation strength (default: 0.1)
    reconsolidation_rate: 0.1
    # Retrieval-induced forgetting penalty (default: 0.05)
    rif_penalty: 0.05
    # Max memories to inject per turn (default: 15)
    max_context: 15
    # Auto-pin after this many accesses (default: 5)
    auto_pin_threshold: 5
    # Semantic dedup threshold (default: 0.85)
    dedup_threshold: 0.85
    # Source confidence defaults by origin
    source_confidence:
      user_correction: 1.0
      user_preference: 0.9
      research_finding: 0.85
      environment_fact: 0.7
      agent_inference: 0.4
```

## How it works

### Architecture

```
  ┌──────────────────────────────────────────────┐
  │              Hermes Agent                     │
  │                                               │
  │  ┌─────────────┐    ┌──────────────────────┐ │
  │  │ Built-in    │    │  Cognitive Memory     │ │
  │  │ Memory Tool │───▶│  Plugin (this repo)   │ │
  │  │ (MEMORY.md) │    │                       │ │
  │  └─────────────┘    │  on_memory_write() ──▶ │ │
  │                     │  CognitiveMemoryStore  │ │
  │  ┌─────────────┐    │  ├── SQLite + FTS5     │ │
  │  │ prefetch()  │◀───│  ├── Decay math        │ │
  │  │ (per turn)  │    │  ├── Reconsolidation   │ │
  │  └─────────────┘    │  ├── RIF penalty       │ │
  │                     │  ├── Dedup + conflict  │ │
  │                     │  └── Pruning + log     │ │
  │  ┌─────────────┐    │                       │ │
  │  │ 4 Agent     │◀───│  cognitive_search      │ │
  │  │ Tools       │    │  cognitive_stats       │ │
  │  │             │    │  cognitive_remember    │ │
  │  │             │    │  cognitive_forget      │ │
  │  └─────────────┘    └──────────────────────┘ │
  └──────────────────────────────────────────────┘
```

The plugin runs **alongside** the built-in memory system. It does not replace
MEMORY.md or USER.md. The built-in `memory` tool still works as before. The
cognitive provider **mirrors** writes and adds decay-weighted retrieval on top.

### Storage

Memories are stored in an SQLite database at
`$HERMES_HOME/cognitive_memory/memory.db`:

```sql
CREATE TABLE memories (
    id           TEXT PRIMARY KEY,
    target       TEXT NOT NULL,          -- 'memory' or 'user'
    content      TEXT NOT NULL,
    importance   REAL NOT NULL,          -- 0.0 to 1.0
    confidence   REAL NOT NULL,          -- 0.0 to 1.0
    created_at   REAL NOT NULL,           -- unix timestamp
    last_access  REAL NOT NULL,           -- unix timestamp
    access_count INTEGER DEFAULT 0,
    origin       TEXT NOT NULL,          -- see Origin classification
    tags         TEXT DEFAULT '[]',      -- JSON array
    reliability  REAL NOT NULL DEFAULT 1.0,  -- 0-1, source trustworthiness
    hard_to_find INTEGER NOT NULL DEFAULT 0, -- 1 = protected, low floor
    pinned       INTEGER NOT NULL DEFAULT 0, -- 1 = never pruned
    temporal     TEXT NOT NULL DEFAULT 'stable', -- timeless | stable | ephemeral
    superseded   INTEGER NOT NULL DEFAULT 0, -- 1 = replaced by newer info
    supersedes   TEXT DEFAULT NULL           -- id of the memory it replaced
);
```

FTS5 is used for keyword search, with ranking combined with the cognitive
scores. If FTS5 is not available (some SQLite builds don't compile it), the
plugin falls back to `LIKE`-based search automatically. When FTS tables are
created over pre-existing content, the index is rebuilt automatically.

### Lifecycle hooks

| Hook | What happens |
|---|---|
| `initialize()` | Create SQLite DB, run migrations (before schema), load config |
| `prefetch(query)` | FTS5 search → rank by `similarity × decayed importance × confidence × reliability` → return top N |
| `sync_turn()` | Count prunable memories; prune below floor if any (runs on background thread) |
| `on_memory_write()` | Mirror built-in memory writes with importance scoring + dedup/conflict check |
| `on_turn_start()` | Apply decay (computed on the fly); log status |
| `on_session_end()` | Prune memories below effective floor, write prune log |

### Decay math — computed on the fly, never stored

Decay is **not** written back to the stored `importance` value. It is
computed at retrieval time from the stored value, `last_access`, and the
current time. This eliminates compounding errors entirely — a memory can
never decay twice from the same baseline.

```
decayed_importance = stored_importance × 1/(1 + decay_rate × hours_since_last_access)
```

Temporal multiplier adjusts the rate:
- `timeless` → rate × 0.3 (survives ~3x longer)
- `stable` → rate × 1.0 (default)
- `ephemeral` → rate × 3.0 (clears ~3x faster)

When a memory is retrieved (prefetch hits it):

```
importance = min(1.0, importance + access_boost)
last_access = now
access_count += 1
# auto-pin at auto_pin_threshold (default 5) accesses
```

Competing memories (same FTS match, lower rank) get:

```
importance = max(0.0, importance - rif_penalty)
```

### Origin classification and protection

Memories are classified by origin on write, which sets initial importance,
confidence, and decay floor:

| Origin | Initial importance | Initial confidence | Base floor | Example |
|---|---|---|---|---|
| `user_correction` | 0.95 | 1.00 | 0.02 | "Never pkill -f hermes" |
| `user_preference` | 0.85 | 0.90 | 0.03 | "Prefers step-by-step plans in chat" |
| `research_finding` | 0.80 | 0.85 | 0.03 | "EURUSD 0.85 corr with DXY, backtested 2y" |
| `environment_fact` | 0.60 | 0.70 | 0.05 | "SSH user is vm_user" |
| `agent_inference` | 0.35 | 0.40 | 0.05 | "Model fallback chain behavior" |

Classification is keyword-based in `decay.py:classify_origin()`. It checks
markers like "never", "do not", "lesson" (corrections), "prefers", "wants"
(preferences), research keywords (see below), "is", "runs on", "uses"
(environment facts), and falls back to inference.

**Research detection** — content with 2+ research keywords is auto-classified
`research_finding`: `research`, `study`, `evidence`, `backtested`, `win rate`,
`expectancy`, `drawdown`, `sharpe`, `correlation`, `benchmark`, `verified`,
`documented`, `peer-reviewed`, `published`, etc.

**Protection layers** (cumulative):
- `reliability` (0-1) — multiplies into search ranking; reliable sources
  rank higher. Pass 0.8–1.0 for verified research, 0.3–0.5 for casual inference.
- `hard_to_find` — +0.15 importance on creation and floor drops to 0.01
  (~200+ days survival). Use for obscure docs, rare correlations, hard-won fixes.
- `pinned` — never pruned, period. Set manually or auto-pinned after 5 accesses.
- Access-count floor: `floor / (1 + access_count × 0.1)` — frequently-used
  memories survive progressively longer.

### Semantic dedup and conflict supersession

When a new memory is added, it is compared against existing non-superseded
memories of the same target using token-set (Jaccard) similarity:

- **Similarity > 0.85** → merge instead of add. The combined entry keeps the
  higher importance, higher reliability, OR of the protection flags, max of
  access_count, and the longer content. No duplicates pile up.
- **Similarity > 0.55 with conflicting values** (numbers changed, negation
  flipped) → the old memory is marked `superseded=1`, importance dropped to 0,
  excluded from search, and pruned at next session end. The new memory records
  `supersedes=<old_id>` for audit. Only the current fact survives.

### Prune log

Every pruned memory is appended to `$HERMES_HOME/cognitive_memory/prune_log.md`
before deletion:

```markdown
## 2026-08-10 21:50
- id=`abc123` | origin=`agent_inference` | imp=0.03 | "Some old inference..."
```

The log itself is never pruned — it is the audit trail of everything the
decay system ever removed. If you see something you wanted to keep, re-add
it and pin it.

## Agent tools

The plugin exposes 4 tools that the LLM can call during a conversation:

### `cognitive_search`

Search the cognitive memory store with decay-weighted ranking.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Search terms |
| `limit` | int | 10 | Max results (clamped 1–30) |
| `target` | string | null | Filter by target: `memory` or `user` |

Returns a JSON array of matches, each with content, importance, confidence,
origin, reliability, temporal class, access_count, and last_access (as ISO
timestamp). Superseded memories are excluded.

### `cognitive_stats`

Show cognitive memory store statistics.

No parameters. Returns:
```json
{
  "total_memories": 15,
  "pinned_count": 2,
  "hard_to_find_count": 1,
  "prunable_count": 0,
  "superseded_count": 0,
  "average_importance": 0.733,
  "by_origin": {"user_correction": 7, "user_preference": 3},
  "by_temporal": {"stable": 12, "timeless": 3},
  "by_target": {"memory": 9, "user": 6},
  "auto_pin_threshold": 5
}
```

### `cognitive_remember`

Store a new memory with cognitive metadata.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `content` | string | required | The fact to remember |
| `target` | string | `memory` | `memory` or `user` |
| `origin` | string | auto-detect | `user_correction`, `user_preference`, `research_finding`, `environment_fact`, `agent_inference` |
| `reliability` | number | 1.0 | 0-1, source trustworthiness |
| `hard_to_find` | bool | false | Protect from pruning (floor 0.01) |
| `pinned` | bool | false | Never prune, regardless of decay |
| `temporal` | string | auto-detect | `timeless`, `stable`, `ephemeral` |
| `tags` | array | [] | Optional tags |

Dedup and conflict supersession run automatically on every write.

### `cognitive_forget`

Delete a memory by its ID.

| Parameter | Type | Description |
|---|---|---|
| `memory_id` | string | required | The memory ID to delete |

## WebUI management panel

If you run [hermes-webui](https://github.com/nesquena/hermes-webui), the
Memory panel gains a **Cognitive Memory** section:

- Browse all memories with cognitive metadata (origin badge, PINNED /
  HARD TO FIND badges, importance bar, reliability, access count,
  last-access age)
- Search + filters (Pinned, Research findings, Hard to find, Timeless,
  Ephemeral)
- **Pin / Unpin / Delete** on every card
- **+ Add memory** form (content, target, origin, temporal, reliability,
  pinned / hard-to-find checkboxes)
- Stats chips: total / pinned / hard-to-find / prunable / superseded
- Collapsible **prune log** viewer

This is an optional integration — it only reads/writes the same SQLite store
the plugin uses, and never imports Hermes Agent code into the WebUI process
(it loads `store.py`/`decay.py` directly, bypassing the package `__init__`).

Install — copy one bridge module + 2 dispatcher lines + JS/CSS blocks:

```bash
cd /root/hermes-webui
cp ../hermes-cognitive-memory/webui_integration/api/cognitive_bridge.py api/
# …then apply the 3 small patches documented in:
# webui_integration/INSTALL.md
sudo systemctl restart hermes-webui
```

Full instructions, the exact patch locations, the API contract, and design
notes are in [`webui_integration/INSTALL.md`](webui_integration/INSTALL.md).
The reference files live under `webui_integration/static/`.

## Seeding existing memories

The `on_memory_write` hook only fires on **new** writes. Memories that
existed before the plugin was installed need to be seeded manually.

A seed script is included at `scripts/seed_from_memory.py`. Usage:

```bash
python scripts/seed_from_memory.py --hermes-home ~/.hermes
```

Use `--dry-run` to preview classifications without writing:

```bash
python scripts/seed_from_memory.py --hermes-home ~/.hermes --dry-run
```

This reads the built-in MEMORY.md and USER.md files, classifies each entry
by origin, and inserts them into the cognitive store with appropriate
importance, confidence, reliability, and temporal scores.

## Relationship to built-in memory

This plugin **complements** the built-in memory system — it does not
replace it:

- The built-in `memory` tool (MEMORY.md / USER.md) still works exactly as
  before
- `on_memory_write()` mirrors every built-in write to the cognitive store
- `prefetch()` injects decay-weighted results into the conversation context
  alongside the built-in memory injection
- The 4 `cognitive_*` tools are additional — they don't interfere with the
  existing `memory` tool
- **The built-in memory files are never modified or deleted by this plugin.**

You can use both systems together. The built-in memory gives you simple,
durable, human-readable entries. The cognitive layer adds decay, ranking,
protection, dedup, and auto-pruning on top.

## Troubleshooting

### DB not created after install

Check that Hermes was actually restarted:

```bash
journalctl -u hermes-webui --since "10 min ago" --no-pager | grep cognitive
```

If you see no cognitive log lines, the plugin wasn't loaded. Verify the
plugin files are in the right place:

```bash
ls ~/.hermes/plugins/cognitive/cognitive_memory/
```

Should contain `__init__.py`, `decay.py`, `store.py`, and `plugin.yaml`.

### "no such column: pinned" / missing columns on existing DB

This was a migration-order bug in earlier versions (fixed in `d3566c6`).
Migrations now run **before** the base schema, and the FTS index is rebuilt
when created over pre-existing content. If you hit it, update `store.py`
from this repo, delete the DB (`rm ~/.hermes/cognitive_memory/memory.db`),
and re-seed.

### FTS5 not available

Some SQLite builds don't compile FTS5. The plugin detects this at startup
and falls back to `LIKE`-based search automatically. You'll see a log line:

```
WARNING cognitive: FTS5 not available, falling back to LIKE search
```

Search results will be slightly lower quality (no ranking by relevance),
but the plugin still works — decay-weighted importance scoring applies
regardless of the search method.

### Config key warnings

When you run `hermes config set memory.cognitive.decay_rate 0.02`, Hermes
warns:

```
⚠ 'memory.cognitive.decay_rate' is not a recognized config key
```

This is expected. These are custom plugin keys, not core Hermes config
keys. They are saved to config.yaml and read by the plugin at startup.

### Database is locked

If you see `sqlite3.OperationalError: database is locked`, it means another
process is accessing the DB. The plugin sets `PRAGMA busy_timeout=5000`
(5-second wait) to handle this. If it persists, check for orphaned Hermes
processes:

```bash
fuser ~/.hermes/cognitive_memory/memory.db
```

### Reset the cognitive store

To wipe all memories and start fresh:

```bash
rm -rf ~/.hermes/cognitive_memory/
sudo systemctl restart hermes-webui
```

Then re-seed from built-in memory:

```bash
python scripts/seed_from_memory.py --hermes-home ~/.hermes
```

## Security

- All SQL queries use parameterized `?` placeholders — no string
  formatting in queries
- LIKE wildcards (`%`, `_`, `\`) are escaped in all LIKE queries to prevent
  wildcard injection
- The plugin runs within Hermes's existing process — no subprocess calls,
  no network access, no file access outside the DB path
- Thread-safe: all DB operations are guarded by an `RLock`
- Connections are closed on schema creation failure (no resource leaks)
- Input validation: limit is clamped to 1–30, target and origin are
  validated against enums
- `PRAGMA busy_timeout=5000` prevents "database is locked" under concurrent
  access

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_store.py -v

# Run with coverage
python -m pytest tests/ --cov=cognitive_memory --cov-report=term-missing

# Smoke test (verify plugin loads)
python -c "from cognitive_memory import CognitiveMemoryProvider; p = CognitiveMemoryProvider(); print(p.name)"
```

### Running the full audit pipeline

```bash
# Static security scan
grep -rn "os.system\|subprocess.*shell=True\|eval(\|exec(\|pickle.loads" cognitive_memory/

# Syntax check
python -m py_compile cognitive_memory/*.py

# Full test suite
python -m pytest tests/ -v
```

## Project structure

```
hermes-cognitive-memory/
├── cognitive_memory/
│   ├── __init__.py      # CognitiveMemoryProvider (provider, tools, hooks)
│   ├── decay.py         # Decay math, origin/temporal classification, dedup
│   ├── store.py         # SQLite + FTS5 storage, dedup, conflict, prune log
│   └── plugin.yaml      # Plugin manifest for Hermes discovery
├── scripts/
│   └── seed_from_memory.py  # Seed existing memories on first install
├── tests/
│   ├── test_decay.py    # Decay math, reconsolidation, RIF, temporal, origins
│   ├── test_store.py    # Store CRUD, dedup, conflict, prune, set_pinned
│   └── test_provider.py # Plugin interface, prefetch, tools, lifecycle
├── webui_integration/
│   ├── api/cognitive_bridge.py  # WebUI backend bridge (optional)
│   ├── static/                  # Reference JS/CSS blocks
│   └── INSTALL.md               # WebUI integration instructions
├── pyproject.toml
├── pytest.ini
├── LICENSE (MIT)
└── README.md
```

## Audit history

- **v1.0** — Initial implementation, 67 tests passing
- **v1.0-audit1** — Independent review found 4 issues (race condition in
  search, resource leak in connect, FTS5 not checked, negative limit).
  All fixed. 67 tests passing.
- **v1.0-audit2** — LIKE wildcard injection in `remove_by_content` found
  and fixed. 2 regression tests added. 69 tests passing.
- **v1.0-audit3** — Targeted verification of all fixes. No new issues.
  69 tests passing. Approved for production.
- **v1.1** — Decay architecture audit: 3 critical bugs found and fixed —
  compounding decay (importance re-decayed from stale `last_access`),
  double decay (search re-decayed already-decayed values), confidence
  compounding. Decay now computed on the fly, never stored. Origin-based
  floors added. `decay_rate` tuned 0.15 → 0.02. 73 tests passing.
- **v1.2** — Research triage: `research_finding` origin (auto-detected),
  `reliability` score, `hard_to_find` flag, `pinned` flag, schema migration
  (3 new columns). 88 tests passing.
- **v1.3** — 6 triage improvements: auto-pin by access count, prune log,
  access-count decay floor, semantic dedup, conflict supersession, temporal
  relevance. 115 tests passing.
- **v1.4** — WebUI management panel: `cognitive_bridge.py` + Memory panel
  section (list/search/filter, pin/unpin/delete, add, prune log viewer).
  119 tests passing. Committed to hermes-webui (`9ee5b680`, unpushed).
- **v1.4-audit** — Live-system audit: found migration-order bug (index on
  `pinned` created before ALTER migrations on pre-existing DBs) and FTS
  rebuild gap. Fixed (`d3566c6`), 3 regression tests added. 122 tests
  passing. Verified live: 15 memories, write path (add→pin→unpin→delete)
  leaves zero residue.

## License

MIT
