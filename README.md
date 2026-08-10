# hermes-cognitive-memory

A Hermes Agent memory provider plugin implementing neuroscience-inspired
cognitive memory mechanisms — decay, reconsolidation, retrieval-induced
forgetting, source-confidence erosion, and importance-weighted retrieval.

## Why

The built-in Hermes memory stores flat text entries that persist forever
until manually removed. Over time, the store fills with stale environment
facts, outdated preferences, and one-time lessons — all at equal weight,
none of them ever forgotten. The store hits its character limit, and the
agent spends context on irrelevant memories.

This plugin adds a **decay-weighted retrieval layer** on top of the existing
memory system. Old, never-accessed memories fade. Frequently-relevant
memories stay strong. The store self-manages.

## What it does

| # | Mechanism | What it does |
|---|---|---|
| 1 | **Ebbinghaus decay** | Memories fade over time: `importance *= 1/(1 + rate × hours)` |
| 2 | **Reconsolidation** | Retrieved weak memories get a bigger boost than strong ones |
| 3 | **Retrieval-induced forgetting** | Competing memories (same topic, lower rank) lose importance |
| 4 | **Source-confidence decay** | Trust in a memory's origin erodes slower than importance |
| 5 | **Access reinforcement** | Each retrieval adds `access_boost` (spaced repetition) |
| 6 | **Importance-weighted retrieval** | `score = fts_rank × decayed_importance × decayed_confidence` |
| 7 | **Origin classification** | `user_correction(0.95) > preference(0.85) > env_fact(0.6) > inference(0.35)` |
| 8 | **Decay-floor pruning** | Memories below `decay_floor` are auto-pruned at session end |

## Install

### Step 1 — copy the plugin

```bash
mkdir -p ~/.hermes/plugins/cognitive
cp -r cognitive_memory/* ~/.hermes/plugins/cognitive/
```

### Step 2 — activate in config

```bash
hermes config set memory.provider cognitive
hermes config set memory.cognitive.decay_rate 0.15
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
    # Decay rate — higher = faster forgetting (default: 0.15)
    decay_rate: 0.15
    # Minimum importance to keep a memory (default: 0.05)
    decay_floor: 0.05
    # Boost on each access (default: 0.3)
    access_boost: 0.3
    # Max memories to inject per turn (default: 15)
    max_context: 15
    # Source confidence defaults by origin
    source_confidence:
      user_correction: 1.0
      user_preference: 0.9
      environment_fact: 0.7
      agent_inference: 0.4
    # Reconsolidation strength (default: 0.1)
    reconsolidation_rate: 0.1
    # Retrieval-induced forgetting penalty (default: 0.05)
    rif_penalty: 0.05
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
  │                     │  └── Pruning          │ │
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
    id          TEXT PRIMARY KEY,
    target      TEXT NOT NULL,         -- 'memory' or 'user'
    content     TEXT NOT NULL,
    importance  REAL NOT NULL,         -- 0.0 to 1.0
    confidence  REAL NOT NULL,         -- 0.0 to 1.0
    created_at  REAL NOT NULL,          -- unix timestamp
    last_access REAL NOT NULL,          -- unix timestamp
    access_count INTEGER DEFAULT 0,
    origin      TEXT NOT NULL,         -- 'user_correction' | 'user_preference' | ...
    tags        TEXT DEFAULT '[]'      -- JSON array
);
```

FTS5 is used for semantic (keyword) search, with ranking combined with the
cognitive scores. If FTS5 is not available (some SQLite builds don't compile
it), the plugin falls back to `LIKE`-based search automatically.

### Lifecycle hooks

| Hook | What happens |
|---|---|
| `initialize()` | Create SQLite DB, ensure schema, load config |
| `prefetch(query)` | FTS5 search → rank by `similarity × importance × decay × confidence` → return top N |
| `sync_turn()` | Apply time-based decay to all memories |
| `on_memory_write()` | Mirror built-in memory writes with importance scoring |
| `on_turn_start()` | Reconsolidation: reinforce/suppress based on last turn's retrievals |
| `on_session_end()` | Prune memories below decay floor |

### Decay math

Each memory has an `importance` score (0.0–1.0). On every turn:

```
elapsed = now - last_access
stability = 1.0 / (1.0 + decay_rate * elapsed / 3600)  # hours
importance *= stability
```

When a memory is retrieved (prefetch hits it):

```
importance = min(1.0, importance + access_boost)
last_access = now
access_count += 1
```

Competing memories (same FTS match, lower rank) get:

```
importance = max(0.0, importance - rif_penalty)
```

### Origin classification

Memories are classified by origin on write, which sets initial importance
and confidence:

| Origin | Initial importance | Initial confidence | Example |
|---|---|---|---|
| `user_correction` | 0.95 | 0.90 | "Never pkill -f hermes" |
| `user_preference` | 0.85 | 0.90 | "Prefers step-by-step plans in chat" |
| `environment_fact` | 0.60 | 0.60 | "SSH user is vm_user" |
| `agent_inference` | 0.50 | 0.50 | "Model fallback chain behavior" |

Classification is keyword-based in `decay.py:classify_origin()`. It checks
for markers like "never", "do not", "lesson" (corrections), "prefers",
"wants" (preferences), "is", "runs on", "uses" (environment facts), and
falls back to inference.

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
origin, access_count, and last_access (as ISO timestamp).

### `cognitive_stats`

Show cognitive memory store statistics.

No parameters. Returns:
```json
{
  "total_memories": 15,
  "average_importance": 0.733,
  "prunable_count": 0,
  "by_target": {"memory": 9, "user": 6}
}
```

### `cognitive_remember`

Store a new memory with cognitive metadata.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `content` | string | required | The fact to remember |
| `target` | string | `memory` | `memory` or `user` |
| `origin` | string | `agent_inference` | `user_correction`, `user_preference`, `environment_fact`, or `agent_inference` |
| `tags` | array | [] | Optional tags for categorization |

### `cognitive_forget`

Delete a memory by its ID.

| Parameter | Type | Description |
|---|---|---|
| `memory_id` | string | required | The memory ID to delete |

## Seeding existing memories

The `on_memory_write` hook only fires on **new** writes. Memories that
existed before the plugin was installed need to be seeded manually.

A seed script is included at `scripts/seed_from_memory.py`. Usage:

```bash
python scripts/seed_from_memory.py --hermes-home ~/.hermes
```

This reads the built-in MEMORY.md and USER.md files, classifies each entry
by origin, and inserts them into the cognitive store with appropriate
importance and confidence scores.

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

You can use both systems together. The built-in memory gives you simple,
durable, human-readable entries. The cognitive layer adds decay, ranking,
and auto-pruning on top.

## WebUI management panel

If you run [hermes-webui](https://github.com/nesquena/hermes-webui), the
Memory panel gains a **Cognitive Memory** section: browse all memories with
their cognitive metadata (origin, importance, reliability, temporal class,
access count), filter/search them, **pin / unpin / delete**, add new
memories, and inspect the prune log.

This is an optional integration — it only reads/writes the same SQLite store
the plugin uses, and never imports Hermes Agent code into the WebUI process.

Install (copy one bridge module + 2 dispatcher lines + JS/CSS blocks):

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

When you run `hermes config set memory.cognitive.decay_rate 0.15`, Hermes
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
│   ├── decay.py          # 8 neuroscience mechanisms + origin classifier
│   ├── store.py           # SQLite + FTS5 storage layer
│   └── plugin.yaml       # Plugin manifest for Hermes discovery
├── scripts/
│   └── seed_from_memory.py  # Seed existing memories on first install
├── tests/
│   ├── test_decay.py      # 22 tests — decay math, reconsolidation, RIF, pruning
│   ├── test_store.py      # 22 tests — add/get/remove/search/decay/prune/effects
│   └── test_provider.py   # 25 tests — plugin interface, prefetch, tools, lifecycle
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

## License

MIT