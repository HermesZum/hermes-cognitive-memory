# hermes-cognitive-memory

A Hermes Agent memory provider plugin implementing neuroscience-inspired
cognitive memory mechanisms — decay, reconsolidation, retrieval-induced
forgetting, source-confidence erosion, and importance-weighted retrieval.

## What it does

The built-in Hermes memory stores flat text entries that persist forever
until manually removed. This plugin adds a **decay-weighted retrieval layer**
on top of that same simple model:

- **Ebbinghaus decay** — memories fade over time, reinforced by access
- **Reconsolidation** — recalling a memory can strengthen or modify it
- **Retrieval-induced forgetting** — recalling one memory suppresses
  competing ones
- **Source-confidence decay** — trust in a memory's origin erodes over time
- **Importance scoring** — user corrections > environment facts > inferences
- **Relevance ranking** — retrieval score = semantic similarity × importance
  × decay factor × source confidence
- **Automatic pruning** — memories below a decay floor are eligible for
  removal, keeping the store focused

## Install

### As a user-installed plugin (recommended)

```bash
# Clone or copy into $HERMES_HOME/plugins/cognitive/
cp -r hermes-cognitive-memory/cognitive_memory ~/.hermes/plugins/cognitive
```

### Activate

Add to `~/.hermes/config.yaml`:

```yaml
memory:
  provider: cognitive
```

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

### Storage

Memories are stored in an SQLite database at
`$HERMES_HOME/cognitive_memory/memory.db` with the following schema:

```sql
CREATE TABLE memories (
    id          TEXT PRIMARY KEY,
    target      TEXT NOT NULL,        -- 'memory' or 'user'
    content     TEXT NOT NULL,
    importance  REAL NOT NULL,        -- 0.0 to 1.0
    confidence  REAL NOT NULL,        -- 0.0 to 1.0
    created_at  REAL NOT NULL,         -- unix timestamp
    last_access REAL NOT NULL,        -- unix timestamp
    access_count INTEGER DEFAULT 0,
    origin      TEXT NOT NULL,        -- 'user_correction' | 'user_preference' | ...
    tags        TEXT DEFAULT '[]'     -- JSON array
);
```

FTS5 is used for semantic (keyword) search, with ranking combined with the
cognitive scores.

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

## Development

```bash
# Run tests
python -m pytest tests/

# Run the plugin standalone (integration smoke test)
python -c "from cognitive_memory import CognitiveMemoryProvider; p = CognitiveMemoryProvider(); print(p.name)"
```

## License

MIT