# WebUI integration — Cognitive Memory management panel

This directory contains the integration that adds a **Cognitive Memory**
management section to the [hermes-webui](https://github.com/nesquena/hermes-webui)
Memory panel: list memories with their cognitive metadata, search/filter them,
pin/unpin, delete, add new ones, and view the prune log.

It does **not** modify the Hermes plugin itself — it only reads/writes the
same SQLite store (`<hermes-home>/cognitive_memory/memory.db`) the plugin uses.

## Files

| File | Destination in hermes-webui |
|---|---|
| `api/cognitive_bridge.py` | `api/cognitive_bridge.py` |
| `panels.js.patch` (see below) | `static/panels.js` (3 small edits) |
| `style.css.patch` (see below) | `static/style.css` (append block) |
| `routes.py.patch` (see below) | `api/routes.py` (2 dispatcher lines) |

## Install

### 1. Copy the bridge module

```bash
cp api/cognitive_bridge.py /path/to/hermes-webui/api/cognitive_bridge.py
```

### 2. Wire the dispatchers in `api/routes.py`

In `handle_get`, immediately after the `if parsed.path == "/api/memory":` block:

```python
    # ── Cognitive memory (GET) — hermes-cognitive-memory plugin store ──
    if parsed.path == "/api/memory/cognitive":
        from api.cognitive_bridge import handle_cognitive_get

        handle_cognitive_get(handler, parsed)
        return True
```

In `handle_post`, immediately after the `if parsed.path == "/api/memory/write":` block:

```python
    # ── Cognitive memory (POST) — pin/unpin/delete/add via plugin store ──
    if parsed.path == "/api/memory/cognitive":
        from api.cognitive_bridge import handle_cognitive_post

        handle_cognitive_post(handler, body)
        return True
```

### 3. Add the section to `static/panels.js`

a) In `MEMORY_SECTIONS`, add the entry (after the `soul` entry):

```js
  { key: 'cognitive', label: 'Cognitive Memory', empty: '', iconKey: 'book-open', readOnly: true },
```

b) At the top of `_renderMemoryDetail()`, add a branch (next to the
`external_notes` branch):

```js
  if (section === 'cognitive') {
    _renderCognitiveMemoryDetail();
    if (!_cognitiveData) _loadCognitiveData();
    return;
  }
```

c) Append the cognitive rendering/manage functions (from the reference
implementation in this repo's `static/panels.js.cognitive.bundle.js`, or copy
them from the upstream commit) after `_renderMemoryEdit()`:
`_loadCognitiveData`, `_renderCognitiveMemoryDetail`, `_cognitiveCardHtml`,
`_cognitiveAge`, `cognitiveAction`, `cognitiveSetQuery`, `cognitiveSetFilter`,
`cognitiveToggleAdd`, `cognitiveAddContentTyped`, `_renderCognitiveAddForm`,
`cognitiveSubmitAdd`, `_showCognitiveAddError` — plus the state variables
`_cognitiveData`, `_cognitiveBusy`, `_cognitiveQuery`, `_cognitiveFilter`,
`_cognitiveAddOpen`, `_cognitiveAddDraft`.

> The full JS block is in `static/panels.js.cognitive.bundle.js` in this
> directory. It is the exact code shipped in the webui commit.

### 4. Append the CSS to `static/style.css`

The block is in `static/style.css.cognitive.css` in this directory. Append it
at the end of the file.

### 5. Restart

```bash
sudo systemctl restart hermes-webui
```

Then open **Memory** → **Cognitive Memory** in the sidebar.

## API

### `GET /api/memory/cognitive`

```json
{
  "available": true,
  "db_path": "/root/.hermes/cognitive_memory/memory.db",
  "memories": [
    {
      "id": "…", "target": "memory", "content": "…",
      "importance": 0.85, "effective_importance": 0.79,
      "confidence": 0.9, "origin": "user_preference",
      "reliability": 1.0, "hard_to_find": false, "pinned": true,
      "temporal": "stable", "superseded": false, "supersedes": null,
      "access_count": 3, "created_at": 1.78e9, "last_access": 1.78e9
    }
  ],
  "stats": {"total": 15, "pinned": 2, "hard_to_find": 1, "superseded": 0,
            "prunable": 1, "by_origin": {"user_preference": 6},
            "by_temporal": {"stable": 13}},
  "prune_log": ["2026-08-10 21:50 | pruned id=abc123 | …"]
}
```

### `POST /api/memory/cognitive`

Body — one of:

```json
{"action": "pin",   "id": "<memory-id>"}
{"action": "unpin", "id": "<memory-id>"}
{"action": "delete", "id": "<memory-id>"}
{"action": "add", "content": "…", "target": "memory", "origin": "research_finding",
 "temporal": "timeless", "reliability": 0.9, "pinned": false, "hard_to_find": false}
```

## Design notes

- **No Hermes Agent imports in the WebUI process.** The bridge loads the
  plugin's `store.py` / `decay.py` under a synthetic package name, skipping
  `__init__.py` (which imports `agent.memory_provider`).
- **Profile-aware.** The store is resolved via `get_active_hermes_home()` and
  cached per home path, so switching profiles in the WebUI switches stores.
- **Concurrency-safe.** `MemoryStore` uses an RLock and `PRAGMA busy_timeout`,
  so the WebUI and the Hermes agent process can share the DB.
- **Read-only to the plugin's data model:** `pinned` writes go through
  `MemoryStore.set_pinned()`; dedup/conflict logic in `add()` is preserved.

## Verification

```bash
# From the hermes-webui repo root:
python3 -m py_compile api/cognitive_bridge.py
node --check static/panels.js        # if node is available
```

The upstream commit also ran an end-to-end bridge test (temp Hermes home,
seeded store, fake handler): list → pin → unpin → delete → add → validation →
missing-store path.
