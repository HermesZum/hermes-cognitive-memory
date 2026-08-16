"""SQLite storage layer for cognitive memory.

Stores memories with cognitive metadata (importance, confidence, origin,
access tracking) and provides FTS5-based retrieval with decay-aware
ranking.

The schema is intentionally simple — one table + one FTS5 virtual table.
SQLite is already used by Hermes for session state, so this adds no new
dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .decay import (
    DecayParams,
    apply_access_reinforcement,
    apply_confidence_decay,
    apply_decay,
    apply_reconsolidation,
    apply_rif_penalty,
    detect_conflict,
    initial_confidence,
    initial_importance,
    semantic_similarity,
    should_prune,
)

from . import embeddings as _emb_mod

logger = logging.getLogger(__name__)

_BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    target        TEXT NOT NULL,
    content       TEXT NOT NULL,
    importance    REAL NOT NULL,
    confidence    REAL NOT NULL,
    created_at    REAL NOT NULL,
    last_access   REAL NOT NULL,
    access_count  INTEGER DEFAULT 0,
    origin        TEXT NOT NULL DEFAULT 'unknown',
    tags          TEXT DEFAULT '[]',
    reliability   REAL NOT NULL DEFAULT 1.0,
    hard_to_find  INTEGER NOT NULL DEFAULT 0,
    pinned        INTEGER NOT NULL DEFAULT 0,
    temporal      TEXT NOT NULL DEFAULT 'stable',
    superseded    INTEGER NOT NULL DEFAULT 0,
    supersedes    TEXT DEFAULT NULL,
    critical      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_target ON memories(target);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_last_access ON memories(last_access);
CREATE INDEX IF NOT EXISTS idx_memories_pinned ON memories(pinned);
CREATE INDEX IF NOT EXISTS idx_memories_temporal ON memories(temporal);
CREATE INDEX IF NOT EXISTS idx_memories_superseded ON memories(superseded);
CREATE INDEX IF NOT EXISTS idx_memories_critical ON memories(critical);

CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id TEXT PRIMARY KEY,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL
);
"""

# Migration SQL for databases created before new columns
_MIGRATION_SQL = [
    "ALTER TABLE memories ADD COLUMN reliability REAL NOT NULL DEFAULT 1.0",
    "ALTER TABLE memories ADD COLUMN hard_to_find INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE memories ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE memories ADD COLUMN temporal TEXT NOT NULL DEFAULT 'stable'",
    "ALTER TABLE memories ADD COLUMN superseded INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE memories ADD COLUMN supersedes TEXT DEFAULT NULL",
    "ALTER TABLE memories ADD COLUMN critical INTEGER NOT NULL DEFAULT 0",
]

_FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync with the base table
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories
    BEGIN
        INSERT INTO memories_fts(rowid, content)
        VALUES (new.rowid, new.content);
    END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories
    BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content)
        VALUES('delete', old.rowid, old.content);
    END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories
    BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, content)
        VALUES('delete', old.rowid, old.content);
        INSERT INTO memories_fts(rowid, content)
        VALUES (new.rowid, new.content);
    END;
"""


def _sanitize_fts_query(query: str) -> str:
    """Sanitize a user query for FTS5 MATCH.

    Splits the query into individual terms and ORs them together
    (``term1 OR term2 OR ...``) so that ANY matching term retrieves a
    memory. This is the standard RAG retrieval behaviour — a natural
    multi-word user message like "What are the security rules?" should
    match memories containing "security" OR "rules", not require the
    exact contiguous phrase.

    Terms are individually double-quoted (FTS5 string literals) so special
    characters (*, NEAR, :, -, +) are inert. Stopwords and very short
    tokens are dropped to avoid noise matches. If no usable terms remain,
    the whole query is returned quoted as a phrase fallback.
    """
    import re
    # Strip FTS5 operators that would change query semantics if left bare
    cleaned = re.sub(r'[\*\+\-\:\(\)\{\}\^\~]', ' ', query)
    tokens = [t for t in cleaned.split() if len(t) > 2 and t.lower() not in _STOPWORDS]
    if not tokens:
        # Fallback: quote the whole thing as a phrase (or a single token)
        safe = query.replace('"', '""')
        return f'"{safe}"'
    quoted = [f'"{t.replace(chr(34), chr(34) * 2)}"' for t in tokens]
    return " OR ".join(quoted)


# English stopwords dropped from FTS term expansion. These are high-frequency
# words that appear in many memories and would otherwise dominate OR-based
# retrieval (e.g. "the" matches 4 memories, skewing re-ranking toward
# top-importance rather than semantically relevant results).
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "this", "that", "what", "are", "was", "were",
    "has", "have", "had", "you", "your", "not", "but", "all", "any", "from",
    "into", "they", "their", "our", "can", "will", "would", "should", "could",
    "when", "where", "which", "who", "whom", "how", "why", "than", "then",
    "there", "here", "about", "over", "under", "between", "before", "after",
    "during", "while", "because", "since", "though", "although", "unless",
    "until", "against", "through", "within", "without", "upon", "among",
    "does", "did", "done", "doing", "been", "being", "its", "it's", "don't",
    "doesn't", "didn't", "won't", "can't", "shouldn't", "wouldn't", "isn't",
    "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't", "let's",
    "that's", "what's", "who's", "it's", "there's", "here's", "i'm", "you're",
    "we're", "they're", "he's", "she's", "my", "me", "we", "us", "him", "her",
    "them", "his", "hers", "its", "own", "same", "such", "only", "also",
    "more", "most", "some", "much", "many", "few", "little", "other", "another",
    "each", "every", "both", "either", "neither", "one", "two", "three",
    "first", "last", "new", "old", "good", "bad", "well", "just", "very",
    "really", "please", "thanks", "thank", "yes", "no", "ok", "okay", "sure",
    "maybe", "perhaps", "probably", "actually", "basically", "essentially",
    "generally", "typically", "usually", "often", "sometimes", "always",
    "never", "ever", "already", "still", "yet", "even", "only", "also",
})


def _strip_operators(query: str) -> str:
    """Remove FTS5 special operators so a query can be tokenized for IDF."""
    import re
    return re.sub(r'[\*\+\-\:\(\)\{\}\^\~"]', ' ', query)


def _term_doc_freq(conn, term: str) -> int:
    """Return the number of active memories containing ``term`` (FTS5)."""
    try:
        cur = conn.execute(
            "SELECT COUNT(*) c FROM memories_fts "
            "JOIN memories m ON m.rowid = memories_fts.rowid "
            "WHERE memories_fts MATCH ? AND m.superseded = 0",
            (f'"{term}"',),
        )
        return cur.fetchone()["c"]
    except sqlite3.OperationalError:
        return 0


class MemoryStore:

    """SQLite-backed memory store with FTS5 and cognitive decay."""

    def __init__(self, db_path: Path, params: DecayParams):
        self._db_path = db_path
        self._params = params
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._connected = False
        self._fts_available = False
        # Embedding backend for hybrid (dense + lexical) retrieval.
        # Lazily resolved on first use; defaults to NoOp (lexical-only).
        self._embedding_backend: Optional[_emb_mod.EmbeddingBackend] = None
        self._embedding_enabled = True
        self._embedding_model = _emb_mod.DEFAULT_MODEL
        self._embedding_url = _emb_mod.DEFAULT_URL
        self._embedding_alpha = 0.6  # weight on lexical vs semantic in fusion
        self._semantic_floor = 0.45  # min cosine for a semantic-only match to count
        # Prune log path — next to the DB
        self._prune_log_path = db_path.parent / "prune_log.md"

    @property
    def prune_log_path(self) -> Path:
        """Path of the prune/change audit log (public accessor)."""
        return self._prune_log_path

    def connect(self) -> None:
        """Open the database connection and ensure schema.

        FTS5 is optional — if the SQLite build doesn't include it, we fall
        back to LIKE-based search by creating the base table only and
        skipping the FTS virtual table + triggers.
        """
        with self._lock:
            if self._connected and self._conn:
                return
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            try:
                conn.row_factory = sqlite3.Row
                # Set a busy timeout so we don't immediately fail if another
                # process holds a write lock — wait up to 5 seconds.
                conn.execute("PRAGMA busy_timeout=5000")
                # Migrate pre-existing databases FIRST: ALTER TABLE ADD COLUMN
                # must run before the base schema's CREATE INDEX statements,
                # which reference the new columns. On a fresh DB these ALTERs
                # fail harmlessly ("no such table") and are swallowed; on an
                # old DB they add the missing columns so the indexes below
                # can be created.
                for sql in _MIGRATION_SQL:
                    try:
                        conn.execute(sql)
                    except sqlite3.OperationalError:
                        pass  # Column already exists (fresh DB) or table absent
                conn.commit()
                # Create base table + indexes (works after migration above)
                conn.executescript(_BASE_SCHEMA_SQL)
                # Try FTS5 — may fail if not compiled into SQLite
                fts_existed = (
                    conn.execute(
                        "SELECT name FROM sqlite_master"
                        " WHERE type='table' AND name='memories_fts'"
                    ).fetchone()
                    is not None
                )
                try:
                    conn.executescript(_FTS_SCHEMA_SQL)
                except sqlite3.OperationalError as e:
                    logger.warning(
                        "cognitive-memory: FTS5 not available (%s), "
                        "falling back to LIKE-based search", e
                    )
                    self._fts_available = False
                else:
                    self._fts_available = True
                    # If the FTS table was just created over pre-existing
                    # content (e.g. an old-schema DB being migrated), its
                    # index is empty/out of sync — rebuild it so the
                    # AFTER UPDATE/INSERT triggers don't hit
                    # "database disk image is malformed".
                    if not fts_existed:
                        has_rows = conn.execute(
                            "SELECT EXISTS(SELECT 1 FROM memories)"
                        ).fetchone()[0]
                        if has_rows:
                            conn.execute(
                                "INSERT INTO memories_fts(memories_fts)"
                                " VALUES('rebuild')"
                            )
                conn.commit()
            except Exception:
                # If anything fails, close the connection before propagating
                conn.close()
                raise
            self._conn = conn
            self._connected = True
            logger.debug("cognitive-memory: connected to %s (fts=%s)",
                         self._db_path, self._fts_available)

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
            self._connected = False

    # -- Embedding configuration (hybrid retrieval) ------------------------

    def configure_embeddings(
        self,
        enabled: bool = True,
        model: str = _emb_mod.DEFAULT_MODEL,
        url: str = _emb_mod.DEFAULT_URL,
        alpha: float = 0.6,
        semantic_floor: float = 0.45,
    ) -> None:
        """Configure hybrid (dense + lexical) retrieval.

        Call before first search if you want semantic recall. If disabled or
        the backend is unreachable, retrieval degrades to lexical-only.
        """
        self._embedding_enabled = enabled
        self._embedding_model = model
        self._embedding_url = url
        self._embedding_alpha = max(0.0, min(1.0, alpha))
        self._semantic_floor = max(0.0, min(1.0, semantic_floor))
        # Force re-resolution on next use.
        self._embedding_backend = None

    def _get_embedding_backend(self) -> Optional[_emb_mod.EmbeddingBackend]:
        """Lazily resolve (and cache) the embedding backend."""
        if self._embedding_backend is None:
            self._embedding_backend = _emb_mod.get_embedding_backend(
                enabled=self._embedding_enabled,
                model=self._embedding_model,
                url=self._embedding_url,
            )
        return self._embedding_backend

    def _embed_and_store(self, memory_id: str, content: str) -> None:
        """Compute an embedding for content and persist it. Best-effort.

        Any failure is logged and swallowed — lexical retrieval must never
        break because of an embedding problem.
        """
        if not self._embedding_enabled:
            return
        backend = self._get_embedding_backend()
        if backend is None or not backend.available:
            return
        try:
            vec = backend.embed(content)
            if not vec:
                return
            with self._lock:
                if not self._conn:
                    return
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_embeddings "
                    "(memory_id, model, dim, vector) VALUES (?, ?, ?, ?)",
                    (memory_id, backend.model, len(vec),
                     _emb_mod.pack_vector(vec)),
                )
                self._conn.commit()
        except Exception as e:  # noqa: BLE001 - never break writes
            logger.warning("cognitive-memory: embed+store failed (%s)", e)

    def _get_embedding(self, memory_id: str) -> Optional[List[float]]:
        """Fetch a stored embedding vector for a memory, or None."""
        try:
            with self._lock:
                if not self._conn:
                    return None
                row = self._conn.execute(
                    "SELECT vector FROM memory_embeddings WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
            if not row:
                return None
            return _emb_mod.unpack_vector(row["vector"])
        except Exception:  # noqa: BLE001
            return None

    def backfill_embeddings(self) -> int:
        """Embed all existing (non-superseded) memories lacking a vector.

        Idempotent: skips rows that already have an embedding. Returns the
        number of memories embedded. Run once after deploying hybrid retrieval.
        """
        if not self._embedding_enabled:
            return 0
        backend = self._get_embedding_backend()
        if backend is None or not backend.available:
            logger.warning("cognitive-memory: backfill skipped (backend unavailable)")
            return 0
        with self._lock:
            if not self._conn:
                return 0
            rows = self._conn.execute(
                "SELECT id, content FROM memories WHERE superseded = 0"
            ).fetchall()
        count = 0
        for row in rows:
            if self._get_embedding(row["id"]) is not None:
                continue
            vec = backend.embed(row["content"])
            if not vec:
                continue
            with self._lock:
                if not self._conn:
                    break
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_embeddings "
                    "(memory_id, model, dim, vector) VALUES (?, ?, ?, ?)",
                    (row["id"], backend.model, len(vec),
                     _emb_mod.pack_vector(vec)),
                )
                self._conn.commit()
            count += 1
        logger.info("cognitive-memory: backfilled %d embeddings", count)
        return count

    # -- Write operations ---------------------------------------------------

    def add(
        self,
        target: str,
        content: str,
        origin: str = "unknown",
        tags: Optional[List[str]] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
        reliability: float = 1.0,
        hard_to_find: bool = False,
        pinned: bool = False,
        temporal: str = "stable",
    ) -> str:
        """Add a new memory. Returns the memory ID.

        Performs semantic deduplication and conflict detection before
        storing. If a near-duplicate exists (similarity > dedup_threshold),
        the memories are merged instead of creating a duplicate. If a
        conflict is detected (high similarity but different numbers/negations),
        the old memory is superseded.

        Args:
            reliability: 0-1, how trustworthy the source is (default 1.0).
            hard_to_find: If True, gets lower decay floor + importance boost.
            pinned: If True, never pruned regardless of decay.
            temporal: 'timeless' (slow decay), 'stable' (normal), 'ephemeral' (fast decay).
        """
        now = time.time()
        if importance is None:
            importance = initial_importance(origin, self._params)
        if confidence is None:
            confidence = initial_confidence(origin, self._params)
        # Hard-to-find memories get an importance boost
        if hard_to_find:
            importance = min(1.0, importance + 0.15)
        tags_json = json.dumps(tags or [])

        with self._lock:
            assert self._conn is not None

            # -- Mechanism 12: Semantic deduplication --
            dup_id = self._find_duplicate_locked(target, content)
            if dup_id:
                self._merge_locked(dup_id, content, importance, confidence,
                                   reliability, hard_to_find, pinned, temporal)
                logger.debug("cognitive-memory: merged with existing %s (dedup)", dup_id)
                return dup_id

            # -- Mechanism 13: Conflict detection / supersession --
            conflict_id = self._find_conflict_locked(target, content)
            if conflict_id:
                self._supersede_locked(conflict_id, content)
                logger.debug("cognitive-memory: superseded %s (conflict)", conflict_id)

            mem_id = str(uuid.uuid4())
            self._conn.execute(
                """INSERT INTO memories
                   (id, target, content, importance, confidence,
                    created_at, last_access, access_count, origin, tags,
                    reliability, hard_to_find, pinned, temporal, superseded, supersedes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (mem_id, target, content, importance, confidence,
                 now, now, origin, tags_json,
                 reliability, int(hard_to_find), int(pinned),
                 temporal, None),
            )
            self._conn.commit()
        # Embed for hybrid retrieval (best-effort; lexical-only if it fails)
        self._embed_and_store(mem_id, content)
        logger.debug(
            "cognitive-memory: added memory %s (origin=%s, importance=%.2f, "
            "reliability=%.2f, pinned=%s, temporal=%s)",
            mem_id, origin, importance, reliability, pinned, temporal,
        )
        return mem_id

    def _find_duplicate_locked(self, target: str, content: str) -> Optional[str]:
        """Find an existing memory that's a semantic duplicate.

        Uses Jaccard token similarity. If above dedup_similarity_threshold,
        it's a duplicate. Superseded memories are not considered (they've
        been replaced).

        MUST be called with self._lock held.
        """
        assert self._conn is not None
        # Only check memories with same target, not superseded
        cur = self._conn.execute(
            "SELECT id, content FROM memories WHERE target = ? AND superseded = 0",
            (target,),
        )
        for row in cur.fetchall():
            sim = semantic_similarity(content, row["content"])
            if sim >= self._params.dedup_similarity_threshold:
                return row["id"]
        return None

    def _merge_locked(
        self, existing_id: str, new_content: str,
        new_importance: float, new_confidence: float,
        new_reliability: float, new_hard_to_find: bool,
        new_pinned: bool, new_temporal: str,
    ) -> None:
        """Merge a new memory into an existing one (dedup).

        Combines content (appends if different enough to be useful),
        keeps the higher importance, higher reliability, max of access_count,
        and ORs the flags.

        MUST be called with self._lock held.
        """
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT content, importance, confidence, reliability, "
            "hard_to_find, pinned, access_count, temporal FROM memories WHERE id = ?",
            (existing_id,),
        )
        row = cur.fetchone()
        if not row:
            return

        # Merge content — keep the longer one (more info)
        merged_content = new_content if len(new_content) > len(row["content"]) else row["content"]

        # Keep higher importance, higher confidence, higher reliability
        merged_importance = max(row["importance"], new_importance)
        merged_confidence = max(row["confidence"], new_confidence)
        merged_reliability = max(row["reliability"], new_reliability)

        # OR the flags
        merged_hard = bool(row["hard_to_find"]) or new_hard_to_find
        merged_pinned = bool(row["pinned"]) or new_pinned

        # Keep the "more stable" temporal (timeless > stable > ephemeral)
        temporal_rank = {"timeless": 0, "stable": 1, "ephemeral": 2}
        old_temporal = row["temporal"] or "stable"
        merged_temporal = min(old_temporal, new_temporal, key=lambda t: temporal_rank.get(t, 1))

        self._conn.execute(
            """UPDATE memories SET
               content = ?, importance = ?, confidence = ?, reliability = ?,
               hard_to_find = ?, pinned = ?, temporal = ?
               WHERE id = ?""",
            (merged_content, merged_importance, merged_confidence, merged_reliability,
             int(merged_hard), int(merged_pinned), merged_temporal, existing_id),
        )
        self._conn.commit()

    def _find_conflict_locked(self, target: str, content: str) -> Optional[str]:
        """Find an existing memory that the new content conflicts with.

        If similarity is above conflict_similarity_threshold AND detect_conflict
        returns True, the old memory should be superseded.

        MUST be called with self._lock held.
        """
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT id, content FROM memories WHERE target = ? AND superseded = 0",
            (target,),
        )
        for row in cur.fetchall():
            sim = semantic_similarity(content, row["content"])
            if sim >= self._params.conflict_similarity_threshold:
                if detect_conflict(row["content"], content):
                    return row["id"]
        return None

    def _supersede_locked(self, old_id: str, new_content: str) -> None:
        """Mark an old memory as superseded by a new one.

        The old memory's importance is set to 0 and superseded=1, so it will
        be pruned at the next session end. The new memory will store a
        reference to the old one in supersedes for audit trail.

        MUST be called with self._lock held.
        """
        assert self._conn is not None
        self._conn.execute(
            "UPDATE memories SET superseded = 1, importance = 0 WHERE id = ?",
            (old_id,),
        )
        self._conn.commit()
        logger.info("cognitive-memory: superseded memory %s (conflict detected)", old_id)

    def replace(self, mem_id: str, new_content: str) -> bool:
        """Replace a memory's content (keeps cognitive scores)."""
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "UPDATE memories SET content = ? WHERE id = ?",
                (new_content, mem_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def remove(self, mem_id: str) -> bool:
        """Delete a memory by ID."""
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "DELETE FROM memories WHERE id = ?", (mem_id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def set_pinned(self, mem_id: str, pinned: bool) -> bool:
        """Set the pinned flag on a memory (manual override from the WebUI).

        Returns True if the memory exists and was updated. Pinned memories
        are never pruned regardless of decay. Unpinning via this method is
        a manual override — auto-pinning on access may re-pin it later if
        the memory keeps being retrieved.
        """
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "UPDATE memories SET pinned = ? WHERE id = ?",
                (int(bool(pinned)), mem_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def remove_by_content(self, content_substring: str) -> int:
        """Remove memories whose content contains the substring. Returns count.

        Escapes LIKE wildcards (% and _) in the substring so they match
        literally, not as pattern characters.
        """
        # Escape LIKE special characters so they match literally
        escaped = content_substring.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "DELETE FROM memories WHERE content LIKE ? ESCAPE '\\'",
                (f"%{escaped}%",),
            )
            self._conn.commit()
            return cur.rowcount

    # -- Read operations ----------------------------------------------------

    def get(self, mem_id: str) -> Optional[Dict[str, Any]]:
        """Get a single memory by ID."""
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (mem_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all memories (excluding superseded)."""
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "SELECT * FROM memories WHERE superseded = 0 ORDER BY importance DESC"
            )
            return [dict(r) for r in cur.fetchall()]

    def get_all_raw(self) -> List[Dict[str, Any]]:
        """Get all memories including superseded (for stats/audit)."""
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "SELECT * FROM memories ORDER BY importance DESC"
            )
            return [dict(r) for r in cur.fetchall()]

    def get_by_target(self, target: str) -> List[Dict[str, Any]]:
        """Get all memories for a target ('memory' or 'user')."""
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "SELECT * FROM memories WHERE target = ? AND superseded = 0 ORDER BY importance DESC",
                (target,),
            )
            return [dict(r) for r in cur.fetchall()]

    # -- IDF helpers (term-frequency weighting for selective retrieval) -------

    def _idf_for_terms(self, terms: List[str]) -> Dict[str, float]:
        """Compute smoothed IDF for each query term.

        idf = ln((N - df + 0.5) / (df + 0.5)) + 1  (BM25-style, always >= 1 so
        common terms don't get zeroed). Higher idf = rarer term = more weight.
        """
        if not self._conn:
            return {}
        total = self.count()
        if total <= 0:
            return {}
        idf: Dict[str, float] = {}
        for t in terms:
            df = _term_doc_freq(self._conn, t)
            df = max(1, df)  # avoid div-by-zero / log(0)
            ratio = (total - df + 0.5) / (df + 0.5)
            idf[t] = 1.0 + math.log1p(ratio)
        return idf

    def _idf_boost_for_row(self, row: Dict[str, Any], idf: Dict[str, float]) -> Tuple[float, bool]:
        """Return (IDF boost multiplier, matched_any_term) for a memory row.

        If the memory's content matches any query term, the boost is the max
        idf among matched terms (rarer matched term → higher boost); matched
        is True. If no query term matches, boost = 1.0 and matched = False
        (no penalty — FTS already matched it, but it may be a non-semantic
        overlap, so the caller can apply the relevance floor).
        """
        if not idf:
            return 1.0, False
        content = (row.get("content") or "").lower()
        matched_vals = [v for t, v in idf.items() if t in content]
        if not matched_vals:
            return 1.0, False
        # Max IDF among matched terms — rewards matching the rarest term
        return max(matched_vals), True

    def search(
        self,
        query: str,
        target: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """FTS5 search with cognitive relevance ranking.

        Returns a list of (memory_dict, score) tuples sorted by score DESC.
        Score = normalized_fts_rank * decayed_importance * decayed_confidence * reliability.

        Superseded memories are excluded from search results.

        The entire operation (fetch + rank + retrieval effects) runs under
        the lock to prevent lost updates from concurrent apply_global_decay()
        or close().
        """
        if not query.strip():
            # No query — return highest-importance memories
            return self._importance_based_retrieval(target, limit)

        max_limit = limit or self._params.max_context
        now = time.time()

        with self._lock:
            if not self._conn:
                return []

            if not self._fts_available:
                # LIKE-based fallback when FTS5 is not available
                return self._like_search(query, target, max_limit, now)

            # FTS5 search — use bm25() ranking (lower = better in SQLite FTS5,
            # so we negate it). We also join back to the base table for
            # cognitive metadata. Exclude superseded memories.
            safe_query = _sanitize_fts_query(query)

            sql = """
                SELECT m.*, bm25(memories_fts) as fts_score
                FROM memories_fts
                JOIN memories m ON m.rowid = memories_fts.rowid
                WHERE memories_fts MATCH ?
                AND m.superseded = 0
            """
            params: list = [safe_query]

            if target:
                sql += " AND m.target = ?"
                params.append(target)

            sql += " ORDER BY fts_score ASC LIMIT ?"
            params.append(max_limit * 3)  # over-fetch, re-rank in Python

            try:
                cur = self._conn.execute(sql, params)
                rows = [dict(r) for r in cur.fetchall()]
            except sqlite3.OperationalError:
                # FTS query syntax error — fall back to LIKE search
                logger.debug("cognitive-memory: FTS query failed, falling back to LIKE")
                return self._like_search(query, target, max_limit, now)

            # Re-rank with cognitive scores — ALL under the lock
            scored = []
            # Semantic query vector (best-effort; None => lexical-only fusion)
            query_vec = None
            backend = self._get_embedding_backend()
            if backend is not None and backend.available:
                query_vec = backend.embed(query)

            # Semantic pass needs ALL non-critical, non-superseded rows (not just
            # FTS hits) so meaning-based matches can surface memories the lexical
            # path misses. Build id->row and look up FTS scores for the lexical part.
            sem_sql = (
                "SELECT * FROM memories WHERE superseded = 0 AND critical = 0"
            )
            sem_params: list = []
            if target:
                sem_sql += " AND target = ?"
                sem_params.append(target)
            sem_rows = [dict(r) for r in self._conn.execute(sem_sql, sem_params).fetchall()]
            sem_by_id = {r["id"]: r for r in sem_rows}
            fts_by_id = {row["id"]: row for row in rows}

            # Pre-compute inverse document frequency (IDF) for each query term so
            # that rare, specific terms weigh more than ubiquitous ones (e.g.
            # "pkill" outranks "hermes" which appears in 19/29 memories). This is
            # standard BM25-style weighting and prevents high-frequency tokens
            # from flooding the selective budget.
            query_terms = [t.lower() for t in _strip_operators(query).split() if len(t) > 2 and t.lower() not in _STOPWORDS]
            idf = self._idf_for_terms(query_terms) if query_terms else {}
            alpha = self._embedding_alpha
            for mid, row in sem_by_id.items():
                fts_row = fts_by_id.get(mid)
                if fts_row is not None:
                    fts_score = fts_row.pop("fts_score", 0.0)
                    normalized_fts = max(0.0, min(1.0, (-fts_score) / 3.0)) if fts_score != 0 else 0.5
                else:
                    # Not a lexical hit — no lexical signal.
                    normalized_fts = 0.0

                # Compute decayed importance on the fly (not stored)
                temporal = row.get("temporal", "stable")
                decayed_importance = apply_decay(
                    row["importance"], row["last_access"], now, self._params, temporal
                )
                decayed_confidence = apply_confidence_decay(
                    row["confidence"], row["last_access"], now, self._params
                )
                reliability = row.get("reliability", 1.0)

                # Multipliers for protected/timeless/recency signals
                hard_to_find = 2.0 if row.get("hard_to_find") else 1.0
                temporal_boost = 1.5 if temporal == "timeless" else 1.0
                recency_boost = 1.0 + max(0.0, (3600.0 - max(0.0, now - row.get("last_access", now))) / 3600.0) * 0.25

                # IDF term-match boost: memories matching rarer query terms rank higher
                idf_boost, matched_term = self._idf_boost_for_row(row, idf)

                # Semantic component (cosine), clamped to [0,1]
                semantic_score = 0.0
                if query_vec is not None:
                    vec = self._get_embedding(mid)
                    if vec is not None:
                        semantic_score = max(0.0, _emb_mod.cosine_similarity(query_vec, vec))

                # Fuse lexical + semantic. lexical component is already [0,1].
                fused = alpha * normalized_fts + (1.0 - alpha) * semantic_score

                # Relevance floor: drop results with NO signal — neither a lexical
                # term match, nor meaningful FTS relevance, nor semantic similarity
                # above the semantic floor. The 0.45 floor is tuned so unrelated
                # short queries (which still score ~0.38-0.40 on nomic-embed-text)
                # don't pollute the selective pool, while genuine paraphrase
                # matches (~0.41-0.69) survive.
                if (not matched_term and normalized_fts < 0.3
                        and semantic_score < self._semantic_floor):
                    continue

                score = (
                    fused
                    * decayed_importance
                    * decayed_confidence
                    * reliability
                    * hard_to_find
                    * temporal_boost
                    * recency_boost
                    * idf_boost
                )
                scored.append((row, score))



            scored.sort(key=lambda x: x[1], reverse=True)

            # Critical safety net: collected separately as its OWN budget tier.
            # Criticals are NOT appended to `scored` (selective pool) — they are
            # returned independently so prefetch() renders them in a dedicated
            # section. This mirrors Letta's core/archival split: the always-on
            # core never competes with relevance-ranked retrieval for slots.
            result_ids = {row["id"] for row, _ in scored}
            critical_rows = self._fetch_critical(target)
            critical = self._critical_batch(critical_rows)

            # Apply access reinforcement + RIF + auto-pin to top results
            if scored:
                self._apply_retrieval_effects_locked(
                    [s[0] for s in scored[:max_limit]],
                    [s[0] for s in scored[max_limit:]],
                )

            return scored[:max_limit], critical

    def _fetch_critical(self, target: Optional[str]) -> List[Dict[str, Any]]:
        """Fetch all non-superseded critical memories (safety tier)."""
        sql = "SELECT * FROM memories WHERE critical = 1 AND superseded = 0"
        params: list = []
        if target:
            sql += " AND target = ?"
            params.append(target)
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _critical_batch(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return up to critical_budget critical memories, ranked by importance."""
        budget = getattr(self._params, "critical_budget", 5)
        rows.sort(key=lambda r: r.get("importance", 0.0), reverse=True)
        return rows[:budget]

    def _like_search(
        self,
        query: str,
        target: Optional[str],
        max_limit: int,
        now: float,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """LIKE-based search fallback when FTS5 is not available.

        MUST be called with self._lock held.
        """
        assert self._conn is not None
        sql = "SELECT * FROM memories WHERE content LIKE ? AND superseded = 0"
        params: list = [f"%{query}%"]
        if target:
            sql += " AND target = ?"
            params.append(target)
        sql += " LIMIT ?"
        params.append(max_limit * 3)

        cur = self._conn.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]

        # Pre-compute IDF for query terms (parity with FTS path)
        query_terms = [t.lower() for t in _strip_operators(query).split() if len(t) > 2 and t.lower() not in _STOPWORDS]
        idf = self._idf_for_terms(query_terms) if query_terms else {}

        scored = []
        for row in rows:
            # Criticals live in their own budget tier — exclude from selective scoring.
            if row.get("critical"):
                continue
            fts_score = 0.0
            normalized_fts = 0.5
            temporal = row.get("temporal", "stable")
            decayed_importance = apply_decay(
                row["importance"], row["last_access"], now, self._params, temporal
            )
            decayed_confidence = apply_confidence_decay(
                row["confidence"], row["last_access"], now, self._params
            )
            reliability = row.get("reliability", 1.0)
            hard_to_find = 2.0 if row.get("hard_to_find") else 1.0
            temporal_boost = 1.5 if temporal == "timeless" else 1.0
            recency_boost = 1.0 + max(0.0, (3600.0 - max(0.0, now - row.get("last_access", now))) / 3600.0) * 0.25
            # IDF boost (LIKE path — same as FTS)
            idf_boost, matched_term = self._idf_boost_for_row(row, idf)
            # Relevance floor (parity with FTS path)
            if not matched_term and normalized_fts < 0.3:
                continue
            score = (
                normalized_fts
                * decayed_importance
                * decayed_confidence
                * reliability
                * hard_to_find
                * temporal_boost
                * recency_boost
                * idf_boost
            )
            scored.append((row, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Critical safety net: same as FTS path — collected as its OWN budget
        # tier, NOT appended to the selective pool.
        result_ids = {row["id"] for row, _ in scored}
        critical_rows = self._fetch_critical(target)
        critical = self._critical_batch(critical_rows)

        if scored:
            self._apply_retrieval_effects_locked(
                [s[0] for s in scored[:max_limit]],
                [s[0] for s in scored[max_limit:]],
            )

        return scored[:max_limit], critical

    def _importance_based_retrieval(
        self,
        target: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """When there's no query, return memories ranked by importance.

        Runs entirely under the lock to prevent race conditions.
        """
        max_limit = limit or self._params.max_context
        now = time.time()

        with self._lock:
            if not self._conn:
                return []

            sql = "SELECT * FROM memories WHERE superseded = 0"
            params: list = []
            if target:
                sql += " AND target = ?"
                params.append(target)
            sql += " ORDER BY importance DESC LIMIT ?"
            params.append(max_limit * 3)
            cur = self._conn.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]

            scored = []
            for row in rows:
                temporal = row.get("temporal", "stable")
                decayed_importance = apply_decay(
                    row["importance"], row["last_access"], now, self._params, temporal
                )
                decayed_confidence = apply_confidence_decay(
                    row["confidence"], row["last_access"], now, self._params
                )
                reliability = row.get("reliability", 1.0)
                score = decayed_importance * decayed_confidence * reliability
                scored.append((row, score))

            scored.sort(key=lambda x: x[1], reverse=True)

            if scored:
                self._apply_retrieval_effects_locked(
                    [s[0] for s in scored[:max_limit]],
                    [s[0] for s in scored[max_limit:]],
                )

            # Return a tuple for parity with search()'s FTS/LIKE paths, which
            # return (results, critical). The empty-query path has no critical
            # tier, but callers (e.g. _handle_search) always unpack as a tuple.
            return scored[:max_limit], []

    def _apply_retrieval_effects_locked(
        self,
        retrieved: List[Dict[str, Any]],
        competitors: List[Dict[str, Any]],
    ) -> None:
        """Apply access reinforcement + reconsolidation + RIF + auto-pin after retrieval.

        - Retrieved memories get access_boost + reconsolidation
        - Competing memories get RIF penalty
        - Auto-pin: if access_count crosses auto_pin_threshold, set pinned=1

        MUST be called with self._lock already held.
        """
        now = time.time()
        assert self._conn is not None
        for mem in retrieved:
            new_importance = apply_access_reinforcement(
                mem["importance"], self._params
            )
            new_importance = apply_reconsolidation(new_importance, self._params)
            new_access_count = mem["access_count"] + 1

            # Mechanism 10: Auto-pin by access frequency
            new_pinned = mem.get("pinned", 0)
            if new_access_count >= self._params.auto_pin_threshold:
                if not new_pinned:
                    new_pinned = 1
                    logger.info(
                        "cognitive-memory: auto-pinned memory %s (access_count=%d)",
                        mem["id"], new_access_count,
                    )

            self._conn.execute(
                """UPDATE memories
                   SET importance = ?, last_access = ?, access_count = ?, pinned = ?
                   WHERE id = ?""",
                (new_importance, now, new_access_count, new_pinned, mem["id"]),
            )

        for comp in competitors:
            new_importance = apply_rif_penalty(
                comp["importance"], self._params
            )
            self._conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (new_importance, comp["id"]),
            )

        self._conn.commit()

    # -- Maintenance --------------------------------------------------------

    def apply_global_decay(self) -> int:
        """Check which memories would be prunable based on current decay.

        Returns count of prunable memories. Does NOT modify stored importance —
        decay is computed on the fly at retrieval/pruning time, never stored.

        Pinned memories are never counted as prunable.
        Superseded memories (importance=0) ARE counted as prunable.
        """
        now = time.time()
        prunable = 0

        with self._lock:
            if not self._conn:
                return 0

            cur = self._conn.execute(
                "SELECT id, importance, last_access, origin, pinned, hard_to_find, "
                "access_count, temporal, superseded FROM memories"
            )
            rows = cur.fetchall()

            for row in rows:
                if row["pinned"]:
                    continue
                temporal = row["temporal"] or "stable"
                decayed = apply_decay(
                    row["importance"], row["last_access"], now, self._params, temporal
                )
                # Superseded memories always have importance 0, use a high floor
                if row["superseded"]:
                    if decayed < self._params.decay_floor:
                        prunable += 1
                    continue
                effective_floor = self._effective_floor(
                    row["origin"], bool(row["hard_to_find"]), row["access_count"]
                )
                if decayed < effective_floor:
                    prunable += 1

        return prunable

    def _effective_floor(self, origin: str, hard_to_find: bool = False,
                         access_count: int = 0) -> float:
        """Get the effective decay floor for a memory.

        Important memories (user corrections, preferences, research) get a LOWER
        floor, meaning they can decay further before being pruned.

        Hard-to-find memories get an even lower floor (0.01).

        Access-count-based floor (mechanism 11): memories accessed more
        frequently get a lower floor, making them survive longer:
            floor = base_floor / (1 + access_count * 0.1)

        So:
        - 0 accesses: floor × 1.0 (normal)
        - 5 accesses: floor × 0.67
        - 10 accesses: floor × 0.5 (survives 2x longer)
        - 20 accesses: floor × 0.33 (survives 3x longer)
        - Auto-pinned (≥ threshold): floor = 0 (never pruned)
        """
        if hard_to_find:
            base = 0.01
        elif origin == "user_correction":
            base = min(0.02, self._params.decay_floor)
        elif origin == "user_preference":
            base = min(0.03, self._params.decay_floor)
        elif origin == "research_finding":
            base = min(0.03, self._params.decay_floor)
        else:
            base = self._params.decay_floor

        # Access-count-based floor reduction
        if access_count > 0:
            base = base / (1.0 + access_count * 0.1)
            base = max(base, 0.001)  # Never go below 0.001

        return base

    def prune(self) -> int:
        """Delete all memories below their origin-specific decay floor.

        Computes decayed importance on the fly — does not rely on stored
        importance being pre-decayed.

        Pinned memories are NEVER pruned, regardless of decay level.
        Hard-to-find memories get a lower floor (0.01) for extra protection.
        Superseded memories (importance=0, superseded=1) are always pruned.

        Before pruning, each deleted memory is logged to prune_log.md for audit.
        """
        now = time.time()
        with self._lock:
            if not self._conn:
                return 0

            cur = self._conn.execute(
                "SELECT id, content, importance, last_access, origin, pinned, "
                "hard_to_find, access_count, temporal, superseded FROM memories"
            )
            rows = cur.fetchall()

            to_delete = []
            for row in rows:
                # Pinned memories are never pruned
                if row["pinned"]:
                    continue

                temporal = row["temporal"] or "stable"
                decayed = apply_decay(
                    row["importance"], row["last_access"], now, self._params, temporal
                )

                if row["superseded"]:
                    # Superseded memories always get pruned if below floor
                    if decayed < self._params.decay_floor:
                        to_delete.append(row)
                    continue

                effective_floor = self._effective_floor(
                    row["origin"], bool(row["hard_to_find"]), row["access_count"]
                )
                if decayed < effective_floor:
                    to_delete.append(row)

            if to_delete:
                # Log each pruned memory for audit
                self._log_pruned(to_delete)

                ids = [r["id"] for r in to_delete]
                placeholders = ",".join("?" * len(ids))
                cur = self._conn.execute(
                    f"DELETE FROM memories WHERE id IN ({placeholders})",
                    ids,
                )
                self._conn.commit()
                logger.info("cognitive-memory: pruned %d decayed memories", cur.rowcount)
                return cur.rowcount
            return 0

    def _log_pruned(self, rows: List[sqlite3.Row]) -> None:
        """Log pruned memories to prune_log.md for audit trail.

        Format:
        2026-08-10 21:50 | pruned id=abc123 | origin=agent_inference | imp=0.03 | "content snippet..."
        """
        from datetime import datetime
        lines = []
        for row in rows:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            content_snippet = row["content"][:80].replace("\n", " ")
            lines.append(
                f"{ts} | pruned id={row['id'][:8]} | origin={row['origin']} | "
                f"imp={row['importance']:.3f} | acc={row['access_count']} | "
                f"\"{content_snippet}...\""
            )

        # Append to prune log (create if not exists)
        log_path = str(self._prune_log_path)
        with open(log_path, "a") as f:
            f.write("\n".join(lines) + "\n")

    def count(self, target: Optional[str] = None, include_superseded: bool = False) -> int:
        """Count memories, optionally filtered by target."""
        with self._lock:
            if not self._conn:
                return 0
            if include_superseded:
                if target:
                    cur = self._conn.execute(
                        "SELECT COUNT(*) FROM memories WHERE target = ?", (target,)
                    )
                else:
                    cur = self._conn.execute("SELECT COUNT(*) FROM memories")
            else:
                if target:
                    cur = self._conn.execute(
                        "SELECT COUNT(*) FROM memories WHERE target = ? AND superseded = 0",
                        (target,),
                    )
                else:
                    cur = self._conn.execute(
                        "SELECT COUNT(*) FROM memories WHERE superseded = 0"
                    )
            return cur.fetchone()[0]

    def total_chars(self, target: Optional[str] = None) -> int:
        """Total character count of all memory content (for budget management)."""
        with self._lock:
            if not self._conn:
                return 0
            if target:
                cur = self._conn.execute(
                    "SELECT SUM(LENGTH(content)) FROM memories WHERE target = ? AND superseded = 0",
                    (target,),
                )
            else:
                cur = self._conn.execute(
                    "SELECT SUM(LENGTH(content)) FROM memories WHERE superseded = 0"
                )
            result = cur.fetchone()[0]
            return result or 0

    def rebuild_fts(self) -> None:
        """Rebuild the FTS index from scratch (maintenance)."""
        with self._lock:
            if not self._conn or not self._fts_available:
                return
            self._conn.execute(
                "INSERT INTO memories_fts(memories_fts) VALUES('rebuild')"
            )
            self._conn.commit()