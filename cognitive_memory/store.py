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
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .decay import (
    DecayParams,
    apply_access_reinforcement,
    apply_confidence_decay,
    apply_decay,
    apply_reconsolidation,
    apply_rif_penalty,
    initial_confidence,
    initial_importance,
    should_prune,
)

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
    tags          TEXT DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_memories_target ON memories(target);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_last_access ON memories(last_access);
"""

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

    Escapes double quotes and strips FTS5 special operators (*, NEAR, :, -, +)
    from the unquoted portion to prevent unexpected query semantics.
    The result is a quoted phrase match — safe and predictable.
    """
    # Escape double quotes for FTS5 string literal
    safe = query.replace('"', '""')
    # Return as a quoted phrase — this is a prefix match, not a full
    # boolean query, so FTS5 special chars are inert inside quotes.
    return f'"{safe}"'


class MemoryStore:
    """SQLite-backed memory store with FTS5 and cognitive decay."""

    def __init__(self, db_path: Path, params: DecayParams):
        self._db_path = db_path
        self._params = params
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._connected = False
        self._fts_available = False

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
                # Create base table first (always works)
                conn.executescript(_BASE_SCHEMA_SQL)
                # Try FTS5 — may fail if not compiled into SQLite
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

    # -- Write operations ---------------------------------------------------

    def add(
        self,
        target: str,
        content: str,
        origin: str = "unknown",
        tags: Optional[List[str]] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> str:
        """Add a new memory. Returns the memory ID."""
        mem_id = str(uuid.uuid4())
        now = time.time()
        if importance is None:
            importance = initial_importance(origin, self._params)
        if confidence is None:
            confidence = initial_confidence(origin, self._params)
        tags_json = json.dumps(tags or [])

        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                """INSERT INTO memories
                   (id, target, content, importance, confidence,
                    created_at, last_access, access_count, origin, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                (mem_id, target, content, importance, confidence,
                 now, now, origin, tags_json),
            )
            self._conn.commit()
        logger.debug(
            "cognitive-memory: added memory %s (origin=%s, importance=%.2f)",
            mem_id, origin, importance,
        )
        return mem_id

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

    def remove_by_content(self, content_substring: str) -> int:
        """Remove memories whose content contains the substring. Returns count.

        Escapes LIKE wildcards (% and _) in the substring so they match
        literally, not as pattern characters.
        """
        # Escape LIKE special characters so they match literally
        # ESCAPE clause uses backslash as the escape character
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
        """Get all memories."""
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
                "SELECT * FROM memories WHERE target = ? ORDER BY importance DESC",
                (target,),
            )
            return [dict(r) for r in cur.fetchall()]

    def search(
        self,
        query: str,
        target: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """FTS5 search with cognitive relevance ranking.

        Returns a list of (memory_dict, score) tuples sorted by score DESC.
        Score = normalized_fts_rank * decayed_importance * decayed_confidence.

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
            # cognitive metadata.
            safe_query = _sanitize_fts_query(query)

            sql = """
                SELECT m.*, bm25(memories_fts) as fts_score
                FROM memories_fts
                JOIN memories m ON m.rowid = memories_fts.rowid
                WHERE memories_fts MATCH ?
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
            for row in rows:
                fts_score = row.pop("fts_score", 0.0)
                # bm25 in SQLite FTS5 returns negative values (lower = better match)
                # We negate and normalize: a perfect match is typically around -2 to -5
                normalized_fts = max(0.0, min(1.0, (-fts_score) / 3.0)) if fts_score != 0 else 0.5

                decayed_importance = apply_decay(
                    row["importance"], row["last_access"], now, self._params
                )
                decayed_confidence = apply_confidence_decay(
                    row["confidence"], row["created_at"], now, self._params
                )

                score = normalized_fts * decayed_importance * decayed_confidence
                scored.append((row, score))

            # Sort by cognitive score descending
            scored.sort(key=lambda x: x[1], reverse=True)

            # Apply access reinforcement + RIF to top results — under the same lock
            if scored:
                self._apply_retrieval_effects_locked(
                    [s[0] for s in scored[:max_limit]],
                    [s[0] for s in scored[max_limit:]],
                )

            return scored[:max_limit]

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
        sql = "SELECT * FROM memories WHERE content LIKE ?"
        params: list = [f"%{query}%"]
        if target:
            sql += " AND target = ?"
            params.append(target)
        sql += " LIMIT ?"
        params.append(max_limit * 3)

        cur = self._conn.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]

        scored = []
        for row in rows:
            decayed_importance = apply_decay(
                row["importance"], row["last_access"], now, self._params
            )
            decayed_confidence = apply_confidence_decay(
                row["confidence"], row["created_at"], now, self._params
            )
            # Without FTS, use importance × confidence as the ranking
            score = decayed_importance * decayed_confidence
            scored.append((row, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        if scored:
            self._apply_retrieval_effects_locked(
                [s[0] for s in scored[:max_limit]],
                [s[0] for s in scored[max_limit:]],
            )

        return scored[:max_limit]

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

            sql = "SELECT * FROM memories"
            params: list = []
            if target:
                sql += " WHERE target = ?"
                params.append(target)
            sql += " ORDER BY importance DESC LIMIT ?"
            params.append(max_limit * 3)
            cur = self._conn.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]

            scored = []
            for row in rows:
                decayed_importance = apply_decay(
                    row["importance"], row["last_access"], now, self._params
                )
                decayed_confidence = apply_confidence_decay(
                    row["confidence"], row["created_at"], now, self._params
                )
                score = decayed_importance * decayed_confidence
                scored.append((row, score))

            scored.sort(key=lambda x: x[1], reverse=True)

            if scored:
                self._apply_retrieval_effects_locked(
                    [s[0] for s in scored[:max_limit]],
                    [s[0] for s in scored[max_limit:]],
                )

            return scored[:max_limit]

    def _apply_retrieval_effects_locked(
        self,
        retrieved: List[Dict[str, Any]],
        competitors: List[Dict[str, Any]],
    ) -> None:
        """Apply access reinforcement + reconsolidation + RIF after retrieval.

        - Retrieved memories get access_boost + reconsolidation
        - Competing memories get RIF penalty

        MUST be called with self._lock already held (uses RLock so re-entry
        is safe, but we avoid the overhead of a second acquire).
        """
        now = time.time()
        assert self._conn is not None
        for mem in retrieved:
            new_importance = apply_access_reinforcement(
                mem["importance"], self._params
            )
            new_importance = apply_reconsolidation(new_importance, self._params)
            self._conn.execute(
                """UPDATE memories
                   SET importance = ?, last_access = ?, access_count = access_count + 1
                   WHERE id = ?""",
                (new_importance, now, mem["id"]),
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
        """Apply time-based decay to ALL memories. Returns count of prunable memories.

        Called once per turn (sync_turn) to keep importance scores current.
        """
        now = time.time()
        prunable = 0

        with self._lock:
            if not self._conn:
                return 0

            cur = self._conn.execute(
                "SELECT id, importance, last_access, created_at, confidence FROM memories"
            )
            rows = cur.fetchall()

            for row in rows:
                decayed_importance = apply_decay(
                    row["importance"], row["last_access"], now, self._params
                )
                decayed_confidence = apply_confidence_decay(
                    row["confidence"], row["created_at"], now, self._params
                )

                self._conn.execute(
                    "UPDATE memories SET importance = ?, confidence = ? WHERE id = ?",
                    (decayed_importance, decayed_confidence, row["id"]),
                )

                if should_prune(decayed_importance, self._params):
                    prunable += 1

            self._conn.commit()

        return prunable

    def prune(self) -> int:
        """Delete all memories below the decay floor. Returns count deleted."""
        with self._lock:
            if not self._conn:
                return 0
            cur = self._conn.execute(
                "DELETE FROM memories WHERE importance < ?",
                (self._params.decay_floor,),
            )
            self._conn.commit()
            return cur.rowcount

    def count(self, target: Optional[str] = None) -> int:
        """Count memories, optionally filtered by target."""
        with self._lock:
            if not self._conn:
                return 0
            if target:
                cur = self._conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE target = ?", (target,)
                )
            else:
                cur = self._conn.execute("SELECT COUNT(*) FROM memories")
            return cur.fetchone()[0]

    def total_chars(self, target: Optional[str] = None) -> int:
        """Total character count of all memory content (for budget management)."""
        with self._lock:
            if not self._conn:
                return 0
            if target:
                cur = self._conn.execute(
                    "SELECT SUM(LENGTH(content)) FROM memories WHERE target = ?",
                    (target,),
                )
            else:
                cur = self._conn.execute("SELECT SUM(LENGTH(content)) FROM memories")
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