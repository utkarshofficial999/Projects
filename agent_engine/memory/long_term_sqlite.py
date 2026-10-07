"""
agent_engine.memory.long_term_sqlite
====================================

SQLite-based long-term memory for structured state persistence.

This module provides the SQLiteLongTermMemory class, which persists
memory entries to a local SQLite database. It supports structured queries
by memory type, metadata filtering, and full-text search.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_engine.memory.base_memory import BaseMemory, MemoryEntry, MemoryType

logger = logging.getLogger(__name__)

# SQL schema for the memory table
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_entries (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding BLOB,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    importance REAL NOT NULL DEFAULT 0.5
);

CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type);
CREATE INDEX IF NOT EXISTS idx_created_at ON memory_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_importance ON memory_entries(importance DESC);
"""


class SQLiteLongTermMemory(BaseMemory):
    """
    Long-term memory backed by a SQLite database.

    Provides persistent storage for memory entries with support for:
    - Structured queries by memory type
    - Metadata filtering
    - Full-text search on content
    - Importance-based prioritization

    Thread-safe: Uses a lock to serialize database access.

    Attributes:
        db_path: Path to the SQLite database file.
        _conn: The underlying SQLite connection.
        _lock: Thread lock for safe concurrent access.
    """

    def __init__(self, db_path: str = "agent_memory.db") -> None:
        """
        Initialize the SQLite long-term memory.

        Args:
            db_path: Path to the SQLite database file. Use ':memory:' for
                     an in-memory database (useful for testing).
        """
        self.db_path: str = db_path
        self._lock: threading.Lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the database connection and create schema."""
        with self._lock:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            logger.debug("Initialized SQLite memory at %s", self.db_path)

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """
        Convert a database row to a MemoryEntry.

        Args:
            row: A sqlite3.Row from the memory_entries table.

        Returns:
            A MemoryEntry instance.
        """
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        embedding = None
        if row["embedding"]:
            import struct
            # Embeddings are stored as packed float32 arrays
            data = row["embedding"]
            n = len(data) // 4
            embedding = list(struct.unpack(f"<{n}f", data))

        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            metadata=metadata,
            embedding=embedding,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            access_count=row["access_count"],
            importance=row["importance"],
        )

    def _entry_to_params(self, entry: MemoryEntry) -> tuple:
        """
        Convert a MemoryEntry to SQL parameters.

        Args:
            entry: The memory entry to convert.

        Returns:
            A tuple of parameters for the INSERT/UPDATE statement.
        """
        import struct

        metadata_json = json.dumps(entry.metadata)
        embedding_blob = None
        if entry.embedding is not None:
            n = len(entry.embedding)
            embedding_blob = struct.pack(f"<{n}f", *entry.embedding)

        return (
            entry.id,
            entry.content,
            entry.memory_type.value,
            metadata_json,
            embedding_blob,
            entry.created_at.isoformat(),
            entry.updated_at.isoformat(),
            entry.access_count,
            entry.importance,
        )

    async def store(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry in the SQLite database.

        If an entry with the same ID already exists, it is updated.

        Args:
            entry: The memory entry to store.

        Returns:
            The unique identifier of the stored entry.
        """
        params = self._entry_to_params(entry)
        sql = """
            INSERT OR REPLACE INTO memory_entries
            (id, content, memory_type, metadata, embedding,
             created_at, updated_at, access_count, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()
        logger.debug("Stored entry %s in SQLite memory", entry.id)
        return entry.id

    async def retrieve(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[MemoryEntry]:
        """
        Retrieve memory entries from the SQLite database.

        Args:
            query: Optional substring to search in content.
            memory_type: Optional filter by memory type.
            limit: Maximum number of entries to return.
            min_importance: Minimum importance score to include.
            metadata_filter: Optional dict of metadata key-value pairs to match.
            **kwargs: Additional parameters (ignored).

        Returns:
            A list of matching memory entries, ordered by importance (desc)
            then created_at (desc).
        """
        conditions: List[str] = []
        params: List[Any] = []

        if memory_type is not None:
            conditions.append("memory_type = ?")
            params.append(memory_type.value)

        if query is not None:
            conditions.append("content LIKE ?")
            params.append(f"%{query}%")

        if min_importance > 0.0:
            conditions.append("importance >= ?")
            params.append(min_importance)

        # Metadata filtering: check each key-value pair
        if metadata_filter:
            for key, value in metadata_filter.items():
                conditions.append(
                    "json_extract(metadata, ?) = ?"
                )
                params.append(f"$.{key}")
                params.append(str(value))

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT * FROM memory_entries
            {where_clause}
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
        """
        params.append(limit)

        with self._lock:
            self._conn.row_factory = sqlite3.Row
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a single memory entry by its ID.

        Args:
            entry_id: The ID of the entry to retrieve.

        Returns:
            The MemoryEntry if found, None otherwise.
        """
        sql = "SELECT * FROM memory_entries WHERE id = ?"
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            cursor = self._conn.execute(sql, (entry_id,))
            row = cursor.fetchone()

        if row is None:
            return None

        entry = self._row_to_entry(row)
        # Increment access count
        await self._increment_access_count(entry_id)
        return entry

    async def _increment_access_count(self, entry_id: str) -> None:
        """Increment the access count for a memory entry."""
        sql = "UPDATE memory_entries SET access_count = access_count + 1 WHERE id = ?"
        with self._lock:
            self._conn.execute(sql, (entry_id,))
            self._conn.commit()

    async def update(self, entry_id: str, **kwargs: Any) -> bool:
        """
        Update an existing memory entry.

        Args:
            entry_id: The ID of the entry to update.
            **kwargs: Fields to update (content, metadata, importance, memory_type).

        Returns:
            True if the entry was found and updated, False otherwise.
        """
        # First, check if the entry exists
        existing = await self.get_by_id(entry_id)
        if existing is None:
            return False

        # Build the update
        updates: List[str] = []
        params: List[Any] = []

        if "content" in kwargs:
            updates.append("content = ?")
            params.append(kwargs["content"])

        if "memory_type" in kwargs:
            updates.append("memory_type = ?")
            params.append(kwargs["memory_type"].value)

        if "metadata" in kwargs:
            # Merge metadata
            merged = {**existing.metadata, **kwargs["metadata"]}
            updates.append("metadata = ?")
            params.append(json.dumps(merged))

        if "importance" in kwargs:
            updates.append("importance = ?")
            params.append(kwargs["importance"])

        if "embedding" in kwargs:
            import struct
            if kwargs["embedding"] is not None:
                n = len(kwargs["embedding"])
                updates.append("embedding = ?")
                params.append(struct.pack(f"<{n}f", *kwargs["embedding"]))
            else:
                updates.append("embedding = NULL")

        if not updates:
            return True  # Nothing to update

        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(entry_id)

        sql = f"UPDATE memory_entries SET {', '.join(updates)} WHERE id = ?"
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()

        return cursor.rowcount > 0

    async def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            entry_id: The ID of the entry to delete.

        Returns:
            True if the entry was found and deleted, False otherwise.
        """
        sql = "DELETE FROM memory_entries WHERE id = ?"
        with self._lock:
            cursor = self._conn.execute(sql, (entry_id,))
            self._conn.commit()
        return cursor.rowcount > 0

    async def clear(self) -> int:
        """
        Clear all memory entries.

        Returns:
            The number of entries that were removed.
        """
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM memory_entries")
            count = cursor.fetchone()[0]
            self._conn.execute("DELETE FROM memory_entries")
            self._conn.commit()
        logger.debug("Cleared %d entries from SQLite memory", count)
        return count

    async def count(self) -> int:
        """
        Get the total number of stored memory entries.

        Returns:
            The count of stored entries.
        """
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM memory_entries")
            return cursor.fetchone()[0]

    async def search(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
    ) -> List[MemoryEntry]:
        """
        Perform a full-text search on memory entries.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            memory_type: Optional filter by memory type.

        Returns:
            A list of matching memory entries.
        """
        return await self.retrieve(
            query=query,
            memory_type=memory_type,
            limit=limit,
        )

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
                logger.debug("Closed SQLite memory connection")

    def __enter__(self) -> "SQLiteLongTermMemory":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SQLiteLongTermMemory(db_path='{self.db_path}')"
