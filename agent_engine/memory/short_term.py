"""
agent_engine.memory.short_term
==============================

Short-term memory implementation using a sliding window buffer.

This module provides the ShortTermMemory class, which maintains a bounded
buffer of recent interactions. When the buffer exceeds its capacity, older
entries are evicted (optionally flushed to long-term memory).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from agent_engine.memory.base_memory import BaseMemory, MemoryEntry, MemoryType

logger = logging.getLogger(__name__)


class ShortTermMemory(BaseMemory):
    """
    Short-term memory with a sliding window of recent interactions.

    This class maintains a bounded deque of MemoryEntry objects. When the
    window is full, the oldest entries are evicted. Optionally, evicted
    entries can be flushed to a long-term memory backend for persistence.

    Attributes:
        max_size: Maximum number of entries to retain in the buffer.
        _buffer: The underlying deque storing memory entries.
        _long_term: Optional long-term memory backend for overflow.
    """

    def __init__(
        self,
        max_size: int = 50,
        long_term_memory: Optional[BaseMemory] = None,
    ) -> None:
        """
        Initialize the ShortTermMemory.

        Args:
            max_size: Maximum number of entries to keep in the sliding window.
            long_term_memory: Optional long-term memory to receive evicted entries.
        """
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer")
        self.max_size: int = max_size
        self._buffer: Deque[MemoryEntry] = deque(maxlen=max_size)
        self._long_term: Optional[BaseMemory] = long_term_memory

    @property
    def size(self) -> int:
        """Current number of entries in the buffer."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """Whether the buffer has reached its maximum capacity."""
        return len(self._buffer) >= self.max_size

    async def store(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry in the short-term buffer.

        If the buffer is full, the oldest entry is evicted. If a long-term
        memory backend is configured, the evicted entry is flushed there.

        Args:
            entry: The memory entry to store.

        Returns:
            The unique identifier of the stored entry.
        """
        # Check if we're about to evict an entry
        if self.is_full:
            evicted = self._buffer[0]
            logger.debug(
                "Short-term buffer full; evicting entry %s", evicted.id
            )
            if self._long_term is not None:
                try:
                    await self._long_term.store(evicted)
                    logger.debug(
                        "Flushed evicted entry %s to long-term memory",
                        evicted.id,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to flush evicted entry %s to long-term memory: %s",
                        evicted.id,
                        str(e),
                    )

        self._buffer.append(entry)
        logger.debug(
            "Stored entry %s in short-term memory (size: %d/%d)",
            entry.id,
            self.size,
            self.max_size,
        )
        return entry.id

    async def retrieve(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[MemoryEntry]:
        """
        Retrieve recent memory entries from the buffer.

        Args:
            query: Optional substring to filter entries by content.
            memory_type: Optional filter by memory type.
            limit: Maximum number of entries to return (most recent first).
            **kwargs: Additional parameters (ignored).

        Returns:
            A list of matching memory entries, most recent first.
        """
        results: List[MemoryEntry] = []
        # Iterate from most recent to oldest
        for entry in reversed(self._buffer):
            if memory_type is not None and entry.memory_type != memory_type:
                continue
            if query is not None and query.lower() not in entry.content.lower():
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    async def get_all(self) -> List[MemoryEntry]:
        """
        Get all entries in the buffer in chronological order.

        Returns:
            A list of all memory entries, oldest first.
        """
        return list(self._buffer)

    async def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """
        Get the most recent N entries.

        Args:
            n: Number of recent entries to retrieve.

        Returns:
            A list of the most recent memory entries, oldest first.
        """
        if n <= 0:
            return []
        if n >= len(self._buffer):
            return list(self._buffer)
        return list(self._buffer)[-n:]

    async def update(self, entry_id: str, **kwargs: Any) -> bool:
        """
        Update an existing memory entry in the buffer.

        Args:
            entry_id: The ID of the entry to update.
            **kwargs: Fields to update (content, metadata, importance, etc.).

        Returns:
            True if the entry was found and updated, False otherwise.
        """
        for i, entry in enumerate(self._buffer):
            if entry.id == entry_id:
                if "content" in kwargs:
                    entry.content = kwargs["content"]
                if "metadata" in kwargs:
                    entry.metadata.update(kwargs["metadata"])
                if "importance" in kwargs:
                    entry.importance = kwargs["importance"]
                if "memory_type" in kwargs:
                    entry.memory_type = kwargs["memory_type"]
                entry.updated_at = entry.updated_at  # Refresh timestamp
                self._buffer[i] = entry
                return True
        return False

    async def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry from the buffer.

        Args:
            entry_id: The ID of the entry to delete.

        Returns:
            True if the entry was found and deleted, False otherwise.
        """
        for i, entry in enumerate(self._buffer):
            if entry.id == entry_id:
                del self._buffer[i]
                return True
        return False

    async def clear(self) -> int:
        """
        Clear all entries from the buffer.

        Returns:
            The number of entries that were removed.
        """
        count = len(self._buffer)
        self._buffer.clear()
        logger.debug("Cleared %d entries from short-term memory", count)
        return count

    async def count(self) -> int:
        """
        Get the number of entries in the buffer.

        Returns:
            The count of stored entries.
        """
        return len(self._buffer)

    def __len__(self) -> int:
        """Return the number of entries in the buffer."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ShortTermMemory(size={self.size}, max_size={self.max_size}, "
            f"has_long_term={self._long_term is not None})"
        )
