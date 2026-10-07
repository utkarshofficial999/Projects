"""
agent_engine.memory.base_memory
===============================

Abstract base classes and data models for the memory subsystem.

This module defines the core abstractions that all memory backends must
implement, along with shared data structures used across the memory layer.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class MemoryType(Enum):
    """Enumeration of memory types for categorizing stored entries."""

    CONVERSATION = "conversation"
    FACT = "fact"
    INSTRUCTION = "instruction"
    OBSERVATION = "observation"
    TOOL_RESULT = "tool_result"
    REFLECTION = "reflection"
    STATE = "state"


@dataclass
class MemoryEntry:
    """
    A single memory entry representing a piece of information stored in memory.

    Attributes:
        id: Unique identifier for this memory entry.
        content: The textual content of the memory.
        memory_type: The category/type of this memory.
        metadata: Additional structured metadata associated with this memory.
        embedding: Optional vector embedding for semantic search.
        created_at: Timestamp when this memory was created.
        updated_at: Timestamp when this memory was last updated.
        access_count: Number of times this memory has been retrieved.
        importance: Importance score (0.0 to 1.0) used for prioritization.
    """

    content: str
    memory_type: MemoryType = MemoryType.CONVERSATION
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    access_count: int = 0
    importance: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this memory entry to a dictionary.

        Returns:
            A dictionary representation of this memory entry.
        """
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Deserialize a memory entry from a dictionary.

        Args:
            data: Dictionary containing memory entry fields.

        Returns:
            A MemoryEntry instance reconstructed from the dictionary.
        """
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
        )


class BaseMemory(ABC):
    """
    Abstract base class for all memory backends.

    Defines the core interface that all memory implementations must provide:
    storing, retrieving, updating, and deleting memory entries.
    """

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry.

        Args:
            entry: The memory entry to store.

        Returns:
            The unique identifier of the stored entry.
        """
        ...

    @abstractmethod
    async def retrieve(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> List[MemoryEntry]:
        """
        Retrieve memory entries matching the given criteria.

        Args:
            query: Optional text query for filtering.
            memory_type: Optional filter by memory type.
            limit: Maximum number of entries to return.
            **kwargs: Additional backend-specific parameters.

        Returns:
            A list of matching memory entries.
        """
        ...

    @abstractmethod
    async def update(self, entry_id: str, **kwargs: Any) -> bool:
        """
        Update an existing memory entry.

        Args:
            entry_id: The ID of the entry to update.
            **kwargs: Fields to update (e.g., content, metadata, importance).

        Returns:
            True if the entry was updated, False if not found.
        """
        ...

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            entry_id: The ID of the entry to delete.

        Returns:
            True if the entry was deleted, False if not found.
        """
        ...

    @abstractmethod
    async def clear(self) -> int:
        """
        Clear all memory entries.

        Returns:
            The number of entries that were removed.
        """
        ...

    @abstractmethod
    async def count(self) -> int:
        """
        Get the total number of stored memory entries.

        Returns:
            The count of stored entries.
        """
        ...
