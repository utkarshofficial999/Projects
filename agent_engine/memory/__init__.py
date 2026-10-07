"""
agent_engine.memory
===================

Dynamic memory management for agent state and context.

This module provides:
- Memory backend abstractions (in-memory, SQLite, Vector Store)
- Memory entry models and serialization
- Memory retrieval and filtering strategies
- Memory compression and summarization

Classes:
    - ShortTermMemory: Sliding window buffer for recent interactions
    - LongTermMemory: Abstract interface for persistent memory
    - SQLiteLongTermMemory: SQLite-based structured state storage
    - VectorStore: FAISS-based semantic recall
    - ContextManager: Intelligent context window management
"""

from agent_engine.memory.base_memory import (
    MemoryEntry,
    MemoryType,
    BaseMemory,
)
from agent_engine.memory.short_term import ShortTermMemory
from agent_engine.memory.long_term_sqlite import SQLiteLongTermMemory
from agent_engine.memory.vector_store import VectorStore
from agent_engine.memory.context_manager import ContextManager

__all__ = [
    "MemoryEntry",
    "MemoryType",
    "BaseMemory",
    "ShortTermMemory",
    "SQLiteLongTermMemory",
    "VectorStore",
    "ContextManager",
]
