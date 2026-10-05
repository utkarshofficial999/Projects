"""
agent_engine.memory
===================

Dynamic memory management for agent state and context.

This module provides the memory subsystem that allows agents to:
- Store and retrieve contextual information
- Maintain conversation history
- Track task state and progress
- Support multiple backends (in-memory, Redis, SQLite)

Planned components:
- MemoryStore: Abstract base for memory backends
- InMemoryStore: In-memory implementation
- MemoryManager: High-level memory management with TTL and compression
"""

__all__ = []
