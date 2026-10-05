"""
agent_engine.core
=================

Core engine components including configuration management and logging infrastructure.

This module contains the foundational building blocks that all other parts of the
agent engine depend on:

- Configuration management using Pydantic Settings
- Centralized structured logging
- Base types and utilities
"""

from agent_engine.core.config import EngineConfig, get_config
from agent_engine.core.logging import setup_logging, get_logger

__all__ = [
    "EngineConfig",
    "get_config",
    "setup_logging",
    "get_logger",
]
