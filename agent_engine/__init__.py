"""
agent_engine
============

Autonomous Multi-Agent Workflow Engine with Tool Calling, Dynamic Memory and Self-Reflection.

This package provides the core infrastructure for building autonomous multi-agent systems
that can plan, execute, reflect, and adapt their workflows dynamically.

Modules:
    core: Core engine components including configuration and logging.
    tools: Tool registry and execution framework.
    memory: Dynamic memory management for agent state and context.
    agents: Agent implementations and orchestration logic.
"""

__version__ = "0.1.0"
__author__ = "Agent Engine Team"

from agent_engine.core.config import EngineConfig, get_config
from agent_engine.core.logging import setup_logging, get_logger

__all__ = [
    "EngineConfig",
    "get_config",
    "setup_logging",
    "get_logger",
    "__version__",
]
