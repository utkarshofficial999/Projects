"""
agent_engine.tools
==================

Tool registry and execution framework.

This package provides:

* :class:`BaseTool` – abstract base class for all tools.
* :class:`ToolRegistry` – dynamic registry with decorator support.
* :class:`ToolSandbox` – subprocess-based execution sandbox.
* :class:`ToolResult` – structured result of a tool invocation.
* :func:`register_tool` – class decorator for the default registry.
* :data:`default_registry` – process-wide default registry.

Built-in tools (``search``, ``calculator``) are auto-registered when
:mod:`agent_engine.tools.builtin` is imported.
"""

from agent_engine.tools.base_tool import BaseTool, ToolExecutionError
from agent_engine.tools.registry import (
    ToolRegistry,
    ToolResult,
    default_registry,
    register_tool,
)
from agent_engine.tools.sandbox import SandboxConfig, ToolSandbox

__all__ = [
    "BaseTool",
    "ToolExecutionError",
    "ToolRegistry",
    "ToolResult",
    "ToolSandbox",
    "SandboxConfig",
    "default_registry",
    "register_tool",
]
