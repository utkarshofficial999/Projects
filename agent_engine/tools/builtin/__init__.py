"""
agent_engine.tools.builtin
==========================

Built-in tools shipped with the engine.

Importing this package auto-registers all built-in tools with the
default registry via their ``@register_tool`` decorators.
"""

from agent_engine.tools.builtin.search_tool import SearchTool
from agent_engine.tools.builtin.calculator_tool import CalculatorTool

__all__ = ["SearchTool", "CalculatorTool"]
