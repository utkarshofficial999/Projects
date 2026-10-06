"""
agent_engine.tools.registry
===========================

Dynamic tool registry with decorator-based registration.

The :class:`ToolRegistry` is the central catalogue of all tools
available to the agent. Tools can be registered in two ways:

1. **Decorator** – use :func:`register_tool` on a class::

       @register_tool
       class MyTool(BaseTool):
           ...

2. **Programmatic** – call :meth:`ToolRegistry.register` at runtime::

       registry.register(MyTool())

The registry also exposes:

* :meth:`get` – look up a tool by name.
* :meth:`list_tools` – enumerate all registered tools.
* :meth:`to_schemas` – bulk-export schemas for LLM function-calling.
* :meth:`execute` – validate + run a tool by name, returning a
  structured :class:`ToolResult`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

from agent_engine.tools.base_tool import BaseTool, ToolExecutionError

logger = logging.getLogger(__name__)


class ToolResult:
    """Structured result of a tool execution.

    Attributes
    ----------
    tool_name : str
        Name of the tool that was executed.
    success : bool
        Whether the tool completed without error.
    output : str
        The tool's output (empty string on failure).
    error : Optional[str]
        Error message if the tool failed, else ``None``.
    """

    __slots__ = ("tool_name", "success", "output", "error")

    def __init__(
        self,
        tool_name: str,
        success: bool,
        output: str = "",
        error: Optional[str] = None,
    ) -> None:
        self.tool_name = tool_name
        self.success = success
        self.output = output
        self.error = error

    def to_context_string(self) -> str:
        """Render the result as a short string for the agent's context.

        On success the output is returned verbatim. On failure a
        structured error message is produced so the agent can
        self-correct.
        """
        if self.success:
            return self.output
        return (
            f"TOOL ERROR [{self.tool_name}]: {self.error}\n"
            f"Please review the error and adjust your approach."
        )

    def __repr__(self) -> str:  # pragma: no cover
        status = "OK" if self.success else "ERR"
        return f"<ToolResult {self.tool_name} {status}>"


class ToolRegistry:
    """Central registry for all tools available to the agent.

    The registry is intentionally a plain class (not a singleton) so
    that tests can create isolated instances. A module-level default
    instance is provided for convenience.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> BaseTool:
        """Register a tool instance.

        Parameters
        ----------
        tool:
            An instantiated :class:`BaseTool` subclass.

        Returns
        -------
        BaseTool
            The same instance (useful as a decorator on instances).

        Raises
        ------
        ValueError
            If a tool with the same name is already registered.
        """
        if not tool.name:
            raise ValueError(
                f"Tool {tool.__class__.__name__} has no name set."
            )
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered."
            )
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)
        return tool

    def register_class(self, tool_cls: Type[BaseTool]) -> Type[BaseTool]:
        """Instantiate and register a tool class.

        This is the workhorse behind the :func:`register_tool`
        decorator.
        """
        instance = tool_cls()
        self.register(instance)
        return tool_cls

    def unregister(self, name: str) -> None:
        """Remove a tool by name. No-op if not found."""
        self._tools.pop(name, None)
        logger.debug("Unregistered tool: %s", name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseTool:
        """Return the tool registered under *name*.

        Raises
        ------
        KeyError
            If no tool with that name exists.
        """
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(sorted(self._tools)) or "(none)"
            raise KeyError(
                f"Tool '{name}' not found. Available: {available}"
            ) from None

    def has(self, name: str) -> bool:
        """Return ``True`` if a tool with *name* is registered."""
        return name in self._tools

    def list_tools(self) -> List[BaseTool]:
        """Return all registered tools (sorted by name)."""
        return [self._tools[k] for k in sorted(self._tools)]

    def list_names(self) -> List[str]:
        """Return all registered tool names (sorted)."""
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # ------------------------------------------------------------------
    # Schema export
    # ------------------------------------------------------------------

    def to_schemas(self) -> List[Dict[str, Any]]:
        """Export all tools as LLM function-calling schemas."""
        return [tool.to_schema() for tool in self.list_tools()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        sandbox: Optional["ToolSandbox"] = None,
    ) -> ToolResult:
        """Validate arguments and execute a tool by name.

        Parameters
        ----------
        name:
            Registered tool name.
        arguments:
            Raw keyword arguments to pass to the tool.
        sandbox:
            Optional :class:`ToolSandbox` instance. When provided, the
            tool's ``execute`` is run inside the sandbox (subprocess
            isolation, timeout, stdout/stderr capture). When ``None``,
            the tool runs in-process.

        Returns
        -------
        ToolResult
            Structured result; never raises for tool-level errors.
        """
        # 1. Resolve tool
        try:
            tool = self.get(name)
        except KeyError as exc:
            return ToolResult(
                tool_name=name,
                success=False,
                error=str(exc),
            )

        # 2. Validate arguments
        try:
            validated = tool.validate_arguments(arguments)
        except ToolExecutionError as exc:
            return ToolResult(
                tool_name=name,
                success=False,
                error=exc.message,
            )

        # 3. Execute
        if sandbox is not None:
            return sandbox.run(tool, validated)

        # In-process execution
        try:
            output = tool.execute(validated)
            return ToolResult(tool_name=name, success=True, output=output)
        except ToolExecutionError as exc:
            return ToolResult(
                tool_name=name, success=False, error=exc.message
            )
        except Exception as exc:  # noqa: BLE001 – catch-all for tool bugs
            logger.exception("Unexpected error in tool '%s'", name)
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Unexpected error: {exc}",
            )


# ----------------------------------------------------------------------
# Module-level default registry + decorator
# ----------------------------------------------------------------------

#: The default, process-wide registry.
default_registry = ToolRegistry()


def register_tool(cls: Type[BaseTool]) -> Type[BaseTool]:
    """Class decorator that registers a tool with the default registry.

    Usage::

        @register_tool
        class MyTool(BaseTool):
            name = "my_tool"
            ...
    """
    default_registry.register_class(cls)
    return cls
