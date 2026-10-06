"""
agent_engine.tools.base_tool
============================

Defines the abstract base class for all tools in the engine.

A *tool* is a callable unit of work that an agent can invoke. Each tool
declares:

* a stable ``name`` (used by the LLM to reference it),
* a human-readable ``description`` (surfaced to the LLM),
* a Pydantic ``arguments_schema`` that validates and documents the
  parameters the tool accepts, and
* an ``execute`` method that performs the actual work.

The base class also provides a small amount of shared machinery:

* :meth:`validate_arguments` – coerces a raw ``dict`` into a validated
  Pydantic model instance.
* :meth:`to_schema` – produces a JSON-schema-style dict that can be
  handed to an LLM's function-calling interface.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Raised when a tool fails during execution.

    Attributes
    ----------
    tool_name : str
        The name of the tool that raised the error.
    message : str
        Human-readable error description.
    """

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"[{tool_name}] {message}")


class BaseTool(abc.ABC):
    """Abstract base class for all engine tools.

    Subclasses **must** set the class-level attributes ``name``,
    ``description``, and ``arguments_schema`` and implement
    :meth:`execute`.

    Example
    -------
    >>> class MyTool(BaseTool):
    ...     name = "my_tool"
    ...     description = "Does a thing."
    ...     arguments_schema = MyToolArgs
    ...
    ...     def execute(self, args: MyToolArgs) -> str:
    ...         return "done"
    """

    #: Unique identifier for the tool (snake_case, no spaces).
    name: str = ""

    #: Short description shown to the LLM.
    description: str = ""

    #: Pydantic model describing the tool's input arguments.
    arguments_schema: Type[BaseModel] = BaseModel  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def execute(self, args: BaseModel) -> str:
        """Execute the tool with validated arguments.

        Parameters
        ----------
        args:
            A validated instance of :attr:`arguments_schema`.

        Returns
        -------
        str
            The tool's output (typically a string the agent can reason
            about).

        Raises
        ------
        ToolExecutionError
            If the tool encounters a runtime failure.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_arguments(self, raw: Dict[str, Any]) -> BaseModel:
        """Validate and coerce *raw* into a typed argument model.

        Parameters
        ----------
        raw:
            Raw keyword arguments (usually parsed from an LLM response).

        Returns
        -------
        BaseModel
            A validated instance of :attr:`arguments_schema`.

        Raises
        ------
        ToolExecutionError
            If validation fails.
        """
        try:
            return self.arguments_schema(**raw)
        except ValidationError as exc:
            errors = exc.errors()
            detail = "; ".join(
                f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
                for e in errors
            )
            raise ToolExecutionError(
                self.name, f"Invalid arguments: {detail}"
            ) from exc

    # ------------------------------------------------------------------
    # Schema export (for LLM function-calling)
    # ------------------------------------------------------------------

    def to_schema(self) -> Dict[str, Any]:
        """Return a JSON-schema-style dict describing this tool.

        The returned structure follows the OpenAI function-calling
        convention so it can be passed directly to an LLM client.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.arguments_schema.model_json_schema(),
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{self.__class__.__name__} name={self.name!r}>"
