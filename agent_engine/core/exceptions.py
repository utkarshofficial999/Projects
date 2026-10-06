"""
Custom exception hierarchy for the agent engine.

This module defines a structured exception hierarchy that allows
for precise error handling and recovery in the multi-agent system.
"""

from typing import Any, Dict, Optional


class AgentEngineError(Exception):
    """
    Base exception for all agent engine errors.

    Attributes:
        message: Human-readable error description.
        context: Optional dictionary of contextual information.
        code: Optional error code for programmatic handling.
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        code: Optional[str] = None
    ) -> None:
        self.message = message
        self.context = context or {}
        self.code = code
        super().__init__(message)

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        """Serialize exception to a dictionary for logging/persistence."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "context": self.context,
        }


class ConfigurationError(AgentEngineError):
    """Raised when configuration is invalid or missing."""
    pass


class SchemaValidationError(AgentEngineError):
    """Raised when data fails Pydantic schema validation."""

    def __init__(
        self,
        message: str,
        validation_errors: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        self.validation_errors = validation_errors or []
        super().__init__(message, context=context, code="SCHEMA_VALIDATION_ERROR")


class ToolExecutionError(AgentEngineError):
    """Raised when a tool execution fails."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        tool_input: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        self.tool_name = tool_name
        self.tool_input = tool_input or {}
        self.original_error = original_error
        super().__init__(
            message,
            context=context,
            code="TOOL_EXECUTION_ERROR"
        )

    def __str__(self) -> str:
        base = super().__str__()
        if self.original_error:
            return f"{base} | Original: {self.original_error}"
        return base


class ToolNotFoundError(AgentEngineError):
    """Raised when a requested tool is not found in the registry."""

    def __init__(self, tool_name: str, context: Optional[Dict[str, Any]] = None) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"Tool '{tool_name}' not found in registry",
            context=context,
            code="TOOL_NOT_FOUND"
        )


class LLMError(AgentEngineError):
    """Raised when an LLM API call fails."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        status_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        self.provider = provider
        self.model = model
        self.status_code = status_code
        super().__init__(message, context=context, code="LLM_ERROR")


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        retry_after: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=429,
            context=context
        )


class AgentError(AgentEngineError):
    """Raised when an agent encounters an operational error."""

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        self.agent_id = agent_id
        super().__init__(message, context=context, code="AGENT_ERROR")


class MemoryError_(AgentEngineError):
    """Raised when memory operations fail. Named with trailing underscore to avoid shadowing built-in."""

    def __init__(
        self,
        message: str,
        scope: Optional[str] = None,
        key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        self.scope = scope
        self.key = key
        super().__init__(message, context=context, code="MEMORY_ERROR")


class WorkflowError(AgentEngineError):
    """Raised when workflow execution encounters an error."""

    def __init__(
        self,
        message: str,
        workflow_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        self.workflow_id = workflow_id
        super().__init__(message, context=context, code="WORKFLOW_ERROR")


class MaxIterationsExceeded(AgentEngineError):
    """Raised when an agent exceeds its maximum iteration limit."""

    def __init__(
        self,
        agent_id: str,
        max_iterations: int,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        self.agent_id = agent_id
        self.max_iterations = max_iterations
        super().__init__(
            f"Agent '{agent_id}' exceeded maximum iterations ({max_iterations})",
            context=context,
            code="MAX_ITERATIONS_EXCEEDED"
        )


class TimeoutError_(AgentEngineError):
    """Raised when an operation exceeds its timeout. Named with trailing underscore to avoid shadowing built-in."""

    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        super().__init__(message, context=context, code="TIMEOUT")
