"""
Pydantic schemas for the agent engine.

This module defines all core data structures used throughout the system,
including messages, tool calls, agent state, and workflow context.
All models are fully serializable for persistence and debugging.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from agent_engine.core.types import (
    AgentStatus,
    MemoryScope,
    MessageType,
    Role,
    WorkflowStatus,
)


def _generate_id() -> str:
    """Generate a unique identifier for entities."""
    return str(uuid4())


def _utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


class Message(BaseModel):
    """
    Represents a single message in the communication protocol.

    This is the fundamental unit of communication between agents,
    users, and tools within the system.

    Attributes:
        id: Unique identifier for the message.
        role: The role of the message sender.
        content: The text content of the message.
        message_type: Categorization of the message type.
        timestamp: When the message was created.
        metadata: Additional metadata (e.g., token counts, model info).
        tool_calls: List of tool calls if this is a tool_call message.
        tool_call_id: Reference to a tool call if this is a tool_result.
        sender_id: Identifier of the sending agent/entity.
        recipient_id: Optional identifier of the intended recipient.
    """
    id: str = Field(default_factory=_generate_id, description="Unique message identifier")
    role: Role = Field(default=Role.USER, description="Role of the message sender")
    content: str = Field(default="", description="Text content of the message")
    message_type: MessageType = Field(default=MessageType.TEXT, description="Type of message")
    timestamp: datetime = Field(default_factory=_utc_now, description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tool_calls: List["ToolCall"] = Field(default_factory=list, description="Tool calls in this message")
    tool_call_id: Optional[str] = Field(default=None, description="Reference to tool call ID")
    sender_id: Optional[str] = Field(default=None, description="ID of sending agent")
    recipient_id: Optional[str] = Field(default=None, description="ID of intended recipient")

    @field_validator("content")
    @classmethod
    def validate_content_not_none(cls, v: Optional[str]) -> str:
        """Ensure content is never None, defaulting to empty string."""
        return v if v is not None else ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialize message from a dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"Message(id={self.id[:8]}..., role={self.role.value}, "
            f"type={self.message_type.value}, content={self.content[:50]!r}...)"
        )


class ToolCall(BaseModel):
    """
    Represents a request to invoke a tool.

    Attributes:
        id: Unique identifier for this tool call.
        name: The name of the tool to invoke.
        arguments: The arguments to pass to the tool.
        timestamp: When the tool call was created.
        metadata: Additional metadata.
    """
    id: str = Field(default_factory=_generate_id, description="Unique tool call identifier")
    name: str = Field(..., description="Name of the tool to invoke")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    timestamp: datetime = Field(default_factory=_utc_now, description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Ensure tool name is not empty."""
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tool call to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Deserialize tool call from a dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        args_preview = str(self.arguments)[:30]
        return f"ToolCall(id={self.id[:8]}..., name={self.name!r}, args={args_preview}...)"


class ToolResult(BaseModel):
    """
    Represents the result of a tool execution.

    Attributes:
        id: Unique identifier for this result.
        tool_call_id: Reference to the ToolCall that produced this result.
        tool_name: Name of the tool that was executed.
        success: Whether the tool execution was successful.
        output: The output data from the tool.
        error: Error message if execution failed.
        duration_ms: Execution time in milliseconds.
        timestamp: When the result was created.
        metadata: Additional metadata.
    """
    id: str = Field(default_factory=_generate_id, description="Unique result identifier")
    tool_call_id: str = Field(..., description="Reference to the originating tool call")
    tool_name: str = Field(..., description="Name of the executed tool")
    success: bool = Field(default=True, description="Whether execution succeeded")
    output: Any = Field(default=None, description="Output data from the tool")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    duration_ms: Optional[float] = Field(default=None, description="Execution time in ms")
    timestamp: datetime = Field(default_factory=_utc_now, description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @model_validator(mode="after")
    def validate_success_error_consistency(self) -> "ToolResult":
        """Ensure that if success is False, an error message is present."""
        if not self.success and not self.error:
            self.error = "Unknown error occurred during tool execution"
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tool result to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Deserialize tool result from a dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"ToolResult({status} tool={self.tool_name!r}, call_id={self.tool_call_id[:8]}...)"


class AgentState(BaseModel):
    """
    Represents the complete state of an agent at a point in time.

    This is used for persistence, debugging, and state restoration.

    Attributes:
        agent_id: Unique identifier for the agent.
        agent_name: Human-readable name of the agent.
        status: Current operational status.
        current_iteration: Number of iterations completed.
        max_iterations: Maximum allowed iterations.
        memory: Snapshot of the agent's memory.
        active_tool_calls: Currently pending tool calls.
        last_message: The most recent message processed.
        error: Current error state if in error status.
        metadata: Additional agent-specific metadata.
        created_at: When the agent state was created.
        updated_at: When the agent state was last updated.
    """
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_name: str = Field(..., description="Human-readable agent name")
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Current status")
    current_iteration: int = Field(default=0, ge=0, description="Iterations completed")
    max_iterations: int = Field(default=100, ge=1, description="Maximum allowed iterations")
    memory: Dict[str, Any] = Field(default_factory=dict, description="Agent memory snapshot")
    active_tool_calls: List[ToolCall] = Field(default_factory=list, description="Pending tool calls")
    last_message: Optional[Message] = Field(default=None, description="Most recent message")
    error: Optional[str] = Field(default=None, description="Current error message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=_utc_now, description="State creation time")
    updated_at: datetime = Field(default_factory=_utc_now, description="Last update time")

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id_not_empty(cls, v: str) -> str:
        """Ensure agent ID is not empty."""
        if not v or not v.strip():
            raise ValueError("Agent ID cannot be empty")
        return v.strip()

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name_not_empty(cls, v: str) -> str:
        """Ensure agent name is not empty."""
        if not v or not v.strip():
            raise ValueError("Agent name cannot be empty")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Deserialize agent state from a dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"AgentState(id={self.agent_id[:8]}..., name={self.agent_name!r}, "
            f"status={self.status.value}, iter={self.current_iteration}/{self.max_iterations})"
        )


class WorkflowContext(BaseModel):
    """
    Represents the context of a workflow execution.

    This encapsulates all information needed to execute and track
    a multi-agent workflow.

    Attributes:
        workflow_id: Unique identifier for the workflow.
        name: Human-readable workflow name.
        description: Description of the workflow's purpose.
        status: Current workflow status.
        agent_states: States of all agents in the workflow.
        message_history: Complete message history for the workflow.
        tool_results: All tool results produced during execution.
        input_data: Initial input data for the workflow.
        output_data: Final output data when completed.
        metadata: Additional workflow metadata.
        created_at: When the workflow was created.
        updated_at: When the workflow was last updated.
        completed_at: When the workflow completed (if applicable).
    """
    workflow_id: str = Field(default_factory=_generate_id, description="Unique workflow identifier")
    name: str = Field(..., description="Workflow name")
    description: str = Field(default="", description="Workflow description")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING, description="Current status")
    agent_states: Dict[str, AgentState] = Field(
        default_factory=dict,
        description="Map of agent_id to AgentState"
    )
    message_history: List[Message] = Field(default_factory=list, description="Complete message history")
    tool_results: List[ToolResult] = Field(default_factory=list, description="All tool results")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Initial input data")
    output_data: Optional[Dict[str, Any]] = Field(default=None, description="Final output data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=_utc_now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=_utc_now, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Ensure workflow name is not empty."""
        if not v or not v.strip():
            raise ValueError("Workflow name cannot be empty")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize workflow context to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowContext":
        """Deserialize workflow context from a dictionary."""
        return cls(**data)

    def add_agent_state(self, state: AgentState) -> None:
        """Add or update an agent state in the workflow."""
        self.agent_states[state.agent_id] = state
        self.updated_at = _utc_now()

    def add_message(self, message: Message) -> None:
        """Add a message to the workflow history."""
        self.message_history.append(message)
        self.updated_at = _utc_now()

    def add_tool_result(self, result: ToolResult) -> None:
        """Add a tool result to the workflow record."""
        self.tool_results.append(result)
        self.updated_at = _utc_now()

    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Retrieve an agent's state by ID."""
        return self.agent_states.get(agent_id)

    def __repr__(self) -> str:
        return (
            f"WorkflowContext(id={self.workflow_id[:8]}..., name={self.name!r}, "
            f"status={self.status.value}, agents={len(self.agent_states)})"
        )


class LLMRequest(BaseModel):
    """
    Represents a request to an LLM provider.

    Attributes:
        model: The model identifier to use.
        messages: The conversation messages to send.
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum number of tokens to generate.
        top_p: Nucleus sampling parameter.
        stop: List of stop sequences.
        tools: List of tool schemas available to the LLM.
        tool_choice: Strategy for tool selection.
        metadata: Additional provider-specific parameters.
    """
    model: str = Field(..., description="Model identifier")
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Max tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Available tool schemas")
    tool_choice: Optional[str] = Field(default=None, description="Tool selection strategy")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific params")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize LLM request to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMRequest":
        """Deserialize LLM request from a dictionary."""
        return cls(**data)


class LLMResponse(BaseModel):
    """
    Represents a response from an LLM provider.

    Attributes:
        id: Unique identifier for the response.
        model: The model that generated the response.
        content: The text content of the response.
        tool_calls: Any tool calls requested by the LLM.
        usage: Token usage statistics.
        finish_reason: Why the generation stopped.
        metadata: Additional provider-specific data.
        timestamp: When the response was received.
    """
    id: str = Field(default_factory=_generate_id, description="Unique response identifier")
    model: str = Field(..., description="Model that generated the response")
    content: str = Field(default="", description="Text content of the response")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Requested tool calls")
    usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        description="Token usage statistics"
    )
    finish_reason: Optional[str] = Field(default=None, description="Why generation stopped")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific data")
    timestamp: datetime = Field(default_factory=_utc_now, description="Response timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize LLM response to a plain dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMResponse":
        """Deserialize LLM response from a dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        return (
            f"LLMResponse(model={self.model!r}, content={self.content[:50]!r}..., "
            f"tool_calls={len(self.tool_calls)}, tokens={self.usage.get('total_tokens', 0)})"
        )


# Re-export for convenience
__all__ = [
    "Message",
    "ToolCall",
    "ToolResult",
    "AgentState",
    "WorkflowContext",
    "LLMRequest",
    "LLMResponse",
]
