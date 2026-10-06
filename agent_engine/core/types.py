"""
Core type definitions and enums for the agent engine.

This module defines the foundational types used throughout the system,
including role identifiers, message types, and status enumerations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union


class Role(str, Enum):
    """
    Represents the role of an entity in a conversation or workflow.

    Attributes:
        SYSTEM: The system prompt or system-level instructions.
        USER: Human user input.
        ASSISTANT: AI agent response.
        TOOL: Result returned from a tool execution.
        AGENT: Message from another agent in a multi-agent setup.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    AGENT = "agent"


class MessageType(str, Enum):
    """
    Categorizes the type of message in the communication protocol.

    Attributes:
        TEXT: Plain text message.
        TOOL_CALL: Request to invoke a tool.
        TOOL_RESULT: Result of a tool invocation.
        STATUS: Status update or heartbeat.
        ERROR: Error notification.
        CONTROL: Control flow signal (e.g., stop, pause).
    """
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATUS = "status"
    ERROR = "error"
    CONTROL = "control"


class AgentStatus(str, Enum):
    """
    Represents the current operational status of an agent.

    Attributes:
        IDLE: Agent is waiting for input.
        THINKING: Agent is processing or reasoning.
        EXECUTING: Agent is executing a tool or action.
        WAITING: Agent is waiting for external dependency.
        ERROR: Agent encountered an error.
        COMPLETED: Agent has finished its task.
        TERMINATED: Agent was explicitly stopped.
    """
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"
    TERMINATED = "terminated"


class WorkflowStatus(str, Enum):
    """
    Represents the overall status of a workflow execution.

    Attributes:
        PENDING: Workflow is queued but not started.
        RUNNING: Workflow is actively executing.
        PAUSED: Workflow is temporarily suspended.
        COMPLETED: Workflow finished successfully.
        FAILED: Workflow terminated due to error.
        CANCELLED: Workflow was explicitly cancelled.
    """
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MemoryScope(str, Enum):
    """
    Defines the scope of memory storage.

    Attributes:
        SESSION: Memory scoped to a single conversation/session.
        AGENT: Memory scoped to a specific agent instance.
        GLOBAL: Memory shared across all agents and sessions.
        WORKFLOW: Memory scoped to a specific workflow execution.
    """
    SESSION = "session"
    AGENT = "agent"
    GLOBAL = "global"
    WORKFLOW = "workflow"


# Type aliases for common patterns
JSONDict = Dict[str, Any]
JSONList = List[Any]
MessageDict = Dict[str, Any]
ToolSchema = Dict[str, Any]
