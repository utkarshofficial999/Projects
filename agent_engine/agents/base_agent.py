"""
agent_engine.agents.base_agent
==============================

Defines the abstract base class for all agents in the multi-agent system.

Each agent encapsulates:
- A unique identifier and role description
- A reference to the shared MessageBus for communication
- Internal state management (idle, working, waiting, error, terminated)
- A lifecycle: start → run → stop
- A handler for incoming messages
- Handoff capability to transfer control to another agent
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from agent_engine.agents.message_bus import Message, MessageBus, MessageType

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Represents the current operational status of an agent."""

    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class AgentState:
    """
    Immutable snapshot of an agent's internal state.

    Attributes:
        agent_id: Unique identifier for this agent instance.
        role: Human-readable role description (e.g., "Researcher").
        status: Current operational status.
        current_task: Description of the task currently being processed, if any.
        metadata: Arbitrary key-value data associated with the agent.
        created_at: Timestamp when the agent was instantiated.
        last_heartbeat: Timestamp of the last activity/heartbeat.
        error: Last error message, if the agent is in ERROR state.
    """

    agent_id: str
    role: str
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent workflow engine.

    Subclasses must implement:
        - ``handle_message(msg)``: Process an incoming message.
        - ``get_capabilities()``: Return a list of capability tags.

    The base class provides:
        - Lifecycle management (start, stop, run loop)
        - Message subscription and dispatch via the MessageBus
        - Handoff mechanism to transfer control to another agent
        - Thread-safe state access
        - Heartbeat tracking for liveness monitoring

    Usage:
        >>> class MyAgent(BaseAgent):
        ...     def handle_message(self, msg: Message) -> None:
        ...         print(f"Received: {msg.payload}")
        ...     def get_capabilities(self) -> list[str]:
        ...         return ["example"]
        ...
        >>> bus = MessageBus()
        >>> agent = MyAgent(role="Example", bus=bus)
        >>> await agent.start()
        >>> await agent.stop()
    """

    def __init__(
        self,
        role: str,
        bus: MessageBus,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            role: Human-readable role name (e.g., "Researcher", "Coder").
            bus: The shared MessageBus instance for inter-agent communication.
            agent_id: Optional unique identifier. Auto-generated if not provided.
            metadata: Optional initial metadata dictionary.
        """
        self._agent_id: str = agent_id or f"{role.lower()}-{uuid.uuid4().hex[:8]}"
        self._role: str = role
        self._bus: MessageBus = bus
        self._state: AgentState = AgentState(
            agent_id=self._agent_id,
            role=role,
            metadata=metadata or {},
        )
        self._running: bool = False
        self._stop_event: asyncio.Event = asyncio.Event()
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._lock: asyncio.Lock = asyncio.Lock()
        self._handoff_target: Optional[str] = None
        self._task: Optional[asyncio.Task] = None

        # Subscribe to messages addressed to this agent
        self._bus.subscribe(self._agent_id, self._on_message)

        logger.debug(
            "Agent initialized: id=%s, role=%s", self._agent_id, self._role
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent."""
        return self._agent_id

    @property
    def role(self) -> str:
        """Human-readable role name."""
        return self._role

    @property
    def state(self) -> AgentState:
        """
        Return a snapshot of the current agent state.

        Returns:
            A copy of the AgentState dataclass.
        """
        return AgentState(
            agent_id=self._state.agent_id,
            role=self._state.role,
            status=self._state.status,
            current_task=self._state.current_task,
            metadata=dict(self._state.metadata),
            created_at=self._state.created_at,
            last_heartbeat=self._state.last_heartbeat,
            error=self._state.error,
        )

    @property
    def is_running(self) -> bool:
        """Whether the agent's main loop is active."""
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the agent's main event loop.

        The agent begins listening for messages and processing them
        in the background. This method is non-blocking; it spawns
        an asyncio task.

        Raises:
            RuntimeError: If the agent is already running.
        """
        if self._running:
            raise RuntimeError(f"Agent {self._agent_id} is already running.")

        self._running = True
        self._stop_event.clear()
        self._set_status(AgentStatus.IDLE)
        self._task = asyncio.create_task(
            self._run_loop(), name=f"agent-{self._agent_id}"
        )
        logger.info("Agent started: id=%s, role=%s", self._agent_id, self._role)

    async def stop(self, timeout: float = 5.0) -> None:
        """
        Gracefully stop the agent.

        Signals the run loop to exit and waits for it to complete.

        Args:
            timeout: Maximum seconds to wait for the agent to finish.

        Raises:
            asyncio.TimeoutError: If the agent does not stop within timeout.
        """
        if not self._running:
            return

        logger.info("Stopping agent: id=%s, role=%s", self._agent_id, self._role)
        self._stop_event.set()

        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "Agent %s did not stop within %.1fs; cancelling.",
                    self._agent_id,
                    timeout,
                )
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        self._running = False
        self._set_status(AgentStatus.TERMINATED)
        self._bus.unsubscribe(self._agent_id)
        logger.info("Agent stopped: id=%s, role=%s", self._agent_id, self._role)

    async def _run_loop(self) -> None:
        """
        Internal main loop: wait for messages and dispatch them.

        This loop runs until ``_stop_event`` is set.
        """
        try:
            while not self._stop_event.is_set():
                try:
                    # Wait for a message or the stop signal
                    msg = await asyncio.wait_for(
                        self._message_queue.get(), timeout=0.1
                    )
                    await self._process_message(msg)
                except asyncio.TimeoutError:
                    # No message; check stop event and continue
                    if self._stop_event.is_set():
                        break
                    continue
        except asyncio.CancelledError:
            logger.debug("Agent %s run loop cancelled.", self._agent_id)
        finally:
            self._running = False
            self._set_status(AgentStatus.TERMINATED)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _on_message(self, msg: Message) -> None:
        """
        Callback invoked by the MessageBus when a message is addressed to this agent.

        Args:
            msg: The incoming message.
        """
        await self._message_queue.put(msg)

    async def _process_message(self, msg: Message) -> None:
        """
        Process a single incoming message.

        Updates the agent state, calls the subclass handler, and
        handles handoff requests.

        Args:
            msg: The message to process.
        """
        self._update_heartbeat()

        try:
            if msg.type == MessageType.HANDOFF_REQUEST:
                await self._handle_handoff(msg)
            elif msg.type == MessageType.TASK_ASSIGNMENT:
                self._set_status(AgentStatus.WORKING)
                self._state.current_task = msg.payload.get("task", "unknown")
                await self.handle_message(msg)
                self._set_status(AgentStatus.IDLE)
                self._state.current_task = None
            elif msg.type == MessageType.BROADCAST:
                await self.handle_message(msg)
            elif msg.type == MessageType.ACK:
                # Acknowledgements are logged but not dispatched to handler
                logger.debug(
                    "Agent %s received ACK from %s", self._agent_id, msg.sender_id
                )
            else:
                await self.handle_message(msg)

        except Exception as exc:
            logger.exception(
                "Agent %s error while processing message %s: %s",
                self._agent_id,
                msg.message_id,
                exc,
            )
            self._set_status(AgentStatus.ERROR)
            self._state.error = str(exc)

    @abstractmethod
    async def handle_message(self, msg: Message) -> None:
        """
        Handle an incoming message. Subclasses must implement this.

        Args:
            msg: The message to process.
        """
        ...

    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """
        Return a list of capability tags this agent supports.

        Returns:
            List of capability strings (e.g., ["search", "summarize"]).
        """
        ...

    # ------------------------------------------------------------------
    # Handoff
    # ------------------------------------------------------------------

    async def request_handoff(self, target_agent_id: str, reason: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Request a handoff of control to another agent.

        Sends a HANDOFF_REQUEST message to the target agent via the bus.

        Args:
            target_agent_id: The agent_id of the target agent.
            reason: Human-readable reason for the handoff.
            context: Optional context data to pass to the target agent.

        Returns:
            True if the handoff request was successfully sent, False otherwise.
        """
        payload: Dict[str, Any] = {
            "reason": reason,
            "context": context or {},
            "source_agent_id": self._agent_id,
        }
        msg = Message(
            message_id=str(uuid.uuid4()),
            sender_id=self._agent_id,
            recipient_id=target_agent_id,
            type=MessageType.HANDOFF_REQUEST,
            payload=payload,
        )
        try:
            await self._bus.publish(msg)
            self._handoff_target = target_agent_id
            logger.info(
                "Agent %s requested handoff to %s: %s",
                self._agent_id,
                target_agent_id,
                reason,
            )
            return True
        except Exception as exc:
            logger.error(
                "Agent %s failed to send handoff to %s: %s",
                self._agent_id,
                target_agent_id,
                exc,
            )
            return False

    async def _handle_handoff(self, msg: Message) -> None:
        """
        Internal handler for incoming handoff requests.

        Transitions the agent to WORKING status and processes the handoff
        context. Subclasses can override for custom behavior.

        Args:
            msg: The HANDOFF_REQUEST message.
        """
        self._set_status(AgentStatus.WORKING)
        context = msg.payload.get("context", {})
        reason = msg.payload.get("reason", "unspecified")
        logger.info(
            "Agent %s received handoff from %s: %s",
            self._agent_id,
            msg.sender_id,
            reason,
        )
        # Default: treat as a task assignment with the handoff context
        await self.handle_message(
            Message(
                message_id=str(uuid.uuid4()),
                sender_id=msg.sender_id,
                recipient_id=self._agent_id,
                type=MessageType.TASK_ASSIGNMENT,
                payload={"task": f"handoff:{reason}", "context": context},
            )
        )
        self._set_status(AgentStatus.IDLE)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_status(self, status: AgentStatus) -> None:
        """
        Update the agent's status.

        Args:
            status: The new AgentStatus.
        """
        self._state.status = status
        self._update_heartbeat()

    def _update_heartbeat(self) -> None:
        """Update the last heartbeat timestamp."""
        self._state.last_heartbeat = time.time()

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set a metadata key on the agent.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self._state.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata value from the agent.

        Args:
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            The metadata value or default.
        """
        return self._state.metadata.get(key, default)
