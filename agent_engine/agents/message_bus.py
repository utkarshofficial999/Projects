"""
agent_engine.agents.message_bus
================================

Implements a thread-safe, asynchronous message bus for inter-agent communication.

The MessageBus supports:
- Point-to-point messaging (agent-to-agent)
- Broadcast messaging (agent-to-all)
- Topic-based pub/sub (optional, for future extensibility)
- Message acknowledgment
- Delivery guarantees (at-least-once via retry)
- Thread-safe state sharing via asyncio primitives

Design:
    - Uses asyncio.Queue per subscriber for ordered delivery
    - Uses asyncio.Lock for shared state mutations
    - Supports both direct callback registration and queue-based consumption
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """
    Enumerates the types of messages that can be sent on the bus.

    Attributes:
        TASK_ASSIGNMENT: A task is being assigned to an agent.
        HANDOFF_REQUEST: An agent is requesting control transfer.
        BROADCAST: A message sent to all subscribed agents.
        ACK: Acknowledgment of message receipt.
        NACK: Negative acknowledgment (message rejected).
        STATUS_UPDATE: An agent is reporting its status.
        SHUTDOWN: Request to shut down an agent.
        CUSTOM: Application-specific message type.
    """

    TASK_ASSIGNMENT = "task_assignment"
    HANDOFF_REQUEST = "handoff_request"
    BROADCAST = "broadcast"
    ACK = "ack"
    NACK = "nack"
    STATUS_UPDATE = "status_update"
    SHUTDOWN = "shutdown"
    CUSTOM = "custom"


@dataclass
class Message:
    """
    Represents a single message on the bus.

    Attributes:
        message_id: Unique identifier for this message.
        sender_id: The agent_id of the sending agent.
        recipient_id: The agent_id of the intended recipient (or "broadcast").
        type: The MessageType of this message.
        payload: Arbitrary data associated with the message.
        timestamp: Unix timestamp when the message was created.
        metadata: Optional additional metadata (e.g., priority, TTL).
    """

    message_id: str
    sender_id: str
    recipient_id: str
    type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate message fields after initialization."""
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.sender_id:
            raise ValueError("sender_id must not be empty.")
        if not self.recipient_id:
            raise ValueError("recipient_id must not be empty.")


@dataclass
class Subscription:
    """
    Represents a subscription of an agent to the message bus.

    Attributes:
        agent_id: The subscribing agent's ID.
        callback: Async callable invoked when a message is delivered.
        message_types: Optional filter; if empty, receives all messages.
    """

    agent_id: str
    callback: Callable[[Message], Coroutine[Any, Any, None]]
    message_types: Set[MessageType] = field(default_factory=set)


class MessageBus:
    """
    Asynchronous, thread-safe message bus for inter-agent communication.

    The bus maintains a registry of subscribers and routes messages
    to the appropriate recipients. It supports:

    - Point-to-point delivery (recipient_id matches a subscriber)
    - Broadcast delivery (recipient_id == "broadcast")
    - Message acknowledgment
    - Delivery retry for at-least-once semantics
    - Graceful shutdown

    Thread Safety:
        All public methods are coroutine-safe. Internal state is protected
        by asyncio.Lock. The bus is designed to be shared across multiple
        agents running in the same event loop.

    Usage:
        >>> bus = MessageBus()
        >>> async def handler(msg: Message):
        ...     print(f"Got {msg.type} from {msg.sender_id}")
        >>> bus.subscribe("agent-1", handler)
        >>> msg = Message(
        ...     message_id="m1",
        ...     sender_id="supervisor",
        ...     recipient_id="agent-1",
        ...     type=MessageType.TASK_ASSIGNMENT,
        ...     payload={"task": "do something"},
        ... )
        >>> await bus.publish(msg)
    """

    BROADCAST_ID: str = "broadcast"

    def __init__(self, max_queue_size: int = 1000, retry_limit: int = 3) -> None:
        """
        Initialize the MessageBus.

        Args:
            max_queue_size: Maximum number of messages per subscriber queue.
            retry_limit: Number of delivery retries for failed callbacks.
        """
        self._subscribers: Dict[str, Subscription] = {}
        self._queues: Dict[str, asyncio.Queue[Message]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._max_queue_size: int = max_queue_size
        self._retry_limit: int = retry_limit
        self._message_log: List[Message] = []
        self._max_log_size: int = 10000
        self._running: bool = False
        self._dispatch_task: Optional[asyncio.Task] = None

        logger.debug(
            "MessageBus initialized: max_queue_size=%d, retry_limit=%d",
            max_queue_size,
            retry_limit,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the bus's internal dispatch loop.

        The dispatch loop continuously checks subscriber queues and
        delivers messages to their callbacks.
        """
        if self._running:
            return
        self._running = True
        self._dispatch_task = asyncio.create_task(
            self._dispatch_loop(), name="message-bus-dispatch"
        )
        logger.info("MessageBus started.")

    async def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the bus and drain all pending messages.

        Args:
            timeout: Maximum seconds to wait for drain completion.
        """
        if not self._running:
            return

        logger.info("MessageBus stopping; draining queues...")
        self._running = False

        # Drain all queues
        for agent_id, queue in self._queues.items():
            while not queue.empty():
                try:
                    msg = queue.get_nowait()
                    await self._deliver(agent_id, msg)
                except asyncio.QueueEmpty:
                    break

        if self._dispatch_task is not None:
            try:
                await asyncio.wait_for(self._dispatch_task, timeout=timeout)
            except asyncio.TimeoutError:
                self._dispatch_task.cancel()
                try:
                    await self._dispatch_task
                except asyncio.CancelledError:
                    pass

        self._subscribers.clear()
        self._queues.clear()
        logger.info("MessageBus stopped.")

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(
        self,
        agent_id: str,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
        message_types: Optional[Set[MessageType]] = None,
    ) -> None:
        """
        Register an agent as a subscriber on the bus.

        Args:
            agent_id: Unique identifier for the subscribing agent.
            callback: Async callable to invoke on message delivery.
            message_types: Optional set of MessageType to filter. If None,
                the agent receives all messages.

        Raises:
            ValueError: If agent_id is empty.
        """
        if not agent_id:
            raise ValueError("agent_id must not be empty.")

        self._subscribers[agent_id] = Subscription(
            agent_id=agent_id,
            callback=callback,
            message_types=message_types or set(),
        )
        self._queues[agent_id] = asyncio.Queue(maxsize=self._max_queue_size)
        logger.debug("Agent %s subscribed to MessageBus.", agent_id)

    def unsubscribe(self, agent_id: str) -> None:
        """
        Remove an agent's subscription from the bus.

        Args:
            agent_id: The agent_id to unsubscribe.
        """
        self._subscribers.pop(agent_id, None)
        self._queues.pop(agent_id, None)
        logger.debug("Agent %s unsubscribed from MessageBus.", agent_id)

    def is_subscribed(self, agent_id: str) -> bool:
        """
        Check if an agent is currently subscribed.

        Args:
            agent_id: The agent_id to check.

        Returns:
            True if subscribed, False otherwise.
        """
        return agent_id in self._subscribers

    @property
    def subscriber_count(self) -> int:
        """Number of currently subscribed agents."""
        return len(self._subscribers)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, message: Message) -> bool:
        """
        Publish a message to the bus.

        Routes the message to the appropriate subscriber queue(s) based
        on the recipient_id. If recipient_id is "broadcast", the message
        is enqueued for all subscribers.

        Args:
            message: The Message to publish.

        Returns:
            True if the message was successfully enqueued, False otherwise.
        """
        async with self._lock:
            self._message_log.append(message)
            # Trim log
            if len(self._message_log) > self._max_log_size:
                self._message_log = self._message_log[-self._max_log_size:]

        if message.recipient_id == self.BROADCAST_ID:
            return await self._broadcast(message)
        else:
            return await self._deliver_to_agent(message)

    async def _deliver_to_agent(self, message: Message) -> bool:
        """
        Deliver a message to a specific agent's queue.

        Args:
            message: The message to deliver.

        Returns:
            True if delivered, False if the recipient is not subscribed.
        """
        queue = self._queues.get(message.recipient_id)
        if queue is None:
            logger.warning(
                "Message %s: recipient %s is not subscribed.",
                message.message_id,
                message.recipient_id,
            )
            return False

        try:
            queue.put_nowait(message)
            logger.debug(
                "Message %s enqueued for %s.", message.message_id, message.recipient_id
            )
            return True
        except asyncio.QueueFull:
            logger.error(
                "Message %s: queue for %s is full; message dropped.",
                message.message_id,
                message.recipient_id,
            )
            return False

    async def _broadcast(self, message: Message) -> bool:
        """
        Broadcast a message to all subscribed agents.

        Args:
            message: The message to broadcast.

        Returns:
            True if broadcast to at least one agent, False if no subscribers.
        """
        delivered = 0
        for agent_id, queue in self._queues.items():
            try:
                queue.put_nowait(message)
                delivered += 1
            except asyncio.QueueFull:
                logger.warning(
                    "Broadcast message %s: queue for %s is full; skipped.",
                    message.message_id,
                    agent_id,
                )

        if delivered > 0:
            logger.debug(
                "Broadcast message %s delivered to %d agents.",
                message.message_id,
                delivered,
            )
        return delivered > 0

    # ------------------------------------------------------------------
    # Dispatch loop
    # ------------------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        """
        Internal loop that continuously delivers messages from queues
        to subscriber callbacks.
        """
        while self._running:
            try:
                # Collect all queues with pending messages
                pending: List[tuple[str, Message]] = []
                for agent_id, queue in self._queues.items():
                    while not queue.empty():
                        try:
                            msg = queue.get_nowait()
                            pending.append((agent_id, msg))
                        except asyncio.QueueEmpty:
                            break

                if pending:
                    for agent_id, msg in pending:
                        await self._deliver(agent_id, msg)
                else:
                    # No pending messages; sleep briefly to avoid busy-wait
                    await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Error in dispatch loop: %s", exc)
                await asyncio.sleep(0.1)

    async def _deliver(self, agent_id: str, message: Message) -> None:
        """
        Deliver a message to an agent's callback with retry logic.

        Args:
            agent_id: The target agent's ID.
            message: The message to deliver.
        """
        subscription = self._subscribers.get(agent_id)
        if subscription is None:
            logger.warning(
                "Delivery failed: agent %s is no longer subscribed.", agent_id
            )
            return

        # Check message type filter
        if subscription.message_types and message.type not in subscription.message_types:
            logger.debug(
                "Message %s filtered out for agent %s (type=%s).",
                message.message_id,
                agent_id,
                message.type.value,
            )
            return

        for attempt in range(1, self._retry_limit + 1):
            try:
                await subscription.callback(message)
                logger.debug(
                    "Message %s delivered to %s (attempt %d).",
                    message.message_id,
                    agent_id,
                    attempt,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Delivery attempt %d/%d failed for message %s to %s: %s",
                    attempt,
                    self._retry_limit,
                    message.message_id,
                    agent_id,
                    exc,
                )
                if attempt < self._retry_limit:
                    await asyncio.sleep(0.1 * attempt)

        logger.error(
            "Message %s delivery to %s failed after %d attempts.",
            message.message_id,
            agent_id,
            self._retry_limit,
        )

    # ------------------------------------------------------------------
    # Acknowledgment
    # ------------------------------------------------------------------

    async def acknowledge(self, message_id: str, agent_id: str) -> None:
        """
        Send an ACK message for a previously received message.

        Args:
            message_id: The ID of the message being acknowledged.
            agent_id: The agent sending the acknowledgment.
        """
        ack_msg = Message(
            message_id=str(uuid.uuid4()),
            sender_id=agent_id,
            recipient_id="supervisor",  # ACKs typically go to the originator
            type=MessageType.ACK,
            payload={"ack_for": message_id},
        )
        await self.publish(ack_msg)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def message_log(self) -> List[Message]:
        """
        Return a copy of the recent message log.

        Returns:
            List of recent Message objects (up to max_log_size).
        """
        return list(self._message_log)

    def get_queue_size(self, agent_id: str) -> int:
        """
        Get the current size of an agent's message queue.

        Args:
            agent_id: The agent_id to check.

        Returns:
            Number of pending messages, or -1 if not subscribed.
        """
        queue = self._queues.get(agent_id)
        if queue is None:
            return -1
        return queue.qsize()
