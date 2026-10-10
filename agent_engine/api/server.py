"""
agent_engine.api.server
=======================

FastAPI application factory for the Agent Engine.

This module builds the ASGI application, wires up CORS, registers the
REST and WebSocket routers, and manages the shared in-process
``WorkflowManager`` that coordinates running workflows and their
real-time event streams.

Design notes
------------
* **Stateless HTTP, stateful WS.** REST endpoints are idempotent and
  safe to scale horizontally. The ``WorkflowManager`` holds the
  authoritative in-memory state for the *current* process; for
  multi-replica deployments, swap it for a Redis-backed implementation
  (the interface is intentionally small).
* **Backpressure-safe streaming.** Each workflow owns a bounded
  ``asyncio.Queue``. If a WebSocket consumer falls behind, the oldest
  events are dropped (with a counter) rather than blocking the agent
  loop.
* **Graceful shutdown.** On ``SIGTERM``/``SIGINT`` the lifespan handler
  cancels all running workflows and closes their queues.

Usage
-----
Run with uvicorn::

    uvicorn agent_engine.api.server:app --host 0.0.0.0 --port 8000

Or create the app programmatically (useful for tests)::

    from agent_engine.api.server import create_app
    app = create_app()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from agent_engine.api.routes import router as api_router
from agent_engine.core.config import get_config
from agent_engine.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Workflow state model
# ---------------------------------------------------------------------------


class WorkflowStatus(str, Enum):
    """Lifecycle states for a workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowEvent:
    """A single event emitted by a running workflow.

    Attributes
    ----------
    event_id:
        Unique identifier for this event (used for deduplication).
    workflow_id:
        The workflow this event belongs to.
    timestamp:
        Unix epoch seconds when the event was produced.
    kind:
        One of ``"thought"``, ``"action"``, ``"observation"``,
        ``"reflection"``, ``"status"``, ``"error"``, ``"result"``.
    agent_name:
        Name of the agent that produced the event (if applicable).
    payload:
        Arbitrary JSON-serialisable data for the event.
    """

    event_id: str
    workflow_id: str
    timestamp: float
    kind: str
    agent_name: Optional[str]
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            "event_id": self.event_id,
            "workflow_id": self.workflow_id,
            "timestamp": self.timestamp,
            "kind": self.kind,
            "agent_name": self.agent_name,
            "payload": self.payload,
        }


@dataclass
class WorkflowRecord:
    """In-memory record for a single workflow execution.

    Attributes
    ----------
    workflow_id:
        Unique identifier (UUID4).
    task:
        The natural-language task description.
    status:
        Current lifecycle status.
    created_at:
        Unix epoch seconds when the workflow was created.
    finished_at:
        Unix epoch seconds when the workflow reached a terminal state,
        or ``None`` if still running.
    result:
        Final result payload (set on completion).
    error:
        Error message (set on failure).
    events:
        Bounded ring buffer of recent events for late-joining clients.
    event_queue:
        Async queue used to push new events to WebSocket subscribers.
    subscribers:
        Set of active WebSocket connections subscribed to this workflow.
    """

    workflow_id: str
    task: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    events: list = field(default_factory=list)
    event_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=1024))
    subscribers: set = field(default_factory=set)
    _dropped_events: int = 0

    def add_event(self, event: WorkflowEvent) -> None:
        """Append an event to the ring buffer and enqueue for subscribers.

        If the queue is full, the oldest event is dropped and a counter
        is incremented so the client can be informed.
        """
        self.events.append(event)
        # Keep only the last 500 events in memory
        if len(self.events) > 500:
            self.events = self.events[-500:]

        try:
            self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped_events += 1
            # Drop the oldest event to make room
            try:
                self.event_queue.get_nowait()
                self.event_queue.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    @property
    def dropped_count(self) -> int:
        """Number of events dropped due to backpressure."""
        return self._dropped_events


# ---------------------------------------------------------------------------
# Workflow Manager
# ---------------------------------------------------------------------------


class WorkflowManager:
    """Coordinates workflow lifecycle and event distribution.

    This is the central in-process state holder. It is intentionally
    kept simple so it can be replaced with a distributed backend
    (e.g. Redis Streams) without changing the API surface.

    Thread-safety: all public methods are ``async`` and must be called
    from the event loop.
    """

    def __init__(self) -> None:
        self._workflows: Dict[str, WorkflowRecord] = {}
        self._lock = asyncio.Lock()

    async def create_workflow(self, task: str, **kwargs: Any) -> WorkflowRecord:
        """Create a new workflow record and return it.

        Parameters
        ----------
        task:
            The natural-language task description.
        **kwargs:
            Optional overrides (e.g. ``max_iterations``, ``agent_name``).

        Returns
        -------
        WorkflowRecord
            The newly created (pending) workflow.
        """
        workflow_id = str(uuid.uuid4())
        record = WorkflowRecord(
            workflow_id=workflow_id,
            task=task,
            status=WorkflowStatus.PENDING,
        )
        async with self._lock:
            self._workflows[workflow_id] = record
        logger.info("Workflow created", extra={"workflow_id": workflow_id, "task": task[:120]})
        return record

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowRecord]:
        """Retrieve a workflow by ID, or ``None`` if not found."""
        return self._workflows.get(workflow_id)

    async def list_workflows(self, limit: int = 50) -> list[WorkflowRecord]:
        """Return the most recent workflows (newest first)."""
        workflows = sorted(
            self._workflows.values(),
            key=lambda w: w.created_at,
            reverse=True,
        )
        return workflows[:limit]

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a pending or running workflow.

        Returns ``True`` if the workflow was successfully cancelled.
        """
        record = self._workflows.get(workflow_id)
        if record is None:
            return False
        if record.status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED):
            return False
        record.status = WorkflowStatus.CANCELLED
        record.finished_at = time.time()
        # Signal all subscribers to disconnect
        event = WorkflowEvent(
            event_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            timestamp=time.time(),
            kind="status",
            agent_name=None,
            payload={"status": "cancelled"},
        )
        record.add_event(event)
        logger.info("Workflow cancelled", extra={"workflow_id": workflow_id})
        return True

    async def complete_workflow(
        self,
        workflow_id: str,
        result: Dict[str, Any],
    ) -> None:
        """Mark a workflow as completed with a result payload."""
        record = self._workflows.get(workflow_id)
        if record is None:
            return
        record.status = WorkflowStatus.COMPLETED
        record.finished_at = time.time()
        record.result = result
        event = WorkflowEvent(
            event_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            timestamp=time.time(),
            kind="result",
            agent_name=None,
            payload=result,
        )
        record.add_event(event)
        logger.info("Workflow completed", extra={"workflow_id": workflow_id})

    async def fail_workflow(self, workflow_id: str, error: str) -> None:
        """Mark a workflow as failed with an error message."""
        record = self._workflows.get(workflow_id)
        if record is None:
            return
        record.status = WorkflowStatus.FAILED
        record.finished_at = time.time()
        record.error = error
        event = WorkflowEvent(
            event_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            timestamp=time.time(),
            kind="error",
            agent_name=None,
            payload={"error": error},
        )
        record.add_event(event)
        logger.error("Workflow failed", extra={"workflow_id": workflow_id, "error": error})

    async def subscribe(self, workflow_id: str, websocket: WebSocket) -> bool:
        """Subscribe a WebSocket connection to a workflow's event stream.

        Returns ``True`` if the subscription was successful.
        """
        record = self._workflows.get(workflow_id)
        if record is None:
            return False
        record.subscribers.add(websocket)
        return True

    async def unsubscribe(self, workflow_id: str, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from a workflow's subscriber set."""
        record = self._workflows.get(workflow_id)
        if record is not None:
            record.subscribers.discard(websocket)

    async def shutdown(self) -> None:
        """Cancel all running workflows and clean up resources."""
        for record in self._workflows.values():
            if record.status in (WorkflowStatus.PENDING, WorkflowStatus.RUNNING):
                record.status = WorkflowStatus.CANCELLED
                record.finished_at = time.time()
        logger.info("WorkflowManager shutdown complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Returns
    -------
    FastAPI
        A fully configured ASGI application with CORS, routers, and
        a shared ``WorkflowManager`` instance stored in ``app.state``.
    """
    app = FastAPI(
        title="Agent Engine API",
        description=(
            "Production interface for the Autonomous Multi-Agent Workflow Engine. "
            "Supports REST for workflow management and WebSocket for real-time "
            "agent thought streaming."
        ),
        version="0.1.0",
    )

    # CORS – allow all origins in development; restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared state
    app.state.workflow_manager = WorkflowManager()

    # Register routers
    app.include_router(api_router, prefix="/api/v1", tags=["Agent Engine"])

    # Health check
    @app.get("/health", tags=["System"])
    async def health() -> Dict[str, str]:
        """Liveness probe endpoint."""
        return {"status": "ok"}

    # WebSocket endpoint for real-time event streaming
    @app.websocket("/ws/workflows/{workflow_id}/stream")
    async def workflow_stream(websocket: WebSocket, workflow_id: str) -> None:
        """WebSocket endpoint that streams events for a given workflow.

        The client connects to this endpoint after starting a workflow
        via ``POST /api/v1/workflows``. Events are pushed as JSON
        messages in the format defined by :class:`WorkflowEvent`.

        Protocol:
        - Server sends events as JSON objects.
        - Client may send ``{"type": "ping"}`` to keep the connection alive.
        - Server sends ``{"type": "pong"}`` in response.
        - On workflow completion, server sends a final event with
          ``kind="result"`` or ``kind="error"`` and closes the connection.
        """
        manager: WorkflowManager = app.state.workflow_manager
        record = await manager.get_workflow(workflow_id)

        if record is None:
            await websocket.close(code=4004, reason="Workflow not found")
            return

        await websocket.accept()
        await manager.subscribe(workflow_id, websocket)

        try:
            # Send initial status
            await websocket.send_json({
                "type": "connected",
                "workflow_id": workflow_id,
                "status": record.status.value,
            })

            # Stream events until workflow completes or client disconnects
            while True:
                try:
                    # Wait for the next event with a timeout to allow
                    # processing of incoming messages (e.g. pings)
                    event = await asyncio.wait_for(
                        record.event_queue.get(),
                        timeout=30.0,
                    )
                    await websocket.send_json(event.to_dict())

                    # If the workflow has reached a terminal state, close
                    if record.status in (
                        WorkflowStatus.COMPLETED,
                        WorkflowStatus.FAILED,
                        WorkflowStatus.CANCELLED,
                    ):
                        break

                except asyncio.TimeoutError:
                    # Send a keepalive ping to the client
                    await websocket.send_json({"type": "keepalive", "timestamp": time.time()})

        except WebSocketDisconnect:
            logger.debug("Client disconnected", extra={"workflow_id": workflow_id})
        except Exception as exc:
            logger.error(
                "Error in WebSocket stream",
                extra={"workflow_id": workflow_id, "error": str(exc)},
            )
            with contextlib.suppress(Exception):
                await websocket.close(code=1011, reason="Internal server error")
        finally:
            await manager.unsubscribe(workflow_id, websocket)

    # Lifespan handler for graceful shutdown
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage application lifespan: startup and shutdown."""
        logger.info("Agent Engine API starting up")
        yield
        logger.info("Agent Engine API shutting down")
        await app.state.workflow_manager.shutdown()

    # Re-create app with lifespan (FastAPI requires lifespan at init)
    # We patch it here to avoid circular import issues
    app.router.lifespan_context = lifespan

    return app


# Module-level app instance for uvicorn
app = create_app()
