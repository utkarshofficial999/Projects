"""
agent_engine.api.routes
=======================

REST API routes for the Agent Engine.

All endpoints are mounted under ``/api/v1`` and provide:

- ``POST   /workflows``          – Start a new workflow
- ``GET    /workflows``          – List recent workflows
- ``GET    /workflows/{id}``     – Get workflow status and result
- ``DELETE /workflows/{id}``     – Cancel a running workflow
- ``GET    /workflows/{id}/events`` – Retrieve buffered events
- ``GET    /agents``             – List available agents
- ``GET    /tools``              – List registered tools

All responses use JSON. Error responses follow the RFC 7807
``application/problem+json`` convention.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from agent_engine.core.config import get_config
from agent_engine.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class CreateWorkflowRequest(BaseModel):
    """Request body for starting a new workflow."""

    task: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Natural-language task description for the agent(s).",
        examples=["Summarize the key points of the attached document."],
    )
    max_iterations: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Override the maximum number of ReAct iterations.",
    )
    agent_name: Optional[str] = Field(
        default=None,
        description="Name of a specific agent to use (if multiple are registered).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata attached to the workflow.",
    )


class WorkflowResponse(BaseModel):
    """Response for a single workflow."""

    workflow_id: str
    task: str
    status: str
    created_at: float
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowListResponse(BaseModel):
    """Response for listing workflows."""

    workflows: List[WorkflowResponse]
    total: int


class EventResponse(BaseModel):
    """A single event from a workflow's event buffer."""

    event_id: str
    workflow_id: str
    timestamp: float
    kind: str
    agent_name: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class EventsResponse(BaseModel):
    """Response for retrieving buffered events."""

    workflow_id: str
    events: List[EventResponse]
    total: int
    dropped: int = 0


class AgentInfo(BaseModel):
    """Information about a registered agent."""

    name: str
    description: str = ""
    model: str = ""


class ToolInfo(BaseModel):
    """Information about a registered tool."""

    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _get_manager(request: Request):
    """Retrieve the WorkflowManager from app state."""
    return request.app.state.workflow_manager


# ---------------------------------------------------------------------------
# Workflow endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/workflows",
    response_model=WorkflowResponse,
    status_code=201,
    summary="Start a new workflow",
    description=(
        "Creates a new workflow execution. The workflow begins in "
        "`pending` status and transitions to `running` once the agent "
        "loop starts. Use the returned `workflow_id` to connect to the "
        "WebSocket stream at `/ws/workflows/{workflow_id}/stream`."
    ),
)
async def create_workflow(
    body: CreateWorkflowRequest,
    request: Request,
) -> WorkflowResponse:
    """Start a new agent workflow.

    This endpoint is non-blocking: it creates the workflow record and
    returns immediately. The actual agent execution is launched as a
    background task.
    """
    manager = _get_manager(request)
    record = await manager.create_workflow(
        task=body.task,
        max_iterations=body.max_iterations,
        agent_name=body.agent_name,
        metadata=body.metadata,
    )

    # Transition to running
    record.status = "running"

    # Launch the agent execution as a background task
    _launch_workflow(record, body)

    return WorkflowResponse(
        workflow_id=record.workflow_id,
        task=record.task,
        status=record.status,
        created_at=record.created_at,
        finished_at=record.finished_at,
        result=record.result,
        error=record.error,
        metadata=body.metadata,
    )


def _launch_workflow(record, body: CreateWorkflowRequest) -> None:
    """Launch the agent execution in a background asyncio task.

    This is intentionally decoupled from the HTTP request so the
    response is returned immediately.
    """
    import asyncio

    async def _run() -> None:
        try:
            from agent_engine.core.react_engine import ReactEngine
            from agent_engine.core.config import get_config

            cfg = get_config()
            engine = ReactEngine(config=cfg)

            # Simulate agent execution with event emission
            # In production, this would call engine.run(task)
            # and hook into the engine's event callbacks.

            # Emit a "thought" event
            thought_event = {
                "event_id": str(uuid.uuid4()),
                "workflow_id": record.workflow_id,
                "timestamp": time.time(),
                "kind": "thought",
                "agent_name": body.agent_name or "default",
                "payload": {
                    "thought": f"Analyzing task: {body.task[:200]}",
                    "iteration": 1,
                },
            }
            from agent_engine.api.server import WorkflowEvent
            record.add_event(WorkflowEvent(**thought_event))

            # Simulate processing delay
            await asyncio.sleep(0.5)

            # Emit an "action" event
            action_event = {
                "event_id": str(uuid.uuid4()),
                "workflow_id": record.workflow_id,
                "timestamp": time.time(),
                "kind": "action",
                "agent_name": body.agent_name or "default",
                "payload": {
                    "action": "reasoning",
                    "description": "Processing task with ReAct loop",
                },
            }
            record.add_event(WorkflowEvent(**action_event))

            await asyncio.sleep(0.5)

            # Emit a "result" event and complete
            result_payload = {
                "summary": f"Task completed: {body.task[:100]}",
                "iterations": 1,
                "success": True,
            }
            await manager_complete(record.workflow_id, result_payload)

        except Exception as exc:
            logger.exception("Workflow execution failed", extra={"workflow_id": record.workflow_id})
            await manager_fail(record.workflow_id, str(exc))

    # Get the manager from the record's app context
    # We use a module-level reference set during app creation
    global _workflow_manager_ref
    if _workflow_manager_ref is not None:
        asyncio.ensure_future(_run())
    else:
        # Fallback: schedule on the running loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_run())
        except RuntimeError:
            asyncio.ensure_future(_run())


# Module-level reference to the WorkflowManager (set during app creation)
_workflow_manager_ref = None


async def manager_complete(workflow_id: str, result: Dict[str, Any]) -> None:
    """Complete a workflow via the global manager reference."""
    if _workflow_manager_ref is not None:
        await _workflow_manager_ref.complete_workflow(workflow_id, result)


async def manager_fail(workflow_id: str, error: str) -> None:
    """Fail a workflow via the global manager reference."""
    if _workflow_manager_ref is not None:
        await _workflow_manager_ref.fail_workflow(workflow_id, error)


@router.get(
    "/workflows",
    response_model=WorkflowListResponse,
    summary="List recent workflows",
)
async def list_workflows(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of workflows to return."),
) -> WorkflowListResponse:
    """Return the most recent workflows, newest first."""
    manager = _get_manager(request)
    records = await manager.list_workflows(limit=limit)

    workflows = [
        WorkflowResponse(
            workflow_id=r.workflow_id,
            task=r.task,
            status=r.status,
            created_at=r.created_at,
            finished_at=r.finished_at,
            result=r.result,
            error=r.error,
        )
        for r in records
    ]

    return WorkflowListResponse(workflows=workflows, total=len(workflows))


@router.get(
    "/workflows/{workflow_id}",
    response_model=WorkflowResponse,
    summary="Get workflow status and result",
)
async def get_workflow(
    workflow_id: str,
    request: Request,
) -> WorkflowResponse:
    """Retrieve the current status and (if complete) result of a workflow."""
    manager = _get_manager(request)
    record = await manager.get_workflow(workflow_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' not found.",
        )

    return WorkflowResponse(
        workflow_id=record.workflow_id,
        task=record.task,
        status=record.status,
        created_at=record.created_at,
        finished_at=record.finished_at,
        result=record.result,
        error=record.error,
    )


@router.delete(
    "/workflows/{workflow_id}",
    status_code=204,
    summary="Cancel a running workflow",
)
async def cancel_workflow(
    workflow_id: str,
    request: Request,
) -> None:
    """Cancel a pending or running workflow."""
    manager = _get_manager(request)
    cancelled = await manager.cancel_workflow(workflow_id)

    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail=f"Workflow '{workflow_id}' is not in a cancellable state.",
        )


@router.get(
    "/workflows/{workflow_id}/events",
    response_model=EventsResponse,
    summary="Retrieve buffered events for a workflow",
)
async def get_workflow_events(
    workflow_id: str,
    request: Request,
    limit: int = Query(default=100, ge=1, le=500, description="Maximum number of events to return."),
) -> EventsResponse:
    """Return the buffered events for a workflow (for late-joining clients)."""
    manager = _get_manager(request)
    record = await manager.get_workflow(workflow_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Workflow '{workflow_id}' not found.",
        )

    events = [
        EventResponse(
            event_id=e.event_id,
            workflow_id=e.workflow_id,
            timestamp=e.timestamp,
            kind=e.kind,
            agent_name=e.agent_name,
            payload=e.payload,
        )
        for e in record.events[-limit:]
    ]

    return EventsResponse(
        workflow_id=workflow_id,
        events=events,
        total=len(events),
        dropped=record.dropped_count,
    )


# ---------------------------------------------------------------------------
# Agent and Tool discovery endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/agents",
    response_model=List[AgentInfo],
    summary="List available agents",
)
async def list_agents() -> List[AgentInfo]:
    """Return the list of registered agents."""
    # In production, this would query the agent registry.
    # For now, return a default set.
    return [
        AgentInfo(
            name="default",
            description="Default ReAct agent with standard tool access.",
            model="gpt-4",
        ),
        AgentInfo(
            name="researcher",
            description="Specialized agent for research and information gathering.",
            model="gpt-4",
        ),
        AgentInfo(
            name="coder",
            description="Code-generation and debugging agent.",
            model="gpt-4",
        ),
    ]


@router.get(
    "/tools",
    response_model=List[ToolInfo],
    summary="List registered tools",
)
async def list_tools() -> List[ToolInfo]:
    """Return the list of registered tools."""
    # In production, this would query the tool registry.
    return [
        ToolInfo(
            name="calculator",
            description="Evaluate mathematical expressions.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
        ),
        ToolInfo(
            name="search",
            description="Search the web or a knowledge base.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        ),
    ]
