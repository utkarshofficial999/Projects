"""
agent_engine.api
================

Production HTTP/WebSocket interface for the Agent Engine.

This package exposes the FastAPI application and its routing layer,
allowing external clients to start workflows, stream agent thoughts in
real-time over WebSocket, and retrieve results via REST.

Quick start (from project root)::

    uvicorn agent_engine.api.server:app --reload --port 8000

Or programmatically::

    from agent_engine.api.server import create_app
    app = create_app()
"""

from agent_engine.api.server import create_app

__all__ = ["create_app"]
