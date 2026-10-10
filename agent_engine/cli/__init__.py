"""
agent_engine.cli
================

Command-line interface for the Agent Engine.

This module provides the CLI entry point for interacting with the
agent engine from the command line.

Usage::

    agent-engine --help
    agent-engine run --task "Summarize the latest news"
    agent-engine serve --port 8000
    agent-engine config
    agent-engine version
"""

from agent_engine.cli.main import cli

__all__ = ["cli"]
