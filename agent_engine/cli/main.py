"""
agent_engine.cli.main
=====================

Main CLI entry point for the Agent Engine.

Provides a Click-based command-line interface with subcommands for:
- config: Display current configuration
- version: Show version information
- run: Run the agent engine (placeholder for future implementation)

Usage:
    agent-engine config
    agent-engine version
    agent-engine run --task "your task description"
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import click

from agent_engine import __version__
from agent_engine.core.config import get_config, reset_config
from agent_engine.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@click.group()
@click.version_option(version=__version__, prog_name="agent-engine")
def cli() -> None:
    """
    Agent Engine - Autonomous Multi-Agent Workflow Engine.

    A powerful framework for building autonomous multi-agent systems
    with tool calling, dynamic memory, and self-reflection capabilities.
    """
    pass


@cli.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "text"]),
    default="json",
    help="Output format for configuration display.",
)
def config(output_format: str) -> None:
    """
    Display the current engine configuration.

    Shows all active configuration values loaded from environment
    variables, .env file, and defaults.

    Example:
        agent-engine config
        agent-engine config --format text
    """
    try:
        cfg = get_config()
        config_dict = cfg.to_dict()

        if output_format == "json":
            click.echo(json.dumps(config_dict, indent=2, default=str))
        else:
            click.echo("=== Agent Engine Configuration ===\n")
            for section, values in config_dict.items():
                click.echo(f"[{section.upper()}]")
                for key, value in values.items():
                    # Mask API keys for security
                    if "api_key" in key.lower():
                        display_value = (
                            value[:4] + "****" if len(value) > 4 else "****"
                        )
                    else:
                        display_value = str(value)
                    click.echo(f"  {key}: {display_value}")
                click.echo()

    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
def version() -> None:
    """
    Display version information.

    Shows the current version of the agent engine package.

    Example:
        agent-engine version
    """
    click.echo(f"agent-engine v{__version__}")


@cli.command()
@click.option(
    "--task",
    "-t",
    required=True,
    help="The task description for the agent to execute.",
)
@click.option(
    "--max-iterations",
    "-m",
    type=int,
    default=None,
    help="Override maximum iterations for this run.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose (DEBUG) logging.",
)
def run(task: str, max_iterations: Optional[int], verbose: bool) -> None:
    """
    Run the agent engine with a given task.

    This command initializes the engine, configures logging, and
    begins executing the specified task.

    Example:
        agent-engine run --task "Summarize the latest news"
        agent-engine run -t "Write a poem" -m 5 -v
    """
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=__import__("agent_engine.core.config", fromlist=["LogLevel"]).LogLevel(log_level))

    logger.info("Starting agent engine", extra={"task": task})

    # Load configuration
    try:
        cfg = get_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)

    # Apply max_iterations override if provided
    if max_iterations is not None:
        cfg.engine.max_iterations = max_iterations

    logger.info(
        "Engine configured",
        extra={
            "provider": cfg.llm.provider.value,
            "model": cfg.llm.model,
            "max_iterations": cfg.engine.max_iterations,
            "reflection_enabled": cfg.engine.reflection_enabled,
        },
    )

    # Placeholder for actual engine execution (to be implemented in later milestones)
    click.echo(f"\nTask: {task}")
    click.echo(f"Provider: {cfg.llm.provider.value}")
    click.echo(f"Model: {cfg.llm.model}")
    click.echo(f"Max Iterations: {cfg.engine.max_iterations}")
    click.echo("\n[INFO] Engine execution will be implemented in a future milestone.")
    click.echo("[INFO] Configuration and logging are fully operational.")


if __name__ == "__main__":
    cli()
