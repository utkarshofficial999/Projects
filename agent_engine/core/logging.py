"""
agent_engine.core.logging
=========================

Centralized structured logging configuration for the Agent Engine.

This module provides:
- A setup_logging() function to configure the root logger with appropriate
  handlers, formatters, and log levels
- A get_logger() helper to obtain named loggers throughout the codebase
- Support for both human-readable and JSON structured log formats
- Optional file logging alongside console output
- Context-aware logging with extra fields

Usage:
    from agent_engine.core.logging import setup_logging, get_logger

    # Initialize logging (call once at application startup)
    setup_logging(level="INFO", json_format=False)

    # Get a logger for your module
    logger = get_logger(__name__)

    # Log messages
    logger.info("Engine started", extra={"engine_id": "eng-001"})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agent_engine.core.config import LoggingConfig, get_config


class StructuredFormatter(logging.Formatter):
    """
    Custom logging formatter that supports both human-readable and JSON output.

    In JSON mode, each log record is serialized as a JSON object with standard
    fields (timestamp, level, logger, message) plus any extra fields provided
    via the `extra` parameter.

    In human-readable mode, logs are formatted as:
        [2024-01-15T10:30:00Z] [INFO] [module.name] message {extra_fields}
    """

    def __init__(
        self,
        json_format: bool = False,
        include_timestamp: bool = True,
        include_context: bool = True,
    ) -> None:
        """
        Initialize the StructuredFormatter.

        Args:
            json_format: If True, output logs as JSON objects.
            include_timestamp: If True, include ISO-8601 timestamps.
            include_context: If True, include extra context fields.
        """
        super().__init__()
        self.json_format = json_format
        self.include_timestamp = include_timestamp
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record.

        Args:
            record: The log record to format.

        Returns:
            str: The formatted log message.
        """
        if self.json_format:
            return self._format_json(record)
        return self._format_human(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            str: JSON-formatted log entry.
        """
        log_entry: Dict[str, Any] = {
            "timestamp": (
                datetime.fromtimestamp(record.created, tz=timezone.utc)
                .isoformat()
                if self.include_timestamp
                else None
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.include_context:
            # Extract extra fields (excluding standard LogRecord attributes)
            standard_attrs = {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "asctime", "taskName",
            }
            extra = {
                k: v for k, v in record.__dict__.items()
                if k not in standard_attrs and not k.startswith("_")
            }
            if extra:
                log_entry["context"] = extra

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str, ensure_ascii=False)

    def _format_human(self, record: logging.LogRecord) -> str:
        """
        Format a log record in human-readable format.

        Args:
            record: The log record to format.

        Returns:
            str: Human-readable log entry.
        """
        timestamp = ""
        if self.include_timestamp:
            ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
            timestamp = f"[{ts.isoformat()}] "

        base = f"{timestamp}[{record.levelname}] [{record.name}] {record.getMessage()}"

        if self.include_context:
            standard_attrs = {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "asctime", "taskName",
            }
            extra = {
                k: v for k, v in record.__dict__.items()
                if k not in standard_attrs and not k.startswith("_")
            }
            if extra:
                base += f" {json.dumps(extra, default=str, ensure_ascii=False)}"

        if record.exc_info:
            base += f"\n{self.formatException(record.exc_info)}"

        return base


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: Optional[bool] = None,
    include_timestamp: Optional[bool] = None,
    include_context: Optional[bool] = None,
) -> logging.Logger:
    """
    Configure the root logger with appropriate handlers and formatters.

    This function should be called once at application startup. It configures
    the root 'agent_engine' logger with:
    - A console handler (stdout)
    - An optional file handler
    - Appropriate log level
    - Structured or human-readable formatting

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, uses the value from global config.
        log_file: Path to log file. If None, only console output.
        json_format: Whether to use JSON format. If None, uses config value.
        include_timestamp: Whether to include timestamps. If None, uses config.
        include_context: Whether to include context. If None, uses config.

    Returns:
        logging.Logger: The configured root 'agent_engine' logger.
    """
    # Get config values as defaults
    config = get_config()
    log_config = config.get_logging_config()

    if level is None:
        level = log_config.level.value
    if json_format is None:
        json_format = log_config.json_format
    if include_timestamp is None:
        include_timestamp = log_config.include_timestamp
    if include_context is None:
        include_context = log_config.include_context

    # Get or create the agent_engine root logger
    logger = logging.getLogger("agent_engine")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any existing handlers to avoid duplicates on re-initialization
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = StructuredFormatter(
        json_format=json_format,
        include_timestamp=include_timestamp,
        include_context=include_context,
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for the agent engine.

    This is the primary way to obtain loggers throughout the codebase.
    It returns a child logger of the 'agent_engine' root logger, ensuring
    consistent configuration.

    Args:
        name: The logger name, typically __name__ of the calling module.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Ensure the root agent_engine logger is configured
    root_logger = logging.getLogger("agent_engine")
    if not root_logger.handlers:
        # If not yet configured, set up with defaults
        setup_logging()

    return root_logger.getChild(name)
