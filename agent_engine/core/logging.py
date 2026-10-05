"""
agent_engine.core.logging
=========================

Centralized structured logging configuration for the Agent Engine.

This module provides:
- A setup_logging() function that configures the root logger with
  structured output (JSON or human-readable console format)
- A get_logger() helper that returns a configured logger instance
- Support for both console and file output
- Structured JSON logging for production environments
- Human-readable console logging for development

Usage:
    from agent_engine.core.logging import setup_logging, get_logger

    # Initialize logging (call once at application startup)
    setup_logging(level="INFO", format="console")

    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("Engine started", extra={"component": "engine"})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agent_engine.core.config import LogFormat, LogLevel


class JSONFormatter(logging.Formatter):
    """
    Custom logging formatter that outputs log records as JSON.

    This formatter produces structured JSON log lines suitable for
    ingestion by log aggregation systems (e.g., ELK, Datadog, CloudWatch).

    Each log line is a single JSON object containing:
    - timestamp: ISO 8601 timestamp with timezone
    - level: Log level name
    - logger: Logger name
    - message: The log message
    - module: Python module name
    - function: Python function name
    - line: Source line number
    - Any additional 'extra' fields passed to the log call
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A JSON string representation of the log record.
        """
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include any extra fields
        reserved_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in reserved_attrs and not key.startswith("_"):
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable console log formatter.

    Produces colored, aligned log lines suitable for terminal output.
    Format: [TIMESTAMP] LEVEL (logger) - message
    """

    # ANSI color codes for log levels
    COLORS: Dict[str, str] = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        """
        Initialize the console formatter.

        Args:
            use_color: Whether to use ANSI color codes.
        """
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record for console output.

        Args:
            record: The log record to format.

        Returns:
            A formatted string for console display.
        """
        formatted = super().format(record)
        if self.use_color:
            color = self.COLORS.get(record.levelname, self.RESET)
            formatted = f"{color}{formatted}{self.RESET}"
        return formatted


def setup_logging(
    level: LogLevel = LogLevel.INFO,
    log_format: LogFormat = LogFormat.CONSOLE,
    log_file: Optional[str] = None,
    enable_file_logging: bool = False,
    use_color: bool = True,
) -> logging.Logger:
    """
    Configure the root logger for the Agent Engine.

    This function should be called once at application startup to initialize
    the logging system. It configures:
    - The root logger with the specified level
    - A console handler with the specified format
    - An optional file handler if file logging is enabled

    Args:
        level: The minimum log level to capture.
        log_format: The output format (JSON or console).
        log_file: Path to the log file (required if enable_file_logging is True).
        enable_file_logging: Whether to enable file logging.
        use_color: Whether to use ANSI colors in console output.

    Returns:
        The configured root logger instance.

    Raises:
        ValueError: If enable_file_logging is True but no log_file is provided.

    Example:
        >>> logger = setup_logging(level=LogLevel.DEBUG, log_format=LogFormat.JSON)
        >>> logger.info("Application started")
    """
    if enable_file_logging and not log_file:
        raise ValueError(
            "log_file must be specified when enable_file_logging is True"
        )

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level.value)

    # Clear existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create and configure the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level.value)

    if log_format == LogFormat.JSON:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter(use_color=use_color))

    root_logger.addHandler(console_handler)

    # Create and configure the file handler if enabled
    if enable_file_logging and log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level.value)

        if log_format == LogFormat.JSON:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(ConsoleFormatter(use_color=False))

        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance for the given name.

    This is a convenience wrapper around logging.getLogger() that ensures
    the logger is properly configured. If logging has not been set up yet,
    it will use the default configuration.

    Args:
        name: The name of the logger (typically __name__).

    Returns:
        A configured logging.Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Hello from my module")
    """
    logger = logging.getLogger(name)

    # If no handlers are configured, add a basic one to avoid "No handlers" warning
    if not logger.handlers and not logging.getLogger().handlers:
        basic_handler = logging.StreamHandler(sys.stdout)
        basic_handler.setFormatter(
            ConsoleFormatter(use_color=True)
        )
        logger.addHandler(basic_handler)
        logger.setLevel(LogLevel.INFO.value)

    return logger
