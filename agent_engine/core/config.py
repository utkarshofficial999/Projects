"""
agent_engine.core.config
========================

Robust configuration system for the Agent Engine using Pydantic Settings.

This module provides a centralized, type-safe configuration system that:
- Loads settings from environment variables and .env files
- Validates all configuration values at startup
- Provides sensible defaults for all parameters
- Supports nested configuration sections
- Offers a singleton pattern for global access to the active configuration

Usage:
    from agent_engine.core.config import get_config

    config = get_config()
    print(config.llm.api_key)
    print(config.engine.max_iterations)
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    GROQ = "groq"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Supported log output formats."""

    JSON = "json"
    CONSOLE = "console"


class LLMConfig(BaseModel):
    """
    Configuration for the LLM client.

    Attributes:
        provider: The LLM provider to use.
        model: The model name/identifier.
        api_key: API key for the LLM provider.
        base_url: Base URL for the LLM API endpoint.
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum number of tokens to generate.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts on failure.
    """

    provider: LLMProvider = Field(
        default=LLMProvider.GROQ,
        description="LLM provider to use",
    )
    model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model name/identifier",
    )
    api_key: str = Field(
        default="",
        description="API key for the LLM provider",
    )
    base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Base URL for the LLM API endpoint",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum number of tokens to generate",
    )
    timeout: float = Field(
        default=60.0,
        ge=1.0,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retry attempts",
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that API key is not empty for cloud providers."""
        if v is None:
            return ""
        return v.strip()

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Ensure base URL has no trailing slash."""
        if v and v.endswith("/"):
            return v.rstrip("/")
        return v


class EngineConfig(BaseModel):
    """
    Configuration for the agent engine core.

    Attributes:
        max_iterations: Maximum number of agent iterations per task.
        max_tool_calls: Maximum number of tool calls per iteration.
        reflection_enabled: Whether self-reflection is enabled.
        reflection_interval: Number of iterations between reflections.
        memory_window: Size of the sliding memory window.
        parallel_agents: Maximum number of parallel agents.
        task_timeout: Timeout for individual task execution in seconds.
    """

    max_iterations: int = Field(
        default=10,
        ge=1,
        description="Maximum number of agent iterations per task",
    )
    max_tool_calls: int = Field(
        default=5,
        ge=1,
        description="Maximum number of tool calls per iteration",
    )
    reflection_enabled: bool = Field(
        default=True,
        description="Whether self-reflection is enabled",
    )
    reflection_interval: int = Field(
        default=3,
        ge=1,
        description="Number of iterations between reflections",
    )
    memory_window: int = Field(
        default=20,
        ge=1,
        description="Size of the sliding memory window",
    )
    parallel_agents: int = Field(
        default=1,
        ge=1,
        description="Maximum number of parallel agents",
    )
    task_timeout: float = Field(
        default=300.0,
        ge=1.0,
        description="Timeout for individual task execution in seconds",
    )


class LoggingConfig(BaseModel):
    """
    Configuration for the logging system.

    Attributes:
        level: The minimum log level to capture.
        format: Output format (json or console).
        log_file: Optional path to a log file.
        enable_file_logging: Whether to write logs to a file.
    """

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log level to capture",
    )
    format: LogFormat = Field(
        default=LogFormat.CONSOLE,
        description="Output format (json or console)",
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Optional path to a log file",
    )
    enable_file_logging: bool = Field(
        default=False,
        description="Whether to write logs to a file",
    )


class MemoryConfig(BaseModel):
    """
    Configuration for the dynamic memory system.

    Attributes:
        backend: Memory backend type (in_memory, redis, sqlite).
        max_entries: Maximum number of entries to store.
        ttl_seconds: Time-to-live for memory entries in seconds.
        compression_enabled: Whether to enable memory compression.
    """

    backend: str = Field(
        default="in_memory",
        description="Memory backend type",
    )
    max_entries: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of entries to store",
    )
    ttl_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Time-to-live for memory entries in seconds",
    )
    compression_enabled: bool = Field(
        default=False,
        description="Whether to enable memory compression",
    )


class AgentEngineSettings(BaseSettings):
    """
    Top-level settings for the Agent Engine.

    This class uses Pydantic Settings to load configuration from:
    1. Environment variables (prefixed with AGENT_ENGINE_)
    2. .env file in the current directory
    3. Default values defined in the model

    Environment variable mapping:
        AGENT_ENGINE_LLM_PROVIDER -> llm.provider
        AGENT_ENGINE_LLM_MODEL -> llm.model
        AGENT_ENGINE_LLM_API_KEY -> llm.api_key
        AGENT_ENGINE_ENGINE_MAX_ITERATIONS -> engine.max_iterations
        AGENT_ENGINE_LOG_LEVEL -> logging.level
        etc.

    Attributes:
        llm: LLM client configuration.
        engine: Core engine configuration.
        logging: Logging system configuration.
        memory: Memory system configuration.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_ENGINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM client configuration",
    )
    engine: EngineConfig = Field(
        default_factory=EngineConfig,
        description="Core engine configuration",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging system configuration",
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory system configuration",
    )

    @field_validator("llm", mode="before")
    @classmethod
    def parse_llm_config(cls, v: object) -> object:
        """
        Parse LLM config from environment variables.

        Handles the case where individual LLM fields are set via env vars
        like AGENT_ENGINE_LLM_PROVIDER, AGENT_ENGINE_LLM_MODEL, etc.
        """
        if isinstance(v, dict):
            return v
        if isinstance(v, LLMConfig):
            return v
        # If it's a string, try to parse as JSON
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return {}

    @field_validator("engine", mode="before")
    @classmethod
    def parse_engine_config(cls, v: object) -> object:
        """Parse engine config from environment variables."""
        if isinstance(v, dict):
            return v
        if isinstance(v, EngineConfig):
            return v
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return {}

    @field_validator("logging", mode="before")
    @classmethod
    def parse_logging_config(cls, v: object) -> object:
        """Parse logging config from environment variables."""
        if isinstance(v, dict):
            return v
        if isinstance(v, LoggingConfig):
            return v
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return {}

    @field_validator("memory", mode="before")
    @classmethod
    def parse_memory_config(cls, v: object) -> object:
        """Parse memory config from environment variables."""
        if isinstance(v, dict):
            return v
        if isinstance(v, MemoryConfig):
            return v
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return {}

    def validate_api_key_for_provider(self) -> None:
        """
        Validate that an API key is provided for cloud-based LLM providers.

        Raises:
            ValueError: If a cloud provider is selected but no API key is provided.
        """
        cloud_providers = {
            LLMProvider.OPENAI,
            LLMProvider.GROQ,
            LLMProvider.DEEPSEEK,
            LLMProvider.ANTHROPIC,
        }
        if self.llm.provider in cloud_providers and not self.llm.api_key:
            raise ValueError(
                f"API key is required for provider '{self.llm.provider.value}'. "
                f"Set AGENT_ENGINE_LLM_API_KEY environment variable."
            )

    def to_dict(self) -> dict:
        """
        Serialize the entire configuration to a dictionary.

        Returns:
            A dictionary representation of the configuration.
        """
        return {
            "llm": self.llm.model_dump(),
            "engine": self.engine.model_dump(),
            "logging": self.logging.model_dump(),
            "memory": self.memory.model_dump(),
        }


@lru_cache(maxsize=1)
def get_config() -> AgentEngineSettings:
    """
    Get the singleton instance of the agent engine configuration.

    This function uses lru_cache to ensure that only one instance of the
    configuration is created and reused throughout the application lifecycle.

    Returns:
        The singleton AgentEngineSettings instance.

    Example:
        >>> config = get_config()
        >>> print(config.llm.provider)
        LLMProvider.GROQ
    """
    settings = AgentEngineSettings()
    # Validate that API key is present for cloud providers
    settings.validate_api_key_for_provider()
    return settings


def reset_config() -> None:
    """
    Reset the cached configuration instance.

    This is primarily useful for testing purposes where you need to
    reload configuration with different environment variables.
    """
    get_config.cache_clear()
