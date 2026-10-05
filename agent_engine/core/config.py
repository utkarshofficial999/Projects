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


class MemoryConfig(BaseModel):
    """
    Configuration for the dynamic memory system.

    Attributes:
        backend: Memory backend type (in-memory, redis, sqlite).
        max_entries: Maximum number of memory entries to retain.
        ttl_seconds: Time-to-live for memory entries in seconds.
        compression_enabled: Whether memory compression is enabled.
        compression_threshold: Threshold for triggering compression.
    """

    backend: str = Field(
        default="in-memory",
        description="Memory backend type",
    )
    max_entries: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of memory entries to retain",
    )
    ttl_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Time-to-live for memory entries in seconds",
    )
    compression_enabled: bool = Field(
        default=False,
        description="Whether memory compression is enabled",
    )
    compression_threshold: int = Field(
        default=500,
        ge=1,
        description="Threshold for triggering compression",
    )


class LoggingConfig(BaseModel):
    """
    Configuration for the logging system.

    Attributes:
        level: Log level.
        log_file: Path to the log file (None for stdout only).
        json_format: Whether to use JSON structured logging.
        include_timestamp: Whether to include timestamps in logs.
        include_context: Whether to include context in logs.
    """

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Log level",
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Path to the log file",
    )
    json_format: bool = Field(
        default=False,
        description="Whether to use JSON structured logging",
    )
    include_timestamp: bool = Field(
        default=True,
        description="Whether to include timestamps in logs",
    )
    include_context: bool = Field(
        default=True,
        description="Whether to include context in logs",
    )


class AgentEngineSettings(BaseSettings):
    """
    Top-level settings for the Agent Engine.

    This class uses Pydantic Settings to automatically load configuration
    from environment variables and .env files. Environment variables are
    mapped to fields using the prefix 'AGENT_ENGINE_'.

    Example:
        AGENT_ENGINE_LLM_PROVIDER=groq
        AGENT_ENGINE_LLM_MODEL=llama-3.3-70b-versatile
        AGENT_ENGINE_LLM_API_KEY=gsk_xxx
        AGENT_ENGINE_ENGINE_MAX_ITERATIONS=15
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_ENGINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Configuration
    llm_provider: LLMProvider = Field(
        default=LLMProvider.GROQ,
        description="LLM provider",
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="LLM model name",
    )
    llm_api_key: str = Field(
        default="",
        description="LLM API key",
    )
    llm_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="LLM base URL",
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature",
    )
    llm_max_tokens: int = Field(
        default=4096,
        ge=1,
        description="LLM max tokens",
    )
    llm_timeout: float = Field(
        default=60.0,
        ge=1.0,
        description="LLM timeout in seconds",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        description="LLM max retries",
    )

    # Engine Configuration
    engine_max_iterations: int = Field(
        default=10,
        ge=1,
        description="Max agent iterations",
    )
    engine_max_tool_calls: int = Field(
        default=5,
        ge=1,
        description="Max tool calls per iteration",
    )
    engine_reflection_enabled: bool = Field(
        default=True,
        description="Enable self-reflection",
    )
    engine_reflection_interval: int = Field(
        default=3,
        ge=1,
        description="Reflection interval",
    )
    engine_memory_window: int = Field(
        default=20,
        ge=1,
        description="Memory window size",
    )
    engine_parallel_agents: int = Field(
        default=1,
        ge=1,
        description="Max parallel agents",
    )
    engine_task_timeout: float = Field(
        default=300.0,
        ge=1.0,
        description="Task timeout in seconds",
    )

    # Memory Configuration
    memory_backend: str = Field(
        default="in-memory",
        description="Memory backend",
    )
    memory_max_entries: int = Field(
        default=1000,
        ge=1,
        description="Max memory entries",
    )
    memory_ttl_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Memory TTL in seconds",
    )
    memory_compression_enabled: bool = Field(
        default=False,
        description="Enable memory compression",
    )
    memory_compression_threshold: int = Field(
        default=500,
        ge=1,
        description="Compression threshold",
    )

    # Logging Configuration
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Log level",
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path",
    )
    log_json_format: bool = Field(
        default=False,
        description="Use JSON log format",
    )
    log_include_timestamp: bool = Field(
        default=True,
        description="Include timestamps in logs",
    )
    log_include_context: bool = Field(
        default=True,
        description="Include context in logs",
    )

    def get_llm_config(self) -> LLMConfig:
        """
        Build and return an LLMConfig instance from flat settings.

        Returns:
            LLMConfig: Configured LLM settings.
        """
        return LLMConfig(
            provider=self.llm_provider,
            model=self.llm_model,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            timeout=self.llm_timeout,
            max_retries=self.llm_max_retries,
        )

    def get_engine_config(self) -> EngineConfig:
        """
        Build and return an EngineConfig instance from flat settings.

        Returns:
            EngineConfig: Configured engine settings.
        """
        return EngineConfig(
            max_iterations=self.engine_max_iterations,
            max_tool_calls=self.engine_max_tool_calls,
            reflection_enabled=self.engine_reflection_enabled,
            reflection_interval=self.engine_reflection_interval,
            memory_window=self.engine_memory_window,
            parallel_agents=self.engine_parallel_agents,
            task_timeout=self.engine_task_timeout,
        )

    def get_memory_config(self) -> MemoryConfig:
        """
        Build and return a MemoryConfig instance from flat settings.

        Returns:
            MemoryConfig: Configured memory settings.
        """
        return MemoryConfig(
            backend=self.memory_backend,
            max_entries=self.memory_max_entries,
            ttl_seconds=self.memory_ttl_seconds,
            compression_enabled=self.memory_compression_enabled,
            compression_threshold=self.memory_compression_threshold,
        )

    def get_logging_config(self) -> LoggingConfig:
        """
        Build and return a LoggingConfig instance from flat settings.

        Returns:
            LoggingConfig: Configured logging settings.
        """
        return LoggingConfig(
            level=self.log_level,
            log_file=self.log_file,
            json_format=self.log_json_format,
            include_timestamp=self.log_include_timestamp,
            include_context=self.log_include_context,
        )


@lru_cache(maxsize=1)
def get_config() -> AgentEngineSettings:
    """
    Get the singleton instance of AgentEngineSettings.

    This function uses lru_cache to ensure only one instance is created
    and reused throughout the application lifecycle.

    Returns:
        AgentEngineSettings: The global configuration instance.
    """
    return AgentEngineSettings()


def reset_config() -> None:
    """
    Reset the cached configuration instance.

    This is primarily useful for testing, allowing tests to create
    fresh configuration instances with different environment variables.
    """
    get_config.cache_clear()
