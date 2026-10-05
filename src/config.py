"""Configuration management for GitAgentic."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file from root
load_dotenv()

class AppConfig(BaseModel):
    # LLM Settings
    llm_base_url: str = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    )
    llm_api_key: str = Field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_model: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    )

    # Project Settings
    project_topic: str = Field(
        default_factory=lambda: os.getenv(
            "PROJECT_TOPIC", 
            "Autonomous Multi-Agent Workflow Engine with Tool Calling, Dynamic Memory and Self-Reflection"
        )
    )
    target_project_path: Path = Field(
        default_factory=lambda: Path(os.getenv("TARGET_PROJECT_PATH", "./target_project"))
    )

    # Git Settings
    git_auto_push: bool = Field(
        default_factory=lambda: os.getenv("GIT_AUTO_PUSH", "true").lower() in ("true", "1", "yes")
    )
    git_remote: str = Field(
        default_factory=lambda: os.getenv("GIT_REMOTE", "origin")
    )
    git_branch: str = Field(
        default_factory=lambda: os.getenv("GIT_BRANCH", "main")
    )
    git_author_name: Optional[str] = Field(
        default_factory=lambda: os.getenv("GIT_AUTHOR_NAME") or None
    )
    git_author_email: Optional[str] = Field(
        default_factory=lambda: os.getenv("GIT_AUTHOR_EMAIL") or None
    )

    # Scheduling Settings (hours)
    min_hours_between_commits: float = Field(
        default_factory=lambda: float(os.getenv("MIN_HOURS_BETWEEN_COMMITS", "18.0"))
    )
    max_hours_between_commits: float = Field(
        default_factory=lambda: float(os.getenv("MAX_HOURS_BETWEEN_COMMITS", "28.0"))
    )

    # Internal state storage
    state_dir: Path = Field(default_factory=lambda: Path("./state"))

    def validate_keys(self) -> None:
        """Validate critical configuration before running."""
        # Ollama local models don't require an API key, but remote providers do
        is_local_ollama = "localhost" in self.llm_base_url or "127.0.0.1" in self.llm_base_url
        if not is_local_ollama and (not self.llm_api_key or self.llm_api_key == "your_api_key_here"):
            raise ValueError(
                "Missing LLM_API_KEY. Please provide your API key in the .env file or environment variable."
            )

def get_config() -> AppConfig:
    """Return loaded application configuration."""
    cfg = AppConfig()
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    return cfg
