"""Coder Agent: Implements code, tests, and documentation for roadmap milestones."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from src.config import AppConfig
from src.llm_client import LLMClient
from src.roadmap_engine import RoadmapStep

logger = logging.getLogger("GitAgentic.Coder")

class StepExecutionResult(BaseModel):
    step_id: str
    commit_message: str
    files_written: List[str]
    summary: str

class CoderAgent:
    """Autonomous software engineer that generates production-grade code."""

    def __init__(self, config: AppConfig, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.target_dir = Path(config.target_project_path).resolve()
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def execute_step(self, step: RoadmapStep) -> StepExecutionResult:
        """Implement the target milestone by writing or updating files in the workspace."""
        logger.info(f"Executing step {step.step_number}: '{step.title}'")

        # 1. Gather context from existing workspace
        existing_tree = self._get_directory_tree(self.target_dir)
        existing_files_content = self._get_relevant_files_content(step.target_files)

        # 2. Formulate LLM prompts
        system_prompt = (
            "You are a Senior Staff Software Engineer and open-source author specialized in Agentic AI, "
            "Python, and distributed systems. Your code is clean, robust, thoroughly commented, type-annotated, "
            "and follows clean architecture principles (SOLID, modular design). "
            "Always include unit tests (pytest) and helpful docstrings."
        )

        user_prompt = f"""
Overall Project Topic: {self.config.project_topic}

Current Milestone:
- Step Number: {step.step_number}
- Phase: {step.phase}
- Title: {step.title}
- Scope & Requirements: {step.description}
- Target Files to create or modify: {json.dumps(step.target_files)}

Current Target Workspace Tree:
{existing_tree}

Existing relevant files content:
{existing_files_content if existing_files_content else "(Workspace is currently empty or starting up)"}

Task:
Produce complete, runnable, production-quality code for this milestone.
Do NOT write placeholder code or `# TODO implement later`. Write full, working implementations.
Include unit tests with `pytest` for the modules being implemented or modified.

Format your response as strict JSON:
{{
  "commit_message": "feat(component): concise conventional commit description",
  "summary": "Brief 1-2 sentence developer summary of changes made",
  "files": [
    {{
      "path": "relative/path/to/file.py",
      "content": "Full file content string"
    }}
  ]
}}
"""

        response = self.llm.generate_json(system_prompt, user_prompt, temperature=0.2)
        
        commit_message = response.get("commit_message", f"feat: implement {step.title}")
        summary = response.get("summary", "")
        files_data = response.get("files", [])

        written_paths: List[str] = []
        for file_entry in files_data:
            rel_path = file_entry.get("path", "").strip()
            content = file_entry.get("content", "")
            if not rel_path or not content:
                continue

            full_path = (self.target_dir / rel_path).resolve()
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            written_paths.append(rel_path)
            logger.info(f"Wrote file: {rel_path} ({len(content)} chars)")

        return StepExecutionResult(
            step_id=step.id,
            commit_message=commit_message,
            files_written=written_paths,
            summary=summary,
        )

    def _get_directory_tree(self, path: Path, max_depth: int = 3, current_depth: int = 0) -> str:
        """Create a compact text representation of existing files."""
        if current_depth > max_depth or not path.exists():
            return ""

        lines = []
        try:
            for item in sorted(path.iterdir()):
                if item.name.startswith((".", "__pycache__", "venv", ".git")):
                    continue
                indent = "  " * current_depth
                if item.is_dir():
                    lines.append(f"{indent}📁 {item.name}/")
                    sub = self._get_directory_tree(item, max_depth, current_depth + 1)
                    if sub:
                        lines.append(sub)
                else:
                    lines.append(f"{indent}📄 {item.name}")
        except Exception:
            pass
        return "\n".join(lines)

    def _get_relevant_files_content(self, target_files: List[str], max_chars: int = 4000) -> str:
        """Read existing files if they already exist so the LLM can edit or append."""
        contents = []
        for rel in target_files:
            p = (self.target_dir / rel).resolve()
            if p.exists() and p.is_file():
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    if len(text) > max_chars:
                        text = text[:max_chars] + "\n...[truncated]..."
                    contents.append(f"--- File: {rel} ---\n{text}")
                except Exception:
                    pass
        return "\n\n".join(contents)
