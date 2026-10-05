"""Roadmap Engine: Generates, manages and persists the multi-step project plan."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.config import AppConfig
from src.llm_client import LLMClient

logger = logging.getLogger("GitAgentic.Roadmap")

class RoadmapStep(BaseModel):
    step_number: int
    id: str
    phase: str
    title: str
    description: str
    target_files: List[str]
    status: str = "pending"  # pending | completed
    completed_at: Optional[str] = None
    commit_hash: Optional[str] = None

class ProjectRoadmap(BaseModel):
    topic: str
    created_at: str
    total_steps: int
    steps: List[RoadmapStep]

class RoadmapEngine:
    """Manages the lifecycle of project steps."""

    def __init__(self, config: AppConfig, llm: LLMClient):
        self.config = config
        self.llm = llm
        self.roadmap_file = config.state_dir / "roadmap.json"
        self._roadmap: Optional[ProjectRoadmap] = None

    def get_or_create_roadmap(self) -> ProjectRoadmap:
        """Load existing roadmap or prompt LLM to generate a new high-quality roadmap."""
        if self._roadmap:
            return self._roadmap

        if self.roadmap_file.exists():
            try:
                with open(self.roadmap_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._roadmap = ProjectRoadmap(**data)
                logger.info(f"Loaded existing roadmap with {len(self._roadmap.steps)} steps.")
                return self._roadmap
            except Exception as e:
                logger.warning(f"Could not load existing roadmap: {e}. Generating new one.")

        logger.info(f"Generating new roadmap for topic: '{self.config.project_topic}'...")
        self._roadmap = self._generate_new_roadmap()
        self._save()
        return self._roadmap

    def _generate_new_roadmap(self) -> ProjectRoadmap:
        """Call LLM to construct a realistic 15-25 step engineering roadmap."""
        system_prompt = (
            "You are a Principal Software Architect and Agentic AI Engineer. "
            "Your task is to plan a comprehensive, state-of-the-art open source repository "
            "for the requested topic. Break down the project into a realistic, sequential "
            "15 to 20 engineering steps. Each step should be discrete enough to be implemented "
            "and committed in a single daily session."
        )

        user_prompt = f"""
Project Topic: "{self.config.project_topic}"

Plan a structured roadmap where each step builds logically on top of previous steps.
Phases should cover:
1. Architecture & Repository Scaffolding (setup, configuration, typing, logger)
2. Core Models, Schemas & Protocols (Pydantic models, abstractions)
3. Core Reasoning Engine / Execution Loop (ReAct / Plan-and-Solve / Graph orchestration)
4. Tool System & Registry (dynamic tool execution, validation, error recovery)
5. Memory & Context Management (short-term buffer, long-term vector store / SQLite state)
6. Agent Collaboration & Multi-Agent Communication (message bus, handoffs, supervisor)
7. Self-Reflection, Evaluation & Error Healing
8. Production Interface (rich CLI / FastAPI server / demo scripts)
9. Automated Unit & Integration Tests (pytest test suite)
10. Documentation, Examples, Architecture Diagrams & Tutorials

Return JSON format:
{{
  "topic": "{self.config.project_topic}",
  "steps": [
    {{
      "step_number": 1,
      "id": "step_01",
      "phase": "Architecture & Repository Scaffolding",
      "title": "Short title (e.g. Repository Scaffolding, Package Setup, and Core Config)",
      "description": "Specific implementation details and requirements for this step.",
      "target_files": ["src/core/config.py", "pyproject.toml", "README.md"]
    }}
  ]
}}
"""

        res = self.llm.generate_json(system_prompt, user_prompt)
        steps_data = res.get("steps", [])

        steps: List[RoadmapStep] = []
        for idx, item in enumerate(steps_data, 1):
            steps.append(
                RoadmapStep(
                    step_number=idx,
                    id=item.get("id", f"step_{idx:02d}"),
                    phase=item.get("phase", "Implementation"),
                    title=item.get("title", f"Milestone {idx}"),
                    description=item.get("description", ""),
                    target_files=item.get("target_files", []),
                    status="pending"
                )
            )

        roadmap = ProjectRoadmap(
            topic=self.config.project_topic,
            created_at=datetime.now(timezone.utc).isoformat(),
            total_steps=len(steps),
            steps=steps,
        )
        return roadmap

    def get_next_pending_step(self) -> Optional[RoadmapStep]:
        """Find the earliest step that hasn't been completed yet."""
        roadmap = self.get_or_create_roadmap()
        for step in roadmap.steps:
            if step.status == "pending":
                return step
        return None

    def mark_step_completed(self, step_id: str, commit_hash: Optional[str] = None) -> None:
        """Mark step as finished and persist state."""
        roadmap = self.get_or_create_roadmap()
        for step in roadmap.steps:
            if step.id == step_id:
                step.status = "completed"
                step.completed_at = datetime.now(timezone.utc).isoformat()
                step.commit_hash = commit_hash
                break
        self._save()

    def get_progress(self) -> Dict[str, Any]:
        """Calculate progress metrics."""
        roadmap = self.get_or_create_roadmap()
        total = len(roadmap.steps)
        completed = sum(1 for s in roadmap.steps if s.status == "completed")
        pct = (completed / total * 100) if total > 0 else 0.0
        return {
            "total": total,
            "completed": completed,
            "pending": total - completed,
            "percentage": round(pct, 1),
            "is_finished": completed == total,
        }

    def reset(self) -> None:
        """Clear state to allow re-initialization."""
        if self.roadmap_file.exists():
            self.roadmap_file.unlink()
        self._roadmap = None

    def _save(self) -> None:
        """Write roadmap to disk."""
        if not self._roadmap:
            return
        with open(self.roadmap_file, "w", encoding="utf-8") as f:
            json.dump(self._roadmap.model_dump(), f, indent=2)
