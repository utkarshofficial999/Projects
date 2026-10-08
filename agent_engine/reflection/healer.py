"""Automatic error healing: corrective prompts, retries, and failure tracking.

This module provides:

- :class:`HealingAttempt` -- a record of a single retry attempt.
- :class:`ReflectionStats` -- tracks success rates and failure patterns so the
  engine can adapt (e.g. increase retries for a flaky tool, or flag a step that
  consistently fails).
- :class:`ErrorHealer` -- builds corrective prompts from critic feedback and
  exposes the retry / stats machinery.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_engine.reflection.critic import CriticVerdict

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data records
# --------------------------------------------------------------------------- #
@dataclass
class HealingAttempt:
    """A single retry attempt within a reflection cycle.

    Attributes
    ----------
    attempt_number:
        1-based index of this attempt.
    corrective_prompt:
        The corrective prompt that was sent to the agent.
    output:
        The output produced by the agent in response.
    success:
        ``True`` if the subsequent critique accepted the output.
    error:
        Error message if the attempt failed (empty on success).
    timestamp:
        Unix timestamp when the attempt was made.
    """

    attempt_number: int
    corrective_prompt: str
    output: str
    success: bool
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "attempt_number": self.attempt_number,
            "corrective_prompt": self.corrective_prompt,
            "output": self.output,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass
class ReflectionStats:
    """Tracks success rates and failure patterns for adaptive behaviour.

    The stats are maintained in-memory and can be serialised for persistence.
    They allow the engine to:

    - Detect steps that consistently fail (high failure rate).
    - Identify the most common failure types (e.g. ``tool_failure`` vs
      ``rejected``).
    - Adapt retry budgets: steps with a high historical success rate get fewer
      retries; flaky steps get more.
    """

    def __init__(self) -> None:
        self._total_attempts: int = 0
        self._total_successes: int = 0
        self._failure_counter: Counter = Counter()
        self._failure_details: Dict[str, List[str]] = defaultdict(list)
        self._per_step_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"attempts": 0, "successes": 0}
        )

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def record_success(self, step_id: Optional[str] = None) -> None:
        """Record a successful evaluation / retry."""
        self._total_attempts += 1
        self._total_successes += 1
        if step_id:
            self._per_step_stats[step_id]["attempts"] += 1
            self._per_step_stats[step_id]["successes"] += 1

    def record_failure(
        self,
        error_type: str,
        detail: str = "",
        step_id: Optional[str] = None,
    ) -> None:
        """Record a failed evaluation / retry.

        Parameters
        ----------
        error_type:
            A short category label (e.g. ``"tool_failure"``, ``"rejected"``,
            ``"execution_error"``).
        detail:
            Human-readable detail (error message, critic reasoning, etc.).
        step_id:
            Optional identifier of the workflow step, for per-step tracking.
        """
        self._total_attempts += 1
        self._failure_counter[error_type] += 1
        if detail:
            # Cap detail storage to avoid unbounded memory growth.
            if len(self._failure_details[error_type]) < 100:
                self._failure_details[error_type].append(detail)
        if step_id:
            self._per_step_stats[step_id]["attempts"] += 1

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def total_attempts(self) -> int:
        """Total number of evaluations / retries recorded."""
        return self._total_attempts

    @property
    def total_successes(self) -> int:
        """Total number of successful evaluations / retries."""
        return self._total_successes

    @property
    def success_rate(self) -> float:
        """Overall success rate in ``[0.0, 1.0]``. Returns 0.0 if no attempts."""
        if self._total_attempts == 0:
            return 0.0
        return self._total_successes / self._total_attempts

    @property
    def failure_patterns(self) -> Dict[str, int]:
        """Mapping of failure type -> count, sorted by frequency (desc)."""
        return dict(self._failure_counter.most_common())

    def get_step_success_rate(self, step_id: str) -> float:
        """Success rate for a specific step. Returns 0.0 if no data."""
        stats = self._per_step_stats.get(step_id)
        if not stats or stats["attempts"] == 0:
            return 0.0
        return stats["successes"] / stats["attempts"]

    def get_adaptive_retry_budget(self, step_id: str, base: int = 3) -> int:
        """Compute an adaptive retry budget for a step.

        Steps with a high historical success rate get fewer retries; steps
        that frequently fail get more (up to a cap).

        Parameters
        ----------
        step_id:
            Identifier of the step.
        base:
            The default retry budget.

        Returns
        -------
        int
            The adjusted retry budget (clamped to ``[1, base * 2]``).
        """
        rate = self.get_step_success_rate(step_id)
        if rate >= 0.9:
            # Very reliable step: use fewer retries.
            return max(1, base - 1)
        if rate < 0.5 and self._per_step_stats[step_id]["attempts"] >= 3:
            # Flaky step: allow more retries.
            return min(base * 2, base + 2)
        return base

    def to_dict(self) -> Dict[str, Any]:
        """Serialise stats to a JSON-friendly dict."""
        return {
            "total_attempts": self._total_attempts,
            "total_successes": self._total_successes,
            "success_rate": round(self.success_rate, 4),
            "failure_patterns": self.failure_patterns,
            "per_step": {
                sid: {
                    "attempts": s["attempts"],
                    "successes": s["successes"],
                    "success_rate": round(
                        s["successes"] / s["attempts"], 4
                    )
                    if s["attempts"]
                    else 0.0,
                }
                for sid, s in self._per_step_stats.items()
            },
        }


# --------------------------------------------------------------------------- #
# Corrective prompt template
# --------------------------------------------------------------------------- #
_CORRECTIVE_PROMPT_TEMPLATE = """\
You are retrying a step that did not fully satisfy the goal. Below is the
feedback from the critic. Use it to produce a corrected, complete output.

## Original Goal
{goal}

## Previous (insufficient) Output
{previous_output}

## Critic Feedback
Status: {status}
Score: {score}
Reasoning: {reasoning}

Issues identified:
{issues}

Suggestions:
{suggestions}

## Instructions
- Address every issue listed above.
- Do not repeat the same mistake.
- Produce a complete, self-contained output that fully satisfies the goal.
- This is attempt {attempt_number}. Be thorough.
"""


class ErrorHealer:
    """Builds corrective prompts and manages retry / stats for error healing.

    Parameters
    ----------
    stats:
        Optional pre-existing :class:`ReflectionStats`. If not provided, a new
        one is created.
    """

    def __init__(self, stats: Optional[ReflectionStats] = None) -> None:
        self.stats = stats if stats is not None else ReflectionStats()

    def build_corrective_prompt(
        self,
        goal: str,
        previous_output: str,
        verdict: CriticVerdict,
        attempt_number: int = 1,
    ) -> str:
        """Build a corrective prompt from critic feedback.

        Parameters
        ----------
        goal:
            The original goal.
        previous_output:
            The output that was rejected / needs revision.
        verdict:
            The :class:`CriticVerdict` containing issues and suggestions.
        attempt_number:
            Which retry attempt this is (1-based).

        Returns
        -------
        str
            A prompt that instructs the agent to fix the identified issues.
        """
        issues_text = "\n".join(f"- {issue}" for issue in verdict.issues) or "- (none listed)"
        suggestions_text = (
            "\n".join(f"- {s}" for s in verdict.suggestions) or "- (none listed)"
        )

        return _CORRECTIVE_PROMPT_TEMPLATE.format(
            goal=goal,
            previous_output=previous_output or "(empty)",
            status=verdict.status.value,
            score=f"{verdict.score:.2f}",
            reasoning=verdict.reasoning,
            issues=issues_text,
            suggestions=suggestions_text,
            attempt_number=attempt_number,
        )

    def record_tool_failure(self, tool_name: str, error: str) -> None:
        """Record a tool-call failure for adaptive tracking.

        Parameters
        ----------
        tool_name:
            Name of the tool that failed.
        error:
            The error message.
        """
        self.stats.record_failure(
            error_type=f"tool_failure:{tool_name}",
            detail=error,
        )
        logger.warning("Tool '%s' failed: %s", tool_name, error)

    def record_step_outcome(
        self,
        step_id: str,
        success: bool,
        error_type: str = "",
        detail: str = "",
    ) -> None:
        """Record the outcome of a workflow step for per-step adaptive tracking.

        Parameters
        ----------
        step_id:
            Identifier of the step.
        success:
            Whether the step succeeded.
        error_type:
            Failure category (only used when ``success`` is ``False``).
        detail:
            Human-readable detail.
        """
        if success:
            self.stats.record_success(step_id=step_id)
        else:
            self.stats.record_failure(
                error_type=error_type or "unknown",
                detail=detail,
                step_id=step_id,
            )
