"""Reflection loop that orchestrates critique, healing, and retry.

The :class:`ReflectionEvaluator` is the top-level entry point for the
self-reflection subsystem. Given a goal and an initial agent output, it:

1. Runs the :class:`~agent_engine.reflection.critic.CriticAgent` to score the output.
2. If the verdict is not acceptable, delegates to the
   :class:`~agent_engine.reflection.healer.ErrorHealer` to build a corrective
   prompt and retry the step.
3. Repeats until the output is accepted, the retry budget is exhausted, or a
   hard error occurs.
4. Records every attempt in :class:`~agent_engine.reflection.healer.ReflectionStats`
   so the engine can adapt its behaviour over time.

The evaluator is intentionally decoupled from any specific agent: it accepts a
``step_executor`` callable that knows how to (re-)run the step, keeping the
reflection logic reusable across agents.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agent_engine.reflection.critic import CriticAgent, CriticVerdict, VerdictStatus
from agent_engine.reflection.healer import ErrorHealer, HealingAttempt, ReflectionStats

logger = logging.getLogger(__name__)

# Type alias for the callable that (re-)executes a workflow step.
# It receives the goal, the current (possibly corrected) prompt/context, and
# returns the new output string.
StepExecutor = Callable[[str, str], Awaitable[str]]


@dataclass
class ReflectionResult:
    """Final outcome of a reflection cycle.

    Attributes
    ----------
    goal:
        The original goal that was being evaluated.
    final_output:
        The best output produced (accepted or the last attempt).
    verdict:
        The final :class:`CriticVerdict` for ``final_output``.
    attempts:
        Every :class:`HealingAttempt` made during the cycle.
    succeeded:
        ``True`` if the final verdict is acceptable.
    elapsed_seconds:
        Wall-clock time spent in the reflection cycle.
    """

    goal: str
    final_output: str
    verdict: CriticVerdict
    attempts: List[HealingAttempt] = field(default_factory=list)
    succeeded: bool = False
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        return {
            "goal": self.goal,
            "final_output": self.final_output,
            "verdict": self.verdict.to_dict(),
            "attempts": [a.to_dict() for a in self.attempts],
            "succeeded": self.succeeded,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


class ReflectionEvaluator:
    """Drives the reflection loop: critique -> heal -> retry.

    Parameters
    ----------
    critic:
        The :class:`CriticAgent` used to score outputs.
    healer:
        The :class:`ErrorHealer` used to build corrective prompts and retry.
    max_retries:
        Maximum number of healing/retry attempts after the initial evaluation.
    """

    def __init__(
        self,
        critic: CriticAgent,
        healer: ErrorHealer,
        max_retries: int = 3,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._critic = critic
        self._healer = healer
        self._max_retries = max_retries

    async def evaluate(
        self,
        goal: str,
        initial_output: str,
        step_executor: StepExecutor,
        context: Optional[str] = None,
    ) -> ReflectionResult:
        """Run the full reflection cycle for a single step.

        Parameters
        ----------
        goal:
            The original user goal.
        initial_output:
            The output produced by the agent on the first attempt.
        step_executor:
            An async callable ``(goal, corrective_context) -> output`` used to
            re-run the step with updated context.
        context:
            Optional extra context passed to the critic.

        Returns
        -------
        ReflectionResult
            The final outcome, including all attempts and the final verdict.
        """
        start = time.monotonic()
        attempts: List[HealingAttempt] = []
        current_output = initial_output
        corrective_context = context or ""

        # --- Initial critique ------------------------------------------- #
        verdict = await self._critic.critique(goal, current_output, corrective_context)
        logger.info(
            "Initial critique: status=%s score=%.2f", verdict.status.value, verdict.score
        )

        retry_count = 0
        while not verdict.is_pass and retry_count < self._max_retries:
            retry_count += 1
            logger.info(
                "Reflection retry %d/%d: building corrective prompt.",
                retry_count,
                self._max_retries,
            )

            # Build a corrective prompt from the critic's feedback.
            corrective_prompt = self._healer.build_corrective_prompt(
                goal=goal,
                previous_output=current_output,
                verdict=verdict,
                attempt_number=retry_count,
            )

            # Execute the step with the corrective context.
            try:
                current_output = await step_executor(goal, corrective_prompt)
            except Exception as exc:  # noqa: BLE001
                logger.error("Step execution failed during retry %d: %s", retry_count, exc)
                attempts.append(
                    HealingAttempt(
                        attempt_number=retry_count,
                        corrective_prompt=corrective_prompt,
                        output="",
                        success=False,
                        error=str(exc),
                    )
                )
                # Record the failure pattern for adaptive behaviour.
                self._healer.stats.record_failure(
                    error_type="execution_error",
                    detail=str(exc),
                )
                # A hard execution error aborts the loop.
                break

            # Re-critique the new output.
            verdict = await self._critic.critique(goal, current_output, corrective_context)
            logger.info(
                "Retry %d critique: status=%s score=%.2f",
                retry_count,
                verdict.status.value,
                verdict.score,
            )

            attempts.append(
                HealingAttempt(
                    attempt_number=retry_count,
                    corrective_prompt=corrective_prompt,
                    output=current_output,
                    success=verdict.is_pass,
                    error="" if verdict.is_pass else verdict.reasoning,
                )
            )

            # Track success/failure for adaptive behaviour.
            if verdict.is_pass:
                self._healer.stats.record_success()
            else:
                self._healer.stats.record_failure(
                    error_type=verdict.status.value,
                    detail=verdict.reasoning,
                )

        succeeded = verdict.is_pass
        elapsed = time.monotonic() - start

        result = ReflectionResult(
            goal=goal,
            final_output=current_output,
            verdict=verdict,
            attempts=attempts,
            succeeded=succeeded,
            elapsed_seconds=elapsed,
        )

        logger.info(
            "Reflection complete: succeeded=%s attempts=%d elapsed=%.3fs",
            succeeded,
            len(attempts),
            elapsed,
        )
        return result

    async def evaluate_with_tool_failure(
        self,
        goal: str,
        initial_output: str,
        tool_error: str,
        step_executor: StepExecutor,
        context: Optional[str] = None,
    ) -> ReflectionResult:
        """Handle the case where a tool call failed (not just a bad output).

        This is a convenience wrapper that pre-seeds the corrective context with
        the tool error so the agent knows *why* the step failed.

        Parameters
        ----------
        goal:
            The original user goal.
        initial_output:
            The (partial or empty) output produced before the tool failure.
        tool_error:
            The error message from the failed tool call.
        step_executor:
            Async callable to re-run the step.
        context:
            Optional extra context.

        Returns
        -------
        ReflectionResult
        """
        # Record the tool failure pattern.
        self._healer.stats.record_failure(
            error_type="tool_failure",
            detail=tool_error,
        )

        # Prepend the tool error to the corrective context so the first
        # corrective prompt already knows about the failure.
        enriched_context = (context or "") + f"\n\nTool failure: {tool_error}"

        return await self.evaluate(
            goal=goal,
            initial_output=initial_output,
            step_executor=step_executor,
            context=enriched_context,
        )
