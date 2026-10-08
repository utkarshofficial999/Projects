"""Critic agent that evaluates agent output against the original goal.

The :class:`CriticAgent` is a specialised, read-only agent whose sole job is to
judge whether a produced artefact (text, code, plan, tool result, ...) actually
satisfies the user's goal. It never mutates state; it only emits a structured
:class:`CriticVerdict` that downstream components (the evaluator and the healer)
can act upon.

Design notes
------------
- **Deterministic fallback.** When no LLM client is available (e.g. in unit
  tests or offline mode) the critic degrades to a lightweight heuristic so the
  rest of the pipeline keeps working.
- **Structured output.** The LLM is asked to return strict JSON so the verdict
  can be parsed reliably. A tolerant parser recovers from minor formatting
  drift.
- **No side effects.** The critic never calls tools or writes to memory; it is
  a pure function of ``(goal, output, context)``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VerdictStatus(str, Enum):
    """Outcome of a single critique pass."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    ERROR = "error"


@dataclass(frozen=True)
class CriticVerdict:
    """Immutable result of a critique.

    Attributes
    ----------
    status:
        High-level outcome (accepted / rejected / needs_revision / error).
    score:
        Normalised quality score in ``[0.0, 1.0]``.
    reasoning:
        Human-readable explanation of the decision.
    issues:
        Concrete, actionable problems that were detected.
    suggestions:
        Concrete suggestions for how to fix the issues.
    raw:
        The raw (possibly un-parsed) model output, kept for debugging.
    """

    status: VerdictStatus
    score: float
    reasoning: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    raw: str = ""

    @property
    def is_pass(self) -> bool:
        """``True`` when the output is acceptable as-is."""
        return self.status in (VerdictStatus.ACCEPTED, VerdictStatus.NEEDS_REVISION) and self.score >= 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dict (used for logging / persistence)."""
        return {
            "status": self.status.value,
            "score": self.score,
            "reasoning": self.reasoning,
            "issues": list(self.issues),
            "suggestions": list(self.suggestions),
        }


# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #
_CRITIC_SYSTEM_PROMPT = """\
You are a meticulous, impartial critic in a multi-agent software workflow.
Your only job is to evaluate whether a produced output satisfies the stated goal.

You MUST respond with a single JSON object and nothing else, using exactly this schema:
{
  "status": "accepted" | "rejected" | "needs_revision",
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<concise explanation>",
  "issues": ["<specific problem 1>", "..."],
  "suggestions": ["<specific fix 1>", "..."]
}

Rules:
- Be specific and actionable; avoid vague praise or criticism.
- If the output is correct and complete, use "accepted" with a high score.
- If it is close but has fixable gaps, use "needs_revision".
- If it is fundamentally wrong or off-topic, use "rejected".
- Never invent requirements that are not implied by the goal.
"""

_CRITIC_USER_TEMPLATE = """\
## Goal
{goal}

## Output to evaluate
{output}

## Additional context
{context}

Evaluate the output against the goal and return the JSON verdict.
"""


class CriticAgent:
    """LLM-backed critic that scores an agent's output against a goal.

    Parameters
    ----------
    llm_client:
        Any object exposing ``async complete(prompt: str, system: str) -> str``.
        When ``None``, the critic uses a deterministic heuristic fallback.
    min_accept_score:
        Minimum score required for an ``ACCEPTED`` verdict.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        min_accept_score: float = 0.7,
    ) -> None:
        self._llm = llm_client
        self._min_accept_score = min_accept_score

    async def critique(
        self,
        goal: str,
        output: str,
        context: Optional[str] = None,
    ) -> CriticVerdict:
        """Evaluate ``output`` against ``goal`` and return a :class:`CriticVerdict`.

        Parameters
        ----------
        goal:
            The original user goal / task description.
        output:
            The artefact produced by another agent.
        context:
            Optional extra context (prior attempts, constraints, etc.).

        Returns
        -------
        CriticVerdict
            A structured, immutable verdict.
        """
        if not goal or not goal.strip():
            logger.warning("Critic received an empty goal; rejecting by default.")
            return CriticVerdict(
                status=VerdictStatus.REJECTED,
                score=0.0,
                reasoning="No goal was provided; cannot evaluate.",
                issues=["Empty goal"],
            )

        if self._llm is not None:
            try:
                return await self._critique_with_llm(goal, output, context)
            except Exception as exc:  # noqa: BLE001 - deliberate broad catch
                logger.error("LLM critique failed (%s); falling back to heuristic.", exc)
                return self._heuristic_critique(goal, output)

        return self._heuristic_critique(goal, output)

    # ------------------------------------------------------------------ #
    # LLM path
    # ------------------------------------------------------------------ #
    async def _critique_with_llm(
        self, goal: str, output: str, context: Optional[str]
    ) -> CriticVerdict:
        """Run the LLM critique and parse the structured response."""
        user_prompt = _CRITIC_USER_TEMPLATE.format(
            goal=goal,
            output=output or "(empty output)",
            context=context or "(none)",
        )
        raw = await self._llm.complete(user_prompt, system=_CRITIC_SYSTEM_PROMPT)
        return self._parse_verdict(raw)

    def _parse_verdict(self, raw: str) -> CriticVerdict:
        """Parse the model's raw text into a :class:`CriticVerdict`.

        Tolerates leading/trailing prose and markdown code fences.
        """
        data = self._extract_json(raw)
        if data is None:
            logger.warning("Could not parse critic JSON; treating as needs_revision.")
            return CriticVerdict(
                status=VerdictStatus.NEEDS_REVISION,
                score=0.4,
                reasoning="Critic returned unparseable output.",
                issues=["Unparseable critic response"],
                raw=raw,
            )

        status = self._coerce_status(data.get("status"))
        score = self._coerce_score(data.get("score"))
        reasoning = str(data.get("reasoning", "")).strip()
        issues = self._coerce_str_list(data.get("issues"))
        suggestions = self._coerce_str_list(data.get("suggestions"))

        # Reconcile status with score for consistency.
        if status == VerdictStatus.ACCEPTED and score < self._min_accept_score:
            status = VerdictStatus.NEEDS_REVISION
        if status == VerdictStatus.REJECTED and score > 0.5:
            status = VerdictStatus.NEEDS_REVISION

        return CriticVerdict(
            status=status,
            score=score,
            reasoning=reasoning or "No reasoning provided.",
            issues=issues,
            suggestions=suggestions,
            raw=raw,
        )

    # ------------------------------------------------------------------ #
    # Heuristic fallback (no LLM)
    # ------------------------------------------------------------------ #
    def _heuristic_critique(self, goal: str, output: str) -> CriticVerdict:
        """A lightweight, deterministic heuristic used when no LLM is available.

        This is intentionally simple: it checks that the output is non-empty,
        has reasonable length, and shares some topical overlap with the goal.
        It exists so the pipeline remains functional in offline / test mode.
        """
        text = (output or "").strip()
        if not text:
            return CriticVerdict(
                status=VerdictStatus.REJECTED,
                score=0.0,
                reasoning="Output is empty.",
                issues=["Empty output"],
            )

        # Token-level overlap heuristic.
        goal_tokens = set(re.findall(r"\w+", goal.lower()))
        out_tokens = set(re.findall(r"\w+", text.lower()))
        overlap = len(goal_tokens & out_tokens) / max(len(goal_tokens), 1)

        # Length sanity: very short outputs are suspicious.
        length_ok = len(text) >= 20

        score = round(min(1.0, 0.4 + 0.6 * overlap), 3)
        if not length_ok:
            score = min(score, 0.3)

        if score >= self._min_accept_score and length_ok:
            status = VerdictStatus.ACCEPTED
        elif score >= 0.4:
            status = VerdictStatus.NEEDS_REVISION
        else:
            status = VerdictStatus.REJECTED

        issues: List[str] = []
        if not length_ok:
            issues.append("Output is suspiciously short.")
        if overlap < 0.2:
            issues.append("Low topical overlap with the goal.")

        return CriticVerdict(
            status=status,
            score=score,
            reasoning=f"Heuristic score {score} (overlap={overlap:.2f}).",
            issues=issues,
            suggestions=["Expand the output to fully address the goal."] if issues else [],
        )

    # ------------------------------------------------------------------ #
    # Parsing helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of a JSON object from model output."""
        if not raw:
            return None
        # Strip markdown code fences if present.
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Try to find the first balanced {...} block.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _coerce_status(value: Any) -> VerdictStatus:
        """Map an arbitrary value onto a :class:`VerdictStatus`."""
        if isinstance(value, VerdictStatus):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            mapping = {
                "accepted": VerdictStatus.ACCEPTED,
                "reject": VerdictStatus.REJECTED,
                "rejected": VerdictStatus.REJECTED,
                "needs_revision": VerdictStatus.NEEDS_REVISION,
                "needs revision": VerdictStatus.NEEDS_REVISION,
                "error": VerdictStatus.ERROR,
            }
            if v in mapping:
                return mapping[v]
        return VerdictStatus.NEEDS_REVISION

    @staticmethod
    def _coerce_score(value: Any) -> float:
        """Coerce an arbitrary value into a float clamped to [0, 1]."""
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, score))

    @staticmethod
    def _coerce_str_list(value: Any) -> List[str]:
        """Coerce an arbitrary value into a list of strings."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value)]
