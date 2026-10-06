"""Parser for ReAct (Reasoning + Acting) LLM output.

The ReAct pattern expects the model to emit a structured block of the form::

    Thought: <reasoning text>
    Action: <tool_name>
    Action Input: <JSON or plain-text argument>

    Observation: <result from the tool>

This module provides :class:`ReActParser`, which extracts the ``Thought``,
``Action``, and ``Action Input`` fields from a raw LLM completion and returns
them as a structured :class:`ReActStep`.

The parser is deliberately lenient: it tolerates extra whitespace, missing
fields (marking them as ``None``), and both JSON and plain-text action inputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ReActStep:
    """A single parsed ReAct step.

    Attributes
    ----------
    thought:
        The model's reasoning text, or ``None`` if absent.
    action:
        The name of the tool the model wants to invoke, or ``None``.
    action_input:
        The parsed argument for the tool. If the raw input is valid JSON it is
        decoded into a Python object; otherwise it is kept as a string.
    raw:
        The original LLM output text, preserved for debugging.
    """

    thought: Optional[str]
    action: Optional[str]
    action_input: Any
    raw: str


class ReActParseError(Exception):
    """Raised when the LLM output cannot be parsed into a valid ReAct step."""


# Pre-compiled regex patterns for the three ReAct fields.
# Each pattern captures the value up to the next field label or end of string.
_THOUGHT_RE = re.compile(
    r"Thought:\s*(?P<value>.*?)(?=\n\s*(?:Action|Observation):|\Z)",
    re.DOTALL,
)
_ACTION_RE = re.compile(
    r"Action:\s*(?P<value>.*?)(?=\n\s*(?:Action Input|Observation):|\Z)",
    re.DOTALL,
)
_ACTION_INPUT_RE = re.compile(
    r"Action Input:\s*(?P<value>.*?)(?=\n\s*(?:Observation):|\Z)",
    re.DOTALL,
)


class ReActParser:
    """Parses raw LLM text into structured :class:`ReActStep` objects.

    The parser is stateless and thread-safe. Instantiate once and reuse across
    multiple calls.
    """

    def parse(self, raw: str) -> ReActStep:
        """Parse a raw LLM completion into a :class:`ReActStep`.

        Parameters
        ----------
        raw:
            The raw text output from the LLM.

        Returns
        -------
        ReActStep
            The parsed step. Missing fields are set to ``None`` (for
            ``thought``/``action``) or an empty string (for ``action_input``).

        Raises
        ------
        ReActParseError
            If the input is empty or contains no recognisable ReAct fields at
            all.
        """
        if not raw or not raw.strip():
            raise ReActParseError("Cannot parse empty LLM output.")

        thought = self._extract(_THOUGHT_RE, raw)
        action = self._extract(_ACTION_RE, raw)
        action_input_raw = self._extract(_ACTION_INPUT_RE, raw)

        # If none of the three fields were found, the output is not ReAct-formatted.
        if thought is None and action is None and action_input_raw is None:
            raise ReActParseError(
                "LLM output does not contain any ReAct fields "
                "(Thought / Action / Action Input)."
            )

        action_input = self._decode_action_input(action_input_raw)

        return ReActStep(
            thought=thought,
            action=action,
            action_input=action_input,
            raw=raw,
        )

    @staticmethod
    def _extract(pattern: re.Pattern[str], text: str) -> Optional[str]:
        """Extract and strip the named group ``value`` from ``text``."""
        match = pattern.search(text)
        if match is None:
            return None
        value = match.group("value").strip()
        return value if value else None

    @staticmethod
    def _decode_action_input(raw_input: Optional[str]) -> Any:
        """Decode the action input, attempting JSON first.

        If the input is valid JSON (object, array, number, or string) it is
        returned as the corresponding Python type. Otherwise the raw string is
        returned unchanged.
        """
        if raw_input is None:
            return ""

        stripped = raw_input.strip()
        if not stripped:
            return ""

        # Attempt JSON decoding. Only try if the input looks like a JSON
        # container or a quoted string to avoid mis-parsing plain text.
        if stripped[0] in "{[\"":
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                pass

        return stripped
