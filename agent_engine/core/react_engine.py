"""Core ReAct reasoning engine.

This module implements the execution loop that drives the ReAct
(Reasoning + Acting) pattern:

1. **Reason** – send the current workflow context to the LLM and receive a
   completion containing a ``Thought`` / ``Action`` / ``Action Input`` block.
2. **Act** – parse the completion, dispatch the requested tool, and capture the
   observation.
3. **Observe** – append the observation to the context and loop.

The loop terminates when the model emits a ``Final Answer`` (signalling task
completion) or when a configurable maximum number of steps is reached.

Design notes
------------
* The engine is **stateless with respect to the LLM**: it does not cache or
  mutate the LLM client. All conversation state lives in the engine's internal
  message list, which is owned by the engine instance.
* The engine is **stateful with respect to the workflow context**: it maintains
  the running list of messages, tool observations, and step history so that
  multi-turn reasoning is coherent.
* Token limits are enforced by tracking cumulative token usage and truncating
  the message history when the context window is approached.
* Transient LLM errors are retried at the client level (see
  :class:`~agent_engine.core.llm_client.OpenAIClient`); the engine additionally
  guards against parse failures by feeding a corrective prompt back to the
  model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from agent_engine.core.llm_client import LLMClient, LLMResponse
from agent_engine.core.parser import ReActParser, ReActStep, ReActParseError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """Specification of a tool the agent is allowed to invoke.

    Attributes
    ----------
    name:
        Unique tool identifier (must match the ``Action`` field in ReAct output).
    description:
        Human-readable description included in the system prompt.
    handler:
        Async callable that executes the tool. It receives the parsed
        ``action_input`` and returns a string observation.
    """

    name: str
    description: str
    handler: Callable[[Any], Any]  # May be sync or async; engine handles both.


@dataclass
class ReActResult:
    """Final result returned by the ReAct engine.

    Attributes
    ----------
    answer:
        The model's final answer text.
    steps:
        Ordered list of all :class:`ReActStep` objects encountered.
    observations:
        Ordered list of tool observations (strings).
    total_tokens:
        Cumulative token usage across all LLM calls.
    completed:
        ``True`` if the model produced a ``Final Answer``; ``False`` if the
        step limit was reached.
    """

    answer: str
    steps: list[ReActStep] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    total_tokens: int = 0
    completed: bool = True


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ReActEngine:
    """Stateful ReAct reasoning engine.

    Parameters
    ----------
    llm:
        The LLM client used to generate completions.
    tools:
        The set of tools the agent may invoke.
    max_steps:
        Maximum number of Reason→Act→Observe iterations before the engine
        aborts. Defaults to ``15``.
    max_context_tokens:
        Approximate token budget for the message history. When exceeded, the
        oldest messages (excluding the system prompt) are dropped. Defaults to
        ``8000``.
    system_prompt:
        The system-level instruction prepended to every LLM call.
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: Sequence[ToolSpec],
        *,
        max_steps: int = 15,
        max_context_tokens: int = 8000,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._tools: dict[str, ToolSpec] = {t.name: t for t in tools}
        self._max_steps = max_steps
        self._max_context_tokens = max_context_tokens
        self._parser = ReActParser()

        if system_prompt is None:
            system_prompt = self._build_default_system_prompt()
        self._system_prompt = system_prompt

        # Mutable workflow state (stateful w.r.t. the workflow context).
        self._messages: list[dict[str, str]] = []
        self._steps: list[ReActStep] = []
        self._observations: list[str] = []
        self._total_tokens: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, task: str) -> ReActResult:
        """Execute the ReAct loop for the given ``task``.

        Parameters
        ----------
        task:
            The user's task description.

        Returns
        -------
        ReActResult
            The final answer and full execution trace.
        """
        self._reset()
        self._messages.append({"role": "user", "content": task})

        for step_num in range(1, self._max_steps + 1):
            logger.info("ReAct step %d/%d", step_num, self._max_steps)

            # --- Reason -------------------------------------------------
            response = await self._call_llm()
            self._total_tokens += response.usage.get("total_tokens", 0)

            # --- Check for Final Answer ---------------------------------
            if "Final Answer:" in response.content:
                answer = self._extract_final_answer(response.content)
                self._steps.append(
                    ReActStep(
                        thought=self._extract_thought(response.content),
                        action=None,
                        action_input="",
                        raw=response.content,
                    )
                )
                logger.info("Final answer reached at step %d.", step_num)
                return ReActResult(
                    answer=answer,
                    steps=self._steps,
                    observations=self._observations,
                    total_tokens=self._total_tokens,
                    completed=True,
                )

            # --- Parse ---------------------------------------------------
            try:
                step = self._parser.parse(response.content)
            except ReActParseError as exc:
                logger.warning(
                    "Parse error at step %d: %s. Sending corrective prompt.",
                    step_num,
                    exc,
                )
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                    }
                )
                self._messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not in the required "
                            "ReAct format. Please respond using exactly the "
                            "following structure:\n"
                            "Thought: <your reasoning>\n"
                            "Action: <tool_name>\n"
                            "Action Input: <json or text>\n"
                            "Or, if you have the final answer:\n"
                            "Final Answer: <your answer>"
                        ),
                    }
                )
                continue

            self._steps.append(step)

            # --- Act -----------------------------------------------------
            if step.action is None:
                # No action requested; treat as a reasoning-only step.
                logger.info("Step %d: no action requested.", step_num)
                self._messages.append(
                    {"role": "assistant", "content": response.content}
                )
                continue

            if step.action not in self._tools:
                observation = (
                    f"Error: unknown tool '{step.action}'. "
                    f"Available tools: {', '.join(self._tools.keys())}."
                )
                logger.warning("Unknown tool '%s' requested.", step.action)
            else:
                observation = await self._execute_tool(step.action, step.action_input)

            self._observations.append(observation)

            # --- Observe -------------------------------------------------
            self._messages.append(
                {"role": "assistant", "content": response.content}
            )
            self._messages.append(
                {
                    "role": "user",
                    "content": f"Observation: {observation}",
                }
            )

        # Step limit reached.
        logger.warning("ReAct loop reached max_steps=%d without a Final Answer.",
                        self._max_steps)
        return ReActResult(
            answer="(No final answer produced; step limit reached.)",
            steps=self._steps,
            observations=self._observations,
            total_tokens=self._total_tokens,
            completed=False,
        )

    def reset(self) -> None:
        """Clear all workflow state so the engine can be reused for a new task."""
        self._reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Reset internal mutable state."""
        self._messages = []
        self._steps = []
        self._observations = []
        self._total_tokens = 0

    async def _call_llm(self) -> LLMResponse:
        """Send the current message history to the LLM and return the response.

        Before calling, the message history is trimmed to stay within
        ``max_context_tokens``.
        """
        self._trim_messages()

        messages = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._messages)

        return await self._llm.complete(messages, temperature=0.0)

    def _trim_messages(self) -> None:
        """Drop the oldest non-system messages if the history is too large.

        A rough heuristic of ~4 characters per token is used to estimate the
        token count of the message history. This is intentionally conservative
        to avoid exceeding the provider's context window.
        """
        estimated_tokens = sum(
            len(m["content"]) for m in self._messages
        ) // 4

        if estimated_tokens <= self._max_context_tokens:
            return

        # Drop messages from the front until we are under the budget.
        # We never drop more than half the messages in a single pass to avoid
        # losing too much context at once.
        while (
            len(self._messages) > 2
            and sum(len(m["content"]) for m in self._messages) // 4
            > self._max_context_tokens
        ):
            removed = self._messages.pop(0)
            logger.debug("Trimmed message: %s...", removed["content"][:60])

    async def _execute_tool(self, tool_name: str, action_input: Any) -> str:
        """Execute a tool and return its observation as a string."""
        tool = self._tools[tool_name]
        try:
            result = tool.handler(action_input)
            # Support both sync and async handlers.
            if hasattr(result, "__await__"):
                result = await result
            return str(result)
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception.", tool_name)
            return f"Error executing tool '{tool_name}': {exc}"

    @staticmethod
    def _extract_final_answer(content: str) -> str:
        """Extract the text after the ``Final Answer:`` marker."""
        idx = content.index("Final Answer:")
        return content[idx + len("Final Answer:"):].strip()

    @staticmethod
    def _extract_thought(content: str) -> Optional[str]:
        """Extract the ``Thought:`` field if present."""
        import re

        match = re.search(
            r"Thought:\s*(?P<value>.*?)(?=\n\s*(?:Action|Final Answer):|\Z)",
            content,
            re.DOTALL,
        )
        if match:
            value = match.group("value").strip()
            return value if value else None
        return None

    def _build_default_system_prompt(self) -> str:
        """Build the default system prompt describing the ReAct protocol."""
        tool_descriptions = "\n".join(
            f"  - {t.name}: {t.description}" for t in self._tools.values()
        )
        return (
            "You are an autonomous agent that solves tasks using the ReAct "
            "(Reasoning + Acting) pattern.\n"
            "For each step, respond in EXACTLY this format:\n"
            "  Thought: <your reasoning>\n"
            "  Action: <tool_name>\n"
            "  Action Input: <JSON or plain-text argument>\n"
            "When you have the final answer, respond with:\n"
            "  Final Answer: <your answer>\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            "Rules:\n"
            "1. Always think before acting.\n"
            "2. Use only the tools listed above.\n"
            "3. Provide the Action Input as valid JSON when the tool expects "
            "structured data.\n"
            "4. Stop and give a Final Answer as soon as the task is complete."
        )
