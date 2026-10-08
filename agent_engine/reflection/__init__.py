"""Self-Reflection, Evaluation & Error-Healing subsystem.

This package provides the machinery that lets the multi-agent engine
critique its own work, detect mistakes, and automatically repair them.

Public API
----------
- :class:`CriticAgent`   -- LLM-backed critic that scores an output against a goal.
- :class:`ReflectionEvaluator` -- orchestrates the reflection loop (critique -> heal -> retry).
- :class:`ErrorHealer`   -- builds corrective prompts and executes bounded retries.
- :class:`ReflectionStats` -- tracks success rates and failure patterns.

Example
-------
>>> from agent_engine.reflection import ReflectionEvaluator
>>> evaluator = ReflectionEvaluator(llm_client, max_retries=3)
>>> result = await evaluator.evaluate(goal, agent_output)
"""

from agent_engine.reflection.critic import CriticAgent, CriticVerdict
from agent_engine.reflection.evaluator import ReflectionEvaluator, ReflectionResult
from agent_engine.reflection.healer import ErrorHealer, HealingAttempt, ReflectionStats

__all__ = [
    "CriticAgent",
    "CriticVerdict",
    "ReflectionEvaluator",
    "ReflectionResult",
    "ErrorHealer",
    "HealingAttempt",
    "ReflectionStats",
]
