"""Abstract LLM client interface and a concrete OpenAI implementation.

This module defines the contract every LLM provider must satisfy
(:class:`LLMClient`) and provides a production-ready OpenAI-backed
implementation (:class:`OpenAIClient`).

Design notes
------------
* The interface is intentionally minimal (a single ``complete`` method) so that
  higher-level reasoning engines remain decoupled from any specific provider.
* The OpenAI client is *stateless* with respect to the model: it holds no
  conversation history. All state (messages, tool results, etc.) is owned by the
  caller (the ReAct engine), which keeps the client trivially reusable and
  thread-safe.
* Retries and timeouts are handled here so that transient network failures do
  not bubble up into the reasoning loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    """Immutable container for a single LLM completion.

    Attributes
    ----------
    content:
        The raw text produced by the model.
    model:
        The model identifier that produced the response.
    usage:
        Token usage statistics (``prompt_tokens``, ``completion_tokens``,
        ``total_tokens``) if the provider reports them.
    raw:
        The raw provider payload, preserved for debugging/inspection.
    """

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: Optional[dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract base class for all LLM providers.

    Subclasses must implement :meth:`complete`. The method is asynchronous so
    that the ReAct engine can be integrated into async runtimes without
    blocking the event loop.
    """

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion for the given chat ``messages``.

        Parameters
        ----------
        messages:
            A sequence of message dicts, each with ``role`` and ``content`` keys.
        temperature:
            Sampling temperature. Defaults to ``0.0`` for deterministic output.
        max_tokens:
            Optional cap on the number of completion tokens.
        **kwargs:
            Provider-specific extra parameters.

        Returns
        -------
        LLMResponse
            The model's completion wrapped in a structured response object.

        Raises
        ------
        LLMError
            If the request ultimately fails after all retries.
        """
        raise NotImplementedError


class LLMError(Exception):
    """Raised when an LLM request fails after exhausting all retries."""


class OpenAIClient(LLMClient):
    """Concrete LLM client backed by the OpenAI Chat Completions API.

    Parameters
    ----------
    api_key:
        OpenAI API key. If omitted, the ``OPENAI_API_KEY`` environment variable
        is used.
    model:
        Default model identifier (e.g. ``"gpt-4o"``).
    max_retries:
        Number of retry attempts for transient errors (429, 5xx, timeouts).
    timeout:
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Pass `api_key` or set "
                "the OPENAI_API_KEY environment variable."
            )
        self._model = model
        self._max_retries = max_retries
        self._timeout = timeout

    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call the OpenAI Chat Completions endpoint with retry logic."""
        import httpx  # Imported lazily to avoid a hard dependency at import time.

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self._model),
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, headers=headers, json=payload)

                    # Retry on rate-limit or server errors.
                    if response.status_code in (429, 500, 502, 503, 504):
                        retry_after = float(response.headers.get("Retry-After", "1.0"))
                        logger.warning(
                            "OpenAI returned %s (attempt %d/%d). Retrying in %.1fs",
                            response.status_code,
                            attempt,
                            self._max_retries,
                            retry_after,
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    response.raise_for_status()
                    data = response.json()

                    choice = data["choices"][0]
                    content = choice["message"]["content"] or ""
                    usage = data.get("usage", {})

                    return LLMResponse(
                        content=content,
                        model=data.get("model", self._model),
                        usage={
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                        raw=data,
                    )

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                logger.warning(
                    "OpenAI request failed (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    exc,
                )
                await asyncio.sleep(min(2 ** attempt, 10))

        raise LLMError(
            f"OpenAI request failed after {self._max_retries} attempts: {last_error}"
        ) from last_error


class MockLLMClient(LLMClient):
    """A deterministic in-memory LLM client for testing and offline development.

    The client returns pre-canned responses in order. When the queue is
    exhausted it repeats the last response. This makes it easy to script
    multi-step ReAct conversations in unit tests.

    Parameters
    ----------
    responses:
        An ordered list of response strings to return on successive calls.
    """

    def __init__(self, responses: Optional[Sequence[str]] = None) -> None:
        self._responses: list[str] = list(responses) if responses else []
        self._call_index = 0
        self._call_log: list[Sequence[dict[str, str]]] = []

    @property
    def call_log(self) -> list[Sequence[dict[str, str]]]:
        """Return the list of message sequences that were sent to the client."""
        return list(self._call_log)

    def reset(self) -> None:
        """Reset the call index so the response queue can be replayed."""
        self._call_index = 0
        self._call_log.clear()

    async def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return the next pre-canned response."""
        self._call_log.append(list(messages))

        if self._call_index < len(self._responses):
            content = self._responses[self._call_index]
        elif self._responses:
            content = self._responses[-1]
        else:
            content = ""

        self._call_index += 1
        return LLMResponse(
            content=content,
            model="mock-model",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
