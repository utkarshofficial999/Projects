"""Unit tests for agent_engine.core.llm_client."""

from __future__ import annotations

import asyncio
import pytest

from agent_engine.core.llm_client import (
    LLMClient,
    LLMError,
    LLMResponse,
    MockLLMClient,
    OpenAIClient,
)


class TestLLMResponse:
    """Tests for the LLMResponse dataclass."""

    def test_basic_construction(self) -> None:
        resp = LLMResponse(content="hello", model="gpt-4o")
        assert resp.content == "hello"
        assert resp.model == "gpt-4o"
        assert resp.usage == {}
        assert resp.raw is None

    def test_with_usage(self) -> None:
        resp = LLMResponse(
            content="hi",
            model="gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert resp.usage["total_tokens"] == 15


class TestMockLLMClient:
    """Tests for the MockLLMClient."""

    def test_returns_responses_in_order(self) -> None:
        client = MockLLMClient(responses=["first", "second", "third"])

        async def run() -> None:
            r1 = await client.complete([{"role": "user", "content": "a"}])
            r2 = await client.complete([{"role": "user", "content": "b"}])
            r3 = await client.complete([{"role": "user", "content": "c"}])
            assert r1.content == "first"
            assert r2.content == "second"
            assert r3.content == "third"

        asyncio.run(run())

    def test_repeats_last_response_when_exhausted(self) -> None:
        client = MockLLMClient(responses=["only"])

        async def run() -> None:
            r1 = await client.complete([{"role": "user", "content": "a"}])
            r2 = await client.complete([{"role": "user", "content": "b"}])
            assert r1.content == "only"
            assert r2.content == "only"

        asyncio.run(run())

    def test_empty_responses_returns_empty_string(self) -> None:
        client = MockLLMClient()

        async def run() -> None:
            r = await client.complete([{"role": "user", "content": "a"}])
            assert r.content == ""

        asyncio.run(run())

    def test_call_log_records_messages(self) -> None:
        client = MockLLMClient(responses=["ok"])

        async def run() -> None:
            await client.complete([{"role": "user", "content": "hello"}])
            await client.complete([{"role": "user", "content": "world"}])

        assert len(client.call_log) == 2
        assert client.call_log[0][0]["content"] == "hello"
        assert client.call_log[1][0]["content"] == "world"

    def test_reset_clears_state(self) -> None:
        client = MockLLMClient(responses=["a", "b"])

        async def run() -> None:
            await client.complete([{"role": "user", "content": "x"}])
            client.reset()
            r = await client.complete([{"role": "user", "content": "y"}])
            assert r.content == "a"  # Back to first response.

        asyncio.run(run())

    def test_is_llm_client_subclass(self) -> None:
        assert issubclass(MockLLMClient, LLMClient)


class TestOpenAIClient:
    """Tests for the OpenAIClient (construction and error handling)."""

    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="API key"):
            OpenAIClient(api_key="")

    def test_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = OpenAIClient()
        assert client._api_key == "test-key"

    def test_explicit_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        client = OpenAIClient(api_key="explicit-key")
        assert client._api_key == "explicit-key"

    def test_is_llm_client_subclass(self) -> None:
        assert issubclass(OpenAIClient, LLMClient)
