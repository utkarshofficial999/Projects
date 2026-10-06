"""Unit tests for agent_engine.core.parser."""

from __future__ import annotations

import pytest

from agent_engine.core.exceptions import ParseError
from agent_engine.core.parser import Action, ParseResult, ReActParser, StepType


@pytest.fixture
def parser() -> ReActParser:
    return ReActParser()


class TestReActParser:
    """Tests for the ReActParser.parse method."""

    def test_parse_single_thought(self, parser: ReActParser) -> None:
        text = "Thought: I need to search for the answer."
        result = parser.parse(text)
        assert len(result.steps) == 1
        assert result.steps[0].type == StepType.THOUGHT
        assert result.steps[0].content == "I need to search for the answer."
        assert result.is_final is False

    def test_parse_action(self, parser: ReActParser) -> None:
        text = "Action: search_web(what is the capital of France?)"
        result = parser.parse(text)
        assert len(result.steps) == 1
        assert result.steps[0].type == StepType.ACTION
        assert result.steps[0].action is not None
        assert result.steps[0].action.name == "search_web"
        assert result.steps[0].action.input == "what is the capital of France?"

    def test_parse_observation(self, parser: ReActParser) -> None:
        text = "Observation: The capital of France is Paris."
        result = parser.parse(text)
        assert len(result.steps) == 1
        assert result.steps[0].type == StepType.OBSERVATION
        assert result.steps[0].content == "The capital of France is Paris."

    def test_parse_final_answer(self, parser: ReActParser) -> None:
        text = "Final Answer: Paris"
        result = parser.parse(text)
        assert result.is_final is True
        assert result.final_answer == "Paris"

    def test_parse_mixed_steps(self, parser: ReActParser) -> None:
        text = (
            "Thought: Let me search.\n"
            "Action: search_web(capital of France)\n"
            "Observation: Paris\n"
            "Thought: I have the answer.\n"
            "Final Answer: Paris"
        )
        result = parser.parse(text)
        assert len(result.steps) == 4
        assert result.steps[0].type == StepType.THOUGHT
        assert result.steps[1].type == StepType.ACTION
        assert result.steps[2].type == StepType.OBSERVATION
        assert result.steps[3].type == StepType.THOUGHT
        assert result.is_final is True
        assert result.final_answer == "Paris"

    def test_parse_with_extra_whitespace(self, parser: ReActParser) -> None:
        text = "  Thought:   I am thinking.  "
        result = parser.parse(text)
        assert result.steps[0].content == "I am thinking."

    def test_parse_case_insensitive(self, parser: ReActParser) -> None:
        text = "thought: lower case works"
        result = parser.parse(text)
        assert result.steps[0].type == StepType.THOUGHT

    def test_parse_empty_string_raises(self, parser: ReActParser) -> None:
        with pytest.raises(ParseError):
            parser.parse("")

    def test_parse_garbage_raises(self, parser: ReActParser) -> None:
        with pytest.raises(ParseError):
            parser.parse("This is not a valid ReAct response at all.")

    def test_parse_preserves_raw(self, parser: ReActParser) -> None:
        text = "Thought: hello"
        result = parser.parse(text)
        assert result.raw == text

    def test_parse_action_with_hyphenated_name(self, parser: ReActParser) -> None:
        text = "Action: my-tool(input here)"
        result = parser.parse(text)
        assert result.steps[0].action is not None
        assert result.steps[0].action.name == "my-tool"

    def test_parse_action_with_underscore_name(self, parser: ReActParser) -> None:
        text = "Action: my_tool(input here)"
        result = parser.parse(text)
        assert result.steps[0].action is not None
        assert result.steps[0].action.name == "my_tool"
