"""Tests per hycoder/agent_executor.py."""

import json
from unittest.mock import MagicMock, patch

import pytest

from hycoder.agent_executor import (
    AgentExecutor,
    build_agent_messages,
    parse_tool_calls,
    format_tool_result,
    format_tools_for_prompt,
    AGENT_SYSTEM_PROMPT,
    MAX_ITERATIONS,
)
from hycoder.tools import ToolResult


class TestFormatToolsForPrompt:
    def test_returns_string_with_tool_list(self):
        result = format_tools_for_prompt()
        assert "read" in result.lower() or "Read" in result
        assert isinstance(result, str)
        assert len(result) > 50

    def test_includes_parameter_descriptions(self):
        result = format_tools_for_prompt()
        assert "path" in result.lower()


class TestBuildAgentMessages:
    def test_basic_structure(self):
        messages = build_agent_messages("test prompt")
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "test prompt"

    def test_system_prompt_replaces_tools_placeholder(self):
        sys_prompt = "You are AI.\n{tools_description}\nEnd."
        messages = build_agent_messages("hi", system_prompt=sys_prompt)
        assert "{tools_description}" not in messages[0]["content"]
        assert "End." in messages[0]["content"]

    def test_custom_system_prompt_no_placeholder(self):
        sys_prompt = "Just a custom prompt."
        messages = build_agent_messages("hi", system_prompt=sys_prompt)
        assert messages[0]["content"] == sys_prompt

    def test_with_agent_config(self):
        agent_config = {
            "instructions": "You are a coding agent.",
            "tools": ["read", "write"],
            "permissions": {"bash": True},
        }
        messages = build_agent_messages("hi", agent_config=agent_config)
        assert messages[0]["content"].startswith("You are a coding agent.")

    def test_with_history(self):
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        messages = build_agent_messages("new prompt", history=history)
        assert len(messages) == 4
        assert messages[1]["content"] == "previous question"
        assert messages[2]["content"] == "previous answer"

    def test_history_truncated_to_last_20(self):
        history = [{"role": "user", "content": f"msg-{i}"} for i in range(30)]
        messages = build_agent_messages("final", history=history)
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) <= 22

    def test_filters_non_user_assistant_from_history(self):
        history = [
            {"role": "system", "content": "system msg"},
            {"role": "user", "content": "user msg"},
            {"role": "tool", "content": "tool result"},
        ]
        messages = build_agent_messages("hi", history=history)
        roles = [m["role"] for m in messages]
        assert roles.count("system") == 1
        assert roles.count("user") == 2
        assert "tool" not in roles


class TestParseToolCalls:
    def test_parse_single_tool_call(self):
        text = '<tool_call>{"name": "read", "arguments": {"path": "file.txt"}}</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "read"
        assert calls[0]["arguments"]["path"] == "file.txt"

    def test_parse_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "read", "arguments": {"path": "a.txt"}}</tool_call>'
            '<tool_call>{"name": "write", "arguments": {"path": "b.txt", "content": "data"}}</tool_call>'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2

    def test_parse_invalid_json_skips(self):
        text = '<tool_call>{invalid json}</tool_call>'
        calls = parse_tool_calls(text)
        assert calls == []

    def test_parse_missing_name_key_skips(self):
        text = '<tool_call>{"not_name": "value"}</tool_call>'
        calls = parse_tool_calls(text)
        assert calls == []

    def test_parse_no_tool_calls(self):
        assert parse_tool_calls("just a normal message") == []

    def test_parse_with_surrounding_text(self):
        text = "Let me read that file.\n<tool_call>{\"name\": \"search\", \"arguments\": {\"pattern\": \"test\"}}</tool_call>\nDone."
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"


class TestFormatToolResult:
    def test_format_success(self):
        result = ToolResult(success=True, output="file content", error="", duration=0.5)
        formatted = format_tool_result("read", result)
        assert "Tool 'read' completed in 0.5s" in formatted
        assert "file content" in formatted
        assert "<tool_result>" in formatted

    def test_format_failure(self):
        result = ToolResult(success=False, output="", error="File not found", duration=1.2)
        formatted = format_tool_result("read", result)
        assert "Tool 'read' failed after 1.2s" in formatted
        assert "File not found" in formatted

    def test_format_empty_success(self):
        result = ToolResult(success=True, output="", error="", duration=0.1)
        formatted = format_tool_result("noop", result)
        assert "(completed successfully)" in formatted

    def test_format_truncates_long_output(self):
        long_output = "x" * 5000
        result = ToolResult(success=True, output=long_output, error="", duration=0.1)
        formatted = format_tool_result("test", result)
        assert len(formatted) < 8000

    def test_contains_json_blob(self):
        result = ToolResult(success=True, output="ok", error="", duration=0.3)
        formatted = format_tool_result("test", result)
        assert '"name": "test"' in formatted
        assert '"success": true' in formatted
        assert '"duration": 0.3' in formatted


class TestAgentExecutor:
    def test_init_defaults(self):
        executor = AgentExecutor(provider=MagicMock())
        assert executor.max_iterations == MAX_ITERATIONS

    def test_init_custom_config(self):
        executor = AgentExecutor(provider=MagicMock(), config={"max_iterations": 3})
        assert executor.max_iterations == 3

    def test_execute_returns_final_text(self):
        mock_provider = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Final answer"
        mock_result.source = "local"
        mock_result.tokens_total = 50
        mock_result.time_s = 1.0
        mock_result.model = "test-model"
        mock_result.cached = False

        def generate_stream(messages):
            yield {"token": "Final ", "done": False}
            yield {"token": "answer", "done": False, "result": None}
            yield {"done": True, "result": mock_result, "text": "Final answer",
                   "source": "local", "tokens": 50, "time_s": 1.0,
                   "model": "test-model", "cached": False}

        mock_provider.generate_chat_stream.side_effect = generate_stream
        executor = AgentExecutor(mock_provider)
        events = list(executor.execute("test prompt"))
        assert len(events) >= 1
        assert events[-1]["type"] == "done"

    def test_execute_with_tool_call(self):
        mock_provider = MagicMock()
        call_count = [0]

        def generate_stream(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                yield {"token": "<tool_call>{\"name\": \"memory\", \"arguments\": {\"action\": \"list\"}}</tool_call>", "done": False}
                yield {"done": True, "result": MagicMock(text="done", source="local",
                       tokens_total=10, time_s=0.1, model="t", cached=False),
                       "text": "done", "source": "local", "tokens": 10, "time_s": 0.1,
                       "model": "t", "cached": False}
            else:
                yield {"token": "final answer", "done": False}
                yield {"done": True, "result": MagicMock(text="final answer", source="local",
                       tokens_total=5, time_s=0.1, model="t", cached=False),
                       "text": "final answer", "source": "local", "tokens": 5, "time_s": 0.1,
                       "model": "t", "cached": False}

        mock_provider.generate_chat_stream.side_effect = generate_stream
        executor = AgentExecutor(mock_provider, config={"max_iterations": 3})
        events = list(executor.execute("test"))
        types = [e["type"] for e in events]
        assert "tool_call" in types or "tool_result" in types

    def test_max_iterations_reached(self):
        mock_provider = MagicMock()
        mock_result = MagicMock(text="still going", source="local", tokens_total=5,
                                time_s=0.1, model="t", cached=False, events=[])

        def generate_stream(messages):
            full_text = "<tool_call>{\"name\": \"memory\", \"arguments\": {\"action\": \"list\"}}</tool_call>"
            yield {"token": full_text, "done": True, "result": mock_result,
                   "text": full_text, "source": "local", "tokens": 5, "time_s": 0.1,
                   "model": "t", "cached": False}

        mock_provider.generate_chat_stream.side_effect = generate_stream
        executor = AgentExecutor(mock_provider, config={"max_iterations": 1})
        events = list(executor.execute("test"))
        types = [e["type"] for e in events]
        assert "info" in types

    def test_execute_handles_error(self):
        mock_provider = MagicMock()
        mock_provider.generate_chat_stream.side_effect = Exception("Provider crashed")
        executor = AgentExecutor(mock_provider)
        events = list(executor.execute("test"))
        assert any(e["type"] == "error" for e in events)
