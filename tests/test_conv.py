"""Tests for conversation logs processor."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conv.models import ConversationRecord, LogEntry, Message, RequestPayload, ResponseChoice, ResponsePayload, ToolDefinition
from conv.processor import (
    coerce_arguments,
    process_session,
    serialize_tool_calls,
    serialize_tool_result,
    transform_messages,
)


class TestCoerceArguments:
    def test_dict_passthrough(self):
        args = {"key": "value"}
        assert coerce_arguments(args) == {"key": "value"}

    def test_string_parsed(self):
        args = '{"key": "value"}'
        assert coerce_arguments(args) == {"key": "value"}

    def test_invalid_string(self):
        args = "not json"
        assert coerce_arguments(args) == "not json"

    def test_none(self):
        assert coerce_arguments(None) is None


class TestSerializeToolCalls:
    def test_single_tool_call(self):
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "arguments": '{"arg": "value"}',
                },
            }
        ]
        result = serialize_tool_calls(tool_calls)
        assert '<tool_call>{"name": "my_tool", "arguments": {"arg": "value"}}</tool_call>' == result

    def test_multiple_tool_calls(self):
        tool_calls = [
            {"id": "1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
            {"id": "2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
        ]
        result = serialize_tool_calls(tool_calls)
        assert "<tool_call>" in result
        assert "tool_a" in result
        assert "tool_b" in result
        assert result.count("<tool_call>") == 2


class TestSerializeToolResult:
    def test_basic(self):
        result = serialize_tool_result("Hello world", "call_123")
        assert result == '<tool_result tool_call_id="call_123">Hello world</tool_result>'

    def test_empty_content(self):
        result = serialize_tool_result("", "call_123")
        assert result == '<tool_result tool_call_id="call_123"></tool_result>'


class TestTransformMessages:
    def test_regular_message_unchanged(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = transform_messages(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_tool_calls_transformed(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{}"},
                    }
                ],
            }
        ]
        result = transform_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "<tool_call>" in result[0]["content"]
        assert "tool_calls" not in result[0]

    def test_tool_result_transformed(self):
        messages = [
            {
                "role": "tool",
                "content": "Result here",
                "tool_call_id": "call_123",
            }
        ]
        result = transform_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert '<tool_result tool_call_id="call_123">Result here</tool_result>' == result[0]["content"]
        assert "tool_call_id" not in result[0]


class TestProcessSession:
    def _make_log_entry(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_content: str = "Response",
    ) -> LogEntry:
        return LogEntry(
            timestamp=datetime.now(timezone.utc),
            session_id="test-session",
            request=RequestPayload(
                model="gpt-4o",
                messages=[Message(**m) for m in messages],
                tools=[ToolDefinition(**t) for t in tools] if tools else None,
            ),
            response=ResponsePayload(
                id="resp-123",
                choices=[
                    ResponseChoice(
                        index=0,
                        message=Message(role="assistant", content=response_content),
                    )
                ],
            ),
        )

    def test_basic_session(self):
        entry = self._make_log_entry(
            messages=[{"role": "user", "content": "Hello"}],
            response_content="Hi there!",
        )
        result = process_session([entry], json_tool_calls=False)
        assert result is not None
        assert len(result.messages) == 2
        assert result.messages[0]["role"] == "user"
        assert result.messages[1]["role"] == "assistant"
        assert result.messages[1]["content"] == "Hi there!"

    def test_empty_session(self):
        result = process_session([])
        assert result is None

    def test_with_tools(self):
        entry = self._make_log_entry(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"type": "function", "function": {"name": "my_tool", "parameters": {}}}],
        )
        result = process_session([entry], json_tool_calls=False)
        assert result is not None
        assert len(result.tools) == 1
        assert result.tools[0]["function"]["name"] == "my_tool"

