"""Unit tests for conv.integrity heuristics."""

from __future__ import annotations

from pathlib import Path

from conv.integrity import IntegritySeverity, analyze_tool_result_heuristics
from conv.models import ConversationRecord


def test_tool_result_error_downgraded_on_success_retry_within_block() -> None:
    record = ConversationRecord(
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool_x", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "tool_x", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"error":"boom"}'},
            {"role": "tool", "tool_call_id": "call_2", "content": "{}"},
            {"role": "assistant", "content": "done"},
        ],
        tools=[],
    )

    findings = analyze_tool_result_heuristics(
        record,
        session="scenario-a",
        input_file=Path("proxy_logs/2026-01-28/scenario-a.jsonl"),
    )
    assert len(findings) == 1
    assert findings[0].severity == IntegritySeverity.WARNING
    assert "tool_x" in findings[0].message
    assert "call_1" in findings[0].message


def test_tool_result_error_reported_when_no_success_retry() -> None:
    record = ConversationRecord(
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool_y", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"meta":{"timeout":true}}',
            },
            {"role": "assistant", "content": "done"},
        ],
        tools=[],
    )

    findings = analyze_tool_result_heuristics(
        record,
        session="scenario-b",
        input_file=Path("proxy_logs/2026-01-28/scenario-b.jsonl"),
    )
    assert len(findings) == 1
    assert findings[0].severity == IntegritySeverity.ERROR
    assert "timeout" in findings[0].message


def test_missing_tool_result_does_not_count_as_success_retry() -> None:
    record = ConversationRecord(
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool_z", "arguments": "{}"},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "tool_z", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"error":"boom"}'},
            {"role": "assistant", "content": "done"},
        ],
        tools=[],
    )

    findings = analyze_tool_result_heuristics(
        record,
        session="scenario-c",
        input_file=Path("proxy_logs/2026-01-28/scenario-c.jsonl"),
    )
    severities = sorted(f.severity for f in findings)
    assert IntegritySeverity.ERROR in severities
