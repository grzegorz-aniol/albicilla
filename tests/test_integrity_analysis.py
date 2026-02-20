"""Unit tests for conv.integrity heuristics."""

from __future__ import annotations

from pathlib import Path

from datetime import datetime, timezone

from conv.integrity import IntegritySeverity, analyze_role_sequence_consistency, analyze_tool_result_heuristics
from conv.models import ConversationRecord, LogEntry, Message, RequestPayload, ResponseChoice, ResponsePayload


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


def test_tool_result_without_matching_call_is_error() -> None:
    record = ConversationRecord(
        messages=[
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "tool_x"}}]},
            {"role": "tool", "tool_call_id": "call_2", "content": "{}"},
            {"role": "assistant", "content": "done"},
        ],
        tools=[],
    )

    findings = analyze_tool_result_heuristics(
        record,
        session="scenario-d",
        input_file=Path("proxy_logs/2026-01-28/scenario-d.jsonl"),
    )
    assert any(f.severity == IntegritySeverity.ERROR for f in findings)
    assert any("no matching tool call" in f.message for f in findings)


def test_tool_result_missing_tool_call_id_is_error() -> None:
    record = ConversationRecord(
        messages=[
            {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "tool_x"}}]},
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": "done"},
        ],
        tools=[],
    )

    findings = analyze_tool_result_heuristics(
        record,
        session="scenario-e",
        input_file=Path("proxy_logs/2026-01-28/scenario-e.jsonl"),
    )
    assert any(f.severity == IntegritySeverity.ERROR for f in findings)
    assert any("missing tool_call_id" in f.message for f in findings)


def test_role_sequence_mismatch_is_error() -> None:
    entry1 = LogEntry(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        session_id="test-session",
        request=RequestPayload(
            model="gpt-4o",
            messages=[Message(role="system", content="start"), Message(role="user", content="hi")],
        ),
        response=ResponsePayload(
            id="resp-1",
            choices=[ResponseChoice(index=0, message=Message(role="assistant", content="ok"))],
        ),
    )
    entry2 = LogEntry(
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        session_id="test-session",
        request=RequestPayload(
            model="gpt-4o",
            messages=[Message(role="user", content="reset")],
        ),
        response=ResponsePayload(
            id="resp-2",
            choices=[ResponseChoice(index=0, message=Message(role="assistant", content="ok"))],
        ),
    )
    findings = analyze_role_sequence_consistency(
        [entry1, entry2],
        session="scenario-f",
        input_file=Path("proxy_logs/2026-01-28/scenario-f.jsonl"),
    )
    assert len(findings) == 1
    assert findings[0].severity == IntegritySeverity.ERROR


def test_role_sequence_allows_acat_to_act_transition() -> None:
    entry1 = LogEntry(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        session_id="test-session",
        request=RequestPayload(
            model="gpt-4o",
            messages=[
                Message(role="system", content="start"),
                Message(role="user", content="hi"),
                Message(role="assistant", content="message"),
                Message(
                    role="assistant",
                    tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "tool_x"}}],
                ),
            ],
        ),
        response=ResponsePayload(
            id="resp-1",
            choices=[ResponseChoice(index=0, message=Message(role="assistant", content="ok"))],
        ),
    )
    entry2 = LogEntry(
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        session_id="test-session",
        request=RequestPayload(
            model="gpt-4o",
            messages=[
                Message(role="system", content="start"),
                Message(role="user", content="hi"),
                Message(
                    role="assistant",
                    content="message",
                    tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "tool_x"}}],
                ),
            ],
        ),
        response=ResponsePayload(
            id="resp-2",
            choices=[ResponseChoice(index=0, message=Message(role="assistant", content="ok"))],
        ),
    )

    findings = analyze_role_sequence_consistency(
        [entry1, entry2],
        session="scenario-h",
        input_file=Path("proxy_logs/2026-01-28/scenario-h.jsonl"),
    )
    assert not findings


def test_role_sequence_act_to_acat_is_error() -> None:
    entry1 = LogEntry(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        session_id="test-session",
        request=RequestPayload(
            model="gpt-4o",
            messages=[
                Message(role="system", content="start"),
                Message(role="user", content="hi"),
                Message(
                    role="assistant",
                    content="message",
                    tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "tool_x"}}],
                ),
            ],
        ),
        response=ResponsePayload(
            id="resp-1",
            choices=[ResponseChoice(index=0, message=Message(role="assistant", content="ok"))],
        ),
    )
    entry2 = LogEntry(
        timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        session_id="test-session",
        request=RequestPayload(
            model="gpt-4o",
            messages=[
                Message(role="system", content="start"),
                Message(role="user", content="hi"),
                Message(role="assistant", content="message"),
                Message(
                    role="assistant",
                    tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "tool_x"}}],
                ),
            ],
        ),
        response=ResponsePayload(
            id="resp-2",
            choices=[ResponseChoice(index=0, message=Message(role="assistant", content="ok"))],
        ),
    )

    findings = analyze_role_sequence_consistency(
        [entry1, entry2],
        session="scenario-i",
        input_file=Path("proxy_logs/2026-01-28/scenario-i.jsonl"),
    )
    assert any(f.severity == IntegritySeverity.ERROR for f in findings)


def test_legacy_function_call_is_error() -> None:
    record = ConversationRecord(
        messages=[
            {
                "role": "assistant",
                "function_call": {"name": "old_tool", "arguments": "{}"},
            },
            {"role": "assistant", "content": "done"},
        ],
        tools=[],
    )

    findings = analyze_tool_result_heuristics(
        record,
        session="scenario-g",
        input_file=Path("proxy_logs/2026-01-28/scenario-g.jsonl"),
    )
    assert any(f.severity == IntegritySeverity.ERROR for f in findings)
    assert any("function_call" in f.message for f in findings)
