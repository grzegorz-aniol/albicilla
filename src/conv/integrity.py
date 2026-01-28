"""Integrity analysis for exported conversation JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .models import ConversationRecord


class IntegritySeverity(StrEnum):
    """Severity of an integrity finding."""

    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True, slots=True)
class IntegrityFinding:
    """A single integrity finding for a processed session."""

    severity: IntegritySeverity
    session: str
    input_file: Path
    message: str


_ERRORISH_KEYS: frozenset[str] = frozenset({"error", "timeout", "missing", "incomplete"})


def analyze_export_record(
    record: ConversationRecord | None,
    *,
    session: str,
    input_file: Path,
    json_tool_calls: bool,
) -> list[IntegrityFinding]:
    """Run structural integrity checks on the record that will be exported.

    This mirrors the spirit of `tests/test_output_integrity.py`, but never raises.
    """
    findings: list[IntegrityFinding] = []

    if record is None:
        findings.append(
            IntegrityFinding(
                severity=IntegritySeverity.ERROR,
                session=session,
                input_file=input_file,
                message="No conversation record produced for session file",
            )
        )
        return findings

    messages = record.messages
    tools = record.tools

    if not isinstance(messages, list) or not messages:
        findings.append(
            IntegrityFinding(
                severity=IntegritySeverity.ERROR,
                session=session,
                input_file=input_file,
                message="messages must be a non-empty list",
            )
        )
        return findings

    if tools is None or not isinstance(tools, list):
        findings.append(
            IntegrityFinding(
                severity=IntegritySeverity.ERROR,
                session=session,
                input_file=input_file,
                message="tools must be a list",
            )
        )
    elif not tools:
        findings.append(
            IntegrityFinding(
                severity=IntegritySeverity.WARNING,
                session=session,
                input_file=input_file,
                message="tools list is empty",
            )
        )
    else:
        for tool_definition in tools:
            if not isinstance(tool_definition, dict):
                findings.append(
                    IntegrityFinding(
                        severity=IntegritySeverity.ERROR,
                        session=session,
                        input_file=input_file,
                        message="tool definition is not an object",
                    )
                )
                continue

            parameters = tool_definition.get("parameters")
            if not isinstance(parameters, dict):
                findings.append(
                    IntegrityFinding(
                        severity=IntegritySeverity.ERROR,
                        session=session,
                        input_file=input_file,
                        message="tool definition lacks parameters dict",
                    )
                )
                continue

            if parameters.get("type") != "object":
                findings.append(
                    IntegrityFinding(
                        severity=IntegritySeverity.ERROR,
                        session=session,
                        input_file=input_file,
                        message="tool definition parameters.type must be object",
                    )
                )

            properties = parameters.get("properties")
            if not isinstance(properties, dict):
                findings.append(
                    IntegrityFinding(
                        severity=IntegritySeverity.ERROR,
                        session=session,
                        input_file=input_file,
                        message="tool definition parameters missing properties dict",
                    )
                )

            required = parameters.get("required", [])
            if not isinstance(required, list):
                findings.append(
                    IntegrityFinding(
                        severity=IntegritySeverity.ERROR,
                        session=session,
                        input_file=input_file,
                        message="tool definition parameters.required must be a list",
                    )
                )

    if json_tool_calls:
        findings.extend(
            _analyze_inline_tool_markup(
                messages,
                session=session,
                input_file=input_file,
            )
        )
    else:
        findings.extend(
            _analyze_untransformed_tool_calls(
                messages,
                session=session,
                input_file=input_file,
            )
        )

    return findings


def analyze_tool_result_heuristics(
    raw_record: ConversationRecord | None,
    *,
    session: str,
    input_file: Path,
) -> list[IntegrityFinding]:
    """Heuristically flag tool results that look like errors.

    A tool result is considered problematic when its JSON payload contains any of:
    `error`, `timeout`, `missing`, `incomplete` (truthy values).

    Problematic calls are downgraded to WARNING if, before the next assistant message,
    the same tool name is called again and produces a non-problematic result.
    """
    if raw_record is None:
        return []

    messages = raw_record.messages
    if not isinstance(messages, list) or not messages:
        return []

    findings: list[IntegrityFinding] = []

    i = 0
    while i < len(messages):
        message = messages[i]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            i += 1
            continue

        tool_calls = _extract_tool_calls_from_assistant(message)
        if not tool_calls:
            i += 1
            continue

        block_start = i + 1
        block_end = _find_next_assistant_index(messages, start=block_start)
        tool_results_by_id = _index_tool_results(messages[block_start:block_end])

        block_findings = _analyze_tool_block(
            tool_calls,
            tool_results_by_id,
            session=session,
            input_file=input_file,
        )
        findings.extend(block_findings)

        i = block_end

    return findings


def _find_next_assistant_index(messages: list[dict[str, Any]], *, start: int) -> int:
    for idx in range(start, len(messages)):
        if isinstance(messages[idx], dict) and messages[idx].get("role") == "assistant":
            return idx
    return len(messages)


def _extract_tool_calls_from_assistant(message: dict[str, Any]) -> list[tuple[str, str]]:
    """Return list of (tool_call_id, tool_name) from an assistant message."""
    tool_calls = message.get("tool_calls")
    extracted: list[tuple[str, str]] = []

    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            function_block = call.get("function")
            if not isinstance(call_id, str) or not call_id:
                continue
            if not isinstance(function_block, dict):
                continue
            name = function_block.get("name")
            if not isinstance(name, str) or not name:
                continue
            extracted.append((call_id, name))

    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        # Legacy / single-tool format does not provide a stable call id in logs.
        name = function_call.get("name")
        if isinstance(name, str) and name:
            extracted.append(("<function_call>", name))

    return extracted


def _index_tool_results(messages: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Map tool_call_id -> list of tool result messages within a block."""
    results: dict[str, list[dict[str, Any]]] = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        results.setdefault(tool_call_id, []).append(message)
    return results


def _analyze_tool_block(
    tool_calls: list[tuple[str, str]],
    tool_results_by_id: dict[str, list[dict[str, Any]]],
    *,
    session: str,
    input_file: Path,
) -> list[IntegrityFinding]:
    """Analyze a single assistant->tool...->assistant block."""
    call_issues: list[tuple[str, str, IntegritySeverity, str]] = []
    by_tool_name: dict[str, list[tuple[str, IntegritySeverity, str]]] = {}

    for call_id, tool_name in tool_calls:
        result_messages = tool_results_by_id.get(call_id, [])
        issue = _classify_tool_result_messages(result_messages)
        if issue is None:
            continue

        severity, details = issue
        call_issues.append((call_id, tool_name, severity, details))
        by_tool_name.setdefault(tool_name, []).append((call_id, severity, details))

    # Downgrade to warning if same tool has a successful retry within the block.
    tool_names_with_success: set[str] = set()
    for call_id, tool_name in tool_calls:
        result_messages = tool_results_by_id.get(call_id)
        if not result_messages:
            continue
        issue = _classify_tool_result_messages(result_messages)
        if issue is None:
            tool_names_with_success.add(tool_name)

    findings: list[IntegrityFinding] = []
    for call_id, tool_name, severity, details in call_issues:
        if severity == IntegritySeverity.ERROR and tool_name in tool_names_with_success:
            severity = IntegritySeverity.WARNING

        findings.append(
            IntegrityFinding(
                severity=severity,
                session=session,
                input_file=input_file,
                message=f"Tool result flagged ({tool_name}, tool_call_id={call_id}): {details}",
            )
        )

    # Missing tool results for known tool_call_ids.
    for call_id, tool_name in tool_calls:
        if call_id not in tool_results_by_id:
            severity = (
                IntegritySeverity.WARNING
                if tool_name in tool_names_with_success
                else IntegritySeverity.ERROR
            )
            findings.append(
                IntegrityFinding(
                    severity=severity,
                    session=session,
                    input_file=input_file,
                    message=f"Missing tool result ({tool_name}, tool_call_id={call_id})",
                )
            )

    return findings


def _classify_tool_result_messages(
    messages: list[dict[str, Any]],
) -> tuple[IntegritySeverity, str] | None:
    if not messages:
        return None

    for message in messages:
        content = message.get("content")
        if content is None or content == "":
            return IntegritySeverity.ERROR, "empty tool result content"

        json_payload = _try_parse_json(content)
        if json_payload is None:
            continue

        keys = _find_errorish_keys(json_payload)
        if keys:
            details = f"errorish keys present: {', '.join(sorted(keys))}"
            return IntegritySeverity.ERROR, details

    return None


def _try_parse_json(content: Any) -> Any | None:
    if isinstance(content, dict) or isinstance(content, list):
        return content

    if isinstance(content, str):
        stripped = content.strip()
        if not stripped:
            return None
        if not ((stripped.startswith("{") and stripped.endswith("}")) or (stripped.startswith("[") and stripped.endswith("]"))):
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None

    return None


def _find_errorish_keys(payload: Any) -> set[str]:
    found: set[str] = set()

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(key, str) and key.lower() in _ERRORISH_KEYS:
                if _is_truthy_value(value):
                    found.add(key.lower())
            found.update(_find_errorish_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            found.update(_find_errorish_keys(item))

    return found


def _is_truthy_value(value: Any) -> bool:
    if value is None:
        return False
    if value is False:
        return False
    if value == 0:
        return False
    if isinstance(value, (str, list, dict)) and not value:
        return False
    return True


def _analyze_inline_tool_markup(
    messages: list[dict[str, Any]],
    *,
    session: str,
    input_file: Path,
) -> list[IntegrityFinding]:
    findings: list[IntegrityFinding] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content", "")
        if role == "assistant" and isinstance(content, str):
            has_tool_markup = "<tool_call>" in content or "</tool_call>" in content
            if has_tool_markup and content.count("<tool_call>") != content.count("</tool_call>"):
                findings.append(
                    IntegrityFinding(
                        severity=IntegritySeverity.ERROR,
                        session=session,
                        input_file=input_file,
                        message="tool_call tags mismatch in assistant content",
                    )
                )
            if "<tool_call>" in content and message.get("tool_calls") is not None:
                findings.append(
                    IntegrityFinding(
                        severity=IntegritySeverity.ERROR,
                        session=session,
                        input_file=input_file,
                        message="tool_calls key should be dropped when json-tool-calls enabled",
                    )
                )

        if role == "tool":
            if not isinstance(content, str) or not content.startswith("<tool_result"):
                findings.append(
                    IntegrityFinding(
                        severity=IntegritySeverity.ERROR,
                        session=session,
                        input_file=input_file,
                        message="tool result missing <tool_result> wrapper",
                    )
                )
            elif not content.endswith("</tool_result>"):
                findings.append(
                    IntegrityFinding(
                        severity=IntegritySeverity.ERROR,
                        session=session,
                        input_file=input_file,
                        message="tool result missing closing </tool_result> tag",
                    )
                )

    return findings


def _analyze_untransformed_tool_calls(
    messages: list[dict[str, Any]],
    *,
    session: str,
    input_file: Path,
) -> list[IntegrityFinding]:
    findings: list[IntegrityFinding] = []

    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list):
            findings.append(
                IntegrityFinding(
                    severity=IntegritySeverity.ERROR,
                    session=session,
                    input_file=input_file,
                    message="assistant tool_calls must be a list when present",
                )
            )
            continue

        for call in tool_calls:
            if not isinstance(call, dict):
                findings.append(
                    IntegrityFinding(
                        severity=IntegritySeverity.ERROR,
                        session=session,
                        input_file=input_file,
                        message="tool_calls entry must be an object",
                    )
                )
                continue
            function_block = call.get("function")
            if not isinstance(function_block, dict):
                findings.append(
                    IntegrityFinding(
                        severity=IntegritySeverity.ERROR,
                        session=session,
                        input_file=input_file,
                        message="tool_calls entry missing function object",
                    )
                )
                continue
            name = function_block.get("name")
            if not isinstance(name, str) or not name:
                findings.append(
                    IntegrityFinding(
                        severity=IntegritySeverity.ERROR,
                        session=session,
                        input_file=input_file,
                        message="tool_calls function.name must be a non-empty string",
                    )
                )

    return findings
