"""Core processing logic for conversation logs."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger
import typer

from .cleanup import CleanupConfig, apply_cleanups
from .models import ConversationRecord, LogEntry, Message, ResponsePayload, SessionToolUsageSample, ToolDefinition
from .scenario import extract_session_name_from_path


class SessionValidationError(ValueError):
    """Raised when a session log fails monotonicity validation."""


@dataclass(frozen=True)
class SessionValidationResult:
    """Validated entries plus a warning flag."""

    entries: list[LogEntry]
    had_warning: bool


def read_session_file(path: Path) -> list[LogEntry]:
    """Parse a JSONL session file and return list of log entries."""
    entries: list[LogEntry] = []

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed JSON in {file}:{line}: {error}",
                    file=path,
                    line=line_number,
                    error=exc,
                )
                continue

            try:
                entry = LogEntry.model_validate(data)
                entries.append(entry)
            except Exception as exc:
                logger.warning(
                    "Skipping invalid entry in {file}:{line}: {error}",
                    file=path,
                    line=line_number,
                    error=exc,
                )
                continue

    return entries


def process_session(
    entries: list[LogEntry],
    json_tool_calls: bool = False,
    *,
    source_path: Path | None = None,
) -> ConversationRecord | None:
    """
    Process a session's log entries and produce a conversation record.

    Takes the last entry, extracts messages and tools from the request,
    appends the final assistant response, and optionally transforms tool calls.
    """
    if not entries:
        return None

    validation = validate_session_entries(entries, source_path=source_path)
    if not validation.entries:
        return None

    return _process_validated_session(
        validation.entries,
        json_tool_calls,
        source_path=source_path,
    )


def _process_validated_session(
    validated_entries: list[LogEntry],
    json_tool_calls: bool = False,
    *,
    source_path: Path | None = None,
) -> ConversationRecord | None:
    """Process an already-validated session entry list."""
    if not validated_entries:
        return None

    last_entry = validated_entries[-1]

    # Extract messages from request
    messages: list[dict[str, Any]] = [
        msg.model_dump(exclude_none=True) for msg in last_entry.request.messages
    ]

    # Extract tools from all requests (or empty list)
    tools = extract_tool_schemas(collect_session_tools(validated_entries))

    # Append final response if available
    if last_entry.response and last_entry.response.choices:
        assistant_message = last_entry.response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_none=True))
    else:
        logger.warning(
            "No response choices in last entry for session {session}",
            session=last_entry.session_id,
        )

    for message in messages:
        if message.get("role") == "developer":
            message["role"] = "system"

    # Transform tool calls if enabled
    if json_tool_calls:
        messages = transform_messages(messages)

    return ConversationRecord(messages=messages, tools=tools)


def process_session_with_stats(
    entries: list[LogEntry],
    json_tool_calls: bool = False,
    *,
    source_path: Path | None = None,
) -> tuple[ConversationRecord | None, int]:
    """Process a session and return the record plus tool-call request count."""
    record = process_session(entries, json_tool_calls=False, source_path=source_path)
    if record is None:
        return None, 0

    tool_call_count = count_tool_call_requests(record.messages)

    if json_tool_calls:
        record = ConversationRecord(messages=transform_messages(record.messages), tools=record.tools)

    return record, tool_call_count


def transform_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Transform messages to use inline JSON format for tool calls and results.

    - Assistant messages with tool_calls: serialize to <tool_call>...</tool_call>
    - Tool messages: wrap content in <tool_result>...</tool_result>
    """
    transformed: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")

        if role == "assistant" and message.get("tool_calls"):
            transformed.append(_transform_assistant_tool_calls(message))
        elif role == "tool" and message.get("tool_call_id"):
            transformed.append(_transform_tool_result(message))
        else:
            # Keep message as-is, but clean up null fields
            cleaned = {k: v for k, v in message.items() if v is not None}
            transformed.append(cleaned)

    return transformed


def count_tool_call_requests(messages: list[dict[str, Any]]) -> int:
    """Count tool-call requests in a normalized session message list.

    Counts assistant-side tool invocations only (skips tool result messages).
    """
    count = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            count += len(tool_calls)
            continue

        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            count += 1
            continue

        content = message.get("content")
        if isinstance(content, list):
            content = _normalize_content_list(content)
        if isinstance(content, str):
            count += content.count("<tool_call>")

    return count


def _assistant_message_has_tool_request(message: dict[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        return True

    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        return True

    content = message.get("content")
    if isinstance(content, list):
        content = _normalize_content_list(content)
    if isinstance(content, str) and "<tool_call>" in content:
        return True

    return False


def count_turn_groups(messages: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Count consecutive role-group "turns" in a session.

    A client turn is a consecutive group of messages where role is either
    "system" or "user", uninterrupted by any other role.

    An assistant turn is a consecutive group of messages where role is
    "assistant", uninterrupted by any other role.

    Tool result messages (role "tool" or legacy "function") are ignored for the
    purpose of grouping.
    """
    client_turns = 0
    assistant_turns = 0
    assistant_tool_turns = 0
    previous_group: str | None = None
    pending_assistant_tool_turn = False

    for message in messages:
        role = message.get("role")
        if role in {"tool", "function"}:
            continue

        if role == "assistant":
            group = "assistant"
        elif role in {"system", "user"}:
            group = "client"
        else:
            if pending_assistant_tool_turn:
                assistant_tool_turns += 1
                pending_assistant_tool_turn = False
            previous_group = None
            continue

        if group != previous_group:
            if previous_group == "assistant" and pending_assistant_tool_turn:
                assistant_tool_turns += 1
                pending_assistant_tool_turn = False
            if group == "assistant":
                assistant_turns += 1
            else:
                client_turns += 1
            previous_group = group

        if group == "assistant" and _assistant_message_has_tool_request(message):
            pending_assistant_tool_turn = True

    if pending_assistant_tool_turn:
        assistant_tool_turns += 1

    return client_turns, assistant_turns, assistant_tool_turns


def _transform_assistant_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    """Transform assistant message with tool_calls to inline format."""
    content_parts: list[str] = []

    # Keep existing content if present
    existing_content = message.get("content")
    if existing_content:
        if isinstance(existing_content, str):
            content_parts.append(existing_content)
        elif isinstance(existing_content, list):
            content_parts.append(_normalize_content_list(existing_content))

    # Serialize tool calls
    tool_calls = message.get("tool_calls", [])
    serialized = serialize_tool_calls(tool_calls)
    if serialized:
        content_parts.append(serialized)

    # Build new message without tool_calls field
    new_message: dict[str, Any] = {"role": "assistant"}

    combined_content = "\n".join(part for part in content_parts if part).strip()
    if combined_content:
        new_message["content"] = combined_content

    # Preserve other fields except tool_calls
    for key, value in message.items():
        if key in ("role", "content", "tool_calls") or value is None:
            continue
        new_message[key] = value

    return new_message


def _transform_tool_result(message: dict[str, Any]) -> dict[str, Any]:
    """Transform tool result message to inline format."""
    tool_call_id = message.get("tool_call_id", "")
    content = message.get("content", "")

    if isinstance(content, list):
        content = _normalize_content_list(content)

    wrapped_content = serialize_tool_result(str(content) if content else "", tool_call_id)

    # Build new message without tool_call_id field
    new_message: dict[str, Any] = {
        "role": "tool",
        "content": wrapped_content,
    }

    # Preserve other fields except tool_call_id
    for key, value in message.items():
        if key in ("role", "content", "tool_call_id") or value is None:
            continue
        new_message[key] = value

    return new_message


def _normalize_content_list(content: list[Any]) -> str:
    """Convert content list (e.g., from vision messages) to string."""
    parts: list[str] = []
    for chunk in content:
        if isinstance(chunk, dict):
            chunk_type = chunk.get("type")
            if chunk_type == "text" and isinstance(chunk.get("text"), str):
                parts.append(chunk["text"])
    return "".join(parts)


def serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Convert tool_calls to <tool_call>...</tool_call> format."""
    serialized: list[str] = []

    for call in tool_calls:
        if not isinstance(call, dict):
            continue

        function_block = call.get("function")
        if not isinstance(function_block, dict):
            continue

        name = function_block.get("name")
        if not isinstance(name, str):
            continue

        arguments = function_block.get("arguments")
        formatted_arguments = coerce_arguments(arguments)

        payload = {"name": name, "arguments": formatted_arguments}
        serialized.append(f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>")

    return "\n".join(serialized)


def serialize_tool_result(content: str, tool_call_id: str) -> str:
    """Wrap tool result in <tool_result>...</tool_result> format."""
    return f'<tool_result tool_call_id="{tool_call_id}">{content}</tool_result>'


def extract_tool_schemas(tools: list[ToolDefinition] | None) -> list[dict[str, Any]]:
    """Return function schemas from request tools, matching dataset structure."""
    if not tools:
        return []

    flattened: list[dict[str, Any]] = []
    for tool in tools:
        function_block = tool.function
        if isinstance(function_block, dict):
            flattened.append(dict(function_block))
            continue
        flattened.append(tool.model_dump(exclude_none=True))
    return flattened


def collect_session_tools(entries: list[LogEntry]) -> list[ToolDefinition]:
    """Collect tool definitions across all entries, deduped by name (first wins)."""
    collected: list[ToolDefinition] = []
    seen_names: set[str] = set()

    for entry in entries:
        tools = entry.request.tools or []
        for tool in tools:
            name = _extract_tool_definition_name(tool)
            if isinstance(name, str) and name:
                if name in seen_names:
                    continue
                seen_names.add(name)
            collected.append(tool)

    return collected


def _extract_tool_definition_name(tool: ToolDefinition) -> str | None:
    function_block = tool.function
    if isinstance(function_block, dict):
        name = function_block.get("name")
        if isinstance(name, str):
            return name
    dumped = tool.model_dump(exclude_none=True)
    name = dumped.get("name")
    if isinstance(name, str):
        return name
    return None


def coerce_arguments(arguments: Any) -> Any:
    """Parse the function arguments string into structured data when possible."""
    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments

    return arguments


def iter_session_files(logs_dir: Path) -> Iterator[Path]:
    """Yield all JSONL session files in the logs directory."""
    for path in sorted(logs_dir.rglob("*.jsonl")):
        if path.is_file():
            yield path


def extract_date_from_path(path: Path, logs_root: Path) -> date | None:
    """Extract date from the file path structure (e.g., proxy_logs/2026-01-13/...)."""
    try:
        relative = path.relative_to(logs_root)
        # Assume structure: YYYY-MM-DD/session.jsonl
        if len(relative.parts) >= 2:
            date_str = relative.parts[0]
            return date.fromisoformat(date_str)
    except (ValueError, IndexError):
        pass

    # Fallback: try to parse from filename or use file modification time
    return None


def validate_session_entries(
    entries: list[LogEntry],
    *,
    source_path: Path | None = None,
) -> SessionValidationResult:
    """Validate file-order monotonicity and handle trailing shorter entries.

    Rules:
    - Timestamps must be strictly increasing in file order.
    - If the last entry's request.messages length is below the session max, use
      the last entry with the maximum length instead (warn and truncate).
    """
    if not entries:
        return SessionValidationResult(entries=[], had_warning=False)
    had_warning = False
    def _format_error(message: str, entry: LogEntry) -> str:
        if source_path is not None:
            return f"{message} (session={entry.session_id}, file={source_path})"
        return f"{message} (session={entry.session_id})"

    expected_session_id = entries[0].session_id
    for idx, entry in enumerate(entries, start=1):
        if entry.session_id != expected_session_id:
            raise SessionValidationError(
                _format_error(
                    "Session ID mismatch at entry {index}".format(index=idx),
                    entry,
                )
            )

    previous_timestamp = None
    for idx, entry in enumerate(entries, start=1):
        timestamp = entry.timestamp

        if previous_timestamp is not None:
            if timestamp <= previous_timestamp:
                raise SessionValidationError(
                    _format_error(
                        "Timestamp not increasing at entry {index}".format(index=idx),
                        entry,
                    )
                )

        previous_timestamp = timestamp

    counts = [len(entry.request.messages) for entry in entries]
    last_count = counts[-1]
    max_count = max(counts)
    if last_count < max_count:
        last_max_index = max(i for i, count in enumerate(counts) if count == max_count)
        last_entry_index = len(entries) - 1
        logger.warning(
            "Last entry {last_index} message count ({last}) below max ({max}); using entry {index} instead",
            last=last_count,
            max=max_count,
            last_index=last_entry_index + 1,
            index=last_max_index + 1,
        )
        had_warning = True
        entries = entries[: last_max_index + 1]

    return SessionValidationResult(entries=entries, had_warning=had_warning)


def apply_cleanups_to_entries(entries: list[LogEntry], cleanup: CleanupConfig) -> list[LogEntry]:
    """Apply cleanup config to each entry's request messages."""
    cleaned_entries: list[LogEntry] = []
    for entry in entries:
        raw_messages = [msg.model_dump(exclude_none=True) for msg in entry.request.messages]
        cleaned_messages = apply_cleanups(raw_messages, cleanup)
        message_models = [Message.model_validate(msg) for msg in cleaned_messages]
        if cleanup.drop_summary_trick_entries and _is_summary_trick_entry(message_models, entry.response):
            continue
        cleaned_request = entry.request.model_copy(update={"messages": message_models})
        cleaned_entries.append(entry.model_copy(update={"request": cleaned_request}))
    return cleaned_entries


def _is_summary_trick_entry(messages: list[Message], response: ResponsePayload | None) -> bool:
    """Detect Goose tool-usage trick entries to drop."""
    if len(messages) < 2:
        return False
    first, second = messages[0], messages[1]
    if first.role != "system" or second.role != "user":
        return False
    system_text = _message_to_text(first)
    user_text = _message_to_text(second)
    if system_text is None or user_text is None:
        return False
    if not system_text.strip().startswith("Reply with only a description in four words or less"):
        return False
    if not user_text.strip().startswith("Here are the first few user messages:"):
        return False
    if response is None or not response.choices:
        return False
    reply = response.choices[0].message
    if reply.role != "assistant":
        return False
    reply_text = _message_to_text(reply)
    if reply_text is None:
        return False
    word_count = len(reply_text.strip().split())
    return 1 <= word_count <= 6


def _message_to_text(message: Message) -> str | None:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text" and isinstance(chunk.get("text"), str):
                parts.append(chunk["text"])
        return "".join(parts)
    return None


def process_logs_directory(
    logs_dir: Path,
    json_tool_calls: bool = False,
) -> dict[date, list[ConversationRecord]]:
    """
    Process all session files in a directory.

    Returns a dictionary mapping dates to lists of conversation records.
    """
    records_by_date: dict[date, list[ConversationRecord]] = {}

    for session_path in iter_session_files(logs_dir):
        logger.debug("Processing session file: {path}", path=session_path)

        entries = read_session_file(session_path)
        if not entries:
            logger.warning("Empty or invalid session file: {path}", path=session_path)
            continue

        record = process_session(
            entries,
            json_tool_calls=json_tool_calls,
            source_path=session_path,
        )
        if record is None:
            continue

        # Determine date for this session
        session_date = extract_date_from_path(session_path, logs_dir)
        if session_date is None:
            # Fallback to timestamp from first entry
            session_date = entries[0].timestamp.date()

        if session_date not in records_by_date:
            records_by_date[session_date] = []
        records_by_date[session_date].append(record)

    return records_by_date


def process_logs_directory_with_tool_usage(
    logs_dir: Path,
    json_tool_calls: bool = False,
    *,
    cleanup: CleanupConfig | None = None,
    trace: bool = False,
    integrity_callback: Callable[
        [Path, str, ConversationRecord | None, ConversationRecord | None, bool, list[LogEntry] | None], None
    ]
    | None = None,
) -> tuple[dict[date, list[ConversationRecord]], dict[str, list[ConversationRecord]], list[SessionToolUsageSample]]:
    """Process all sessions and return per-session tool usage samples."""
    records_by_date: dict[date, list[ConversationRecord]] = {}
    records_by_scenario: dict[str, list[ConversationRecord]] = {}
    samples: list[SessionToolUsageSample] = []

    for session_path in iter_session_files(logs_dir):
        logger.debug("Processing session file: {path}", path=session_path)

        entries = read_session_file(session_path)
        if cleanup is not None:
            entries = apply_cleanups_to_entries(entries, cleanup)
        scenario = extract_session_name_from_path(session_path)
        if not entries:
            logger.warning("Empty or invalid session file: {path}", path=session_path)
            if integrity_callback is not None:
                integrity_callback(session_path, scenario, None, None, json_tool_calls, None)
            continue

        try:
            validation = validate_session_entries(entries, source_path=session_path)
        except SessionValidationError:
            if trace:
                emit_role_trace(entries)
            raise
        if not validation.entries:
            if integrity_callback is not None:
                integrity_callback(session_path, scenario, None, None, json_tool_calls, None)
            continue

        if trace and validation.had_warning:
            emit_role_trace(entries)

        raw_record = _process_validated_session(
            validation.entries,
            json_tool_calls=False,
            source_path=session_path,
        )
        if raw_record is None:
            if integrity_callback is not None:
                integrity_callback(session_path, scenario, None, None, json_tool_calls, validation.entries)
            continue

        tool_call_count = count_tool_call_requests(raw_record.messages)
        client_turns, assistant_turns, assistant_tool_turns = count_turn_groups(raw_record.messages)
        record = raw_record
        if json_tool_calls:
            record = ConversationRecord(messages=transform_messages(raw_record.messages), tools=raw_record.tools)

        if integrity_callback is not None:
            integrity_callback(session_path, scenario, raw_record, record, json_tool_calls, validation.entries)

        session_date = extract_date_from_path(session_path, logs_dir)
        if session_date is None:
            session_date = entries[0].timestamp.date()

        records_by_date.setdefault(session_date, []).append(record)
        records_by_scenario.setdefault(scenario, []).append(record)

        samples.append(
            SessionToolUsageSample(
                date=session_date,
                session=scenario,
                tool_call_count=tool_call_count,
                tool_definition_names=extract_tool_names(record.tools),
                client_turns=client_turns,
                assistant_turns=assistant_turns,
                assistant_turns_with_tools=assistant_tool_turns,
            )
        )

    return records_by_date, records_by_scenario, samples


def emit_role_trace(entries: list[LogEntry]) -> None:
    """Log per-request role traces for a session file."""
    logger.info("")
    base_sequences: list[list[str]] = []
    display_sequences: list[str] = []
    for index, entry in enumerate(entries, start=1):
        roles = build_role_trace_tokens(entry.request.messages)
        tools = entry.request.tools or []
        mcps = extract_tool_mcp_prefixes(tools)
        mcps_label = ",".join(mcps) if mcps else "-"
        response_marker = build_response_marker(entry)
        response_id = entry.response.id if entry.response is not None else "-"
        roles_with_response = _join_trace_tokens(roles, response_marker=response_marker)
        base_sequences.append(roles)
        display_sequences.append(roles_with_response)
        line = (
            f"<{index:03d}>: roles={roles_with_response} "
            f"msgs={len(entry.request.messages)} "
            f"tools={len(tools)} mcps={mcps_label} resp.id={response_id}"
        )
        logger.info(line)

    for idx in range(len(base_sequences) - 1):
        current = base_sequences[idx]
        nxt = base_sequences[idx + 1]
        if not _tokens_start_with(nxt, current):
            prefix_len = _common_prefix_len_with_assistant_equivalence(current, nxt)
            prev_start_idx = prefix_len if prefix_len > 0 else 1
            next_start_idx = prefix_len if prefix_len > 0 else 1
            logger.info("")
            logger.info(
                "CTX-SWITCH at <{cur:03d}> -> <{nxt:03d}>: roles {cur_roles} -> {next_roles}",
                cur=idx + 1,
                nxt=idx + 2,
                cur_roles=display_sequences[idx],
                next_roles=display_sequences[idx + 1],
            )
            logger.info("Previous chat (entry <{cur:03d}>):", cur=idx + 1)
            _log_entry_messages(
                entries[idx],
                start_idx=prev_start_idx,
                max_count=5,
            )
            logger.info("New chat (entry <{nxt:03d}>):", nxt=idx + 2)
            _log_entry_messages(
                entries[idx + 1],
                start_idx=next_start_idx,
                max_count=5,
            )


def build_role_trace(messages: list[Any]) -> str:
    """Build an ASCII role trace for the message sequence."""
    return _join_trace_tokens(build_role_trace_tokens(messages))


def build_normalized_role_trace(messages: list[Message]) -> str:
    """Build a role trace with developer normalized to system."""
    return _join_trace_tokens(build_normalized_role_trace_tokens(messages))


def build_role_trace_tokens(messages: list[Any]) -> list[str]:
    """Build a role trace token list for the message sequence."""
    trace: list[str] = []
    for message in messages:
        role = _get_message_field(message, "role")
        content = _get_message_field(message, "content")
        tool_calls = _get_message_field(message, "tool_calls")
        if role == "user":
            trace.append(classify_user_message(content))
        elif role == "system":
            trace.append("S")
        elif role == "assistant":
            trace.append(_classify_assistant_message(content, tool_calls))
        elif role == "tool":
            trace.append("T")
        elif role == "developer":
            trace.append("D")
        elif isinstance(role, str) and role:
            trace.append(role[:1].upper())
    return trace


def build_normalized_role_trace_tokens(messages: list[Message]) -> list[str]:
    """Build a role trace token list with developer normalized to system."""
    trace: list[str] = []
    for message in messages:
        role = message.role
        if role == "developer":
            role = "system"

        if role == "user":
            trace.append(classify_user_message(message.content))
        elif role == "system":
            trace.append("S")
        elif role == "assistant":
            trace.append(_classify_assistant_message(message.content, message.tool_calls))
        elif role == "tool":
            trace.append("T")
        elif isinstance(role, str) and role:
            trace.append(role[:1].upper())
    return trace


def _classify_assistant_message(content: Any, tool_calls: Any) -> str:
    has_text = _has_non_whitespace_text(content)
    has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
    if has_text and has_tool_calls:
        return "Act"
    if has_text:
        return "Ac"
    if has_tool_calls:
        return "At"
    return "A"


def _has_non_whitespace_text(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text" and isinstance(chunk.get("text"), str):
                parts.append(chunk["text"])
        return bool("".join(parts).strip())
    return False


def _join_trace_tokens(tokens: list[str], *, response_marker: str = "") -> str:
    if response_marker:
        tokens = [*tokens, response_marker]
    return " ".join(tokens)


def _tokens_start_with(sequence: list[str], prefix: list[str]) -> bool:
    if len(sequence) < len(prefix):
        return False
    return sequence[: len(prefix)] == prefix


def _common_prefix_len_with_assistant_equivalence(left: list[str], right: list[str]) -> int:
    """Return matched prefix length allowing AcAt -> Act folding."""
    left_idx = 0
    right_idx = 0
    matched = 0
    while left_idx < len(left) and right_idx < len(right):
        left_token = left[left_idx]
        right_token = right[right_idx]
        if left_token == right_token:
            left_idx += 1
            right_idx += 1
            matched += 1
            continue
        if (
            left_token == "Ac"
            and left_idx + 1 < len(left)
            and left[left_idx + 1] == "At"
            and right_token == "Act"
        ):
            left_idx += 2
            right_idx += 1
            matched += 1
            continue
        break
    return matched


def is_prefix_with_assistant_equivalence(previous: list[str], nxt: list[str]) -> bool:
    """Return True if previous matches the start of nxt with AcAt -> Act folding."""
    prev_idx = 0
    next_idx = 0
    while prev_idx < len(previous) and next_idx < len(nxt):
        prev_token = previous[prev_idx]
        next_token = nxt[next_idx]
        if prev_token == next_token:
            prev_idx += 1
            next_idx += 1
            continue
        if (
            prev_token == "Ac"
            and prev_idx + 1 < len(previous)
            and previous[prev_idx + 1] == "At"
            and next_token == "Act"
        ):
            prev_idx += 2
            next_idx += 1
            continue
        return False
    return prev_idx == len(previous)


def _get_message_field(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _start_from_last_common_prev(*, current_len: int, overlap: int) -> int:
    """Return 1-based index for last common role in the previous sequence."""
    if current_len <= 0:
        return 1
    if overlap <= 0:
        return 1
    return max(1, current_len - overlap + 1)


def _start_from_last_common_next(*, overlap: int) -> int:
    """Return 1-based index for last common role in the next sequence."""
    if overlap <= 0:
        return 1
    return max(1, overlap)


def build_response_marker(entry: LogEntry) -> str:
    """Return *A if response contains an assistant message, otherwise -."""
    response = entry.response
    if response is None or not response.choices:
        return ""
    message = response.choices[0].message
    role = getattr(message, "role", None)
    if role == "assistant":
        return "*A"
    return ""


def classify_user_message(content: Any) -> str:
    """Return U for user text, R for user tool_result-only content."""
    if isinstance(content, list):
        tool_result_only = True
        has_chunks = False
        for chunk in content:
            if not isinstance(chunk, dict):
                tool_result_only = False
                break
            has_chunks = True
            if chunk.get("type") != "tool_result":
                tool_result_only = False
                break
        if tool_result_only and has_chunks:
            return "R"
        return "U"
    return "U"


def extract_tool_mcp_prefixes(tools: list[ToolDefinition]) -> list[str]:
    """Extract MCP prefixes from tool definitions."""
    prefixes: set[str] = set()
    for tool in tools:
        name: str | None = None
        function_block = tool.function
        if isinstance(function_block, dict):
            fn_name = function_block.get("name")
            if isinstance(fn_name, str):
                name = fn_name
        if name is None and hasattr(tool, "model_dump"):
            dumped = tool.model_dump(exclude_none=True)
            dumped_name = dumped.get("name")
            if isinstance(dumped_name, str):
                name = dumped_name
        if isinstance(name, str) and "__" in name:
            prefixes.add(name.split("__", maxsplit=1)[0])
    return sorted(prefixes)


def _log_entry_messages(entry: LogEntry, *, start_idx: int = 1, max_count: int = 0) -> None:
    """Log request/response messages for a single entry."""
    emitted = 0
    for idx, message in enumerate(entry.request.messages, start=1):
        if idx < start_idx:
            continue
        if 0 < max_count <= emitted:
            break
        role = getattr(message, "role", "")
        content = _stringify_message_content(getattr(message, "content", None))
        tool_calls = getattr(message, "tool_calls", None)
        if role == "assistant" and not content and tool_calls:
            tool_names = _extract_tool_call_names(tool_calls)
            tool_names_label = ",".join(tool_names) if tool_names else "<unknown>"
            logger.info(
                "  msg[{idx:02d}] role={role} tool_call={tool_names}",
                idx=idx,
                role=role,
                tool_names=tool_names_label,
            )
        else:
            logger.info(
                "  msg[{idx:02d}] role={role} content={content}",
                idx=idx,
                role=role,
                content=content,
            )
        emitted += 1

    response = entry.response
    if response is None or not response.choices:
        return
    msg = response.choices[0].message
    role = getattr(msg, "role", "")
    content = _stringify_message_content(getattr(msg, "content", None))
    tool_calls = getattr(msg, "tool_calls", None)
    if role == "assistant" and not content and tool_calls:
        tool_names = _extract_tool_call_names(tool_calls)
        tool_names_label = ",".join(tool_names) if tool_names else "<unknown>"
        logger.info(
            "  resp role={role} tool_call={tool_names}",
            role=role,
            tool_names=tool_names_label,
        )
    else:
        logger.info("  resp role={role} content={content}", role=role, content=content)


def _stringify_message_content(content: Any, *, limit: int = 800) -> str:
    """Render message content as a single line, with truncation."""
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if not isinstance(chunk, dict):
                continue
            chunk_type = chunk.get("type")
            if chunk_type == "text" and isinstance(chunk.get("text"), str):
                parts.append(chunk["text"])
            elif chunk_type == "tool_result":
                chunk_content = chunk.get("content")
                if isinstance(chunk_content, str) and chunk_content:
                    parts.append(chunk_content)
                else:
                    parts.append("<tool_result>")
        text = "".join(parts)
    else:
        text = str(content)

    text = text.replace("\n", "\\n")
    if len(text) > limit:
        return f"{text[:limit]}…[{len(text)} chars]"
    return text


def _extract_tool_call_names(tool_calls: Any) -> list[str]:
    """Extract tool names from tool_calls payloads."""
    names: list[str] = []
    if not isinstance(tool_calls, list):
        return names
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function_block = call.get("function")
        if isinstance(function_block, dict):
            name = function_block.get("name")
            if isinstance(name, str):
                names.append(name)
                continue
        name = call.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def extract_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    """Extract tool names from a flattened `ConversationRecord.tools` list."""
    names: set[str] = set()
    for tool in tools:
        name = tool.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names
