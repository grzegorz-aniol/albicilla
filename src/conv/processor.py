"""Core processing logic for conversation logs."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

from .models import ConversationRecord, LogEntry, SessionToolUsageRow, ToolDefinition

_GENERATED_SESSION_UNIX_NS_SUFFIX = re.compile(r"^(?P<prefix>.+)-(?P<unix_ns>\d{19})$")
_GENERATED_SESSION_UUID_SUFFIX = re.compile(
    r"^(?P<prefix>.+)-(?P<uuid>[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})$",
    flags=re.IGNORECASE,
)


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
) -> ConversationRecord | None:
    """
    Process a session's log entries and produce a conversation record.

    Takes the last entry, extracts messages and tools from the request,
    appends the final assistant response, and optionally transforms tool calls.
    """
    if not entries:
        return None

    # Sort by timestamp and take the last entry
    sorted_entries = sorted(entries, key=lambda e: e.timestamp)
    last_entry = sorted_entries[-1]

    # Extract messages from request
    messages: list[dict[str, Any]] = [
        msg.model_dump(exclude_none=True) for msg in last_entry.request.messages
    ]

    # Extract tools from request (or empty list)
    tools = extract_tool_schemas(last_entry.request.tools)

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
) -> tuple[ConversationRecord | None, int]:
    """Process a session and return the record plus tool-call request count."""
    record = process_session(entries, json_tool_calls=False)
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
        if isinstance(content, str):
            count += content.count("<tool_call>")

    return count


def extract_session_name_from_path(session_path: Path) -> str:
    """Extract a stable session name from a log filename.

    For generated filenames like `arxiv-python-<unix_ns>.jsonl` or
    `market-brief-<uuid>.jsonl`, returns the scenario prefix.
    Also strips short, numeric prompt/task IDs such as `google-maps-es-01.jsonl`,
    returning the stable session prefix (`google-maps-es`) for grouping.
    Otherwise returns the filename stem.
    """
    stem = session_path.stem
    return _normalize_session_stem_for_grouping(stem)


def _normalize_session_stem_for_grouping(stem: str) -> str:
    """Normalize a session filename stem to a stable scenario/session name.

    This function is intentionally conservative: it removes only well-known
    generated suffixes (UUID, 19-digit unix-ns) and short numeric suffixes when
    they look like an appended prompt/task ID.
    """
    normalized = stem
    # Allow multiple suffixes (e.g. "<scenario>-01-<uuid>" or "<scenario>-01-<unix_ns>").
    for _ in range(3):
        updated = _strip_generated_session_suffix(normalized)
        updated = _strip_numeric_prompt_suffix(updated)
        if updated == normalized:
            break
        normalized = updated
    return normalized


def _strip_generated_session_suffix(stem: str) -> str:
    match = _GENERATED_SESSION_UNIX_NS_SUFFIX.match(stem)
    if match is not None:
        return match.group("prefix")
    match = _GENERATED_SESSION_UUID_SUFFIX.match(stem)
    if match is not None:
        return match.group("prefix")
    return stem


def _strip_numeric_prompt_suffix(stem: str) -> str:
    parts = stem.split("-")
    if len(parts) < 3:
        return stem

    last = parts[-1]
    prev = parts[-2]

    # Treat a short numeric trailing segment as a prompt/task ID when the
    # preceding segment isn't also numeric (avoids stripping YYYY-MM-DD-like
    # suffixes).
    if last.isdigit() and 1 <= len(last) <= 6 and not prev.isdigit():
        prefix = "-".join(parts[:-1])
        return prefix or stem
    return stem




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

        record = process_session(entries, json_tool_calls=json_tool_calls)
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
    integrity_callback: Callable[
        [Path, str, ConversationRecord | None, ConversationRecord | None, bool], None
    ]
    | None = None,
) -> tuple[dict[date, list[ConversationRecord]], list[SessionToolUsageRow]]:
    """Process all sessions and also return a per-session tool usage report."""
    records_by_date: dict[date, list[ConversationRecord]] = {}
    tool_calls_by_date_and_scenario: dict[tuple[date, str], int] = {}
    tool_names_by_date_and_scenario: dict[tuple[date, str], set[str]] = {}
    session_count_by_date_and_scenario: dict[tuple[date, str], int] = {}

    for session_path in iter_session_files(logs_dir):
        logger.debug("Processing session file: {path}", path=session_path)

        entries = read_session_file(session_path)
        scenario = extract_session_name_from_path(session_path)
        if not entries:
            logger.warning("Empty or invalid session file: {path}", path=session_path)
            if integrity_callback is not None:
                integrity_callback(session_path, scenario, None, None, json_tool_calls)
            continue

        raw_record = process_session(entries, json_tool_calls=False)
        if raw_record is None:
            if integrity_callback is not None:
                integrity_callback(session_path, scenario, None, None, json_tool_calls)
            continue

        tool_call_count = count_tool_call_requests(raw_record.messages)
        record = raw_record
        if json_tool_calls:
            record = ConversationRecord(messages=transform_messages(raw_record.messages), tools=raw_record.tools)

        if integrity_callback is not None:
            integrity_callback(session_path, scenario, raw_record, record, json_tool_calls)

        session_date = extract_date_from_path(session_path, logs_dir)
        if session_date is None:
            session_date = entries[0].timestamp.date()

        records_by_date.setdefault(session_date, []).append(record)

        key = (session_date, scenario)
        tool_calls_by_date_and_scenario[key] = tool_calls_by_date_and_scenario.get(key, 0) + tool_call_count
        session_count_by_date_and_scenario[key] = session_count_by_date_and_scenario.get(key, 0) + 1
        if key not in tool_names_by_date_and_scenario:
            tool_names_by_date_and_scenario[key] = set()
        tool_names_by_date_and_scenario[key].update(extract_tool_names(record.tools))

    keys = set(tool_calls_by_date_and_scenario) | set(tool_names_by_date_and_scenario)
    report_rows = [
        SessionToolUsageRow(
            date=day,
            session=scenario,
            session_count=session_count_by_date_and_scenario.get((day, scenario), 0),
            tool_call_count=tool_calls_by_date_and_scenario.get((day, scenario), 0),
            tool_definition_count=len(
                tool_names_by_date_and_scenario.get((day, scenario), set())
            ),
        )
        for (day, scenario) in sorted(keys)
    ]

    return records_by_date, report_rows


def extract_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    """Extract tool names from a flattened `ConversationRecord.tools` list."""
    names: set[str] = set()
    for tool in tools:
        name = tool.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names
