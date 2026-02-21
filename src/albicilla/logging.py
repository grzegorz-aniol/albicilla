"""JSONL logging utilities for the proxy."""

import asyncio
import json
import re
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .config import Settings
from .models import LogEntry

_session_locks: dict[str, asyncio.Lock] = {}
_session_locks_guard = threading.Lock()
_session_prefix_map: dict[str, str] = {}
_session_prefix_guard = threading.Lock()


def sanitize_session_id(session_id: str) -> str:
    """Sanitize session ID to prevent path traversal.

    Args:
        session_id: Raw session identifier.

    Returns:
        Sanitized session ID containing only alphanumerics and underscores.
    """
    # Replace any non-alphanumeric chars (except underscore/hyphen) with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
    # Ensure it's not empty
    return sanitized or "unknown"


def set_session_file_prefix(session_id: str, prefix: str | None) -> None:
    """Set or clear the log file prefix for a session ID."""
    cleaned_session_id = session_id.strip()
    if not cleaned_session_id:
        return
    cleaned_prefix = prefix.strip() if prefix is not None else ""
    with _session_prefix_guard:
        if not cleaned_prefix:
            _session_prefix_map.pop(cleaned_session_id, None)
            return
        _session_prefix_map[cleaned_session_id] = cleaned_prefix


def clear_session_file_prefixes() -> None:
    """Clear all session file prefix overrides."""
    with _session_prefix_guard:
        _session_prefix_map.clear()


def _resolve_log_session_id(session_id: str) -> str:
    safe_session_id = sanitize_session_id(session_id)
    cleaned_session_id = session_id.strip()
    if not cleaned_session_id:
        return safe_session_id
    with _session_prefix_guard:
        prefix = _session_prefix_map.get(cleaned_session_id)
    if not prefix:
        return safe_session_id
    safe_prefix = sanitize_session_id(prefix)
    return f"{safe_prefix}-{safe_session_id}"


def configure_logging(verbose: bool) -> None:
    """Configure Loguru logging level and sinks.

    Args:
        verbose: Enable DEBUG logging when True, otherwise INFO.
    """
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=level)


def get_log_path(settings: Settings, session_id: str) -> Path:
    """Get the log file path for a session.

    Args:
        settings: Server settings containing log_root.
        session_id: The session identifier (will be sanitized).

    Returns:
        Path to the JSONL log file.
    """
    safe_session_id = _resolve_log_session_id(session_id)
    date_folder = datetime.now(UTC).strftime("%Y-%m-%d")
    return settings.log_root / date_folder / f"{safe_session_id}.jsonl"


def _get_log_path_for_safe_session(settings: Settings, safe_session_id: str) -> Path:
    date_folder = datetime.now(UTC).strftime("%Y-%m-%d")
    return settings.log_root / date_folder / f"{safe_session_id}.jsonl"


def _get_session_lock(safe_session_id: str) -> asyncio.Lock:
    with _session_locks_guard:
        lock = _session_locks.get(safe_session_id)
        if lock is None:
            lock = asyncio.Lock()
            _session_locks[safe_session_id] = lock
        return lock


def _build_entry(
    session_id: str,
    request_data: dict,
    response_data: dict,
) -> tuple[str, LogEntry]:
    safe_session_id = sanitize_session_id(session_id)
    entry = LogEntry(
        timestamp=datetime.now(UTC).isoformat(),
        session_id=safe_session_id,
        request=request_data,
        response=response_data,
    )
    log_session_id = _resolve_log_session_id(session_id)
    return log_session_id, entry


def _write_entry(log_path: Path, entry: LogEntry) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")


async def append_session_entry_async(
    settings: Settings,
    session_id: str,
    request_data: dict,
    response_data: dict,
) -> None:
    """Append a log entry to the session's JSONL file (async-safe).

    Args:
        settings: Server settings.
        session_id: The session identifier.
        request_data: The request payload as a dict.
        response_data: The response payload as a dict.

    Raises:
        IOError: If the log write fails.
    """
    log_session_id, entry = _build_entry(session_id, request_data, response_data)
    log_path = _get_log_path_for_safe_session(settings, log_session_id)
    lock = _get_session_lock(log_session_id)

    try:
        async with lock:
            _write_entry(log_path, entry)
        logger.debug(f"Logged entry to {log_path}")
    except OSError as e:
        logger.error(f"Failed to write log entry to {log_path}: {e}")
        raise IOError(f"Log write failed: {e}") from e


def append_session_entry(
    settings: Settings,
    session_id: str,
    request_data: dict,
    response_data: dict,
) -> None:
    """Append a log entry to the session's JSONL file.

    Args:
        settings: Server settings.
        session_id: The session identifier.
        request_data: The request payload as a dict.
        response_data: The response payload as a dict.

    Raises:
        IOError: If the log write fails.
    """
    log_session_id, entry = _build_entry(session_id, request_data, response_data)
    log_path = _get_log_path_for_safe_session(settings, log_session_id)

    try:
        _write_entry(log_path, entry)
        logger.debug(f"Logged entry to {log_path}")
    except OSError as e:
        logger.error(f"Failed to write log entry to {log_path}: {e}")
        raise IOError(f"Log write failed: {e}") from e
