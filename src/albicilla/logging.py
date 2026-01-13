"""JSONL logging utilities for the proxy."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from .config import Settings
from .models import LogEntry


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


def get_log_path(settings: Settings, session_id: str) -> Path:
    """Get the log file path for a session.

    Args:
        settings: Server settings containing log_root.
        session_id: The session identifier (will be sanitized).

    Returns:
        Path to the JSONL log file.
    """
    safe_session_id = sanitize_session_id(session_id)
    date_folder = datetime.now(UTC).strftime("%Y-%m-%d")
    return settings.log_root / date_folder / f"{safe_session_id}.jsonl"


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
    log_path = get_log_path(settings, session_id)

    # Ensure directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = LogEntry(
        timestamp=datetime.now(UTC).isoformat(),
        session_id=sanitize_session_id(session_id),
        request=request_data,
        response=response_data,
    )

    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.model_dump(), ensure_ascii=False) + "\n")
        logger.debug(f"Logged entry to {log_path}")
    except OSError as e:
        logger.error(f"Failed to write log entry to {log_path}: {e}")
        raise IOError(f"Log write failed: {e}") from e