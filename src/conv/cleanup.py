"""Message cleanup utilities for exported conversation records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_GOOSE_INFO_MSG_RE = re.compile(r"\A\s*<info-msg>\s*.*?\s*</info-msg>\s*\Z", re.DOTALL)
_GOOSE_SYSTEM_INSTRUCTION_RE = re.compile(
    r"general-purpose\s+AI\s+agent\s+called\s+goose",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CleanupConfig:
    """Options for cleaning up noisy/injected messages during conversion."""

    drop_goose_info_user_messages: bool = True
    drop_summary_trick_entries: bool = True
    drop_empty_assistant_followed_by_assistant: bool = True


def apply_cleanups(messages: list[dict[str, Any]], config: CleanupConfig) -> list[dict[str, Any]]:
    """Apply configured cleanup steps to a message list."""
    cleaned = messages
    if config.drop_goose_info_user_messages:
        cleaned = _drop_goose_info(cleaned)
        cleaned = _remove_goose_system_instruction(cleaned)
    if config.drop_empty_assistant_followed_by_assistant:
        cleaned = _drop_empty_assistant_followed_by_assistant(cleaned)
    return cleaned


def _remove_goose_system_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize Goose default system instruction messages.

    Replaces any `role=system` message whose text matches the default Goose
    bootstrap prompt (identified by the phrase `general-purpose AI agent called goose`)
    with a concise, stable instruction.
    """
    replacement_text = "You are a general-purpose AI agent"
    kept: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "system":
            kept.append(message)
            continue

        content = message.get("content")
        text = _content_to_text(content)
        if text is not None and _GOOSE_SYSTEM_INSTRUCTION_RE.search(text):
            updated = dict(message)
            updated["content"] = replacement_text
            kept.append(updated)
            continue

        kept.append(message)
    return kept


def _drop_goose_info(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop Goose-injected `<info-msg>...</info-msg>` messages.

    Only drops messages where:
    - `role` is `"user"`, and
    - the entire textual content is wrapped in `<info-msg>...</info-msg>` tags.
    """
    kept: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "user":
            kept.append(message)
            continue

        content = message.get("content")
        text = _content_to_text(content)
        if text is not None and _GOOSE_INFO_MSG_RE.match(text):
            continue

        kept.append(message)
    return kept


def _drop_empty_assistant_followed_by_assistant(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop empty assistant messages that are immediately followed by another assistant message."""
    kept: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        if (
            message.get("role") == "assistant"
            and idx + 1 < len(messages)
            and messages[idx + 1].get("role") == "assistant"
            and _is_empty_assistant_message(message)
        ):
            continue
        kept.append(message)
    return kept


def _is_empty_assistant_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    text = _content_to_text(content)
    if text is None or text.strip():
        return False
    if message.get("tool_calls"):
        return False
    if message.get("tool_call_id"):
        return False
    return True


def _content_to_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text" and isinstance(chunk.get("text"), str):
                parts.append(chunk["text"])
        return "".join(parts)
    return None
