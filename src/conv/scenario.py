"""Scenario/session naming utilities shared across exports and statistics."""

import re
from pathlib import Path

_GENERATED_SESSION_UNIX_NS_SUFFIX = re.compile(r"^(?P<prefix>.+)-(?P<unix_ns>\d{19})$")
_GENERATED_SESSION_UUID_SUFFIX = re.compile(
    r"^(?P<prefix>.+)-(?P<uuid>[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})$",
    flags=re.IGNORECASE,
)


def extract_session_name_from_path(session_path: Path) -> str:
    """Extract a stable session/scenario name from a log filename.

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
