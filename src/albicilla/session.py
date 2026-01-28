"""Session resolution for the proxy."""

import threading
import time

from fastapi import Request

from .config import Settings

# In-memory mapping from bearer token to a generated session ID
# Thread-safe via a shared lock.
_token_session_map: dict[str, str] = {}
_token_lock = threading.Lock()
_session_prefix: str | None = None


def set_session_prefix(prefix: str | None) -> None:
    """Set a global prefix for generated session IDs."""
    global _session_prefix
    if prefix is None:
        _session_prefix = None
        return
    cleaned = prefix.strip()
    if not cleaned:
        _session_prefix = None
        return
    _session_prefix = cleaned


def clear_session_prefix() -> None:
    """Clear the global session prefix."""
    global _session_prefix
    _session_prefix = None


def _generate_session_id(default_prefix: str) -> str:
    prefix = _session_prefix or default_prefix
    # Use a fixed-width, digits-only unix timestamp suffix for stable
    # lexicographic sorting (and to avoid mixed alpha/decimal UUID strings).
    suffix = f"{time.time_ns():019d}"
    return f"{prefix}-{suffix}"


async def resolve_session_id(request: Request, payload_user: str | None, settings: Settings) -> str:
    """Resolve the session ID using the defined fallback strategy.

    Resolution order:
    1. `user` field from request payload
    2. X-Session-Id header
    3. Bearer token mapping (persists in-memory for process lifetime)
    4. Timestamp fallback (per request)

    Args:
        request: The FastAPI request object.
        payload_user: The `user` field from the request payload, if provided.
        settings: Server settings.

    Returns:
        The resolved session identifier.
    """
    # 1. Check request payload user field
    if payload_user:
        return payload_user

    # 2. Check session header
    session_header = request.headers.get(settings.session_header)
    if session_header:
        return session_header

    # 3. Check Authorization bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            with _token_lock:
                if token not in _token_session_map:
                    _token_session_map[token] = _generate_session_id("session")
                return _token_session_map[token]

    # 4. Fallback to a generated timestamp-based ID
    return _generate_session_id("anon")


def clear_token_map() -> None:
    """Clear the token-to-session mapping. Useful for testing."""
    with _token_lock:
        _token_session_map.clear()
