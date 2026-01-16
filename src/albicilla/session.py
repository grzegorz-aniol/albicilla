"""Session resolution for the proxy."""

import asyncio
from uuid import uuid4

from fastapi import Request

from .config import Settings

# In-memory mapping from bearer token to session UUID
# Thread-safe via asyncio.Lock
_token_session_map: dict[str, str] = {}
_token_lock = asyncio.Lock()


async def resolve_session_id(request: Request, payload_user: str | None, settings: Settings) -> str:
    """Resolve the session ID using the defined fallback strategy.

    Resolution order:
    1. `user` field from request payload
    2. X-Session-Id header
    3. Bearer token mapping (persists in-memory for process lifetime)
    4. UUID fallback (per request)

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
            async with _token_lock:
                if token not in _token_session_map:
                    _token_session_map[token] = f"session-{uuid4().hex[:12]}"
                return _token_session_map[token]

    # 4. Fallback to UUID
    return f"anon-{uuid4().hex[:12]}"


def clear_token_map() -> None:
    """Clear the token-to-session mapping. Useful for testing."""
    _token_session_map.clear()