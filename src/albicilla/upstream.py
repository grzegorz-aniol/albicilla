"""Upstream API client for forwarding requests."""

from dataclasses import dataclass

import httpx
from loguru import logger

from .config import Settings
from .models import ChatCompletionRequest

# Headers that should not be forwarded to upstream
EXCLUDED_REQUEST_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "upgrade",
    "accept-encoding",  # Let httpx handle encoding
}

# Headers from upstream that should not be forwarded to client
EXCLUDED_RESPONSE_HEADERS = {
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "content-encoding",
}


class UpstreamError(Exception):
    """Error from upstream service."""

    def __init__(self, status_code: int, detail: str, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.detail = detail
        self.headers = headers or {}
        super().__init__(f"Upstream error {status_code}: {detail}")


@dataclass
class UpstreamResponse:
    """Response from upstream service."""

    data: dict
    headers: dict[str, str]


def filter_request_headers(headers: dict[str, str]) -> dict[str, str]:
    """Filter headers to forward to upstream.

    Args:
        headers: Original request headers.

    Returns:
        Filtered headers safe to forward.
    """
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in EXCLUDED_REQUEST_HEADERS
    }


def filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    """Filter headers to forward from upstream response.

    Args:
        headers: Upstream response headers.

    Returns:
        Filtered headers safe to forward to client.
    """
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in EXCLUDED_RESPONSE_HEADERS
    }


async def forward_request(
    payload: ChatCompletionRequest,
    settings: Settings,
    request_headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> UpstreamResponse:
    """Forward a chat completion request to the upstream service.

    Args:
        payload: The chat completion request to forward.
        settings: Server settings containing upstream endpoint.
        request_headers: Headers from the original request to forward.
        timeout: Request timeout in seconds.

    Returns:
        UpstreamResponse containing response data and headers.

    Raises:
        UpstreamError: If the upstream service returns an error.
    """
    url = settings.upstream_chat_completions_url

    # Prepare headers - start with filtered original headers
    headers = filter_request_headers(request_headers or {})
    # Remove any existing content-type variations and set our own
    headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
    headers["Content-Type"] = "application/json"

    # Debug logging for header forwarding
    auth_header = headers.get("authorization") or headers.get("Authorization")
    auth_preview = f"{auth_header[:20]}..." if auth_header and len(auth_header) > 20 else auth_header
    logger.debug(f"Original headers keys: {list(request_headers.keys()) if request_headers else []}")
    logger.debug(f"Forwarding headers keys: {list(headers.keys())}")
    logger.debug(f"Authorization header: {auth_preview}")

    # Prepare request body - exclude None values for cleaner requests
    request_data = payload.model_dump(mode="json", exclude_none=True)

    # Ensure model is set - use default if not provided or empty
    original_model = request_data.get("model")
    if not original_model:  # Handles None, missing key, and empty string
        logger.info(f"Request missing model parameter (was: {original_model!r}), using default: {settings.default_model}")
        request_data["model"] = settings.default_model

    logger.debug(f"Forwarding request to {url} with model: {request_data['model']}")
    logger.debug(f"Request body keys: {list(request_data.keys())}")
    logger.debug(f"Request body model value: {request_data.get('model')!r}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                json=request_data,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException as e:
            logger.error(f"Upstream timeout: {e}")
            raise UpstreamError(504, "Upstream service timeout") from e
        except httpx.ConnectError as e:
            logger.error(f"Upstream connection error: {e}")
            raise UpstreamError(502, f"Cannot connect to upstream: {settings.upstream_endpoint}") from e
        except httpx.RequestError as e:
            logger.error(f"Upstream request error: {e}")
            raise UpstreamError(502, f"Upstream request failed: {e}") from e

    response_headers = filter_response_headers(response.headers)

    if response.status_code >= 400:
        logger.warning(f"Upstream returned {response.status_code}: {response.text[:500]}")
        raise UpstreamError(response.status_code, response.text, response_headers)

    try:
        return UpstreamResponse(data=response.json(), headers=response_headers)
    except ValueError as e:
        logger.error(f"Invalid JSON from upstream: {response.text[:500]}")
        raise UpstreamError(502, "Invalid JSON response from upstream", response_headers) from e
