"""Upstream API client for forwarding requests."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

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

    def __init__(
        self,
        status_code: int,
        detail: str | None = None,
        headers: dict[str, str] | None = None,
        body: bytes | str | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.body = self._ensure_body_bytes(body, detail)
        self.detail = detail if detail is not None else self._body_as_text()
        super().__init__(f"Upstream error {status_code}: {self.detail}")

    @staticmethod
    def _ensure_body_bytes(body: bytes | str | None, detail: str | None) -> bytes | None:
        if body is not None:
            return body.encode("utf-8") if isinstance(body, str) else body
        if detail is None:
            return None
        return detail.encode("utf-8")

    def _body_as_text(self) -> str:
        if self.body is None:
            return ""
        return self.body.decode("utf-8", errors="ignore")

    def excerpt(self, limit: int = 100) -> str:
        """Return a shortened preview of the upstream error."""
        if self.detail:
            return self.detail[:limit]
        if self.body is None:
            return ""
        return self._body_as_text()[:limit]


@dataclass
class UpstreamResponse:
    """Response from upstream service."""

    data: dict
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class StreamingContext:
    """Context for streaming response with buffered content for logging."""

    headers: dict[str, str]
    first_chunk: dict | None = None
    last_chunk: dict | None = None
    choices: dict[int, "ChoiceAccumulator"] = field(default_factory=dict)
    _line_buffer: str = ""

    def process_chunk(self, chunk: bytes) -> None:
        """Process a chunk of bytes, extracting content for logging.

        Handles partial lines across chunk boundaries.
        """
        text = self._line_buffer + chunk.decode("utf-8", errors="ignore")
        lines = text.split("\n")

        # Last element may be incomplete - save for next chunk
        self._line_buffer = lines[-1]

        for line in lines[:-1]:
            self._process_line(line)

    def flush_buffer(self) -> None:
        """Process any remaining data in the line buffer."""
        if self._line_buffer:
            self._process_line(self._line_buffer)
            self._line_buffer = ""

    def _process_line(self, line: str) -> None:
        """Process a single SSE line."""
        line = line.strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            return

        try:
            chunk_data = json.loads(line[6:])
            if self.first_chunk is None:
                self.first_chunk = chunk_data
            self.last_chunk = chunk_data

            # Extract deltas across all choices
            if choices := chunk_data.get("choices"):
                for choice in choices:
                    index = int(choice.get("index", 0))
                    accumulator = self.choices.setdefault(index, ChoiceAccumulator())
                    if delta := choice.get("delta"):
                        accumulator.add_delta(delta)
                    if (finish_reason := choice.get("finish_reason")) is not None:
                        accumulator.finish_reason = finish_reason
        except json.JSONDecodeError:
            pass

    def content_part_count(self) -> int:
        """Count the buffered content parts across all choices."""
        return sum(len(choice.content_parts) for choice in self.choices.values())

    def build_complete_response(self) -> dict[str, Any]:
        """Build a complete response from buffered chunks.

        Returns a response matching non-streaming format for consistent logging.
        """
        if not self.first_chunk:
            return {}

        choices = [
            {
                "index": index,
                "message": accumulator.build_message(),
                "finish_reason": accumulator.finish_reason,
            }
            for index, accumulator in sorted(self.choices.items())
        ]

        response: dict[str, Any] = {
            "id": self.first_chunk.get("id"),
            "object": "chat.completion",
            "created": self.first_chunk.get("created"),
            "model": self.first_chunk.get("model"),
            "choices": choices,
        }

        # Include usage if present in last chunk (OpenAI includes it with stream_options)
        if self.last_chunk and (usage := self.last_chunk.get("usage")):
            response["usage"] = usage

        return response


@dataclass
class FunctionCallAccumulator:
    """Accumulate streaming function call fields."""

    name: str | None = None
    arguments_parts: list[str] = field(default_factory=list)

    def add_delta(self, data: dict[str, Any]) -> None:
        """Apply a function call delta to the accumulator."""
        if "name" in data:
            self.name = data.get("name")
        if "arguments" in data:
            arguments = data.get("arguments")
            if arguments is not None:
                self.arguments_parts.append(arguments)

    def build(self) -> dict[str, str] | None:
        """Build a function call payload."""
        if not self.name and not self.arguments_parts:
            return None
        return {
            "name": self.name or "",
            "arguments": "".join(self.arguments_parts),
        }


@dataclass
class ToolCallAccumulator:
    """Accumulate streaming tool call fields."""

    id: str | None = None
    type: str | None = None
    function: FunctionCallAccumulator = field(default_factory=FunctionCallAccumulator)

    def add_delta(self, data: dict[str, Any]) -> None:
        """Apply a tool call delta to the accumulator."""
        if "id" in data:
            self.id = data.get("id")
        if "type" in data:
            self.type = data.get("type")
        if (function_data := data.get("function")) and isinstance(function_data, dict):
            self.function.add_delta(function_data)

    def build(self) -> dict[str, Any] | None:
        """Build a tool call payload."""
        function_payload = self.function.build()
        if not self.id and not self.type and not function_payload:
            return None
        return {
            "id": self.id,
            "type": self.type or "function",
            "function": function_payload or {"name": "", "arguments": ""},
        }


@dataclass
class ChoiceAccumulator:
    """Accumulate streaming content and tool calls for a choice."""

    role: str | None = None
    content_parts: list[str] = field(default_factory=list)
    tool_calls: dict[int, ToolCallAccumulator] = field(default_factory=dict)
    function_call: FunctionCallAccumulator | None = None
    finish_reason: str | None = None

    def add_delta(self, delta: dict[str, Any]) -> None:
        """Apply a delta update to the choice accumulator."""
        if "role" in delta:
            self.role = delta.get("role")
        if "content" in delta:
            content = delta.get("content")
            if content is not None:
                self.content_parts.append(content)
        if (tool_calls := delta.get("tool_calls")) and isinstance(tool_calls, list):
            for tool_delta in tool_calls:
                index = int(tool_delta.get("index", 0))
                accumulator = self.tool_calls.setdefault(index, ToolCallAccumulator())
                if isinstance(tool_delta, dict):
                    accumulator.add_delta(tool_delta)
        if (function_call := delta.get("function_call")) and isinstance(function_call, dict):
            if self.function_call is None:
                self.function_call = FunctionCallAccumulator()
            self.function_call.add_delta(function_call)

    def build_message(self) -> dict[str, Any]:
        """Build the message payload from accumulated deltas."""
        content = "".join(self.content_parts) if self.content_parts else None
        tool_calls_payload = [
            payload
            for _, accumulator in sorted(self.tool_calls.items())
            if (payload := accumulator.build()) is not None
        ]
        function_call_payload = (
            self.function_call.build() if self.function_call is not None else None
        )

        message: dict[str, Any] = {
            "role": self.role or "assistant",
            "content": content,
        }
        if tool_calls_payload:
            message["tool_calls"] = tool_calls_payload
        if function_call_payload:
            message["function_call"] = function_call_payload
        return message


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
    logger.debug(f"Original headers keys: {list(request_headers.keys()) if request_headers else []}")
    logger.debug(f"Forwarding headers keys: {list(headers.keys())}")
    if auth_header:
        logger.debug("Authorization header set for upstream request")
    else:
        logger.debug("Authorization header missing for upstream request")

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
        raise UpstreamError(
            response.status_code,
            detail=response.text,
            headers=response_headers,
            body=response.content,
        )

    try:
        return UpstreamResponse(data=response.json(), headers=response_headers)
    except ValueError as e:
        logger.error(f"Invalid JSON from upstream: {response.text[:500]}")
        raise UpstreamError(502, "Invalid JSON response from upstream", response_headers) from e


async def forward_streaming_request(
    payload: ChatCompletionRequest,
    settings: Settings,
    request_headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> tuple[AsyncIterator[bytes], StreamingContext]:
    """Forward a streaming chat completion request to the upstream service.

    Returns an async iterator that yields raw bytes for immediate forwarding,
    and a StreamingContext that collects content for logging.

    Args:
        payload: The chat completion request to forward.
        settings: Server settings containing upstream endpoint.
        request_headers: Headers from the original request to forward.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (byte iterator, streaming context).

    Raises:
        UpstreamError: If the upstream service returns an error.
    """
    url = settings.upstream_chat_completions_url

    # Prepare headers
    headers = filter_request_headers(request_headers or {})
    headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
    headers["Content-Type"] = "application/json"

    # Prepare request body
    request_data = payload.model_dump(mode="json", exclude_none=True)
    if not request_data.get("model"):
        request_data["model"] = settings.default_model

    # Ensure stream is set to true
    request_data["stream"] = True

    logger.debug(f"Forwarding streaming request to {url}")

    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))

    try:
        response = await client.send(
            client.build_request("POST", url, json=request_data, headers=headers),
            stream=True,
        )
    except httpx.TimeoutException as e:
        await client.aclose()
        logger.error(f"Upstream timeout: {e}")
        raise UpstreamError(504, "Upstream service timeout") from e
    except httpx.ConnectError as e:
        await client.aclose()
        logger.error(f"Upstream connection error: {e}")
        raise UpstreamError(502, f"Cannot connect to upstream: {settings.upstream_endpoint}") from e
    except httpx.RequestError as e:
        await client.aclose()
        logger.error(f"Upstream request error: {e}")
        raise UpstreamError(502, f"Upstream request failed: {e}") from e

    if response.status_code >= 400:
        error_bytes = await response.aread()
        await response.aclose()
        await client.aclose()
        logger.warning(f"Upstream returned {response.status_code}: {error_bytes[:500]}")
        raise UpstreamError(
            response.status_code,
            detail=error_bytes.decode("utf-8", errors="ignore"),
            headers=filter_response_headers(response.headers),
            body=error_bytes,
        )

    response_headers = filter_response_headers(response.headers)
    logger.debug(f"Upstream streaming response header keys: {list(response_headers.keys())}")
    context = StreamingContext(headers=response_headers)

    async def byte_iterator() -> AsyncIterator[bytes]:
        """Iterate over response bytes, forwarding immediately while buffering for logging."""
        try:
            async for chunk in response.aiter_bytes():
                # Process for logging (non-blocking, just string operations)
                context.process_chunk(chunk)
                # Yield immediately - no latency added
                yield chunk
            # Flush any remaining buffered content
            context.flush_buffer()
        finally:
            await response.aclose()
            await client.aclose()

    return byte_iterator(), context