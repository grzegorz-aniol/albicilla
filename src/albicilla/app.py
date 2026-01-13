"""FastAPI application factory and router for Albicilla."""

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from .config import Settings
from .logging import append_session_entry
from .models import ChatCompletionRequest
from .session import resolve_session_id
from .upstream import (
    StreamingContext,
    UpstreamError,
    forward_request,
    forward_streaming_request,
)

# Global settings instance (set by create_app or overridden in tests)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Dependency to get current settings."""
    if _settings is None:
        return Settings()
    return _settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Optional settings instance. If None, defaults are used.

    Returns:
        Configured FastAPI application.
    """
    global _settings
    _settings = settings or Settings()

    app = FastAPI(
        title="Albicilla - LLM passthrough proxy",
        description="A proxy that forwards requests to upstream OpenAI-compatible APIs and logs to JSONL files.",
        version="0.1.0",
    )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: Request,
        payload: ChatCompletionRequest,
        settings: Settings = Depends(get_settings),
    ) -> JSONResponse | StreamingResponse:
        """Handle chat completion requests.

        Forwards the request to upstream, logs both request and response,
        and returns the upstream response with headers.
        Supports both streaming and non-streaming modes.
        """
        # 1. Resolve session ID
        session_id = await resolve_session_id(request, payload.user, settings)

        # 2. Get request headers to forward
        request_headers = dict(request.headers)

        # Debug: log authorization header
        auth_header = request_headers.get("authorization")
        auth_preview = f"{auth_header[:30]}..." if auth_header and len(auth_header) > 30 else auth_header
        logger.debug(f"[{session_id}] Received Authorization header: {auth_preview}")

        # Log incoming request
        model = payload.model or settings.default_model
        n_messages = len(payload.messages)
        is_streaming = payload.stream or False
        logger.info(
            f"[{session_id}] → Received request: model={model}, messages={n_messages}, stream={is_streaming}"
        )

        # Prepare request data for logging
        request_data = payload.model_dump(mode="json")

        # 3. Handle streaming vs non-streaming
        if is_streaming:
            return await _handle_streaming_request(
                payload, settings, request_headers, session_id, request_data
            )
        else:
            return await _handle_non_streaming_request(
                payload, settings, request_headers, session_id, request_data
            )

    async def _handle_streaming_request(
        payload: ChatCompletionRequest,
        settings: Settings,
        request_headers: dict[str, str],
        session_id: str,
        request_data: dict,
    ) -> StreamingResponse:
        """Handle a streaming chat completion request."""
        try:
            byte_iterator, context = await forward_streaming_request(
                payload, settings, request_headers
            )
        except UpstreamError as e:
            logger.warning(f"[{session_id}] ✗ Upstream error: {e.status_code} {e.detail[:100]}")
            raise HTTPException(status_code=e.status_code, detail=e.detail, headers=e.headers) from e

        async def stream_and_log():
            """Yield bytes and log when complete."""
            async for chunk in byte_iterator:
                yield chunk

            # Stream complete - build and log the full response
            response_data = context.build_complete_response()
            response_id = response_data.get("id", "unknown")
            response_model = response_data.get("model", "unknown")
            content_len = len(context.content_parts)
            logger.info(
                f"[{session_id}] ← Streaming complete: id={response_id}, model={response_model}, chunks={content_len}"
            )

            try:
                append_session_entry(settings, session_id, request_data, response_data)
                logger.debug(f"[{session_id}] Logged streaming response to session file")
            except IOError as e:
                logger.error(f"[{session_id}] Failed to write log: {e}")

        return StreamingResponse(
            stream_and_log(),
            media_type="text/event-stream",
            headers={
                **context.headers,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    async def _handle_non_streaming_request(
        payload: ChatCompletionRequest,
        settings: Settings,
        request_headers: dict[str, str],
        session_id: str,
        request_data: dict,
    ) -> JSONResponse:
        """Handle a non-streaming chat completion request."""
        try:
            upstream_response = await forward_request(payload, settings, request_headers)
        except UpstreamError as e:
            logger.warning(f"[{session_id}] ✗ Upstream error: {e.status_code} {e.detail[:100]}")
            raise HTTPException(status_code=e.status_code, detail=e.detail, headers=e.headers) from e

        response_data = upstream_response.data
        response_headers = upstream_response.headers

        # Log successful response
        response_id = response_data.get("id", "unknown")
        response_model = response_data.get("model", "unknown")
        finish_reason = response_data.get("choices", [{}])[0].get("finish_reason", "unknown")
        logger.info(
            f"[{session_id}] ← Received response: id={response_id}, model={response_model}, finish_reason={finish_reason}"
        )

        # Append to log file
        try:
            append_session_entry(settings, session_id, request_data, response_data)
            logger.debug(f"[{session_id}] Logged to session file")
        except IOError as e:
            logger.error(f"[{session_id}] Failed to write log: {e}")

        return JSONResponse(content=response_data, headers=response_headers)

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    return app