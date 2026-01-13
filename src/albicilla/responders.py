"""Synthetic response builder for the proxy."""

import time
from uuid import uuid4

from .config import Settings
from .models import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ResponseMessage,
)


def build_response(request: ChatCompletionRequest, settings: Settings) -> ChatCompletionResponse:
    """Build a synthetic OpenAI-compatible response.

    Args:
        request: The incoming chat completion request.
        settings: Server settings including response template.

    Returns:
        A valid ChatCompletionResponse with synthetic content.
    """
    model = request.model or settings.default_model
    n_messages = len(request.messages)

    # Format the response content using the template
    content = settings.response_template.format(
        n_messages=n_messages,
        model=model,
    )

    response_message = ResponseMessage(
        role="assistant",
        content=content,
    )

    choice = ChatCompletionChoice(
        index=0,
        message=response_message,
        finish_reason="stop",
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=None,
    )