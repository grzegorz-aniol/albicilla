"""Pydantic models for conversation log processing."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class Message(BaseModel):
    """Individual message in a conversation."""

    role: str
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    model_config = {"extra": "allow"}


class ToolDefinition(BaseModel):
    """Tool definition structure."""

    type: str = "function"
    function: dict[str, Any]

    model_config = {"extra": "allow"}


class RequestPayload(BaseModel):
    """The request structure sent to the LLM."""

    model: str
    messages: list[Message]
    tools: list[ToolDefinition] | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class ResponseChoice(BaseModel):
    """A single choice in the response."""

    index: int
    message: Message
    finish_reason: str | None = None
    logprobs: Any | None = None

    model_config = {"extra": "allow"}


class ResponsePayload(BaseModel):
    """The response structure from the LLM."""

    id: str
    object: str = "chat.completion"
    created: int | None = None
    model: str | None = None
    choices: list[ResponseChoice]
    usage: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class LogEntry(BaseModel):
    """Represents a single JSONL log line with request-response pair."""

    timestamp: datetime
    session_id: str
    request: RequestPayload
    response: ResponsePayload | None = None

    model_config = {"extra": "allow"}


class ConversationRecord(BaseModel):
    """Output format for processed conversations."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]

