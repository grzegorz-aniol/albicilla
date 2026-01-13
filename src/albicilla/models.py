"""Pydantic models for OpenAI-compatible chat completions API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


# --- Request Models ---


class FunctionCall(BaseModel):
    """Function call within a tool call."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call in a message."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatCompletionMessage(BaseModel):
    """A message in the chat completion request."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool responses


class ResponseFormat(BaseModel):
    """Response format specification."""

    type: Literal["text", "json_object"] = "text"


class FunctionDefinition(BaseModel):
    """Function definition for tools."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    """Tool definition."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str | None = None
    messages: list[ChatCompletionMessage]
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    tools: list[Tool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: ResponseFormat | None = None

    model_config = {"extra": "allow"}  # Allow unknown fields for logging completeness


# --- Response Models ---


class ResponseMessage(BaseModel):
    """Message in a chat completion response."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatCompletionChoice(BaseModel):
    """A choice in the chat completion response."""

    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: None = None  # Omitted per plan - not needed

    system_fingerprint: str | None = None


# --- Log Entry Model ---


class LogEntry(BaseModel):
    """Schema for JSONL log entries."""

    timestamp: str  # ISO 8601 UTC
    session_id: str
    request: dict[str, Any]
    response: dict[str, Any]

