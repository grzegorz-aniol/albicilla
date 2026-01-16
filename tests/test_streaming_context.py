"""Tests for streaming response aggregation."""

import json

from albicilla.upstream import StreamingContext


def _sse_line(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def test_streaming_context_aggregates_choices_and_tool_calls():
    context = StreamingContext(headers={})

    context.process_chunk(
        _sse_line(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "gpt-4",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "Hello "}},
                    {"index": 1, "delta": {"role": "assistant", "content": "Alt "}},
                ],
            }
        )
    )

    context.process_chunk(
        _sse_line(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "do", "arguments": "{\"a\":"},
                                }
                            ]
                        },
                    }
                ]
            }
        )
    )

    context.process_chunk(
        _sse_line(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
                        "finish_reason": "tool_calls",
                    },
                    {"index": 1, "delta": {"content": "path"}, "finish_reason": "stop"},
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        )
    )

    context.flush_buffer()
    response = context.build_complete_response()

    assert response["id"] == "chatcmpl-1"
    assert response["model"] == "gpt-4"
    assert response["usage"]["total_tokens"] == 3

    choices = {choice["index"]: choice for choice in response["choices"]}
    assert choices[0]["message"]["content"] == "Hello "
    assert choices[0]["finish_reason"] == "tool_calls"
    assert choices[1]["message"]["content"] == "Alt path"
    assert choices[1]["finish_reason"] == "stop"

    tool_calls = choices[0]["message"]["tool_calls"]
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "do"
    assert tool_calls[0]["function"]["arguments"] == "{\"a\":1}"
