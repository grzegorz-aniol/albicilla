"""Validation tests for JSONL output sessions."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from collections.abc import Iterator

import pytest
from jsonschema import Draft202012Validator

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def _iter_jsonl_files(directory: Path) -> Iterator[Path]:
    for jsonl_path in sorted(p for p in directory.glob("*.jsonl") if p.is_file()):
        yield jsonl_path


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, json.loads(stripped)
            except json.JSONDecodeError as exc:  # pragma: no cover - immediate failure
                raise AssertionError(
                    f"Malformed JSON in {path}:{line_number}: {exc.msg}"
                ) from exc


def _assert_messages_and_tools(payload: dict) -> None:
    messages = payload.get("messages")
    tools = payload.get("tools")
    assert isinstance(messages, list) and messages, "messages must be a non-empty list"
    assert tools is None or isinstance(tools, list)


def _assert_tool_parameters(tools: list[dict]) -> None:
    for tool_definition in tools:
        parameters = tool_definition.get("parameters")
        assert isinstance(parameters, dict), "tool definition lacks parameters dict"
        assert parameters.get("type") == "object", "parameters.type must be object"
        properties = parameters.get("properties")
        assert isinstance(
            properties, dict
        ), "parameters missing properties dict"
        required = parameters.get("required", [])
        assert isinstance(required, list), "parameters.required must be a list"


def _assert_messages(payload: dict) -> None:
    # * NOTE: Test fixtures assume json-tool-calls were enabled when outputs were generated,
    # so we treat the flag as true even if params metadata is absent.
    json_tool_calls_enabled = payload.get("params", {}).get("json-tool-calls", True)
    for message in payload["messages"]:
        role = message.get("role")
        if role == "assistant":
            _assert_assistant_message(message, json_tool_calls_enabled)
        if role == "tool":
            content = message.get("content", "")
            assert isinstance(content, str) and content.startswith(
                "<tool_result"
            ), "tool result missing wrapper"
            assert content.endswith(
                "</tool_result>"
            ), "tool result missing closing tag"


def _assert_assistant_message(
    message: dict,
    json_tool_calls_enabled: bool,
) -> None:
    content = message.get("content", "")
    if not isinstance(content, str):
        return

    has_tool_markup = "<tool_call>" in content or "</tool_call>" in content
    if not has_tool_markup:
        return

    if not json_tool_calls_enabled:
        warnings.warn(
            "tool_call markup present but json-tool-calls flag disabled",
            stacklevel=2,
        )

    assert content.count("<tool_call>") == content.count(
        "</tool_call>"
    ), "tool_call tags mismatch"
    assert message.get("tool_calls") is None, "tool_calls key should be dropped"


def _assert_tools_used(payload: dict, path: Path, line_number: int) -> None:
    tools = payload.get("tools")
    assert isinstance(tools, list), "tools must be a list"
    if not tools:
        warnings.warn(
            f"tools list is empty in {path.name}:{line_number}",
            stacklevel=2,
        )


@pytest.fixture(scope="session")
def tool_schema_validator(tool_definitions_schema: dict[str, object]) -> Draft202012Validator:
    """Compile reusable validator for tool schemas."""
    return Draft202012Validator(tool_definitions_schema)


@pytest.mark.parametrize("jsonl_path", _iter_jsonl_files(OUTPUT_DIR))
def test_output_jsonl_records(
    jsonl_path: Path,
    tool_schema_validator: Draft202012Validator,
) -> None:
    """Ensure each JSONL payload matches expectations."""
    has_payload = False

    for line_number, payload in _iter_jsonl(jsonl_path):
        has_payload = True
        _assert_messages(payload)
        _assert_tools_used(payload, jsonl_path, line_number)

        tools = payload.get("tools") or []
        assert isinstance(tools, list) or tools is None, "tools must be a list"
        if tools:
            tool_schema_validator.validate(tools)
            _assert_tool_parameters(tools)

    assert has_payload, f"File {jsonl_path} must contain at least one record"