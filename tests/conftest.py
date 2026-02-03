"""Shared pytest fixtures for conversation-related tests."""

from pathlib import Path
from typing import Final

import pytest

DEFAULT_LOGS_DIR: Final[Path] = Path(__file__).parent.parent / "output"

TOOL_DEFINITIONS_SCHEMA: Final[dict[str, object]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://albicilla.dev/tests/tool-definitions.schema.json",
    "title": "Sample tool definitions",
    "description": "Schema derived from SAMPLE_TOOL_DEFINITIONS in tests/test_conv.py.",
    "type": "array",
    "minItems": 0,

    "items": {"$ref": "#/$defs/functionBlock"},
    "$defs": {
        "functionBlock": {
            "type": "object",
            "required": ["name", "description", "parameters"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "description": {"type": "string", "minLength": 1},
                "parameters": {"$ref": "#/$defs/objectParameters"},
            },
            "additionalProperties": False,
        },
        "objectParameters": {
            "type": "object",
            "required": ["type", "properties"],
            "properties": {
                "type": {"type": "string", "const": "object"},
                "properties": {
                    "type": "object",
                    "minProperties": 0,
                    "additionalProperties": {
                        "anyOf": [
                            {"$ref": "#/$defs/functionArgument"},
                            {"type": "object"},
                        ]
                    },
                },
                "required": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                    "default": [],
                },
            },
            "additionalProperties": True,
        },
        "functionArgument": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "string",
                        "integer",
                        "number",
                        "array",
                        "boolean",
                        "object",
                    ],
                },
                "description": {"type": "string", "minLength": 1},
                "format": {"type": "string", "minLength": 1},
                "items": {"$ref": "#/$defs/arrayItems"},
            },
            "additionalProperties": True,
            "allOf": [
                {
                    "if": {"properties": {"type": {"const": "array"}}},
                    "then": {"required": ["items"]},
                }
            ],
        },
        "arrayItems": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "string",
                        "integer",
                        "number",
                        "boolean",
                        "object",
                        "array",
                    ],
                },
                "description": {"type": "string", "minLength": 1},
                "format": {"type": "string", "minLength": 1},
            },
            "additionalProperties": True,
        },
    },
}


@pytest.fixture(scope="session")
def tool_definitions_schema() -> dict[str, object]:
    """JSON Schema describing the SAMPLE_TOOL_DEFINITIONS structure."""
    return TOOL_DEFINITIONS_SCHEMA


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--logs-dir",
        action="store",
        default=None,
        help=(
            "Directory containing JSONL logs used by tests/test_output_integrity.py "
            f"(default: {DEFAULT_LOGS_DIR})."
        ),
    )


@pytest.fixture(scope="session")
def logs_dir(pytestconfig: pytest.Config) -> Path:
    """Directory containing JSONL logs used by output integrity tests."""
    return _resolve_logs_dir(pytestconfig)


def _resolve_logs_dir(config: pytest.Config) -> Path:
    value = config.getoption("--logs-dir")
    if value:
        logs_dir = Path(value).expanduser()
    else:
        logs_dir = DEFAULT_LOGS_DIR

    if not logs_dir.exists():
        raise pytest.UsageError(f"--logs-dir path does not exist: {logs_dir}")
    if not logs_dir.is_dir():
        raise pytest.UsageError(f"--logs-dir path is not a directory: {logs_dir}")

    return logs_dir


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "jsonl_path" not in metafunc.fixturenames:
        return

    logs_dir = _resolve_logs_dir(metafunc.config)
    jsonl_paths = sorted(p for p in logs_dir.glob("*.jsonl") if p.is_file())
    if not jsonl_paths:
        raise pytest.UsageError(f"No .jsonl files found in {logs_dir}")

    metafunc.parametrize("jsonl_path", jsonl_paths)
