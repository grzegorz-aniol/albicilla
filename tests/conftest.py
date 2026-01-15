"""Shared pytest fixtures for conversation-related tests."""

from typing import Final

import pytest

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