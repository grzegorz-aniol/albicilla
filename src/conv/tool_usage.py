"""Tool-usage statistics aggregation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from enum import Enum

from .models import SessionToolUsageRow, SessionToolUsageSample


class ToolUsageGroupBy(str, Enum):
    none = "none"
    date = "date"
    scenario = "scenario"


def aggregate_tool_usage(
    samples: Iterable[SessionToolUsageSample],
    *,
    group_by: ToolUsageGroupBy,
) -> list[SessionToolUsageRow]:
    totals: dict[tuple[date | None, str], dict[str, object]] = {}

    for sample in samples:
        match group_by:
            case ToolUsageGroupBy.date:
                key = (sample.date, "")
            case ToolUsageGroupBy.scenario:
                key = (None, sample.session)
            case ToolUsageGroupBy.none:
                key = (None, "")
            case _:
                raise ValueError(f"Unsupported group_by: {group_by}")

        current = totals.get(key)
        if current is None:
            current = {
                "date": key[0],
                "session": key[1],
                "session_count": 0,
                "tool_call_count": 0,
                "tool_definition_names": set(),
                "client_turns": 0,
                "assistant_turns": 0,
                "assistant_turns_with_tools": 0,
            }
            totals[key] = current

        current["session_count"] = int(current["session_count"]) + 1
        current["tool_call_count"] = int(current["tool_call_count"]) + sample.tool_call_count
        tool_names = current["tool_definition_names"]
        assert isinstance(tool_names, set)
        tool_names.update(sample.tool_definition_names)
        current["client_turns"] = int(current["client_turns"]) + sample.client_turns
        current["assistant_turns"] = int(current["assistant_turns"]) + sample.assistant_turns
        current["assistant_turns_with_tools"] = (
            int(current["assistant_turns_with_tools"]) + sample.assistant_turns_with_tools
        )

    rows: list[SessionToolUsageRow] = []
    for (_, _), current in totals.items():
        tool_names = current["tool_definition_names"]
        assert isinstance(tool_names, set)
        rows.append(
            SessionToolUsageRow(
                date=current["date"] if current["date"] is not None else None,
                session=str(current["session"]),
                session_count=int(current["session_count"]),
                tool_call_count=int(current["tool_call_count"]),
                tool_definition_count=len(tool_names),
                client_turns=int(current["client_turns"]),
                assistant_turns=int(current["assistant_turns"]),
                assistant_turns_with_tools=int(current["assistant_turns_with_tools"]),
            )
        )

    match group_by:
        case ToolUsageGroupBy.date:
            return sorted(rows, key=lambda item: item.date or date.min)
        case ToolUsageGroupBy.scenario:
            return sorted(rows, key=lambda item: item.session)
        case ToolUsageGroupBy.none:
            return rows
        case _:
            return rows
