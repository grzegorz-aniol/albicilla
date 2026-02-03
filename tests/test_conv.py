"""Tests for conversation logs processor."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conv.models import ConversationRecord, LogEntry, Message, RequestPayload, ResponseChoice, ResponsePayload, ToolDefinition
from conv.processor import (
    count_tool_call_requests,
    count_turn_groups,
    coerce_arguments,
    extract_session_name_from_path,
    extract_tool_schemas,
    process_session,
    process_logs_directory_with_tool_usage,
    process_session_with_stats,
    serialize_tool_calls,
    serialize_tool_result,
    transform_messages,
)
from conv.tool_usage import ToolUsageGroupBy, aggregate_tool_usage

TIP_SPLIT_PROMPT = (
    "Ile wyniesie indywidualna kwota napiwku dla każdej osoby w naszej grupie "
    "archeologów, jeśli całkowity rachunek za kolację wynosi 250 złotych, a w "
    "grupie jest nas pięcioro?"
)
TIP_SPLIT_FINAL_REPLY = (
    "Indywidualna kwota napiwku dla każdej osoby w waszej grupie archeologów "
    "wyniesie 50 złotych."
)
TIP_SPLIT_TOOL_CALL_ID = "call_tip_split"
TIP_SPLIT_ARGUMENTS = {"total_bill": 250, "number_of_people": 5}
TIP_SPLIT_ARGUMENTS_JSON = json.dumps(TIP_SPLIT_ARGUMENTS, ensure_ascii=False)
TIP_SPLIT_RESULT_PAYLOAD = {"calculate_tip_split": {"individual_tip_amount": 50}}
TIP_SPLIT_RESULT_CONTENT = json.dumps(TIP_SPLIT_RESULT_PAYLOAD, ensure_ascii=False)

SAMPLE_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_recipes",
            "description": "Wyszukiwanie przepisów kulinarnych według określonych kryteriów",
            "parameters": {
                "type": "object",
                "properties": {
                    "cuisine": {
                        "type": "string",
                        "description": "Kuchnia przepisu",
                    },
                    "ingredient": {
                        "type": "string",
                        "description": "Składnik do wyszukania w przepisach",
                    },
                    "diet": {
                        "type": "string",
                        "description": "Ograniczenie dietetyczne dla przepisów",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_tip_split",
            "description": "Obliczanie indywidualnej kwoty napiwku dla grupy",
            "parameters": {
                "type": "object",
                "properties": {
                    "total_bill": {
                        "type": "number",
                        "description": "Całkowita kwota rachunku",
                    },
                    "number_of_people": {
                        "type": "integer",
                        "description": "Liczba osób w grupie",
                    },
                },
                "required": ["total_bill", "number_of_people"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_recipes",
            "description": "Wyszukiwanie przepisów na podstawie preferencji użytkownika",
            "parameters": {
                "type": "object",
                "properties": {
                    "cuisine": {
                        "type": "string",
                        "description": "Preferowana kuchnia",
                    },
                    "ingredients": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Składniki do uwzględnienia w przepisach",
                    },
                    "max_time": {
                        "type": "integer",
                        "description": "Maksymalny czas przygotowania w minutach",
                    },
                },
                "required": ["cuisine", "ingredients"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_books",
            "description": "Rekomendacja książek",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {
                        "type": "string",
                        "description": "Preferowany gatunek książek",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maksymalna cena, jaką użytkownik jest skłonny zapłacić",
                    },
                },
                "required": ["genre", "max_price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_loan_emi",
            "description": "Oblicz równą miesięczną ratę spłaty kredytu na podstawie kwoty kredytu, oprocentowania i okresu kredytowania",
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_amount": {
                        "type": "number",
                        "description": "Kwota kredytu",
                    },
                    "interest_rate": {
                        "type": "number",
                        "description": "Oprocentowanie roczne",
                    },
                    "tenure": {
                        "type": "integer",
                        "description": "Okres kredytowania w miesiącach",
                    },
                },
                "required": ["loan_amount", "interest_rate", "tenure"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_task",
            "description": "Utwórz nowe zadanie w liście rzeczy do zrobienia",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Tytuł zadania",
                    },
                    "due_date": {
                        "type": "string",
                        "format": "date",
                        "description": "Termin wykonania zadania",
                    },
                },
                "required": ["title", "due_date"],
            },
        },
    },
]

EXPECTED_TOOL_SCHEMAS = [tool["function"] for tool in SAMPLE_TOOL_DEFINITIONS]
TOOL_CALL_PAYLOAD = {
    "name": "calculate_tip_split",
    "arguments": TIP_SPLIT_ARGUMENTS,
}
EXPECTED_TOOL_CALL_CONTENT = (
    f"<tool_call>{json.dumps(TOOL_CALL_PAYLOAD, ensure_ascii=False)}</tool_call>"
)
EXPECTED_TOOL_RESULT_CONTENT = (
    f'<tool_result tool_call_id="{TIP_SPLIT_TOOL_CALL_ID}">{TIP_SPLIT_RESULT_CONTENT}</tool_result>'
)

SAMPLE_SESSION_JSON = {
    "messages": [
        {"role": "user", "content": TIP_SPLIT_PROMPT},
        {
            "role": "assistant",
            "content": '<tool_call>{"name": "calculate_tip_split", "arguments": {'
            '"total_bill": 250, "number_of_people": 5}}</tool_call>',
        },
        {
            "role": "tool",
            "content": '<tool_result tool_call_id="call_tip_split">'
            '{"calculate_tip_split": {"individual_tip_amount": 50}}</tool_result>',
        },
        {
            "role": "assistant",
            "content": TIP_SPLIT_FINAL_REPLY,
        },
    ],
    "tools": EXPECTED_TOOL_SCHEMAS,
}

EXPECTED_SESSION_RECORD = ConversationRecord(
    messages=SAMPLE_SESSION_JSON["messages"],
    tools=EXPECTED_TOOL_SCHEMAS,
)


class TestSampleJson:
    def _log_entry_from_sample(self, json_payload: dict, *, json_tool_calls: bool) -> LogEntry:
        messages = json_payload["messages"]
        tools = [ToolDefinition(type="function", function=function) for function in json_payload["tools"]]
        request_messages = [Message(**message) for message in messages[:-1]]
        response_message = Message(**messages[-1])
        response = ResponsePayload(
            id="resp-1",
            object="chat.completion",
            choices=[
                ResponseChoice(
                    index=0,
                    message=response_message,
                )
            ],
        )
        return LogEntry(
            timestamp=datetime.now(timezone.utc),
            session_id="session-sample",
            request=RequestPayload(
                model="gpt-4o-mini",
                messages=request_messages,
                tools=tools,
            ),
            response=response,
        )

    def _coerce_json_tool_call_messages(self, json_payload: dict) -> list[dict]:
        coerced: list[dict] = []
        for message in json_payload["messages"]:
            if message["role"] != "assistant":
                coerced.append(message)
                continue

            serialized = message["content"]
            if "<tool_call>" not in serialized:
                coerced.append(message)
                continue

            tool_calls = []
            for raw_call in serialized.split("</tool_call>"):
                if "<tool_call>" not in raw_call:
                    continue
                body = raw_call.split("<tool_call>")[-1]
                tool_calls.append(
                    {
                        "id": "call_tip_split",
                        "type": "function",
                        "function": json.loads(body),
                    }
                )
            coerced.append(
                {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                    "content": None,
                }
            )
        return coerced

    def test_process_session_matches_expected_schema(self):
        entry = self._log_entry_from_sample(SAMPLE_SESSION_JSON, json_tool_calls=True)
        result = process_session([entry], json_tool_calls=True)
        assert result == EXPECTED_SESSION_RECORD

    def test_transform_messages_validates_xml_wrapping(self):
        messages = self._coerce_json_tool_call_messages(SAMPLE_SESSION_JSON)
        transformed = transform_messages(messages)
        assert transformed[1]["content"] == SAMPLE_SESSION_JSON["messages"][1]["content"]
        assert transformed[2]["content"] == SAMPLE_SESSION_JSON["messages"][2]["content"]


class TestExtractToolSchemas:
    def test_returns_function_blocks(self):
        definitions = [ToolDefinition(**definition) for definition in SAMPLE_TOOL_DEFINITIONS]
        schemas = extract_tool_schemas(definitions)
        assert schemas == EXPECTED_TOOL_SCHEMAS

    def test_empty_input(self):
        assert extract_tool_schemas(None) == []


class TestCoerceArguments:
    def test_dict_passthrough(self):
        args = {"key": "value"}
        assert coerce_arguments(args) == {"key": "value"}

    def test_string_parsed(self):
        args = '{"key": "value"}'
        assert coerce_arguments(args) == {"key": "value"}

    def test_invalid_string(self):
        args = "not json"
        assert coerce_arguments(args) == "not json"

    def test_none(self):
        assert coerce_arguments(None) is None


class TestSerializeToolCalls:
    def test_single_tool_call(self):
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "arguments": '{"arg": "value"}',
                },
            }
        ]
        result = serialize_tool_calls(tool_calls)
        assert '<tool_call>{"name": "my_tool", "arguments": {"arg": "value"}}</tool_call>' == result

    def test_multiple_tool_calls(self):
        tool_calls = [
            {"id": "1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
            {"id": "2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
        ]
        result = serialize_tool_calls(tool_calls)
        assert "<tool_call>" in result
        assert "tool_a" in result
        assert "tool_b" in result
        assert result.count("<tool_call>") == 2


class TestSerializeToolResult:
    def test_basic(self):
        result = serialize_tool_result("Hello world", "call_123")
        assert result == '<tool_result tool_call_id="call_123">Hello world</tool_result>'

    def test_empty_content(self):
        result = serialize_tool_result("", "call_123")
        assert result == '<tool_result tool_call_id="call_123"></tool_result>'


class TestTransformMessages:
    def test_regular_message_unchanged(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = transform_messages(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_assistant_tool_calls_transformed(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{}"},
                    }
                ],
            }
        ]
        result = transform_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "<tool_call>" in result[0]["content"]
        assert "tool_calls" not in result[0]

    def test_tool_result_transformed(self):
        messages = [
            {
                "role": "tool",
                "content": "Result here",
                "tool_call_id": "call_123",
            }
        ]
        result = transform_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert '<tool_result tool_call_id="call_123">Result here</tool_result>' == result[0]["content"]
        assert "tool_call_id" not in result[0]


class TestProcessSession:
    def _make_log_entry(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_content: str = "Response",
    ) -> LogEntry:
        return LogEntry(
            timestamp=datetime.now(timezone.utc),
            session_id="test-session",
            request=RequestPayload(
                model="gpt-4o",
                messages=[Message(**m) for m in messages],
                tools=[ToolDefinition(**t) for t in tools] if tools else None,
            ),
            response=ResponsePayload(
                id="resp-123",
                choices=[
                    ResponseChoice(
                        index=0,
                        message=Message(role="assistant", content=response_content),
                    )
                ],
            ),
        )

    def test_basic_session(self):
        entry = self._make_log_entry(
            messages=[{"role": "user", "content": "Hello"}],
            response_content="Hi there!",
        )
        result = process_session([entry], json_tool_calls=False)
        assert result is not None
        assert len(result.messages) == 2
        assert result.messages[0]["role"] == "user"
        assert result.messages[1]["role"] == "assistant"
        assert result.messages[1]["content"] == "Hi there!"

    def test_developer_message_normalized(self):
        entry = self._make_log_entry(
            messages=[
                {"role": "developer", "content": "Follow policy."},
                {"role": "user", "content": "Hello"},
            ],
        )
        result = process_session([entry], json_tool_calls=False)
        assert result is not None
        assert result.messages[0]["role"] == "system"
        assert result.messages[0]["content"] == "Follow policy."

    def test_empty_session(self):
        result = process_session([])
        assert result is None

    def test_with_tools(self):
        entry = self._make_log_entry(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"type": "function", "function": {"name": "my_tool", "parameters": {}}}],
        )
        result = process_session([entry], json_tool_calls=False)
        assert result is not None
        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "my_tool"


class TestToolUsageCounting:
    def test_counts_tool_calls_from_assistant_tool_calls(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}, {"id": "2"}]},
            {"role": "tool", "content": "ok", "tool_call_id": "1"},
            {"role": "assistant", "content": "done"},
        ]
        assert count_tool_call_requests(messages) == 2

    def test_counts_tool_calls_from_inline_tool_call_tags(self):
        messages = [
            {"role": "assistant", "content": "<tool_call>{}</tool_call>\n<tool_call>{}</tool_call>"},
        ]
        assert count_tool_call_requests(messages) == 2

    def test_extract_session_name_strips_timestamp_suffix(self):
        path = Path("arxiv-python-1769035200000000001.jsonl")
        assert extract_session_name_from_path(path) == "arxiv-python"

    def test_extract_session_name_strips_uuid_suffix(self):
        path = Path("market-brief-019c0179-fae3-7c9f-98ed-fc605980e052.jsonl")
        assert extract_session_name_from_path(path) == "market-brief"

    def test_extract_session_name_strips_numeric_prompt_suffix(self):
        path = Path("google-maps-es-01.jsonl")
        assert extract_session_name_from_path(path) == "google-maps-es"

    def test_extract_session_name_does_not_strip_two_part_numeric_suffix(self):
        path = Path("gpt-4.jsonl")
        assert extract_session_name_from_path(path) == "gpt-4"

    def test_extract_session_name_strips_prompt_suffix_after_uuid(self):
        path = Path("google-maps-es-01-019c0179-fae3-7c9f-98ed-fc605980e052.jsonl")
        assert extract_session_name_from_path(path) == "google-maps-es"

    def test_extract_session_name_strips_prompt_suffix_after_timestamp(self):
        path = Path("github-sk-05-1769035200000000001.jsonl")
        assert extract_session_name_from_path(path) == "github-sk"

    def test_process_session_with_stats_keeps_count_before_transform(self):
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc),
            session_id="session-1",
            request=RequestPayload(
                model="gpt-4o-mini",
                messages=[
                    Message(role="user", content="Hello"),
                    Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "tool_a", "arguments": "{}"},
                            }
                        ],
                    ),
                    Message(role="tool", content="{}", tool_call_id="call_1"),
                ],
            ),
            response=ResponsePayload(
                id="resp-1",
                choices=[ResponseChoice(index=0, message=Message(role="assistant", content="Done"))],
            ),
        )

        record, tool_call_count = process_session_with_stats([entry], json_tool_calls=True)
        assert record is not None
        assert tool_call_count == 1
        assert "<tool_call>" in record.messages[1]["content"]

    def test_process_logs_directory_with_tool_usage(self, tmp_path: Path):
        day_dir = tmp_path / "2026-01-21"
        day_dir.mkdir()
        session_file_1 = day_dir / "arxiv-python-1769035200000000001.jsonl"
        session_file_2 = day_dir / "arxiv-python-1769035200000000002.jsonl"

        tools_1 = [
            ToolDefinition(type="function", function={"name": "tool_a", "parameters": {}}),
            ToolDefinition(type="function", function={"name": "tool_b", "parameters": {}}),
        ]
        tools_2 = [
            ToolDefinition(type="function", function={"name": "tool_a", "parameters": {}}),
            ToolDefinition(type="function", function={"name": "tool_c", "parameters": {}}),
        ]

        entry = LogEntry(
            timestamp=datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc),
            session_id="session-1",
            request=RequestPayload(
                model="gpt-4o-mini",
                messages=[
                    Message(role="user", content="Hi"),
                    Message(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "tool_a", "arguments": "{}"},
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "tool_b", "arguments": "{}"},
                            },
                        ],
                    ),
                    Message(role="tool", content="{}", tool_call_id="call_1"),
                    Message(role="tool", content="{}", tool_call_id="call_2"),
                ],
                tools=tools_1,
            ),
            response=ResponsePayload(
                id="resp-1",
                choices=[ResponseChoice(index=0, message=Message(role="assistant", content="Done"))],
            ),
        )
        session_file_1.write_text(
            json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        entry_2 = entry.model_copy(
            update={
                "session_id": "session-2",
                "request": entry.request.model_copy(update={"tools": tools_2}),
            }
        )
        session_file_2.write_text(
            json.dumps(entry_2.model_dump(mode="json"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        records_by_date, records_by_scenario, tool_usage_samples = process_logs_directory_with_tool_usage(tmp_path)
        assert len(records_by_date) == 1
        assert len(records_by_scenario) == 1
        assert "arxiv-python" in records_by_scenario
        tool_usage = aggregate_tool_usage(tool_usage_samples, group_by=ToolUsageGroupBy.date)
        assert len(tool_usage) == 1
        assert tool_usage[0].date is not None
        assert tool_usage[0].date.isoformat() == "2026-01-21"
        assert tool_usage[0].session == ""
        assert tool_usage[0].session_count == 2
        assert tool_usage[0].tool_call_count == 4
        assert tool_usage[0].tool_definition_count == 3
        assert tool_usage[0].client_turns == 2
        assert tool_usage[0].assistant_turns == 2
        assert tool_usage[0].assistant_turns_with_tools == 2

    def test_process_logs_directory_groups_prompt_id_sessions(self, tmp_path: Path):
        day_dir = tmp_path / "2026-01-21"
        day_dir.mkdir(parents=True)

        session_file_1 = day_dir / "google-maps-es-01.jsonl"
        session_file_2 = day_dir / "google-maps-es-02.jsonl"

        entry = LogEntry(
            timestamp=datetime(2026, 1, 21, tzinfo=timezone.utc),
            session_id="google-maps-es-01",
            request=RequestPayload(
                model="gpt-4o-mini",
                messages=[Message(role="user", content="Hello")],
                tools=[
                    {"type": "function", "function": {"name": "tool_a", "parameters": {}}},
                ],
            ),
            response=ResponsePayload(
                id="resp-1",
                choices=[ResponseChoice(index=0, message=Message(role="assistant", content="Done"))],
            ),
        )

        session_file_1.write_text(
            json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        entry_2 = entry.model_copy(update={"session_id": "google-maps-es-02"})
        session_file_2.write_text(
            json.dumps(entry_2.model_dump(mode="json"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        _, records_by_scenario, tool_usage_samples = process_logs_directory_with_tool_usage(tmp_path)
        assert len(records_by_scenario) == 1
        tool_usage = aggregate_tool_usage(tool_usage_samples, group_by=ToolUsageGroupBy.scenario)
        assert len(tool_usage) == 1
        assert tool_usage[0].date is None
        assert tool_usage[0].session == "google-maps-es"
        assert tool_usage[0].session_count == 2

    def test_write_per_scenario_merges_across_dates(self, tmp_path: Path):
        from conv.main import write_per_scenario

        day_one = tmp_path / "2026-01-21"
        day_two = tmp_path / "2026-01-22"
        day_one.mkdir(parents=True)
        day_two.mkdir(parents=True)

        session_one = day_one / "market-brief-1769035200000000001.jsonl"
        session_two = day_two / "market-brief-1769035200000000002.jsonl"

        entry = LogEntry(
            timestamp=datetime(2026, 1, 21, 12, 0, tzinfo=timezone.utc),
            session_id="session-1",
            request=RequestPayload(
                model="gpt-4o-mini",
                messages=[Message(role="user", content="Hi")],
            ),
            response=ResponsePayload(
                id="resp-1",
                choices=[ResponseChoice(index=0, message=Message(role="assistant", content="Done"))],
            ),
        )

        session_one.write_text(
            json.dumps(entry.model_dump(mode="json"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        session_two.write_text(
            json.dumps(entry.model_copy(update={"session_id": "session-2"}).model_dump(mode="json"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        _, records_by_scenario, _ = process_logs_directory_with_tool_usage(tmp_path)
        assert set(records_by_scenario) == {"market-brief"}
        assert len(records_by_scenario["market-brief"]) == 2

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        files_written = write_per_scenario(records_by_scenario, output_dir, anonymize=False)
        assert files_written == 1

        output_path = output_dir / "market-brief.jsonl"
        content = output_path.read_text(encoding="utf-8").splitlines()
        assert len(content) == 2


class TestToolUsageReportWriter:
    def test_writes_csv(self, tmp_path: Path):
        from conv.main import write_tool_usage_report
        from conv.models import SessionToolUsageRow

        output_path = tmp_path / "tool-usage.csv"
        write_tool_usage_report(
            [
                SessionToolUsageRow(
                    date=datetime(2026, 1, 21, tzinfo=timezone.utc).date(),
                    session="arxiv",
                    session_count=5,
                    tool_call_count=3,
                    tool_definition_count=2,
                    client_turns=7,
                    assistant_turns=6,
                    assistant_turns_with_tools=3,
                ),
                SessionToolUsageRow(
                    date=datetime(2026, 1, 22, tzinfo=timezone.utc).date(),
                    session="yt",
                    session_count=1,
                    tool_call_count=0,
                    tool_definition_count=0,
                    client_turns=1,
                    assistant_turns=1,
                    assistant_turns_with_tools=0,
                ),
            ],
            output_path,
        )

        content = output_path.read_text(encoding="utf-8").splitlines()
        assert (
            content[0]
            == "date,scenario,session_count,tool_call_count,tool_definition_count,client_turns,assistant_turns,assistant_turns_with_tools"
        )
        assert "2026-01-21,arxiv,5,3,2,7,6,3" in content
        assert "2026-01-22,yt,1,0,0,1,1,0" in content


class TestTurnCounting:
    def test_counts_turn_groups(self):
        messages = [
            {"role": "system", "content": "A"},
            {"role": "user", "content": "B"},
            {"role": "assistant", "content": "C"},
            {"role": "assistant", "content": "D"},
            {"role": "tool", "content": "E"},
            {"role": "user", "content": "F"},
            {"role": "assistant", "content": "G"},
        ]

        client_turns, assistant_turns, assistant_tool_turns = count_turn_groups(messages)
        assert client_turns == 2
        assert assistant_turns == 2
        assert assistant_tool_turns == 0

    def test_counts_assistant_turn_groups_with_tool_requests(self):
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
            {"role": "assistant", "content": "still assistant"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "no tools"},
            {"role": "user", "content": "B"},
            {"role": "assistant", "content": "<tool_call>{}</tool_call>"},
        ]

        _, assistant_turns, assistant_tool_turns = count_turn_groups(messages)
        assert assistant_turns == 2
        assert assistant_tool_turns == 2