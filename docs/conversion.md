# Conversion: Squashing Requests/Responses Into One Chat History

This document describes the **core conversion algorithm** that turns a session's
request/response log entries into **one consolidated chat history** with
`messages` and `tools`. It focuses on the conversion logic only. For validation
and cleanup rules, see:

- `docs/integration-check.md`
- `docs/goose-chat-cleanup.md`

## Input Model (Session Logs)

Each session file contains **multiple JSONL entries**, where **each entry**
captures one request/response pair:

- `request.messages`: the full chat history **as sent** to the model at that time
- `request.tools`: tool definitions **as sent** with that request
- `response.choices[0].message`: the assistant reply for that request

## Conversion Algorithm (Per Session File)

The converter **does not merge messages across entries**. Instead, it treats the
last suitable entry as a **snapshot of the complete conversation** and uses it
as the single source of truth.

Steps:

1. **Read all entries** in file order.
2. **Optional cleanup** (if enabled): apply message cleanup rules to *every*
   entry before validation and conversion.
3. **Validate entries** (monotonic timestamps, consistent session id, and
   trailing shorter-entry truncation). If the last entry has fewer request
   messages than the session maximum, truncate to the last entry with the
   maximum length and continue.
4. **Select the last validated entry** (post-truncation).
5. **Build `messages`** from `last_entry.request.messages`.
6. **Append the final assistant response** from
   `last_entry.response.choices[0].message` (if present).
7. **Normalize roles**: any `developer` role is mapped to `system`.
8. **Build `tools`** from all entries' `request.tools` (deduped by tool name,
   first definition wins; empty list if no tools).
9. **Optional tool-call serialization** (when `--json-tool-calls` is enabled):
   transform tool calls and tool results into inline `<tool_call>` /
   `<tool_result>` blocks within message content.

Result: a **single record**:

```json
{"messages": [...], "tools": [...]}
```

## What Data Is Read From Where

**Conversion uses all entries only for validation/cleanup**. The **actual chat
history** and **tool definitions** come **only** from the **last validated
entry**.

Specifically:

- **Chat history**: `last_entry.request.messages`
- **Assistant reply**: `last_entry.response.choices[0].message`
- **Tool definitions**: all entries' `request.tools` (deduped by name)

No per-entry merging happens. The assumption is that **each request already
includes the full prior conversation**, so the last request is the most complete
snapshot.

## Tool Definitions Flattening

Tool definitions are taken from the final request and flattened to match the
training dataset schema:

- If a tool definition has a `function` object, that is emitted directly.
- Otherwise, the full tool object is serialized.

## Tool-Call Serialization (When Enabled)

When `--json-tool-calls` is enabled, the message list is rewritten to inline
tool calls and tool results:

1. Assistant tool calls: each `assistant` message with `tool_calls` becomes a
   plain content message. Each tool call is serialized as
   `<tool_call>{"name": "...", "arguments": ...}</tool_call>`. If `arguments`
   is a JSON string, it is parsed into structured JSON. If the assistant already
   has text content, tool calls are appended on a new line. The `tool_calls`
   field is removed from the message.
2. Tool results: each `tool` message with `tool_call_id` is wrapped as
   `<tool_result tool_call_id="...">...</tool_result>` and the `tool_call_id`
   field is removed from the message.

When `--json-tool-calls` is **disabled** (default), the original OpenAI-style
`tool_calls` and `tool` messages are preserved.
