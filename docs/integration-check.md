# Integration checks during JSONL conversion

This document describes checks performed during conversion to JSONL. These checks verify structure and integrity; they do not include cleanup/mutation rules (see `docs/goose-chat-cleanup.md`).

## Session validation (per log file)

1. Session ID consistency
- All entries in a session file must share the same `session_id`.
- Failure raises `SessionValidationError` and aborts the current run.

2. Timestamp monotonicity
- Entry timestamps must be strictly increasing in file order.
- Failure raises `SessionValidationError` and aborts the current run.

3. Trailing short entry handling
- If the last entry has fewer request messages than the session maximum, the session is truncated to the last entry that has the maximum length.
- This is logged as a warning and continues.

## Export integrity analysis (per session)

4. Record presence
- A conversation record must be produced for the session.
- Missing record is an ERROR finding.

5. Messages list shape
- `messages` must be a non-empty list.
- Failure is an ERROR finding.

6. Tools list shape
- `tools` must be a list.
- Missing/invalid list is an ERROR finding.
- Empty list is an ERROR finding.

7. Tool definition schema
For each tool definition:
- Must be an object (ERROR if not).
- `parameters` must be an object (ERROR if not).
- `parameters` must be an object (ERROR if not).
- `parameters.type` must be `object` (ERROR if not).
- `parameters.properties` must be an object (ERROR if not).
- `parameters.required` must be a list (ERROR if not).

## Tool call representation checks

8. When `--json-tool-calls` is enabled
- Assistant content must have balanced `<tool_call>` tags (ERROR if mismatch).
- Assistant messages must not also contain `tool_calls` field (ERROR if present).
- Tool messages must wrap content in `<tool_result>...</tool_result>` (ERROR if missing tags).

9. When `--json-tool-calls` is disabled
- `assistant.tool_calls` must be a list when present (ERROR if not).
- Each `tool_calls` entry must be an object (ERROR if not).
- Each `tool_calls` entry must have a `function` object (ERROR if not).
- Each `function.name` must be a non-empty string (ERROR if not).

## Tool result heuristics (warnings/errors)

10. Tool result payload inspection (heuristic)
- Tool result content is parsed as JSON when possible.
- If any JSON field named `error`, `timeout`, `missing`, or `incomplete` is present with a truthy value, the tool result is flagged as an ERROR.
- Empty tool result content is flagged as an ERROR.
- If the same tool is retried within the same assistant->tool...->assistant block and succeeds, these ERRORs are downgraded to WARNINGs.
- Missing tool results for known `tool_call_id` values are flagged as ERROR, downgraded to WARNING if a successful retry exists in the same block.

11. Reverse tool result check (errors)
- Tool results without a matching tool call in the same block are flagged as ERROR.
- Tool results with missing `tool_call_id` are flagged as ERROR.

12. Legacy function_call format (errors)
- Any assistant message using `function_call` is flagged as ERROR.

13. Role sequence consistency (errors)
- Each entry's role trace must be a prefix of the next (developer normalized to system).
- Assistant roles are tokenized as `Ac` (content), `At` (tool_calls), `Act` (content+tool_calls), or `A` (neither).
- For validation, a prior sequence containing `Ac At` is treated as equivalent to a later `Act`.
- Violations are flagged as ERROR.

## Output

- Findings are written to `output/integrity-report.txt` when `--integrity-analysis` is enabled (default).
- Conversion proceeds even with WARNING/ERROR findings; they are reported but do not halt processing.
