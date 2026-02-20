# Goose chat cleanup

This document describes the Goose-specific cleanup rules applied to chat messages during conversion.

## When it runs

- Cleanup runs when `--cleanup-goose` is enabled (default).
- It runs before validation and before JSONL export.

## Rules

1. Drop Goose info user messages
- Applies only to messages with `role == "user"`.
- If the entire textual content is wrapped in `<info-msg>...</info-msg>` (no other text), the message is removed.

2. Normalize Goose system bootstrap
- Applies only to messages with `role == "system"`.
- If the message text matches the Goose bootstrap regex (contains the phrase `general-purpose AI agent called goose`, case-insensitive), replace the full content with:
  `You are a general-purpose AI agent`

3. Drop Goose summary-trick entries
- Applies at the entry level (entire log entry is removed).
- A log entry is dropped when all of the following are true:
  - The first message is `system` and starts with `Reply with only a description in four words or less`.
  - The second message is `user` and starts with `Here are the first few user messages:`.
  - The assistant reply exists and contains 1–6 words.

4. Drop empty assistant messages followed by another assistant message
- Applies only to messages with `role == "assistant"`.
- If `content` is empty or whitespace-only, and there are no `tool_calls` or `tool_call_id`,
  then drop the message when the next message is also `assistant`.

## Notes

- These rules are the only Goose-specific mutations/removals.
- Non-Goose cleanup and integrity checks are documented separately.
