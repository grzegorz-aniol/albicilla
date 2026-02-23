## Sessions and Log Naming

Albicilla writes one JSONL file per session. The session ID comes from headers or fallbacks and is used in:
- The log entry field `session_id`
- The log filename (sanitized)

### Session Header

Default session header: `agent-session-id` (configurable via `--session-header` / `PROXY_SESSION_HEADER`).

### Modes

Required header mode (default):
- Resolution order:
  - Configured session header (default `agent-session-id`)
  - `X-Session-Id` fallback
- If neither header is present, request is rejected with `400`.

Legacy mode (`--no-require-session-header`):
- Resolution order:
  - `X-Session-Id` header
  - Bearer token mapping (only if `--allow-bearer-token-fallback`)
  - Single-session fallback (`single-session`)
- Consequence: without a session ID header, requests may be logged under a shared session file
  (bearer-mapped or `single-session`), which can mix multiple clients into one log.

### Session Prefix Mapping (REST)

Use this to control **log filename prefixes per session**, regardless of how the session ID was resolved.
This is needed because the client decides how to namespace or group log files, and the server only
sees the session ID after it is resolved.

Endpoint:
- `POST /sessions`

Payload:
```
{
  "session_prefix": "<prefix or null/blank to clear>",
  "session_id": "<session_id>"
}
```

Effect:
- For that `session_id`, log files are named: `<prefix>-<session_id>.jsonl`
- The log entry field `session_id` remains the resolved session ID (without prefix).
- Sending `session_prefix: null` or blank clears the override for that `session_id`.

