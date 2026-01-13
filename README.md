# Albicilla

OpenAI-compatible logging pass-trough proxy that captures LLM requests/responses for auditing.

## Features

- Drop-in replacement for OpenAI API endpoints
- Supports `/v1/chat/completions` endpoint (both streaming and non-streaming requests)
- Forwards requests to upstream LLM and logs request/response pairs
- Per-session JSONL logging with automatic date-based organization
- Session grouping via `user` field, headers, or bearer tokens

## Installation

```bash
uv sync
```

## Usage

Start the proxy server:

```bash
uv run albicilla-proxy --upstream-endpoint https://api.openai.com --log-root ./proxy_logs --host 127.0.0.1 --port 9000
```

Or with environment variables:

```bash
export PROXY_UPSTREAM_ENDPOINT=https://api.openai.com
export PROXY_LOG_ROOT=./proxy_logs
export PROXY_HOST=127.0.0.1
export PROXY_PORT=9000
uv run albicilla-proxy
```

## Conversation Converter

Use the bundled `albicilla-conv` CLI to turn captured proxy logs into JSONL datasets that can be fed to fine-tuning or analytics workflows. All commands expect `--logs` to point at the directory that the proxy writes to and `--output` to target a directory where converted files will be stored.

#### Output schema (flattened by default)

Each processed record mirrors the OpenAI fine-tuning schema but is normalized so downstream tooling does not need to inspect raw request envelopes:

- `messages` contains the chronology of the request conversation plus the final assistant reply. When `--json-tool-calls` is enabled, tool calls are emitted inline using `<tool_call>` / `<tool_result>` wrappers so that the assistant content remains valid JSON even after multiple calls.
- `tools` is derived from the original request’s `tools` array and flattened to just the function definition block (`{"name", "description", "parameters"}` etc.). This makes the schema explicit for analytics/training without carrying the surrounding wrapper structure.

### Per-day output (native tool calls by default)

```bash
uv run albicilla-conv --logs ./proxy_logs --output ./output
```

This creates day-based subdirectories inside `./output` and keeps the tool call payloads exactly as captured by the proxy.

### Enable tool structured output (JSON tool calls)

```bash
uv run albicilla-conv --logs ./proxy_logs --output ./output --json-tool-calls
```

Wraps tool invocations and results with inline structured tags so downstream consumers can reliably parse them:

```json
{
  "messages": [
    {
      "role": "assistant",
      "content": "<tool_call>{\"name\": \"retrieve_user\", \"arguments\": {\"user_id\": 42}}</tool_call>"
    },
    {
      "role": "tool",
      "content": "<tool_result tool_call_id=\"call_abc\">{\"name\": \"Ada Lovelace\"}</tool_result>"
    }
  ]
}
```

### Single-file export

```bash
uv run albicilla-conv --logs ./proxy_logs --output ./output --single-file
```

Collects all sessions into one `conversations.jsonl` file inside the output directory.

### Verbose logging

```bash
uv run albicilla-conv --logs ./proxy_logs --output ./output -v
```

Adds progress information to stderr, which is useful when processing large log directories.

## Example Request

```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Log Directory Structure

```
proxy_logs/
└── 2026-01-13/
    └── session-id.jsonl
```

Each line in the JSONL file contains:
- `timestamp`: ISO 8601 UTC timestamp
- `session_id`: Resolved session identifier
- `request`: Full request payload
- `response`: Full response payload

## Configuration

| Option | Env Variable | Default | Description |
|--------|--------------|---------|-------------|
| `--upstream-endpoint` | `PROXY_UPSTREAM_ENDPOINT` | *(required)* | Upstream OpenAI-compatible API base URL |
| `--log-root` | `PROXY_LOG_ROOT` | `./proxy_logs` | Root directory for logs |
| `--host` | `PROXY_HOST` | `0.0.0.0` | Server bind address |
| `--port` | `PROXY_PORT` | `9000` | Server port |

## Development

Run tests:

```bash
uv run pytest
```

## License

MIT
