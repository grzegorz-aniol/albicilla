# Albicilla

OpenAI-compatible logging proxy that captures LLM requests/responses to JSONL files for training and auditing.

## Features

- Drop-in replacement for OpenAI API endpoints
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

