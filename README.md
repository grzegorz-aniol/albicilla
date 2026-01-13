# Albicilla

OpenAI-compatible logging proxy that captures LLM requests/responses to JSONL files for training and auditing.

## Features

- Drop-in replacement for OpenAI API endpoints
- Per-session JSONL logging with automatic date-based organization
- Session grouping via `user` field, headers, or bearer tokens
- Synthetic response mode for offline testing

## Installation

```bash
uv sync
```

## Usage

Start the proxy server:

```bash
uv run albicilla-proxy --log-root ./proxy_logs --host 127.0.0.1 --port 9000
```

Or with environment variables:

```bash
export LOG_ROOT=./proxy_logs
export APP_HOST=127.0.0.1
export APP_PORT=9000
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
| `--log-root` | `LOG_ROOT` | `./proxy_logs` | Root directory for logs |
| `--host` | `APP_HOST` | `0.0.0.0` | Server bind address |
| `--port` | `APP_PORT` | `9000` | Server port |

## Development

Run tests:

```bash
uv run pytest
```

## License

MIT

