# Albicilla

OpenAI-compatible logging pass-trough proxy that captures LLM requests/responses for auditing.

## Features

- Drop-in replacement for OpenAI API endpoints
- Supports `/v1/chat/completions` endpoint (both streaming and non-streaming requests)
- Forwards requests to upstream LLM and logs request/response pairs
- Per-session JSONL logging with automatic date-based organization
- Session grouping via required session header (default `agent-session-id`), with `X-Session-Id` fallback
- Per-session log writes are serialized in-process; multi-worker setups need external locking

## Installation

```bash
uv sync
```

## Usage

Start the proxy server:

```bash
uv run albicilla-proxy --upstream https://api.openai.com --log-root ./proxy_logs --host 127.0.0.1 --port 9000
```

Session headers are enforced by default. Set a custom header name with `--session-header`, disable enforcement with
`--no-require-session-header`, and (optionally) allow bearer fallback with `--allow-bearer-token-fallback`.

Or with environment variables:

```bash
export PROXY_UPSTREAM_ENDPOINT=https://api.openai.com
export PROXY_DEFAULT_MODEL=gpt-4o-mini
export PROXY_LOG_ROOT=./proxy_logs
export PROXY_HOST=127.0.0.1
export PROXY_PORT=9000
export PROXY_VERBOSE=false
export PROXY_SESSION_HEADER=agent-session-id
export PROXY_REQUIRE_SESSION_HEADER=true
export PROXY_ALLOW_BEARER_FALLBACK=false
uv run albicilla-proxy
```

### Running with Docker

Build the image locally:

```bash
docker build -t albicilla-proxy .
```

Start the container with the proxy bound to `localhost:9000`, mount the log directory for inspection, and pass any additional CLI arguments via `command`:

```bash
docker run --rm \
  -p 9000:9000 \
  -v "$(pwd)/proxy_logs:/app/proxy_logs" \
  albicilla-proxy \
  --upstream https://api.openai.com \
  --log-root /app/proxy_logs
```

All proxy flags can be overridden by appending to the end of the `docker run` invocation, letting you keep configuration alongside compose files or orchestration manifests.

### Configuring clients

Albicilla currently proxies **only** the `/v1/chat/completions` endpoint. Any client that can target an OpenAI-compatible chat completions API can reuse your upstream API key and simply point its base URL at the proxy (for example `http://127.0.0.1:9000/v1`). Setting `OPENAI_API_BASE` or the tool’s equivalent `base_url` flag is usually enough.

> Keep using your real OpenAI (or other upstream) API key. Albicilla captures the traffic while forwarding it to the upstream provider.

#### OpenAI SDK (Python)

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9000/v1",
    api_key=os.environ["OPENAI_API_KEY"],
)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Ping"}],
)
print(completion.choices[0].message.content)
```

#### OpenAI SDK (JavaScript/TypeScript)

```ts
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "http://127.0.0.1:9000/v1",
});

const result = await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "Ping" }],
});
console.log(result.choices[0].message);
```

#### LangChain

```python
import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base="http://127.0.0.1:9000/v1",
)
response = llm.invoke("Summarize today's weather")
print(response.content)
```

#### Pydantic.AI

```python
import os
from pydantic_ai import Agent
from pydantic_ai.clients.openai import OpenAIChatCompletionsClient

client = OpenAIChatCompletionsClient(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="http://127.0.0.1:9000/v1",
)
weather_bot = Agent("Summarize weather alerts", client=client, model="gpt-4o-mini")
print(weather_bot.run("Any storms expected?"))
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
| `--upstream` | `PROXY_UPSTREAM_ENDPOINT` | *(required)* | Upstream OpenAI-compatible API base URL |
| `--model` | `PROXY_DEFAULT_MODEL` | `gpt-4o-mini` | Default model when none is provided |
| `--log-root` | `PROXY_LOG_ROOT` | `./proxy_logs` | Root directory for logs |
| `--host` | `PROXY_HOST` | `0.0.0.0` | Server bind address |
| `--port` | `PROXY_PORT` | `9000` | Server port |
| `--reload` | `PROXY_RELOAD` | `false` | Enable auto-reload for development |
| `--verbose` | `PROXY_VERBOSE` | `false` | Enable debug logging |

## Development

Run tests:

```bash
uv run pytest
```

## License

MIT
