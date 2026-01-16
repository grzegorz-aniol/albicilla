"""Integration tests for the OpenAI-Compatible Logging Proxy.

These tests start an actual server and make real HTTP requests.
"""

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest


class MockUpstreamHandler(BaseHTTPRequestHandler):
    """Mock upstream OpenAI-compatible server."""

    def do_POST(self):
        """Handle POST requests to chat completions."""
        if self.path == "/v1/chat/completions":
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            request_data = json.loads(body) if body else {}

            # Build mock response
            model = request_data.get("model", "mock-model")
            messages = request_data.get("messages", [])

            response = {
                "id": "chatcmpl-integration-test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Mock response to {len(messages)} messages",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress request logging."""
        pass


@pytest.fixture(scope="module")
def mock_upstream():
    """Start a mock upstream server."""
    server = HTTPServer(("127.0.0.1", 18766), MockUpstreamHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    # Wait for server to be ready
    for _ in range(30):
        try:
            resp = httpx.post(
                "http://127.0.0.1:18766/v1/chat/completions",
                json={"messages": []},
                timeout=1.0,
            )
            if resp.status_code == 200:
                break
        except httpx.ConnectError:
            pass
        time.sleep(0.1)

    yield "http://127.0.0.1:18766"

    server.shutdown()


@pytest.fixture(scope="module")
def server(tmp_path_factory, mock_upstream):
    """Start the proxy server for integration testing."""
    log_dir = tmp_path_factory.mktemp("integration_logs")
    port = 18765  # Use non-standard port to avoid conflicts

    # Start server as subprocess, pointing to mock upstream
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "albicilla.main",
            "--upstream",
            mock_upstream,
            "--port",
            str(port),
            "--log-root",
            str(log_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent.parent,
    )

    # Wait for server to be ready
    base_url = f"http://127.0.0.1:{port}"
    max_retries = 30
    for i in range(max_retries):
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                break
        except httpx.ConnectError:
            pass
        time.sleep(0.1)
    else:
        proc.terminate()
        stdout, stderr = proc.communicate(timeout=5)
        pytest.fail(f"Server failed to start.\nstdout: {stdout}\nstderr: {stderr}")

    yield {"url": base_url, "log_dir": log_dir, "port": port}

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


class TestIntegration:
    """Integration tests with real HTTP requests."""

    def test_health_endpoint(self, server):
        """Health endpoint responds correctly."""
        resp = httpx.get(f"{server['url']}/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_chat_completion_basic(self, server):
        """Basic chat completion request works end-to-end."""
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello, integration test!"}
            ],
        }

        resp = httpx.post(
            f"{server['url']}/v1/chat/completions",
            json=payload,
        )

        assert resp.status_code == 200
        data = resp.json()

        # Verify response structure
        assert data["object"] == "chat.completion"
        assert data["model"] == "gpt-4"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "1 messages" in data["choices"][0]["message"]["content"]

    def test_chat_completion_with_session_header(self, server):
        """Session ID from header is used for logging."""
        payload = {
            "messages": [{"role": "user", "content": "Test with header"}]
        }

        resp = httpx.post(
            f"{server['url']}/v1/chat/completions",
            json=payload,
            headers={"X-Session-Id": "integration-test-session"},
        )

        assert resp.status_code == 200

        # Verify log file was created with correct session name
        log_files = list(server["log_dir"].rglob("integration-test-session.jsonl"))
        assert len(log_files) == 1

        # Verify log content
        with open(log_files[0]) as f:
            entry = json.loads(f.readline())

        assert entry["session_id"] == "integration-test-session"
        assert entry["request"]["messages"][0]["content"] == "Test with header"

    def test_chat_completion_with_user_field(self, server):
        """User field in payload takes precedence over headers."""
        payload = {
            "messages": [{"role": "user", "content": "Test with user field"}],
            "user": "payload-user-session",
        }

        resp = httpx.post(
            f"{server['url']}/v1/chat/completions",
            json=payload,
            headers={"X-Session-Id": "should-be-ignored"},
        )

        assert resp.status_code == 200

        # Verify log file uses user field, not header
        log_files = list(server["log_dir"].rglob("payload-user-session.jsonl"))
        assert len(log_files) == 1

    def test_chat_completion_with_bearer_token(self, server):
        """Bearer token creates consistent session mapping."""
        payload = {"messages": [{"role": "user", "content": "Bearer test"}]}
        headers = {"Authorization": "Bearer integration-test-token"}

        # Make two requests with same token
        resp1 = httpx.post(
            f"{server['url']}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        resp2 = httpx.post(
            f"{server['url']}/v1/chat/completions",
            json=payload,
            headers=headers,
        )

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        # Both should go to same log file (bearer-* pattern)
        bearer_logs = list(server["log_dir"].rglob("session-*.jsonl"))
        assert len(bearer_logs) == 1

        # Should have 2 entries
        with open(bearer_logs[0]) as f:
            entries = f.readlines()
        assert len(entries) == 2

    def test_multiple_messages_in_request(self, server):
        """Request with multiple messages is handled correctly."""
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Second message"},
            ],
            "user": "multi-message-test",
        }

        resp = httpx.post(
            f"{server['url']}/v1/chat/completions",
            json=payload,
        )

        assert resp.status_code == 200
        data = resp.json()

        # Response should mention 4 messages
        assert "4 messages" in data["choices"][0]["message"]["content"]
        assert data["model"] == "gpt-4o"

    def test_invalid_payload_returns_422(self, server):
        """Invalid payload returns validation error."""
        # Missing required 'messages' field
        payload = {"model": "gpt-4"}

        resp = httpx.post(
            f"{server['url']}/v1/chat/completions",
            json=payload,
        )

        assert resp.status_code == 422  # Unprocessable Entity

    def test_logs_are_date_partitioned(self, server):
        """Logs are organized in date folders."""
        payload = {
            "messages": [{"role": "user", "content": "Date partition test"}],
            "user": "date-partition-test",
        }

        resp = httpx.post(
            f"{server['url']}/v1/chat/completions",
            json=payload,
        )

        assert resp.status_code == 200

        # Find log file and verify date folder structure
        log_files = list(server["log_dir"].rglob("date-partition-test.jsonl"))
        assert len(log_files) == 1

        # Parent should be a date folder (YYYY-MM-DD format)
        date_folder = log_files[0].parent.name
        assert len(date_folder) == 10  # YYYY-MM-DD
        assert date_folder.count("-") == 2

    def test_response_contains_id(self, server):
        """Each response contains an ID from upstream."""
        payload = {"messages": [{"role": "user", "content": "ID test"}]}

        resp = httpx.post(
            f"{server['url']}/v1/chat/completions",
            json=payload,
        )
        assert resp.status_code == 200
        assert "id" in resp.json()

    def test_extra_fields_are_preserved_in_logs(self, server):
        """Unknown fields in request are logged for completeness."""
        payload = {
            "messages": [{"role": "user", "content": "Extra fields test"}],
            "user": "extra-fields-test",
            "custom_field": "custom_value",
            "another_field": {"nested": True},
        }

        resp = httpx.post(
            f"{server['url']}/v1/chat/completions",
            json=payload,
        )

        assert resp.status_code == 200

        # Verify extra fields are in the log
        log_files = list(server["log_dir"].rglob("extra-fields-test.jsonl"))
        with open(log_files[0]) as f:
            entry = json.loads(f.readline())

        assert entry["request"]["custom_field"] == "custom_value"
        assert entry["request"]["another_field"] == {"nested": True}

