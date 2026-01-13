"""Tests for the OpenAI-Compatible Logging Proxy."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from albicilla.app import create_app
from albicilla.config import Settings
from albicilla.logging import sanitize_session_id
from albicilla.models import ChatCompletionRequest, ChatCompletionMessage
from albicilla.responders import build_response
from albicilla.session import clear_token_map
from albicilla.upstream import UpstreamResponse


def make_mock_upstream_response(model: str = "gpt-4o-mini") -> UpstreamResponse:
    """Create a mock upstream response."""
    return UpstreamResponse(
        data={
            "id": "chatcmpl-mock123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Mock response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        headers={"x-request-id": "mock-request-id"},
    )


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """Create a temporary log directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def settings(temp_log_dir: Path) -> Settings:
    """Create settings with temporary log directory."""
    return Settings(log_root=temp_log_dir, upstream_endpoint="http://localhost:11434")


@pytest.fixture
def mock_upstream():
    """Mock the upstream forward_request function."""
    with patch("albicilla.app.forward_request", new_callable=AsyncMock) as mock:
        mock.return_value = make_mock_upstream_response()
        yield mock


@pytest.fixture
def client(settings: Settings, mock_upstream) -> TestClient:
    """Create a test client with configured app and mocked upstream."""
    clear_token_map()
    app = create_app(settings)
    return TestClient(app)


@pytest.fixture
def minimal_payload() -> dict:
    """Create a minimal valid chat completion request."""
    return {
        "messages": [
            {"role": "user", "content": "Hello, world!"}
        ]
    }


class TestSchemaRoundTrip:
    """Test that request/response schemas work correctly."""

    def test_minimal_payload_validates(self, client: TestClient, minimal_payload: dict):
        """Ensure minimal payload validates and returns expected structure."""
        response = client.post("/v1/chat/completions", json=minimal_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "chat.completion"
        assert "id" in data
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_full_payload_validates(self, client: TestClient, mock_upstream):
        """Ensure a full payload with all fields validates."""
        mock_upstream.return_value = make_mock_upstream_response(model="gpt-4")
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "n": 1,
            "stream": False,
            "max_tokens": 100,
            "user": "test-user",
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.json()["model"] == "gpt-4"

    def test_unknown_fields_allowed(self, client: TestClient, minimal_payload: dict):
        """Ensure unknown fields are accepted (extra='allow')."""
        minimal_payload["custom_field"] = "custom_value"
        response = client.post("/v1/chat/completions", json=minimal_payload)
        assert response.status_code == 200


class TestLogging:
    """Test JSONL logging functionality."""

    def test_log_file_created(
        self, client: TestClient, minimal_payload: dict, temp_log_dir: Path
    ):
        """POST payload creates JSONL file with request/response."""
        minimal_payload["user"] = "test-session"
        response = client.post("/v1/chat/completions", json=minimal_payload)
        assert response.status_code == 200

        # Find the log file
        log_files = list(temp_log_dir.rglob("*.jsonl"))
        assert len(log_files) == 1
        assert "test-session" in log_files[0].name

        # Verify content
        with open(log_files[0]) as f:
            entry = json.loads(f.readline())

        assert "timestamp" in entry
        assert entry["session_id"] == "test-session"
        assert entry["request"]["messages"][0]["content"] == "Hello, world!"
        assert "response" in entry

    def test_multiple_requests_append(
        self, client: TestClient, minimal_payload: dict, temp_log_dir: Path
    ):
        """Multiple requests to same session append to same file."""
        minimal_payload["user"] = "multi-test"

        for _ in range(3):
            response = client.post("/v1/chat/completions", json=minimal_payload)
            assert response.status_code == 200

        log_files = list(temp_log_dir.rglob("*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            lines = f.readlines()
        assert len(lines) == 3


class TestSessionResolution:
    """Test session ID resolution fallback order."""

    def test_user_field_takes_precedence(
        self, client: TestClient, minimal_payload: dict, temp_log_dir: Path
    ):
        """User field in payload takes precedence over headers."""
        minimal_payload["user"] = "payload-user"
        response = client.post(
            "/v1/chat/completions",
            json=minimal_payload,
            headers={"X-Session-Id": "header-session"},
        )
        assert response.status_code == 200

        log_files = list(temp_log_dir.rglob("*.jsonl"))
        assert any("payload-user" in f.name for f in log_files)

    def test_header_fallback(
        self, client: TestClient, minimal_payload: dict, temp_log_dir: Path
    ):
        """X-Session-Id header is used when user field missing."""
        response = client.post(
            "/v1/chat/completions",
            json=minimal_payload,
            headers={"X-Session-Id": "header-session"},
        )
        assert response.status_code == 200

        log_files = list(temp_log_dir.rglob("*.jsonl"))
        assert any("header-session" in f.name for f in log_files)

    def test_bearer_token_mapping(
        self, client: TestClient, minimal_payload: dict, temp_log_dir: Path
    ):
        """Bearer token creates consistent session mapping."""
        headers = {"Authorization": "Bearer test-token-123"}

        # First request
        response1 = client.post("/v1/chat/completions", json=minimal_payload, headers=headers)
        assert response1.status_code == 200

        # Second request with same token
        response2 = client.post("/v1/chat/completions", json=minimal_payload, headers=headers)
        assert response2.status_code == 200

        # Should have only one log file (same session)
        log_files = list(temp_log_dir.rglob("*.jsonl"))
        assert len(log_files) == 1
        assert "bearer-" in log_files[0].name

    def test_uuid_fallback(
        self, client: TestClient, minimal_payload: dict, temp_log_dir: Path
    ):
        """UUID fallback when no session identifiers provided."""
        response = client.post("/v1/chat/completions", json=minimal_payload)
        assert response.status_code == 200

        log_files = list(temp_log_dir.rglob("*.jsonl"))
        assert len(log_files) == 1
        assert "anon-" in log_files[0].name


class TestUpstreamErrors:
    """Test error handling for upstream failures."""

    def test_upstream_error_returns_error_status(
        self, settings: Settings, minimal_payload: dict, temp_log_dir: Path
    ):
        """Upstream error is propagated to client."""
        from albicilla.upstream import UpstreamError

        clear_token_map()
        with patch("albicilla.app.forward_request", new_callable=AsyncMock) as mock:
            mock.side_effect = UpstreamError(503, "Service unavailable")
            app = create_app(settings)
            client = TestClient(app)

            response = client.post("/v1/chat/completions", json=minimal_payload)
            assert response.status_code == 503
            assert "Service unavailable" in response.json()["detail"]

    def test_upstream_timeout_returns_504(
        self, settings: Settings, minimal_payload: dict
    ):
        """Upstream timeout returns 504 Gateway Timeout."""
        from albicilla.upstream import UpstreamError

        clear_token_map()
        with patch("albicilla.app.forward_request", new_callable=AsyncMock) as mock:
            mock.side_effect = UpstreamError(504, "Upstream service timeout")
            app = create_app(settings)
            client = TestClient(app)

            response = client.post("/v1/chat/completions", json=minimal_payload)
            assert response.status_code == 504

    def test_upstream_connection_error_returns_502(
        self, settings: Settings, minimal_payload: dict
    ):
        """Upstream connection error returns 502 Bad Gateway."""
        from albicilla.upstream import UpstreamError

        clear_token_map()
        with patch("albicilla.app.forward_request", new_callable=AsyncMock) as mock:
            mock.side_effect = UpstreamError(502, "Cannot connect to upstream")
            app = create_app(settings)
            client = TestClient(app)

            response = client.post("/v1/chat/completions", json=minimal_payload)
            assert response.status_code == 502


class TestSanitization:
    """Test session ID sanitization."""

    def test_sanitize_removes_path_traversal(self):
        """Sanitize removes path traversal attempts."""
        assert sanitize_session_id("../../../etc/passwd") == "_________etc_passwd"
        assert sanitize_session_id("normal-session_123") == "normal-session_123"
        assert sanitize_session_id("") == "unknown"
        assert sanitize_session_id("test/with/slashes") == "test_with_slashes"


class TestResponseBuilder:
    """Test synthetic response generation."""

    def test_response_uses_template(self, settings: Settings):
        """Response content uses configured template."""
        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                ChatCompletionMessage(role="user", content="Hello"),
                ChatCompletionMessage(role="user", content="World"),
            ],
        )

        response = build_response(request, settings)

        assert "2 messages" in response.choices[0].message.content
        assert "gpt-4" in response.choices[0].message.content

    def test_default_model_used(self, settings: Settings):
        """Default model is used when not specified in request."""
        request = ChatCompletionRequest(
            messages=[ChatCompletionMessage(role="user", content="Hello")]
        )

        response = build_response(request, settings)

        assert response.model == settings.default_model


class TestHeaderForwarding:
    """Test header forwarding to upstream."""

    def test_filter_request_headers_preserves_authorization(self):
        """Authorization header is preserved when filtering."""
        from albicilla.upstream import filter_request_headers

        headers = {
            "authorization": "Bearer sk-test-key-12345",
            "host": "localhost:9000",
            "content-type": "application/json",
            "content-length": "123",
            "user-agent": "test-client",
        }

        filtered = filter_request_headers(headers)

        assert "authorization" in filtered
        assert filtered["authorization"] == "Bearer sk-test-key-12345"
        assert "host" not in filtered
        assert "content-length" not in filtered
        assert "user-agent" in filtered

    def test_authorization_header_passed_to_upstream(
        self, settings: Settings, minimal_payload: dict
    ):
        """Authorization header from request is passed to forward_request."""
        clear_token_map()

        with patch("albicilla.app.forward_request", new_callable=AsyncMock) as mock:
            mock.return_value = make_mock_upstream_response()
            app = create_app(settings)
            client = TestClient(app)

            client.post(
                "/v1/chat/completions",
                json=minimal_payload,
                headers={"Authorization": "Bearer sk-my-secret-key"},
            )

            # Check that forward_request was called with headers containing authorization
            call_args = mock.call_args
            request_headers = call_args.kwargs.get("request_headers") or call_args.args[2]
            assert "authorization" in request_headers
            assert request_headers["authorization"] == "Bearer sk-my-secret-key"


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check(self, client: TestClient):
        """Health check returns ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

