"""Configuration for the proxy server."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Proxy server settings with env/CLI override support."""

    log_root: Path = Path("./proxy_logs")
    host: str = "0.0.0.0"
    port: int = 9000
    default_model: str = "gpt-4o-mini"
    response_template: str = "Proxy logged {n_messages} messages for {model}."
    upstream_endpoint: str  # Required - upstream OpenAI-compatible API base URL

    # Constants (not configurable via env)
    session_header: str = "X-Session-Id"

    model_config = {
        "env_prefix": "PROXY_",
        "env_file": ".env",
        "extra": "ignore",
    }

    @property
    def upstream_chat_completions_url(self) -> str:
        """Get the full URL for chat completions endpoint."""
        base = self.upstream_endpoint.rstrip("/")
        # Handle both base URL and /v1 URL formats
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"