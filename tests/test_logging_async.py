import asyncio
import json

from albicilla.config import Settings
from albicilla.logging import append_session_entry_async, get_log_path


def test_append_session_entry_async_serializes_lines(tmp_path):
    settings = Settings(
        upstream_endpoint="http://example.com",
        log_root=tmp_path,
    )
    request_data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    }
    response_data = {
        "id": "resp-1",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
    }

    async def _run() -> None:
        await asyncio.gather(
            append_session_entry_async(settings, "test-session", request_data, response_data),
            append_session_entry_async(settings, "test-session", request_data, response_data),
        )

    asyncio.run(_run())

    log_path = get_log_path(settings, "test-session")
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        entry = json.loads(line)
        assert entry["session_id"] == "test-session"
        assert "request" in entry
        assert "response" in entry
