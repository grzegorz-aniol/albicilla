.PHONY: test test-unit test-integration test-all run-proxy

default:
	@echo "Specify a target"

# Run the OpenAI-compatible logging proxy
run-proxy:
	LOGURU_LEVEL=DEBUG uv run albicilla-proxy -u http://localhost:1234

run-proxy-openai:
	LOGURU_LEVEL=DEBUG uv run albicilla-proxy -u https://api.openai.com

# Run all tests
test: test-all

# Run unit tests only (fast, no server startup)
test-unit:
	uv run pytest tests/test_proxy.py tests/test_processing.py -v

# Run integration tests only (starts actual server)
test-integration:
	uv run pytest tests/test_proxy_integration.py -v

# Run all tests
test-all:
	uv run pytest tests/ -v

