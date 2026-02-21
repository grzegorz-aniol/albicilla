.PHONY: test test-unit test-integration test-all run-proxy

default:
	@echo "Specify a target"

# Run the OpenAI-compatible logging proxy
run-proxy:
	LOGURU_LEVEL=DEBUG albicilla-proxy -u http://localhost:1234

run-proxy-openai:
	LOGURU_LEVEL=DEBUG albicilla-proxy -u https://api.openai.com

# Run all tests
test:
	uv run pytest tests/ -v
