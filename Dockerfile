FROM python:3.13-slim AS base

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY src/ src/

ENV PROXY_LOG_ROOT=/app/proxy_logs \
    PROXY_HOST=0.0.0.0 \
    PROXY_PORT=9000

EXPOSE 9000
VOLUME ["/app/proxy_logs"]

ENTRYPOINT ["uv", "run", "albicilla-proxy"]
CMD []
