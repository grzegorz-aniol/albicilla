"""CLI entrypoint for the OpenAI-Compatible Logging Proxy."""

from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from loguru import logger

from .app import create_app
from .config import Settings
from .logging import configure_logging

app = typer.Typer(
    name="albicilla",
    help="Albicilla - LLM passthrough proxy with JSONL logging.",
)


@app.command()
def serve(
    upstream: Annotated[
        str,
        typer.Option(
            "--upstream", "-u",
            help="Upstream OpenAI-compatible API endpoint (required).",
            envvar="PROXY_UPSTREAM_ENDPOINT",
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model", "-m",
            help="Default model to use when not specified in request.",
            envvar="PROXY_DEFAULT_MODEL",
        ),
    ] = "gpt-4o-mini",
    log_root: Annotated[
        Path,
        typer.Option(
            "--log-root",
            "-l",
            help="Root directory for log files.",
            envvar="PROXY_LOG_ROOT",
        ),
    ] = Path("./proxy_logs"),
    host: Annotated[
        str,
        typer.Option(
            "--host",
            "-h",
            help="Host to bind the server to.",
            envvar="PROXY_HOST",
        ),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to bind the server to.",
            envvar="PROXY_PORT",
        ),
    ] = 9000,
    reload: Annotated[
        bool,
        typer.Option(
            "--reload",
            "-r",
            help="Enable auto-reload for development.",
            envvar="PROXY_RELOAD",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable debug logging.",
            envvar="PROXY_VERBOSE",
        ),
    ] = False,
) -> None:
    """Start the OpenAI-compatible logging proxy server."""
    # Create settings from CLI args (env vars are also loaded)
    settings = Settings(
        upstream_endpoint=upstream,
        default_model=model,
        log_root=log_root,
        host=host,
        port=port,
        verbose=verbose,
    )

    configure_logging(settings.verbose)
    logger.info(f"Starting proxy server on {settings.host}:{settings.port}")
    logger.info(f"Upstream endpoint: {settings.upstream_endpoint}")
    logger.info(f"Default model: {settings.default_model}")
    logger.info(f"Logging requests to: {settings.log_root.absolute()}")

    # Ensure log directory exists
    settings.log_root.mkdir(parents=True, exist_ok=True)

    app_instance = create_app(settings)
    uvicorn.run(
        app_instance,
        host=settings.host,
        port=settings.port,
        reload=reload,
    )


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
