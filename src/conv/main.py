"""CLI interface for conversation logs processor."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer
from loguru import logger

from .models import ConversationRecord, SessionToolUsageRow
from .processor import process_logs_directory_with_tool_usage

DEFAULT_LOGS_DIR = Path.cwd() / "proxy_logs"
DEFAULT_OUTPUT_DIR = Path.cwd() / "output"
DEFAULT_TOOL_REPORT_NAME = "tool-usage.csv"

app = typer.Typer(
    help="Process proxy logs and export conversations as JSONL.",
    pretty_exceptions_enable=True,
)


@app.command()
def process(
    logs: Path = typer.Option(
        DEFAULT_LOGS_DIR,
        "--logs",
        "-l",
        help="Directory containing JSONL log files",
    ),
    output: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output",
        "-o",
        help="Directory where output files should be written",
    ),
    single_file: bool = typer.Option(
        False,
        "--single-file",
        "-s",
        help="Merge all sessions into one output file",
    ),
    json_tool_calls: bool = typer.Option(
        False,
        "--json-tool-calls/--no-json-tool-calls",
        "-j/-J",
        help="Transform tool_calls into inline JSON format",
    ),
    tool_report: bool = typer.Option(
        True,
        "--tool-report/--no-tool-report",
        help="Write a per-session tool usage report",
    ),
    tool_report_path: Path | None = typer.Option(
        None,
        "--tool-report-path",
        help="Path for the tool usage report (defaults to output/tool-usage.jsonl)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Process proxy logs and export session JSONL files."""
    # Configure logging
    if verbose:
        logger.enable("conv")
    else:
        logger.disable("conv")

    logs_dir = logs.expanduser()
    output_dir = output.expanduser()

    if not logs_dir.exists() or not logs_dir.is_dir():
        raise typer.BadParameter(
            f"Logs directory '{logs_dir}' does not exist or is not a directory."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all sessions
    logger.info("Processing logs from {path}", path=logs_dir)
    records_by_date, tool_usage = process_logs_directory_with_tool_usage(
        logs_dir, json_tool_calls=json_tool_calls
    )

    if not records_by_date:
        typer.echo("No sessions found to process.")
        raise typer.Exit(code=0)

    # Write output
    total_records = sum(len(records) for records in records_by_date.values())

    if single_file:
        output_path = output_dir / "conversations.jsonl"
        all_records = [
            record for records in records_by_date.values() for record in records
        ]
        write_records_to_file(all_records, output_path)
        typer.echo(f"Wrote {total_records} records to {output_path}")
    else:
        files_written = write_per_day(records_by_date, output_dir)
        typer.echo(
            f"Wrote {total_records} records across {files_written} files to {output_dir}"
        )

    if tool_report:
        report_path = tool_report_path or (output_dir / DEFAULT_TOOL_REPORT_NAME)
        write_tool_usage_report(tool_usage, report_path)
        typer.echo(f"Wrote tool usage report to {report_path}")


def write_records_to_file(records: list[ConversationRecord], output_path: Path) -> None:
    """Write conversation records to a JSONL file."""
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record.model_dump(), ensure_ascii=False)
            handle.write(line + "\n")


def write_per_day(
    records_by_date: dict[date, list[ConversationRecord]],
    output_dir: Path,
) -> int:
    """Write one JSONL file per day. Returns number of files written."""
    files_written = 0

    for day, records in sorted(records_by_date.items()):
        output_path = output_dir / f"{day.isoformat()}.jsonl"
        write_records_to_file(records, output_path)
        logger.info(
            "Wrote {count} records to {path}",
            count=len(records),
            path=output_path,
        )
        files_written += 1

    return files_written


def write_tool_usage_report(tool_usage: list[SessionToolUsageRow], output_path: Path) -> None:
    """Write a per-session tool usage report as CSV (comma-separated)."""
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("date,scenario,session_count,tool_call_count,tool_definition_count\n")
        for row in sorted(tool_usage, key=lambda item: (item.date, item.session)):
            scenario = _csv_escape(row.session)
            handle.write(
                f"{row.date.isoformat()},{scenario},{row.session_count},{row.tool_call_count},{row.tool_definition_count}\n"
            )


def _csv_escape(value: str) -> str:
    """Escape a CSV field without using the csv module."""
    if any(ch in value for ch in (",", "\"", "\n", "\r")):
        escaped = value.replace('"', '""')
        return f'"{escaped}"'
    return value


def main() -> None:
    """CLI entrypoint."""
    app()


if __name__ == "__main__":
    main()
