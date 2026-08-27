"""Read forward-pass cost records and say what the attention pin costs end to end.

The decision logic -- when a ratio may be printed and when it must be
withheld -- is :mod:`sdpa_benchmark_report`'s, unchanged and shared rather
than restated. That is deliberate: the whole point of this measurement is to
be read against the per-call one, and two readers would be two places for
"is this ratio trustworthy" to be answered differently.

WHAT THIS REPORT ADDS. A column the per-call table has no use for: the
absolute milliseconds per pass. A multiplier alone cannot say whether a cost
is affordable, and end-to-end that is the question -- 5x on a pass that takes
two milliseconds and 5x on one that takes two seconds are different facts.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import RunRecord

from model_trainer.cli import _test_hooks
from model_trainer.cli.forward_benchmark import forward_prefix
from model_trainer.cli.record_reports import read_run_records
from model_trainer.cli.sdpa_benchmark import PINNED_KEY
from model_trainer.cli.sdpa_benchmark_report import memory_growth, slowdown
from model_trainer.core.run_fingerprint import describe_run_fingerprint
from model_trainer.core.services.model.cost_labels import (
    DEFAULT_KEY,
    SECONDS_SUFFIX,
    labelled,
)

_log = get_logger(__name__)

DIR_FLAG = "--dir"

_FLAGS = (DIR_FLAG,)

#: What a cell shows when the record does not carry that timing.
ABSENT = "--"


def milliseconds(values: dict[str, float], prefix: str, backend: str) -> str:
    """Render one arm's absolute time for one row.

    Args:
        values: The record's observations by name.
        prefix: What was measured.
        backend: Which arm.

    Returns:
        Milliseconds per pass, or :data:`ABSENT`.
    """
    seconds = values.get(labelled(prefix, backend, SECONDS_SUFFIX))
    if seconds is None:
        return ABSENT
    return f"{seconds * 1e3:.1f}"


def report_lines(named_records: tuple[tuple[str, RunRecord], ...]) -> tuple[str, ...]:
    """Render the whole report.

    Args:
        named_records: ``(filename, record)`` pairs, in column order.

    Returns:
        The lines to print: one block per run.
    """
    lines: list[str] = []
    for index, (name, record) in enumerate(named_records):
        values = {o["name"]: o["value"] for o in record["observations"]}
        lines.append(f"[{index}] {name}  {describe_run_fingerprint(record['fingerprint'])}")
        lines.append(f"  {'row':<20} {'default ms':>11} {'math ms':>10} {'slower':>12}   memory")
        for shape in _test_hooks.forward_shapes():
            prefix = forward_prefix(shape)
            lines.append(
                f"  {shape['name']:<20} "
                f"{milliseconds(values, prefix, DEFAULT_KEY):>11} "
                f"{milliseconds(values, prefix, PINNED_KEY):>10} "
                f"{slowdown(values, prefix):>12}   {memory_growth(values, prefix)}"
            )
        lines.append("")
    return tuple(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the end-to-end cost table for a directory of records.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the report is printed.

    Raises:
        ValueError: When a flag is unknown, repeated, or missing.
        FileNotFoundError: When the directory is absent or holds no records.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    named_records = read_run_records(pathlib.Path(cli_args.require_flag(parsed, DIR_FLAG)))
    for line in report_lines(named_records):
        _log.info("%s", line)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level="INFO",
        format_mode="text",
        service_name="forward-benchmark-report",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "ABSENT",
    "entrypoint",
    "main",
    "memory_growth",
    "milliseconds",
    "report_lines",
    "slowdown",
]


if __name__ == "__main__":
    entrypoint()
