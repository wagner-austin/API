"""Read attention-cost records and say what pinning the math backend costs.

The record holds seconds, spread, peak bytes and a fitted flag per shape per
backend. This turns them into the two ratios a decision actually needs -- how
much slower, and how much more memory -- and refuses to print a ratio it
cannot stand behind.

WHY A RATIO CAN BE WITHHELD. Three cases are not slowdowns and are named
instead of divided:

* the pinned backend did not FIT, which is the strongest cost result there is
  and is not a number;
* a measurement's spread is a large fraction of its median, which on a shared
  cluster means the node was busy and the median must not be compared with
  another one -- the split-K benchmark learned this from a run with 54%, 85%
  and 90% batch spreads;
* the two measurements are both below the per-call floor, where launch
  overhead dominates and the arithmetic cannot be resolved at all.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import RunRecord

from model_trainer.cli.record_reports import read_run_records
from model_trainer.cli.sdpa_benchmark import PINNED_KEY
from model_trainer.core.run_fingerprint import describe_run_fingerprint
from model_trainer.core.services.model.cost_labels import (
    DEFAULT_KEY,
    FITTED_SUFFIX,
    PEAK_SUFFIX,
    SECONDS_SUFFIX,
    SPREAD_SUFFIX,
    TRUE_VALUE,
    labelled,
)
from model_trainer.core.services.model.sdpa_shapes import cost_prefix, cost_shapes

_log = get_logger(__name__)

DIR_FLAG = "--dir"

_FLAGS = (DIR_FLAG,)

#: What a row says when the pinned backend ran out of memory.
DID_NOT_FIT = "DID NOT FIT"

#: What a row says when a record claims a call fitted and then carries no
#: timing for it. Kept distinct from :data:`DID_NOT_FIT`, which is a fact
#: about the card, because this is a fact about the record -- and reporting
#: a truncated record as a memory limit would put a defect in the file into
#: a table of hardware results.
INCOMPLETE = "incomplete record"

#: What a row says when a measurement is too noisy to divide. A spread this
#: large a fraction of the median means the node was scheduling other work.
NOISY = "too noisy"

#: Spread over median above which a measurement is called noisy.
NOISE_LIMIT = 0.20

#: Seconds per call below which launch overhead dominates. Measured on this
#: cluster by the split-K benchmark: 136 us on the V100, 115 on the A30, 104
#: on the A100. The largest is used, so "clears the floor" means clears it on
#: every card rather than on the fastest one.
OVERHEAD_FLOOR = 136e-6

#: What a row says when both measurements sit under that floor.
BELOW_FLOOR = "below floor"


def _values(record: RunRecord) -> dict[str, float]:
    """Index one record's observations by name."""
    return {o["name"]: o["value"] for o in record["observations"]}


def fitted(values: dict[str, float], prefix: str, backend: str) -> bool:
    """Say whether one piece of work fitted in memory under one backend.

    Args:
        values: The record's observations by name.
        prefix: What was measured.
        backend: Which backend.

    Returns:
        True when it fitted. A record that does not mention it reads as not
        fitted, which is conservative: it keeps a ratio from being printed
        for something that may never have run.
    """
    return values.get(labelled(prefix, backend, FITTED_SUFFIX)) == TRUE_VALUE


def slowdown(values: dict[str, float], prefix: str) -> str:
    """Render how much slower pinning the math backend made one measurement.

    Args:
        values: The record's observations by name.
        prefix: What was measured.

    Returns:
        A multiplier, or one of :data:`DID_NOT_FIT`, :data:`INCOMPLETE`,
        :data:`NOISY` or :data:`BELOW_FLOOR`.
    """
    if not fitted(values, prefix, PINNED_KEY):
        return DID_NOT_FIT
    base = values.get(labelled(prefix, DEFAULT_KEY, SECONDS_SUFFIX))
    pinned = values.get(labelled(prefix, PINNED_KEY, SECONDS_SUFFIX))
    if base is None or pinned is None:
        # Distinct from DID NOT FIT, which is a fact about the card. This is
        # a record that says a call fitted and then carries no timing for it,
        # which is a fact about the record.
        return INCOMPLETE
    for seconds, arm in ((base, DEFAULT_KEY), (pinned, PINNED_KEY)):
        spread = values.get(labelled(prefix, arm, SPREAD_SUFFIX))
        if spread is not None and seconds > 0.0 and spread / seconds > NOISE_LIMIT:
            return NOISY
    if base < OVERHEAD_FLOOR and pinned < OVERHEAD_FLOOR:
        return BELOW_FLOOR
    return f"{pinned / base:.1f}x"


def memory_growth(values: dict[str, float], prefix: str) -> str:
    """Render how much more memory pinning the math backend needed.

    Args:
        values: The record's observations by name.
        prefix: What was measured.

    Returns:
        A multiplier and the pinned peak in MiB, or :data:`DID_NOT_FIT`.
    """
    if not fitted(values, prefix, PINNED_KEY):
        return DID_NOT_FIT
    base = values.get(labelled(prefix, DEFAULT_KEY, PEAK_SUFFIX))
    pinned = values.get(labelled(prefix, PINNED_KEY, PEAK_SUFFIX))
    if base is None or pinned is None or base <= 0.0:
        return "not recorded"
    return f"{pinned / base:.1f}x ({pinned / 2**20:.0f} MiB)"


def report_lines(named_records: tuple[tuple[str, RunRecord], ...]) -> tuple[str, ...]:
    """Render the whole report.

    Args:
        named_records: ``(filename, record)`` pairs, in column order.

    Returns:
        The lines to print: one block per run, each a table of slowdown and
        memory growth per shape.
    """
    lines: list[str] = []
    for index, (name, record) in enumerate(named_records):
        values = _values(record)
        lines.append(f"[{index}] {name}  {describe_run_fingerprint(record['fingerprint'])}")
        lines.append(f"  {'shape':<22} {'batch':>5} {'seq':>5} {'slower':>12}   memory")
        for shape in cost_shapes():
            prefix = cost_prefix(shape)
            lines.append(
                f"  {shape['name']:<22} {shape['batch']:>5} {shape['sequence_len']:>5} "
                f"{slowdown(values, prefix):>12}   {memory_growth(values, prefix)}"
            )
        lines.append("")
    return tuple(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the cost table for a directory of records.

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
        service_name="sdpa-benchmark-report",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "BELOW_FLOOR",
    "DID_NOT_FIT",
    "INCOMPLETE",
    "NOISE_LIMIT",
    "NOISY",
    "OVERHEAD_FLOOR",
    "entrypoint",
    "fitted",
    "main",
    "memory_growth",
    "report_lines",
    "slowdown",
]


if __name__ == "__main__":
    entrypoint()
