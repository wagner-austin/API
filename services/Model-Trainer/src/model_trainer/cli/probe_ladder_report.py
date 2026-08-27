"""Read several ladder records and say where the cards stop agreeing.

The ladder writes one record per card. The finding is not in any one of them:
it is the rung at which their values stop being identical. Doing that
comparison by eye means reading seventeen-digit decimals in columns, which is
how "these agree" gets said about numbers that do not -- so it is done here,
by :func:`~platform_core.run_record.agree_across_runs`, and printed.

WHY THE RESULT IS NOT WRITTEN TO A FILE. It is derived entirely from the
records, and a stored derivative goes stale silently the moment another card
is added: a reader who finds it cannot tell whether it covers three cards or
five. The records are the durable artifact; this is a view of them, and
re-running it is cheap.

THE CONFIGURATION DIFFERENCES ARE COMPUTED RATHER THAN PRINTED FOR THE READER
TO CHECK, and a difference on a confounding axis is called out rather than
listed -- both live in :mod:`record_reports`, which the forward-trace report
shares. The command still prints the whole report when an axis is confounded:
refusing would hide the numbers from someone who has a reason to look at them,
and a warning they cannot miss does the job without deciding for them.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import (
    ObservationAgreement,
    RunAgreement,
    RunRecord,
    agree_across_runs,
)

from model_trainer.cli.record_reports import (
    CONFOUNDING_AXES,
    VALUE_DIGITS,
    configuration_lines,
    read_run_records,
)
from model_trainer.core.run_fingerprint import describe_run_fingerprint
from model_trainer.core.services.model.probe_shapes import (
    PROBE_AXES,
    ProbeAxis,
    probe_label,
    require_probe_shape,
)

_log = get_logger(__name__)

DIR_FLAG = "--dir"

_FLAGS = (DIR_FLAG,)


def rung_agreement(agreement: RunAgreement, rung: str) -> ObservationAgreement | None:
    """Find one rung's entry in an agreement result.

    Args:
        agreement: The computed agreement.
        rung: The rung name to look up.

    Returns:
        Its entry, or None when not every run reported that rung.

    Raises:
        KeyError: If the rung is not one the probe ladder declares.
    """
    label = probe_label(require_probe_shape(rung))
    for entry in agreement["shared"]:
        if entry["name"] == label:
            return entry
    return None


def first_disagreement(agreement: RunAgreement, axis: ProbeAxis) -> str | None:
    """Name the first rung along an axis whose runs did not all agree.

    "First" is in the axis's own order, which is why
    :data:`~model_trainer.core.services.model.known_answer_probe.PROBE_AXES`
    declares one. A record sorts its observations by name, and the rung labels
    sort alphabetically -- ``large`` before ``medium`` before ``small`` --
    so reading a record's own order would name a threshold at random.

    Args:
        agreement: The computed agreement.
        axis: The axis to walk.

    Returns:
        The first rung with more than one distinct value, or None when every
        rung the runs shared agreed exactly. Rungs no run shared are skipped;
        they are reported as unmatched elsewhere rather than counted as
        agreement.
    """
    for rung in axis["rungs"]:
        entry = rung_agreement(agreement, rung)
        if entry is not None and entry["distinct"] > 1:
            return rung
    return None


def _rung_line(rung: str, entry: ObservationAgreement | None) -> str:
    """Render one rung's row.

    Args:
        rung: The rung name.
        entry: Its agreement entry, or None when not every run reported it.

    Returns:
        A single line.
    """
    if entry is None:
        return f"  {rung:<16} (not reported by every run)"
    values = " ".join(f"{value:.{VALUE_DIGITS}g}" for value in entry["values"])
    return f"  {rung:<16} distinct={entry['distinct']} spread={entry['spread']:.3e}  {values}"


def report_lines(named_records: tuple[tuple[str, RunRecord], ...]) -> tuple[str, ...]:
    """Render the whole report.

    Args:
        named_records: ``(filename, record)`` pairs, in the order their values
            should appear in each row.

    Returns:
        The lines to print.

    Raises:
        ValueError: Propagated from
            :func:`~platform_core.run_record.agree_across_runs` when fewer
            than two runs are given or they answer different experiments.
    """
    records = tuple(record for _, record in named_records)
    agreement = agree_across_runs(records)

    lines = [f"{agreement['runs']} runs, experiment {agreement['experiment']}"]
    lines += [
        f"  [{index}] {name}  {describe_run_fingerprint(record['fingerprint'])}"
        for index, (name, record) in enumerate(named_records)
    ]
    lines.append("")
    lines.append("configuration")
    lines += list(configuration_lines(named_records))

    for axis in PROBE_AXES:
        lines.append("")
        lines.append(f"axis {axis['name']}")
        lines += [_rung_line(rung, rung_agreement(agreement, rung)) for rung in axis["rungs"]]
        broke = first_disagreement(agreement, axis)
        if broke is None:
            lines.append("  -> every shared rung agreed exactly")
        else:
            lines.append(f"  -> agreement breaks at rung {broke!r}")

    if agreement["unmatched"]:
        lines.append("")
        lines.append("observations not reported by every run:")
        lines += [f"  {name}" for name in agreement["unmatched"]]

    return tuple(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the agreement report for a directory of ladder records.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the report is printed. Deliberately 0 even when the runs
        disagree: a disagreement is the measurement, not a failure, and a
        non-zero exit would make a shell treat the answer as an error.

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
        service_name="probe-ladder-report",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "CONFOUNDING_AXES",
    "VALUE_DIGITS",
    "configuration_lines",
    "entrypoint",
    "first_disagreement",
    "main",
    "read_run_records",
    "report_lines",
    "rung_agreement",
]


if __name__ == "__main__":
    entrypoint()
