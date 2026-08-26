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

WHY IT COMPUTES THE CONFIGURATION DIFFERENCES RATHER THAN PRINTING
FINGERPRINTS AND TRUSTING THE READER. An earlier revision did the latter, on
the reasoning that the caller submitted the runs and knows what they are.
That is the same shape as every defect this apparatus exists to catch: a
check a person has to remember to perform by eye is not a check. So each run
is differenced against the first with
:func:`~platform_core.comparability.find_differences` and the axes are named.

A difference on :const:`CONFOUNDING_AXES` is called out rather than listed,
because it does not qualify the reading -- it destroys it. Two runs in
different images that disagree at a rung have told you nothing about cards.
The command still prints the whole report: refusing would hide the numbers
from someone who has a reason to look at them, and a warning they cannot
miss does the job without deciding for them.

AN AXIS NOBODY RECORDED IS REPORTED SEPARATELY, because differencing cannot
see it. Two runs that each failed to record an image digest have EQUAL
digests as far as :func:`find_differences` is concerned, and the report read
"identical to the reference" for a pair that had no idea what they ran in.
:func:`~platform_core.known_answer_registry.incomplete_axes` already names
empty axes for the registry, which refuses to store an entry carrying one;
the same call answers it here.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import find_differences
from platform_core.json_utils import load_json_str
from platform_core.known_answer_registry import incomplete_axes
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import (
    ObservationAgreement,
    RunAgreement,
    RunRecord,
    agree_across_runs,
    decode_run_record,
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

#: Digits used when printing a probe value. Seventeen significant digits is
#: what it takes to write an IEEE double so it reads back unchanged, and
#: anything shorter would print two differing values identically -- which is
#: the exact mistake this command exists to prevent.
VALUE_DIGITS = 17

#: Fingerprint axes whose difference makes a cross-card reading meaningless
#: rather than merely qualified. Two runs in different images that disagree at
#: a rung have not told you anything about their cards -- the image is the
#: variable the whole apparatus pins so that the card can be the one that
#: moves. Determinism is deliberately absent: it is a nested record whose
#: differences the per-axis listing already names, and a change to it is a
#: finding rather than a confound.
CONFOUNDING_AXES = ("image_digest",)


def read_ladder_records(directory: pathlib.Path) -> tuple[tuple[str, RunRecord], ...]:
    """Read every record in a directory, in filename order.

    Every ``.json`` file is read and decoded. Nothing is skipped on the way:
    a file that does not decode fails the command, because a directory of
    ladder records with one unreadable member is a directory whose agreement
    result would silently cover fewer cards than the reader thinks.

    Args:
        directory: Directory holding one ladder record per run.

    Returns:
        ``(filename, record)`` pairs, sorted by filename so the value columns
        are in the same order every time this is run.

    Raises:
        FileNotFoundError: If the directory does not exist, or holds no
            ``.json`` file at all.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"no such directory: {directory}")
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no .json records in {directory}")
    return tuple(
        (path.name, decode_run_record(load_json_str(path.read_text(encoding="utf-8"))))
        for path in paths
    )


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


def configuration_lines(named_records: tuple[tuple[str, RunRecord], ...]) -> tuple[str, ...]:
    """Report how each run's configuration differs from the first.

    The first run is the reference simply because a set needs one; which run
    holds the position does not change which axes are named.

    Args:
        named_records: ``(filename, record)`` pairs, in report order.

    Returns:
        One line per run, plus a warning line for every confounding axis found
        anywhere in the set.
    """
    reference = named_records[0][1]["fingerprint"]
    lines = [f"  [0] {named_records[0][0]}  (reference)"]
    confounded: list[str] = []

    for index, (name, record) in enumerate(named_records[1:], start=1):
        differences = find_differences(reference, record["fingerprint"])
        axes = [d["axis"] for d in differences]
        described = ", ".join(axes) if axes else "identical to the reference"
        lines.append(f"  [{index}] {name}  differs on: {described}")
        confounded.extend(axis for axis in axes if axis in CONFOUNDING_AXES)

    # An axis nobody recorded compares EQUAL to the same gap in another run,
    # so two runs that each failed to record their image read as "identical
    # configuration". They are not; they are two runs that cannot say. The
    # empty axis has to be named separately from the differing ones, because
    # the difference machinery is structurally unable to see it.
    for index, (name, record) in enumerate(named_records):
        empty = incomplete_axes(record["fingerprint"])
        if empty:
            lines.append(
                f"  !! [{index}] {name} recorded no {', '.join(empty)}. An unrecorded axis"
            )
            lines.append(
                "     matches every other unrecorded one, so agreement here is not evidence."
            )
            # NOT added to `confounded`. That warning reads "these runs do not
            # all share one X", which is false here -- they share the absence.
            # The line above already says the true thing. A set where only SOME
            # runs recorded the axis reaches `confounded` through the
            # differencing loop, because empty differs from non-empty there.

    for axis in sorted(set(confounded)):
        lines.append(f"  !! these runs do not all share one {axis}. A disagreement between")
        lines.append(
            "     them is not a fact about the cards, and this report cannot separate them."
        )
    return tuple(lines)


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

    named_records = read_ladder_records(pathlib.Path(cli_args.require_flag(parsed, DIR_FLAG)))
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
    "read_ladder_records",
    "report_lines",
    "rung_agreement",
]


if __name__ == "__main__":
    entrypoint()
