"""Read several sdpa-backend records and say which kernel each card chose.

The record holds facts -- a digest per forced backend, a digest for the
unforced call, an availability flag and torch's eligibility opinion. This
derives the answer from them: the backend whose forced digest equals the
unforced one is the backend the dispatcher selected.

TWO OUTCOMES OF THAT METHOD ARE NOT ANSWERS, AND THIS SAYS SO RATHER THAN
PICKING. If no forced digest matches, the unforced call produced something
none of the forced runs reproduced. If several match, two backends agree bit
for bit here and the method cannot separate them. Both are printed as what
they are; a report that silently returned the first match would be inventing
a selection.

IT ALSO CHECKS TORCH AGAINST ITSELF. ``can_use_*`` is an opinion about a
configuration and forcing the backend is what happened. Where the two
disagree -- eligible but no kernel, or ineligible yet it ran -- that is a
finding about torch rather than about the card, and it gets its own line.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import RunRecord

from model_trainer.cli.record_reports import (
    agreement_groups,
    configuration_lines,
    read_run_records,
)
from model_trainer.core.run_fingerprint import describe_run_fingerprint
from model_trainer.core.services.model.sdpa_shapes import (
    AVAILABLE_SUFFIX,
    BACKEND_KEYS,
    DEFAULT_KEY,
    DIGEST_SUFFIX,
    ELIGIBLE_SUFFIX,
    TRUE_VALUE,
    SdpaShape,
    sdpa_label,
    sdpa_shapes,
)

_log = get_logger(__name__)

DIR_FLAG = "--dir"

_FLAGS = (DIR_FLAG,)

#: What a shape's row says when no forced backend reproduced the unforced
#: call. Named rather than left blank: "nothing matched" is a finding, and a
#: blank cell reads as a formatting fault.
NONE_MATCHED = "NONE-MATCHED"


def _values(record: RunRecord) -> dict[str, float]:
    """Index one record's observations by name."""
    return {o["name"]: o["value"] for o in record["observations"]}


def selection_for(record: RunRecord, shape: SdpaShape) -> tuple[str, ...]:
    """Name the backends whose forced output matched the unforced call.

    Args:
        record: One card's record.
        shape: The attention call to read.

    Returns:
        Matching backend keys in declaration order; empty when none matched
        or when the record does not carry this shape.
    """
    values = _values(record)
    default = values.get(sdpa_label(shape, DEFAULT_KEY, DIGEST_SUFFIX))
    if default is None:
        return ()
    return tuple(
        name
        for name in BACKEND_KEYS
        if values.get(sdpa_label(shape, name, DIGEST_SUFFIX)) == default
    )


def describe_selection(selected: tuple[str, ...]) -> str:
    """Render a selection result, including the two that are not answers.

    Args:
        selected: Backends whose digest matched.

    Returns:
        The backend name, :data:`NONE_MATCHED`, or the ambiguous set joined
        by ``=`` to show they were indistinguishable rather than that one won.
    """
    if not selected:
        return NONE_MATCHED
    return "=".join(selected)


def disagreements(record: RunRecord, shape: SdpaShape) -> tuple[str, ...]:
    """Name backends where torch's opinion and the forced run disagree.

    Args:
        record: One card's record.
        shape: The attention call to read.

    Returns:
        One phrase per disagreeing backend, empty when they all agree or the
        record carries no opinion for this shape.
    """
    values = _values(record)
    out: list[str] = []
    for name in BACKEND_KEYS:
        eligible = values.get(sdpa_label(shape, name, ELIGIBLE_SUFFIX))
        available = values.get(sdpa_label(shape, name, AVAILABLE_SUFFIX))
        if eligible is None or available is None:
            continue
        if eligible == TRUE_VALUE and available != TRUE_VALUE:
            out.append(f"{name}: torch says eligible, forcing it found no kernel")
        if eligible != TRUE_VALUE and available == TRUE_VALUE:
            out.append(f"{name}: torch says ineligible, yet forcing it ran")
    return tuple(out)


def output_agreement(named_records: tuple[tuple[str, RunRecord], ...], shape: SdpaShape) -> str:
    """Say which cards produced the same attention output.

    The bridge back to the forward trace, and the reason this report is not
    just a selection table. Two cards selecting the SAME backend and still
    disagreeing bitwise is a different finding from two cards selecting
    different backends, and a selection table alone cannot tell them apart.

    Args:
        named_records: ``(filename, record)`` pairs, in report order.
        shape: The call to read.

    Returns:
        Run indices grouped by shared digest, or ``"not reported"`` when some
        run did not carry this shape.
    """
    values: list[float] = []
    for _, record in named_records:
        digest = _values(record).get(sdpa_label(shape, DEFAULT_KEY, DIGEST_SUFFIX))
        if digest is None:
            return "not reported"
        values.append(digest)
    return agreement_groups(tuple(values))


def shape_lines(
    named_records: tuple[tuple[str, RunRecord], ...], shape: SdpaShape
) -> tuple[str, ...]:
    """Render one attention call's row across every card.

    Args:
        named_records: ``(filename, record)`` pairs, in report order.
        shape: The call to report.

    Returns:
        Its lines: the selection per card, which cards' outputs agree, then
        any torch-versus-reality disagreements.
    """
    label = f"{shape['rung']:<12} h{shape['heads']:<3} s{shape['sequence_len']:<4}"
    picks = [describe_selection(selection_for(record, shape)) for _, record in named_records]
    lines = [
        f"  {label} {'  '.join(f'[{i}] {p:<12}' for i, p in enumerate(picks))}"
        f" outputs={output_agreement(named_records, shape)}"
    ]

    for index, (name, record) in enumerate(named_records):
        for phrase in disagreements(record, shape):
            lines.append(f"    !! [{index}] {name} {shape['rung']}: {phrase}")
    return tuple(lines)


def report_lines(named_records: tuple[tuple[str, RunRecord], ...]) -> tuple[str, ...]:
    """Render the whole report.

    Args:
        named_records: ``(filename, record)`` pairs, in the order their
            columns should appear.

    Returns:
        The lines to print.

    Raises:
        ValueError: When fewer than two runs are given. A selection table
            over one card is a fact about that card, and this report exists
            to put cards beside each other.
    """
    if len(named_records) < 2:
        raise ValueError(f"a selection table needs at least two runs, got {len(named_records)}")

    lines = [f"{len(named_records)} runs"]
    lines += [
        f"  [{index}] {name}  {describe_run_fingerprint(record['fingerprint'])}"
        for index, (name, record) in enumerate(named_records)
    ]
    lines.append("")
    lines.append("configuration")
    lines += list(configuration_lines(named_records))
    lines.append("")
    lines.append("backend the dispatcher selected, by forced-output identity")
    for shape in sdpa_shapes():
        lines += list(shape_lines(named_records, shape))
    return tuple(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the backend-selection table for a directory of records.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the report is printed.

    Raises:
        ValueError: When a flag is unknown, repeated, or missing, or when
            fewer than two records were found.
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
        service_name="sdpa-probe-report",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "NONE_MATCHED",
    "describe_selection",
    "disagreements",
    "entrypoint",
    "main",
    "output_agreement",
    "report_lines",
    "selection_for",
    "shape_lines",
]


if __name__ == "__main__":
    entrypoint()
