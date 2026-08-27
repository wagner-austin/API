"""Read several forward traces and name the operation where they first differ.

The trace writes one record per run, each holding a digest for every tensor
that crossed a module boundary. The finding is not in any one of them: it is
the FIRST tensor, in execution order, whose digests stop matching -- because
every later difference is downstream of it and says nothing more.

WHY FIRST-IN-EXECUTION-ORDER IS THE WHOLE POINT. A transformer carries a
difference forward: once one tensor differs, the residual stream differs for
the rest of the model, and a report that listed every differing tensor would
list thousands and bury the one that matters. The step counter is assigned by
the hooks at run time and zero-padded into the observation name, so sorting
the record's own observations recovers the order the kernels ran in.

WHAT THE LOSS ROW IS FOR. It is the instrument's control, not a result. It
must equal what the untraced ladder recorded for the same rung on the same
card under the same condition; the report prints it so that a reader can
check the trace against the ladder rather than take it on trust. Hooks that
changed the arithmetic would show up here first.

WHY IT DOES NOT WRITE A FILE. Same reason the ladder report does not: it is
derived entirely from the records, and a stored derivative goes stale
silently the moment another card is added. The records are the durable
artifact; this is a view of them.
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
    VALUE_DIGITS,
    configuration_lines,
    read_run_records,
)
from model_trainer.core.run_fingerprint import describe_run_fingerprint
from model_trainer.core.services.model.trace_plan import (
    DIGEST_SUFFIX,
    WORKSPACE_NAME,
    TraceName,
    describe_workspace,
    parse_trace_name,
    trace_loss_name,
)

_log = get_logger(__name__)

DIR_FLAG = "--dir"

_FLAGS = (DIR_FLAG,)

#: How many differing tensors to list after the first. The first names the
#: operation; a handful after it says whether the difference stayed inside
#: one module or spread immediately, which is the difference between "one
#: kernel disagrees" and "the residual stream diverged".
FOLLOWING_SHOWN = 5


def describe_condition(record: RunRecord) -> str:
    """Say which split-K condition one record ran under.

    The fingerprint cannot answer this -- nothing in it carries
    ``CUBLASLT_WORKSPACE_SIZE``, so two runs differing only in the variable
    the experiment manipulates difference as identical. The trace records it
    as an observation instead, and this reads it back.

    Args:
        record: One trace record.

    Returns:
        The condition, or ``"NOT RECORDED"`` for a record written before the
        trace recorded it. Named loudly rather than rendered as "unset",
        because "this run did not set the variable" and "this run cannot say
        whether it set the variable" are different facts and only one of them
        is a measurement.
    """
    for observation in record["observations"]:
        if observation["name"] == WORKSPACE_NAME:
            return describe_workspace(observation["value"])
    return "NOT RECORDED"


def _execution_order(pair: tuple[TraceName, ObservationAgreement]) -> tuple[int, int]:
    """Sort key putting traced tensors in the order the hooks saw them.

    A named function rather than a lambda: a lambda's parameter would be
    inferred as Any, and this package forbids expressions of that type.

    Args:
        pair: A parsed name and its agreement entry.

    Returns:
        ``(step, index within the step)``.
    """
    return pair[0]["step"], pair[0]["index"]


def rung_digests(
    agreement: RunAgreement, rung: str
) -> tuple[tuple[TraceName, ObservationAgreement], ...]:
    """Collect one rung's digest observations, in execution order.

    Only digests. The sums are recorded for magnitude and are read once a
    difference is located; including them here would double every row and put
    two answers to different questions in one column.

    Args:
        agreement: The computed agreement over every run.
        rung: The rung to collect.

    Returns:
        ``(parsed name, agreement entry)`` pairs, sorted by step then by the
        index within a step.
    """
    found: list[tuple[TraceName, ObservationAgreement]] = []
    for entry in agreement["shared"]:
        parsed = parse_trace_name(entry["name"])
        if parsed is not None and parsed["rung"] == rung and parsed["suffix"] == DIGEST_SUFFIX:
            found.append((parsed, entry))
    return tuple(sorted(found, key=_execution_order))


def divergences(
    digests: tuple[tuple[TraceName, ObservationAgreement], ...],
) -> tuple[tuple[TraceName, ObservationAgreement], ...]:
    """Keep only the tensors the runs did not all agree on.

    Args:
        digests: One rung's digest observations, in execution order.

    Returns:
        The differing ones, in the same order.
    """
    return tuple(pair for pair in digests if pair[1]["distinct"] > 1)


def agreement_groups(values: tuple[float, ...]) -> str:
    """Say which runs agreed with which, by run index.

    ``distinct=2`` over three runs says two agree and one does not. It does
    NOT say WHICH, and which is the entire finding: the earlier ladder work
    established that the odd card MOVES between rungs and between conditions
    -- the V100 is alone at ``xl`` by default and the A30 is alone at ``xl``
    with split-K removed. A count cannot show that and a column of 48-bit
    digests is unreadable, so this renders the partition instead.

    Args:
        values: One observation's value from each run, in run order.

    Returns:
        Run indices grouped by shared value, groups ordered by first
        appearance and joined by ``|`` -- e.g. ``"0,2|1"`` for three runs
        where the second one is the odd one out.
    """
    groups: list[list[int]] = []
    seen: dict[float, int] = {}
    for index, value in enumerate(values):
        at = seen.get(value)
        if at is None:
            seen[value] = len(groups)
            groups.append([index])
        else:
            groups[at].append(index)
    return "|".join(",".join(str(index) for index in group) for group in groups)


def _tensor_line(name: TraceName, entry: ObservationAgreement) -> str:
    """Render one traced tensor's row.

    Args:
        name: Its parsed observation name.
        entry: Its agreement across the runs.

    Returns:
        A single line naming the step, the module, how many distinct values
        the runs produced and which runs shared which.
    """
    return (
        f"    step {name['step']:>5}  {name['kind']:<3} "
        f"{name['module_class']}.{name['path']}#{name['index']}  "
        f"distinct={entry['distinct']} runs={agreement_groups(entry['values'])}"
    )


def _loss_line(agreement: RunAgreement, rung: str) -> str:
    """Render one rung's control row.

    Args:
        agreement: The computed agreement.
        rung: The rung.

    Returns:
        A line carrying every run's loss, or a line saying no run reported
        one -- which would mean the record is not a trace this can read.
    """
    wanted = trace_loss_name(rung)
    for entry in agreement["shared"]:
        if entry["name"] == wanted:
            values = " ".join(f"{value:.{VALUE_DIGITS}g}" for value in entry["values"])
            groups = agreement_groups(entry["values"])
            return f"  loss distinct={entry['distinct']} runs={groups}  {values}"
    return "  loss not reported by every run"


def rung_lines(agreement: RunAgreement, rung: str) -> tuple[str, ...]:
    """Render one rung's whole section.

    Args:
        agreement: The computed agreement over every run.
        rung: The rung to report.

    Returns:
        The lines: the control, the count of traced and differing tensors,
        and the first differing tensor with a few of its successors.
    """
    digests = rung_digests(agreement, rung)
    if not digests:
        return (f"rung {rung}", "  no traced tensors shared by every run")

    differing = divergences(digests)
    lines = [
        f"rung {rung}",
        _loss_line(agreement, rung),
        f"  {len(digests)} tensors traced by every run, {len(differing)} differ",
    ]
    if not differing:
        lines.append("  -> every traced tensor is bit-identical across these runs")
        return tuple(lines)

    first, first_entry = differing[0]
    lines.append(
        f"  -> first difference at step {first['step']}: "
        f"{first['module_class']}.{first['path']} ({first['kind']}put #{first['index']}), "
        f"runs={agreement_groups(first_entry['values'])}"
    )
    lines.append(f"  first {min(len(differing), FOLLOWING_SHOWN + 1)} differing tensors:")
    lines.append(_tensor_line(first, first_entry))
    lines += [_tensor_line(name, entry) for name, entry in differing[1 : FOLLOWING_SHOWN + 1]]
    return tuple(lines)


def traced_rungs(agreement: RunAgreement) -> tuple[str, ...]:
    """Name every rung the runs share, in the order they were traced.

    Read from the observations rather than from
    :data:`~model_trainer.core.services.model.trace_plan.TRACE_RUNGS`, so a
    report over records written by an older trace names what those records
    actually contain instead of silently reporting nothing for a rung they
    never walked.

    Args:
        agreement: The computed agreement.

    Returns:
        The rung names, ordered by the first step each one traced.

    """
    first_step: dict[str, int] = {}
    for entry in agreement["shared"]:
        parsed = parse_trace_name(entry["name"])
        if parsed is None:
            continue
        seen = first_step.get(parsed["rung"])
        if seen is None or parsed["step"] < seen:
            first_step[parsed["rung"]] = parsed["step"]
    ordered = sorted((step, rung) for rung, step in first_step.items())
    return tuple(rung for _, rung in ordered)


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
        f"  [{index}] {name}  {describe_run_fingerprint(record['fingerprint'])} "
        f"cublaslt_workspace={describe_condition(record)}"
        for index, (name, record) in enumerate(named_records)
    ]
    lines.append("")
    lines.append("configuration")
    lines += list(configuration_lines(named_records))

    for rung in traced_rungs(agreement):
        lines.append("")
        lines += list(rung_lines(agreement, rung))

    if agreement["unmatched"]:
        lines.append("")
        lines.append(
            f"{len(agreement['unmatched'])} observations not reported by every run "
            "(a module graph that differs between runs appears here, not above):"
        )
        lines += [f"  {name}" for name in agreement["unmatched"][:FOLLOWING_SHOWN]]

    return tuple(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the divergence report for a directory of trace records.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the report is printed. Deliberately 0 even when the runs
        diverge: a divergence is the measurement, not a failure.

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
        service_name="probe-trace-report",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "FOLLOWING_SHOWN",
    "agreement_groups",
    "describe_condition",
    "divergences",
    "entrypoint",
    "main",
    "report_lines",
    "rung_digests",
    "rung_lines",
    "traced_rungs",
]


if __name__ == "__main__":
    entrypoint()
