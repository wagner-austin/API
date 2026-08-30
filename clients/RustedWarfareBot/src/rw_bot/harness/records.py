"""Filing what a batch's numbers were produced under, beside the numbers.

A batch already writes one scorecard per match. What it did not write was
anything saying what those scorecards were produced UNDER -- which game build,
which machine, which image -- so two batches could be compared on nothing but
the assumption that nothing had moved between them.

:mod:`rw_bot.provenance` decides what a record IS; this writes one per arm at
the end of a batch, next to the results it summarises.

WRITTEN BY THE THING THAT RAN THE EXPERIMENT, not by the analyser. An analysis
tool reads a batch long afterwards, on a machine that may not be the one that
played it -- so a fingerprint captured there would describe the reader rather
than the run. The sweep is the only place that is definitely the run.

REWRITTEN ON EVERY PASS, because a batch is resumable: a run that plays the
last four matches of a twelve-match batch must leave a record covering all
twelve, not four. The scorecards are the store and the record is derived from
them, so recomputing is what keeps the two from disagreeing.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import determinism_record
from platform_core.environment_record import stdlib_host_probe
from platform_core.json_utils import dump_json_str
from platform_core.run_record import NO_PAYLOAD, encode_run_record, run_record_sidecar

from rw_bot.harness import _test_hooks
from rw_bot.harness.results_layout import RESULT_SUFFIX, TRACE_SUFFIX
from rw_bot.harness.scorecards import MatchRow, parse_match_row, row_order
from rw_bot.provenance import arm_run_record, summarize_arm, sweep_fingerprint

#: Column of a trace line holding the extractor count, and how many columns a
#: line must have before that column means anything.
_EXTRACTOR_COLUMN = 4
_TRACE_COLUMNS = 12

#: Indent a filed record is written with. Read by people at a terminal as
#: often as by tools -- "what was this batch played under" is asked by hand.
RECORD_INDENT = 2

#: What the simulation runs on. The engine is Java on the CPU; there is no
#: card in this experiment and no record should imply one.
DETERMINISM_DEVICE = "cpu"


def read_extractors(lines: Sequence[str]) -> tuple[int, int]:
    """Return the extractor peak and the peak-to-end drop from a trace.

    The figure every verdict this project has produced turns on, and the one
    a scorecard cannot carry: a match reporting ``extractors 0 -> 0`` had in
    fact held a peak of fourteen before collapsing ([[policy-trace]]).

    Args:
        lines: The trace's lines, header included.

    Returns:
        The peak and how many of it were gone by the end. Both zero for a
        trace with no sample lines, which is what an interrupted match leaves.
    """
    peak = end = 0
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= _TRACE_COLUMNS and parts[0].isdigit():
            value = int(parts[_EXTRACTOR_COLUMN])
            peak = max(peak, value)
            end = value
    return peak, peak - end


def read_batch_rows(out_dir: Path, traces_root: Path, batch: str) -> tuple[MatchRow, ...]:
    """Read every filed result of a batch as a row.

    Args:
        out_dir: Where the batch's results are filed.
        traces_root: Root of the trace tree.
        batch: The batch's name, which is also its trace namespace.

    Returns:
        One row per result, ordered by arm then seed so a record built from
        them does not change with the filesystem's listing order.

    Raises:
        OSError: When the results directory cannot be read.
        ValueError: When a result filename does not name an arm and a seed.
    """
    rows: list[MatchRow] = []
    for name in _test_hooks.list_names(out_dir):
        if not name.endswith(RESULT_SUFFIX):
            continue
        stem = name[: -len(RESULT_SUFFIX)]
        trace = traces_root / batch / f"{stem}{TRACE_SUFFIX}"
        peak, dropped = (
            read_extractors(_test_hooks.read_text_lines(trace))
            if _test_hooks.path_exists(trace)
            else (0, 0)
        )
        text = "\n".join(_test_hooks.read_text_lines(out_dir / name))
        rows.append(parse_match_row(stem, text, peak, dropped))
    return tuple(sorted(rows, key=row_order))


def batch_fingerprint(source_game_dir: str) -> RunFingerprint:
    """Describe what a batch's numbers were produced under.

    Assembled here rather than in the sweep's command line, so the entry point
    does not have to know what a determinism record is: it knows which game
    directory it played from, which is the only thing it decides.

    The platform comes from the running interpreter, which is right HERE and
    would be wrong almost anywhere else in this package: everything else
    composes a command for a machine it is not on, while this describes a
    batch that already played on this one.

    Args:
        source_game_dir: The pinned game directory the batch played from.
            Passed whole, because the record names the engine, the runtime and
            the assets and all three are found inside it.

    Returns:
        The fingerprint.

    Raises:
        FileNotFoundError: When the game jar or the runtime's release file is
            absent. Propagated because a record saying "some build" about an
            obfuscated binary says nothing, and this project's claims are
            valid for one build only.
        JvmReleaseError: ``RW-JVM-002`` when the runtime states no version.
        TreeIdentityError: When the runtime or the asset tree is absent, empty
            or holds a symbolic link.
        UnknownCoreCountError: When the platform will not report its core
            count, rather than recording a guess.
    """
    return sweep_fingerprint(
        # Nothing is pinned: the simulation is Java and this harness sets no
        # thread or BLAS environment, so an empty record is the accurate
        # statement rather than an omission.
        determinism_record(DETERMINISM_DEVICE, {}),
        _test_hooks.get_env,
        stdlib_host_probe(_test_hooks.count_cores),
        Path(source_game_dir),
        _test_hooks.read_platform(),
    )


def write_arm_records(
    out_dir: Path, batch: str, rows: Sequence[MatchRow], fingerprint: RunFingerprint
) -> tuple[str, ...]:
    """File one run record per arm, beside that arm's results.

    Args:
        out_dir: Where the batch's results are filed.
        batch: The batch's name, which the record's label carries.
        rows: Every filed result of the batch.
        fingerprint: What the batch was played under.

    Returns:
        The arms a record was written for, in order.

    Raises:
        OSError: When a record cannot be written.
    """
    arms = sorted({row["arm"] for row in rows})
    for arm in arms:
        record = arm_run_record(batch, summarize_arm(rows, arm), fingerprint, NO_PAYLOAD)
        # Indented, because these are read by people as often as by tools --
        # "what was this batch played under" is a question asked at a terminal.
        text = dump_json_str(encode_run_record(record), indent=RECORD_INDENT)
        _test_hooks.write_text_lines(run_record_sidecar(out_dir / arm), text.splitlines())
    return tuple(arms)


__all__ = [
    "DETERMINISM_DEVICE",
    "RECORD_INDENT",
    "batch_fingerprint",
    "read_batch_rows",
    "read_extractors",
    "write_arm_records",
]
