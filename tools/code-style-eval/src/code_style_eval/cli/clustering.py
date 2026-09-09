"""CLI: how much of this comparison's sample size is real.

Reports the design effect of a paired comparison at all three clustering
units, for one checker or for the combined outcome, optionally restricted to
the items both arms ran to a stop token.

WHY THIS IS NOT A FIELD ON THE COMPARISON REPORT. A design effect requires a
clustering unit, and choosing one is a claim about which files are correlated
that this corpus does not settle -- *k* is 14, 64 or 336 depending on the
answer. Folding a single unit into every comparison would make that choice
silently, once, for every figure. Reported separately and at three units, the
spread is the output.

WHY IT REPORTS RATHER THAN CORRECTS. It prints the design effect and the
effective *n*; it does not rewrite the p-value beside them. Deflating the
discordant table by the design effect and re-running the exact test is the
standard effective-sample-size shortcut, and it is an APPROXIMATION -- a
properly clustered McNemar (Durkalski's, and its relatives) is a different
statistic, not a rescaling of this one. Emitting a corrected p here would
publish that approximation as though it were the test. The numbers a reader
needs to do the shortcut deliberately, and to say they did, are all printed.

Usage:
    code-style-eval-clustering --baseline base.outcomes.jsonl \\
        --candidate cand.outcomes.jsonl --checker guards \\
        [--baseline-generation base.generation.jsonl \\
         --candidate-generation cand.generation.jsonl]
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core.clustering import clustered_paired_power
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.power_types import ClusteredPairedPower

from code_style_eval.cli import _test_hooks
from code_style_eval.contracts.generation import decode_generation_outcome
from code_style_eval.contracts.outcomes import CHECKERS, ItemOutcome, decode_item_outcome
from code_style_eval.core.clustering import (
    ClusteringUnit,
    both_finished,
    grouped_differences,
    paired_differences,
)

_BASELINE_FLAG = "--baseline"
_CANDIDATE_FLAG = "--candidate"
_CHECKER_FLAG = "--checker"
_BASELINE_GENERATION_FLAG = "--baseline-generation"
_CANDIDATE_GENERATION_FLAG = "--candidate-generation"

#: Flags every invocation must carry.
REQUIRED_FLAGS: tuple[str, ...] = (_BASELINE_FLAG, _CANDIDATE_FLAG, _CHECKER_FLAG)

#: Flags that restrict to the both-finished stratum. Both or neither: one
#: alone would restrict by one arm's truncations and read as a stratum
#: defined by both, which is a different denominator wearing the same name.
GENERATION_FLAGS: tuple[str, ...] = (_BASELINE_GENERATION_FLAG, _CANDIDATE_GENERATION_FLAG)

#: Every flag this CLI accepts, which is also the list an unknown one is
#: reported against.
KNOWN_FLAGS: tuple[str, ...] = REQUIRED_FLAGS + GENERATION_FLAGS


def read_outcomes(path: pathlib.Path) -> dict[str, ItemOutcome]:
    """Read one arm's outcomes, keyed by item id.

    Args:
        path: Outcome file, one JSON object per line.

    Returns:
        The outcomes, keyed by the item they scored.

    Raises:
        ValueError: If the file scores the same item twice.
    """
    outcomes: dict[str, ItemOutcome] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        outcome = decode_item_outcome(narrow_json_to_dict(load_json_str(line)))
        if outcome["item_id"] in outcomes:
            raise ValueError(f"{path} scores '{outcome['item_id']}' more than once")
        outcomes[outcome["item_id"]] = outcome
    return outcomes


def read_finished(path: pathlib.Path) -> tuple[str, ...]:
    """Read the item ids one arm ran to a stop token.

    Args:
        path: Generation manifest, one JSON object per line.

    Returns:
        The finished ids, in file order. Unfinished rows are dropped here
        rather than carried as False, because every caller wants the set.
    """
    return tuple(
        record["item_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [decode_generation_outcome(narrow_json_to_dict(load_json_str(line)))]
        if record["finished"]
    )


def build_records(
    differences: dict[str, int],
) -> tuple[ClusteredPairedPower, ...]:
    """Compute the design effect at every clustering unit.

    Args:
        differences: The per-item paired difference series.

    Returns:
        One record per :class:`~code_style_eval.core.clustering.ClusteringUnit`,
        in declaration order -- coarsest first, so a reader meets the
        largest design effect before the reassuring one.

    Raises:
        AppError: With ``POWER_CLUSTERS_INSUFFICIENT`` when a unit's grouping
            cannot support the estimate. Propagated rather than skipped: a
            table silently missing its coarsest row reads as though that row
            were fine.
    """
    return tuple(
        clustered_paired_power(grouped_differences(differences, unit), unit.value)
        for unit in ClusteringUnit
    )


def render(
    records: Sequence[ClusteredPairedPower], checker: str, differences: dict[str, int]
) -> tuple[str, ...]:
    """Format the table.

    Args:
        records: One record per clustering unit.
        checker: What was scored.
        differences: The series the records were computed from, for the
            counts printed above the table.

    Returns:
        The lines to emit.
    """
    baseline_only = sum(1 for value in differences.values() if value < 0)
    candidate_only = sum(1 for value in differences.values() if value > 0)
    lines = [
        f"checker                  {checker}",
        f"items                    {len(differences)}",
        f"discordant               {baseline_only + candidate_only} "
        f"(baseline-only {baseline_only}, candidate-only {candidate_only})",
        f"net                      {candidate_only - baseline_only:+d}",
        "",
        f"{'clustering unit':22s} {'k':>5s} {'m0':>7s} {'largest':>8s} "
        f"{'ICC':>8s} {'DE':>6s} {'eff n':>8s}",
    ]
    lines.extend(
        f"{record['unit']:22s} {record['clusters']:5d} "
        f"{record['average_cluster_size']:7.2f} {record['largest_cluster']:8d} "
        f"{record['intracluster_correlation']:+8.4f} {record['design_effect']:6.3f} "
        f"{record['effective_sample_size']:8.1f}"
        for record in records
    )
    lines.append("")
    lines.append(
        "A DE of 1.000 beside a negative ICC is the floor, not a corpus that "
        "landed there: clustering cannot raise a sample size."
    )
    return tuple(lines)


def parse_arguments(
    tokens: Sequence[str],
) -> tuple[pathlib.Path, pathlib.Path, str, tuple[pathlib.Path, pathlib.Path] | None]:
    """Parse the command line.

    Args:
        tokens: Arguments excluding the program name.

    Returns:
        A tuple of (baseline path, candidate path, checker, generation
        manifests or None when the run is not restricted to a stratum).

    Raises:
        ValueError: If a flag is unknown, missing or valueless, if the
            checker names nothing, or if exactly one generation manifest is
            given. The last is refused rather than honoured: restricting by
            one arm's truncations produces a stratum that is not the
            both-finished one and cannot be told apart from it downstream.
    """
    values: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in KNOWN_FLAGS:
            raise ValueError(f"unknown argument '{token}'; known flags: {KNOWN_FLAGS}")
        if index + 1 >= len(tokens):
            raise ValueError(f"{token} requires a value")
        values[token] = tokens[index + 1]
        index += 2
    for required in REQUIRED_FLAGS:
        if required not in values:
            raise ValueError(f"{required} is required")
    checker = values[_CHECKER_FLAG]
    if checker != "all" and checker not in CHECKERS:
        raise ValueError(f"{_CHECKER_FLAG} must be 'all' or one of {CHECKERS}; got {checker!r}")
    present = [flag for flag in GENERATION_FLAGS if flag in values]
    if len(present) == 1:
        raise ValueError(
            f"{present[0]} was given without its pair; the both-finished stratum "
            "is defined by BOTH arms finishing, and restricting by one arm's "
            "truncations silently reports a different denominator under that name"
        )
    generation = (
        (
            pathlib.Path(values[_BASELINE_GENERATION_FLAG]),
            pathlib.Path(values[_CANDIDATE_GENERATION_FLAG]),
        )
        if len(present) == 2
        else None
    )
    return (
        pathlib.Path(values[_BASELINE_FLAG]),
        pathlib.Path(values[_CANDIDATE_FLAG]),
        checker,
        generation,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Report the design effect at every clustering unit.

    Args:
        argv: Arguments excluding the program name. Defaults to the process
            arguments.

    Returns:
        Exit code 0 when the table was emitted.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    baseline_path, candidate_path, checker, generation = parse_arguments(tokens)

    differences = paired_differences(
        read_outcomes(baseline_path), read_outcomes(candidate_path), checker
    )
    label = checker
    if generation is not None:
        differences = both_finished(
            read_finished(generation[0]), read_finished(generation[1]), differences
        )
        label = f"{checker} (both-finished)"

    for line in render(build_records(differences), label, differences):
        _test_hooks.emit(line)
    return 0


def entrypoint() -> None:
    """Console-script entry point."""
    raise SystemExit(main())


__all__ = [
    "GENERATION_FLAGS",
    "KNOWN_FLAGS",
    "REQUIRED_FLAGS",
    "build_records",
    "entrypoint",
    "main",
    "parse_arguments",
    "read_finished",
    "read_outcomes",
    "render",
]


# Without this, `python -m code_style_eval.cli.clustering` imports the module
# and exits 0 having reported nothing, which looks exactly like a run that
# produced no output. The sibling CLIs carry the same guard for the same
# reason, and it cost a real scoring run on 2026-09-04.
if __name__ == "__main__":
    entrypoint()
