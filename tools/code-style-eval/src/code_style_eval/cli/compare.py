"""CLI: compare two scored arms over the items they share.

Reads two outcome files written by ``code-style-eval`` and reports the paired
comparison. Kept separate from scoring so a sweep can be re-compared without
re-running any checker, and so a third arm can be added later by scoring it
alone and comparing it against the same baseline file.

Usage:
    code-style-eval-compare --baseline base.jsonl --candidate cand.jsonl \\
        --out report.json
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.run_record import encode_run_record, run_record_sidecar

from code_style_eval.cli import _test_hooks
from code_style_eval.contracts.outcomes import (
    ComparisonReport,
    ItemOutcome,
    decode_item_outcome,
    encode_comparison_report,
)
from code_style_eval.core.provenance import comparison_run_record
from code_style_eval.core.scoring import (
    exact_mcnemar_p,
    mid_p_mcnemar_p,
    net_improvement,
    paired_counts,
    pass_rate,
)

_BASELINE_FLAG = "--baseline"
_CANDIDATE_FLAG = "--candidate"
_OUT_FLAG = "--out"
_LABEL_FLAG = "--label"

_FLAGS: tuple[str, ...] = (_BASELINE_FLAG, _CANDIDATE_FLAG, _OUT_FLAG, _LABEL_FLAG)


def read_outcomes(path: pathlib.Path) -> dict[str, ItemOutcome]:
    """Read one arm's outcomes, keyed by item id.

    Args:
        path: Outcome file, one JSON object per line.

    Returns:
        The outcomes, keyed by the item they scored.

    Raises:
        ValueError: If the file scores the same item twice. Two rows for one
            item would let whichever came last decide the arm's verdict on
            it, silently.
    """
    outcomes: dict[str, ItemOutcome] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        outcome = decode_item_outcome(narrow_json_to_dict(load_json_str(line)))
        item_id = outcome["item_id"]
        if item_id in outcomes:
            raise ValueError(f"{path} scores '{item_id}' more than once")
        outcomes[item_id] = outcome
    return outcomes


def arm_name(outcomes: dict[str, ItemOutcome], path: pathlib.Path) -> str:
    """Read the single arm name an outcome file records.

    Args:
        outcomes: The file's outcomes.
        path: The file, for the error message.

    Returns:
        The arm name, or the empty string when the file is empty.

    Raises:
        ValueError: If the file mixes arms. Comparing a file that already
            mixes two models against another would produce a table whose
            rows came from three.
    """
    names = {outcome["arm"] for outcome in outcomes.values()}
    if len(names) > 1:
        raise ValueError(f"{path} mixes arms: {sorted(names)}")
    return next(iter(names)) if names else ""


def build_report(
    baseline: dict[str, ItemOutcome],
    candidate: dict[str, ItemOutcome],
    *,
    baseline_arm: str,
    candidate_arm: str,
) -> ComparisonReport:
    """Compute every figure the comparison reports.

    Rates are taken over the SHARED items rather than over each file's own
    length, so the two rates and the 2x2 table share one denominator. An arm
    scored on more items would otherwise report a rate the table cannot
    explain.

    Args:
        baseline: Baseline outcomes by item id.
        candidate: Candidate outcomes by item id.
        baseline_arm: Baseline arm name.
        candidate_arm: Candidate arm name.

    Returns:
        The report.
    """
    shared = sorted(set(baseline) & set(candidate))
    counts = paired_counts(baseline, candidate)
    return ComparisonReport(
        baseline_arm=baseline_arm,
        candidate_arm=candidate_arm,
        shared_items=len(shared),
        baseline_pass_rate=pass_rate([baseline[item] for item in shared]),
        candidate_pass_rate=pass_rate([candidate[item] for item in shared]),
        counts=counts,
        net_improvement=net_improvement(counts),
        mid_p=mid_p_mcnemar_p(counts),
        exact_p=exact_mcnemar_p(counts),
    )


def render(report: ComparisonReport) -> list[str]:
    """Render the report as lines for a human reader.

    Args:
        report: The computed report.

    Returns:
        The lines, without trailing newlines.
    """
    counts = report["counts"]
    return [
        f"items scored by both      {report['shared_items']}",
        f"{report['baseline_arm']} pass rate".ljust(26) + f"{report['baseline_pass_rate']:.3f}",
        f"{report['candidate_arm']} pass rate".ljust(26) + f"{report['candidate_pass_rate']:.3f}",
        f"both passed               {counts['both_passed']}",
        f"baseline only             {counts['baseline_only']}",
        f"candidate only            {counts['candidate_only']}",
        f"neither                   {counts['neither']}",
        f"net items fixed           {report['net_improvement']:+d}",
        f"mid-p                     {report['mid_p']:.6f}",
        f"exact conditional p       {report['exact_p']:.6f}",
    ]


def parse_arguments(
    tokens: Sequence[str],
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, str]:
    """Parse the command line.

    Args:
        tokens: Arguments excluding the program name.

    Returns:
        A tuple of (baseline path, candidate path, output path, label).

    Raises:
        ValueError: If a flag is unknown, missing, or has no value.
    """
    values: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in _FLAGS:
            raise ValueError(f"unknown argument '{token}'; known flags: {_FLAGS}")
        if index + 1 >= len(tokens):
            raise ValueError(f"{token} requires a value")
        values[token] = tokens[index + 1]
        index += 2
    for required in _FLAGS:
        if required not in values:
            raise ValueError(f"{required} is required")
    return (
        pathlib.Path(values[_BASELINE_FLAG]),
        pathlib.Path(values[_CANDIDATE_FLAG]),
        pathlib.Path(values[_OUT_FLAG]),
        values[_LABEL_FLAG],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Compare two arms and write the report.

    Args:
        argv: Arguments excluding the program name. Defaults to the process
            arguments.

    Returns:
        Exit code 0 when the report was written.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    baseline_path, candidate_path, out_path, label = parse_arguments(tokens)

    baseline = read_outcomes(baseline_path)
    candidate = read_outcomes(candidate_path)
    report = build_report(
        baseline,
        candidate,
        baseline_arm=arm_name(baseline, baseline_path),
        candidate_arm=arm_name(candidate, candidate_path),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        dump_json_str(encode_comparison_report(report), compact=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # The record is written beside the comparison, never instead of it, and
    # covers the inputs the comparison was computed from. Without it the
    # rates and p-values above name no models, no decoding settings and no
    # machine, which makes them unciteable the moment the run is over.
    sidecar = run_record_sidecar(out_path)
    sidecar.write_text(
        dump_json_str(
            encode_run_record(
                comparison_run_record(
                    report,
                    label,
                    [baseline_path, candidate_path],
                    _test_hooks.record_distributions,
                )
            ),
            compact=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for line in render(report):
        _test_hooks.emit(line)
    _test_hooks.emit(f"run record               {sidecar}")
    return 0


def entrypoint() -> None:
    """Console-script entry point."""
    raise SystemExit(main())


__all__ = [
    "arm_name",
    "build_report",
    "entrypoint",
    "main",
    "parse_arguments",
    "read_outcomes",
    "render",
]


# Without this, `python -m code_style_eval.cli.compare` imports the module,
# defines these functions and exits 0 -- having scored nothing while
# reporting success. That is worse than a crash: the console script works, so
# the two invocation forms disagree, and the silent one looks exactly like a
# run that legitimately produced no output.
#
# It cost a real scoring run on 2026-09-04, against 226 generated files that
# had just taken an A30 thirty-three minutes to produce. The sibling packages
# `model_trainer` and `hpc3` each carry this guard and a test that enforces
# it; this package carried neither.
if __name__ == "__main__":
    entrypoint()
