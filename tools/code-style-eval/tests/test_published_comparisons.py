"""Every committed comparison must still be reproducible from its outcomes.

WHY THIS EXISTS. Two wiki pages report figures this package computed, and
until 2026-09-09 this package computed them with its own copy of the McNemar
arithmetic. That copy is gone -- :mod:`platform_core.power_distributions`
owns both variants now -- and the migration was accepted on the basis that no
published number moved. This file is what keeps that true tomorrow.

WHY IT READS ``runs/`` RATHER THAN A FIXTURE. A fixture written beside the
implementation agrees with the implementation by construction, which is the
defect this monorepo has now shipped twice: certification records rendered
from the file's own name, and four guard cases that passed while nothing
could fire. The committed ``comparison.json`` files were written by an
EARLIER version of this code, on a cluster, from generated files nobody can
regenerate cheaply. Rebuilding them from their own ``*.outcomes.jsonl`` and
demanding byte equality is therefore a comparison against a past
implementation rather than against the present one's assumptions.

WHAT IT WOULD CATCH. Any change to the projection, to either p-value, to the
2x2 tabulation, or to the shared-item rule that moves a figure a published
page cites. It cannot catch a change that alters the outcomes files
themselves; those are evidence, and their integrity is git's job.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from code_style_eval.cli.compare import build_report, read_outcomes
from code_style_eval.contracts.outcomes import ComparisonReport, decode_comparison_report

_RUNS = pathlib.Path(__file__).resolve().parent.parent / "runs"

#: The mid-p values two wiki pages print, pinned as literals beside the run
#: that produced each. The reproduction test below would still pass if a
#: committed artifact and the code changed together; these would not.
#:
#: ``sweep-v1`` is the "all scored" row of the instrument-limits page (3 v 3,
#: the TIE form, which a version that doubles the tail returns 1.0 for), and
#: ``sweep-v1-cap384`` and ``sweep-v2-greedy`` are its two 3 v 2 rows.
#: ``gen-v1`` is the 4 v 3 table published as 0.7265625.
_PUBLISHED_MID_P: tuple[tuple[str, int, float], ...] = (
    ("gen-v1", 226, 0.7265625),
    ("gen-v2", 875, 0.361594608053565),
    ("sweep-v1", 226, 0.84375),
    ("sweep-v1-cap384", 392, 0.6875),
    ("sweep-v2-greedy", 226, 0.6875),
    ("sweep-v3-nodeps", 226, 0.84375),
)


def _committed_report(directory: pathlib.Path) -> ComparisonReport:
    """Decode the comparison a run directory committed.

    Decoding rather than reading raw JSON is deliberate: the decoder rejects
    a report whose 2x2 table disagrees with its own denominator, so a corrupt
    artifact fails here as a decode error rather than surviving into a
    comparison against an equally corrupt rebuild.

    Args:
        directory: The run directory.

    Returns:
        The committed report.
    """
    raw = (directory / "comparison.json").read_text(encoding="utf-8")
    return decode_comparison_report(narrow_json_to_dict(load_json_str(raw)))


def _committed_runs() -> tuple[pathlib.Path, ...]:
    """Find every run directory holding a comparison and both outcome files.

    Returns:
        The run directories, sorted by name. A directory missing any of the
        three is not a partial run to repair here -- several hold generation
        output only -- so it is skipped rather than failed.
    """
    return tuple(
        sorted(
            directory
            for directory in _RUNS.iterdir()
            if (directory / "comparison.json").is_file()
            and (directory / "base.outcomes.jsonl").is_file()
            and (directory / "candidate.outcomes.jsonl").is_file()
        )
    )


class TestEveryCommittedComparisonReproduces:
    """The drift detector for every figure this package has published."""

    def test_the_run_directories_are_all_present(self) -> None:
        """Fail loudly if the artifacts moved, rather than testing nothing.

        A parametrised sweep over a directory listing reports a cheerful
        green when the listing is empty, which is the shape that lets a
        deleted corpus read as a passing suite.
        """
        found = tuple(directory.name for directory in _committed_runs())

        assert found == tuple(name for name, _, _ in _PUBLISHED_MID_P)

    @pytest.mark.parametrize("run_name", [name for name, _, _ in _PUBLISHED_MID_P])
    def test_rebuilding_from_the_outcomes_gives_the_committed_report(self, run_name: str) -> None:
        """Every field, not only the p-values.

        Args:
            run_name: The run directory to rebuild.
        """
        directory = _RUNS / run_name
        committed = _committed_report(directory)

        rebuilt = build_report(
            read_outcomes(directory / "base.outcomes.jsonl"),
            read_outcomes(directory / "candidate.outcomes.jsonl"),
            baseline_arm=committed["baseline_arm"],
            candidate_arm=committed["candidate_arm"],
        )

        assert rebuilt == committed

    @pytest.mark.parametrize(("run_name", "shared_items", "mid_p"), _PUBLISHED_MID_P)
    def test_the_published_mid_p_is_the_literal_a_wiki_page_prints(
        self, run_name: str, shared_items: int, mid_p: float
    ) -> None:
        """Exact float equality, because the pages print exact digits.

        Args:
            run_name: The run directory.
            shared_items: The denominator the page reports.
            mid_p: The mid-p value the page reports.
        """
        directory = _RUNS / run_name
        committed = _committed_report(directory)

        rebuilt = build_report(
            read_outcomes(directory / "base.outcomes.jsonl"),
            read_outcomes(directory / "candidate.outcomes.jsonl"),
            baseline_arm=committed["baseline_arm"],
            candidate_arm=committed["candidate_arm"],
        )

        assert rebuilt["shared_items"] == shared_items
        assert rebuilt["mid_p"] == mid_p
