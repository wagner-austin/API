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
from code_style_eval.core.provenance import payload_digest

_RUNS = pathlib.Path(__file__).resolve().parent.parent / "runs"

#: Every committed comparison, with its denominator and mid-p pinned as
#: literals. The reproduction test below would still pass if a committed
#: artifact and the code changed together; these would not.
#:
#: FOUR OF THE SIX ARE PUBLISHED FIGURES, and the other two are pinned
#: anyway -- they are the same instrument's output and catch drift equally,
#: but the distinction is recorded so nobody reads this as six citations:
#:
#:   sweep-v1        the "all scored" row of the instrument-limits page --
#:                   3 v 3, the TIE form, which a version that doubles the
#:                   tail returns 1.0 for instead of 0.84375
#:   sweep-v3-nodeps the same generations scored before the corpus group
#:                   existed; the page reports this headline as byte-identical
#:   gen-v1          the 4 v 3 table published as 0.7265625, which
#:                   platform_core pinned its own tests to
#:   gen-v2          the 875-item aggregate of the results page
#:   sweep-v1-cap384 cited by the pages for its perplexity.json only
#:   sweep-v2-greedy not cited by either page
#:
#: The page's other two strata rows (n=90 and n=49) are SUBSETS of sweep-v1
#: rather than run directories, and are not reachable from here. They also
#: report 0.688, which is what makes them easy to mistake for the two runs
#: above; they are not the same measurement.
_COMMITTED_MID_P: tuple[tuple[str, int, float], ...] = (
    ("gen-v1", 226, 0.7265625),
    ("gen-v2", 875, 0.361594608053565),
    ("sweep-v1", 226, 0.84375),
    ("sweep-v1-cap384", 392, 0.6875),
    ("sweep-v2-greedy", 226, 0.6875),
    ("sweep-v3-nodeps", 226, 0.84375),
)


def _rebuild(directory: pathlib.Path, committed: ComparisonReport) -> ComparisonReport:
    """Recompute a run's comparison from the outcome files beside it.

    The digest is recomputed from those same two files rather than copied
    from the committed report, so an equality check against the committed
    report tests the IDENTITY as well as the figures: a comparison.json whose
    digest names bytes other than its neighbours' fails here.

    Args:
        directory: The run directory.
        committed: The committed report, for the arm names.

    Returns:
        The rebuilt report.
    """
    baseline = directory / "base.outcomes.jsonl"
    candidate = directory / "candidate.outcomes.jsonl"
    return build_report(
        read_outcomes(baseline),
        read_outcomes(candidate),
        baseline_arm=committed["baseline_arm"],
        candidate_arm=committed["candidate_arm"],
        payload_digest=payload_digest([baseline, candidate]),
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

        assert found == tuple(name for name, _, _ in _COMMITTED_MID_P)

    @pytest.mark.parametrize("run_name", [name for name, _, _ in _COMMITTED_MID_P])
    def test_rebuilding_from_the_outcomes_gives_the_committed_report(self, run_name: str) -> None:
        """Every field, not only the p-values.

        Args:
            run_name: The run directory to rebuild.
        """
        directory = _RUNS / run_name
        committed = _committed_report(directory)

        rebuilt = _rebuild(directory, committed)

        assert rebuilt == committed

    @pytest.mark.parametrize(("run_name", "shared_items", "mid_p"), _COMMITTED_MID_P)
    def test_the_mid_p_is_the_pinned_literal(
        self, run_name: str, shared_items: int, mid_p: float
    ) -> None:
        """Exact float equality, because the pages print exact digits.

        Rebuilding alone cannot catch an artifact and the code changing
        together; a literal written down here can.

        Args:
            run_name: The run directory.
            shared_items: The denominator the comparison reports.
            mid_p: The mid-p value the comparison reports.
        """
        directory = _RUNS / run_name
        committed = _committed_report(directory)

        rebuilt = _rebuild(directory, committed)

        assert rebuilt["shared_items"] == shared_items
        assert rebuilt["mid_p"] == mid_p

    @pytest.mark.parametrize("run_name", [name for name, _, _ in _COMMITTED_MID_P])
    def test_the_committed_digest_names_the_outcome_files_beside_it(self, run_name: str) -> None:
        """The property that makes a QUOTED figure traceable.

        Every other test here compares numbers against numbers, and numbers
        do not say what produced them: ``gen-v1`` and ``sweep-v1`` both report
        226 shared items, so "226 items" identifies neither. The digest is the
        only field that ties a report to bytes, and this asserts it names the
        outcome files actually sitting beside it rather than some other run's.

        Args:
            run_name: The run directory to check.
        """
        directory = _RUNS / run_name
        committed = _committed_report(directory)

        recomputed = payload_digest(
            [directory / "base.outcomes.jsonl", directory / "candidate.outcomes.jsonl"]
        )

        assert committed["payload_digest"] == recomputed

    def test_no_two_runs_share_a_digest(self) -> None:
        """The digests must actually discriminate, or they prove nothing.

        Six runs whose reports all carried the same digest would satisfy every
        other assertion in this file while identifying nothing. This is the
        check that the field does the job it was added for.
        """
        digests = {
            name: _committed_report(_RUNS / name)["payload_digest"]
            for name, _, _ in _COMMITTED_MID_P
        }

        assert len(set(digests.values())) == len(digests)
