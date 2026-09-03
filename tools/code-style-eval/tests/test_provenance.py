"""What a comparison records about the run that produced it."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.determinism_record import UNPINNED_STACK

from code_style_eval.contracts.outcomes import ComparisonReport, PairedCounts
from code_style_eval.core.provenance import (
    EXPERIMENT,
    SCORING_DISTRIBUTIONS,
    comparison_observations,
    comparison_run_record,
    payload_digest,
    scoring_fingerprint,
)


def _report() -> ComparisonReport:
    """Build a comparison whose every figure is distinct.

    Distinct values matter: a record that mapped two figures onto one
    observation name would still round-trip if the values were equal.

    Returns:
        The report.
    """
    return ComparisonReport(
        baseline_arm="base",
        candidate_arm="candidate",
        shared_items=10,
        baseline_pass_rate=0.3,
        candidate_pass_rate=0.4,
        counts=PairedCounts(both_passed=2, baseline_only=1, candidate_only=3, neither=4),
        net_improvement=2,
        mid_p=0.25,
        exact_p=0.5,
    )


class TestDigestingThePayload:
    """The digest is what ties a comparison to the bytes behind it."""

    def _file(self, path: pathlib.Path, body: str) -> pathlib.Path:
        """Write a file.

        Args:
            path: Where to write.
            body: Contents.

        Returns:
            The path.
        """
        path.write_text(body, encoding="utf-8")
        return path

    def test_the_same_files_digest_the_same(self, tmp_path: pathlib.Path) -> None:
        """Otherwise no two runs could ever be shown to agree."""
        first = self._file(tmp_path / "a.jsonl", "one")
        second = self._file(tmp_path / "b.jsonl", "two")

        assert payload_digest([first, second]) == payload_digest([first, second])

    def test_the_order_given_does_not_change_the_digest(self, tmp_path: pathlib.Path) -> None:
        """Sorted before hashing, so an argument order cannot alter it."""
        first = self._file(tmp_path / "a.jsonl", "one")
        second = self._file(tmp_path / "b.jsonl", "two")

        assert payload_digest([first, second]) == payload_digest([second, first])

    def test_changed_content_changes_the_digest(self, tmp_path: pathlib.Path) -> None:
        """The property the digest exists for."""
        path = self._file(tmp_path / "a.jsonl", "one")
        before = payload_digest([path])
        path.write_text("two", encoding="utf-8")

        assert payload_digest([path]) != before

    def test_swapping_two_files_names_changes_the_digest(self, tmp_path: pathlib.Path) -> None:
        """Names are hashed with the bytes, so a rename is a difference.

        Without the name, an outcome file for the baseline and one for the
        candidate could be exchanged and the digest would not notice, which
        is exactly the mix-up that would invert a comparison.
        """
        first = self._file(tmp_path / "base.jsonl", "one")
        second = self._file(tmp_path / "cand.jsonl", "two")
        before = payload_digest([first, second])

        first.write_text("two", encoding="utf-8")
        second.write_text("one", encoding="utf-8")

        assert payload_digest([first, second]) != before

    def test_covering_nothing_is_refused(self) -> None:
        """A digest over no files is a constant every run would share."""
        with pytest.raises(ValueError, match="at least one file"):
            _ = payload_digest([])


class TestTheFingerprint:
    """Three axes are unknown here, and say so rather than guess."""

    def test_the_absent_axes_are_empty_not_invented(self) -> None:
        """Scoring is CPU work outside any image; claiming a GPU would lie."""
        fingerprint = scoring_fingerprint()

        assert fingerprint["image_digest"] == ""
        assert fingerprint["gpu_model"] == ""
        assert fingerprint["driver_version"] == ""

    def test_the_determinism_posture_is_recorded_as_unpinned(self) -> None:
        """Explicitly unpinned beats a missing key, which reads as forgotten.

        Asserted against the constant rather than its current spelling, so a
        rename upstream cannot leave this test passing against a literal the
        production code no longer emits.
        """
        assert scoring_fingerprint()["determinism"]["stack"] == UNPINNED_STACK

    def test_the_host_is_captured(self) -> None:
        """The axis that separates two runs when the GPU axes are all empty."""
        host = scoring_fingerprint()["host"]

        assert host["platform"] != ""
        assert host["logical_cores"] > 0

    def test_every_checker_version_is_captured(self) -> None:
        """A ruff release that adds a rule moves the rate on its own."""
        recorded = {package["name"] for package in scoring_fingerprint()["packages"]}

        assert recorded == set(SCORING_DISTRIBUTIONS)


class TestTheObservations:
    """Every figure the comparison concluded is named."""

    def test_every_reported_figure_is_carried(self) -> None:
        """A figure on the page but not in the record cannot be compared."""
        names = {observation["name"] for observation in comparison_observations(_report())}

        assert names == {
            "shared_items",
            "baseline_pass_rate",
            "candidate_pass_rate",
            "both_passed",
            "baseline_only",
            "candidate_only",
            "neither",
            "net_improvement",
            "mid_p",
            "exact_p",
        }

    def test_the_counts_travel_beside_the_rates(self) -> None:
        """Three of three and thirty of thirty are both 1.0.

        Only one is evidence, so a record carrying the rate without the
        denominator cannot be read.
        """
        values = {o["name"]: o["value"] for o in comparison_observations(_report())}

        assert values["shared_items"] == 10.0
        assert values["baseline_pass_rate"] == 0.3

    def test_the_figures_are_not_transposed(self) -> None:
        """Each name carries its own figure, not a neighbour's."""
        values = {o["name"]: o["value"] for o in comparison_observations(_report())}

        assert values["baseline_only"] == 1.0
        assert values["candidate_only"] == 3.0
        assert values["mid_p"] == 0.25
        assert values["exact_p"] == 0.5


class TestTheRecord:
    """The whole record, as it lands beside the comparison."""

    def test_the_record_names_the_experiment_and_the_run(self, tmp_path: pathlib.Path) -> None:
        """Experiment pairs runs; label distinguishes them within it.

        Args:
            tmp_path: Directory for the covered file.
        """
        covered = tmp_path / "base.jsonl"
        covered.write_text("x", encoding="utf-8")

        record = comparison_run_record(_report(), "sweep-v3", [covered])

        assert record["experiment"] == EXPERIMENT
        assert record["label"] == "sweep-v3"

    def test_observations_are_sorted_by_name(self, tmp_path: pathlib.Path) -> None:
        """Canonical order, so two records list them the same way.

        Args:
            tmp_path: Directory for the covered file.
        """
        covered = tmp_path / "base.jsonl"
        covered.write_text("x", encoding="utf-8")

        record = comparison_run_record(_report(), "s", [covered])
        names = [observation["name"] for observation in record["observations"]]

        assert names == sorted(names)

    def test_an_unlabelled_run_is_refused(self, tmp_path: pathlib.Path) -> None:
        """A run with no label cannot be told apart from another.

        Args:
            tmp_path: Directory for the covered file.
        """
        covered = tmp_path / "base.jsonl"
        covered.write_text("x", encoding="utf-8")

        with pytest.raises(ValueError, match="label"):
            _ = comparison_run_record(_report(), "", [covered])
