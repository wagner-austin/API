"""The RunRecord a benchmark writes beside its manifest."""

from __future__ import annotations

import pytest
from platform_core.run_record import decode_run_record, encode_run_record

from covenant_ml.benchmarking.reporting import summarize_gap
from covenant_ml.benchmarking.run_records import (
    BENCHMARK_EXPERIMENT,
    benchmark_observations,
    benchmark_run_record,
    manifest_digest,
)
from covenant_ml.benchmarking.types import BenchmarkManifest

from .test_reporting import make_manifest, make_result

#: Two arms at two seeds, with the arms deliberately unequal so a record that
#: transposed them would not still pass.
_RESULTS = [
    make_result("cleargbm", 1, 2.0, 30.0),
    make_result("cleargbm", 2, 2.4, 34.0),
    make_result("lightgbm", 1, 1.0, 15.0),
    make_result("lightgbm", 2, 1.4, 17.0),
]


def _manifest() -> BenchmarkManifest:
    """Build a finished two-arm manifest.

    Returns:
        The manifest.
    """
    return make_manifest(_RESULTS)


class TestDigestingTheManifest:
    """The digest ties a sidecar to the manifest it summarises."""

    def test_the_same_manifest_digests_the_same(self) -> None:
        """Otherwise no sidecar could be shown to belong to its manifest."""
        assert manifest_digest(_manifest()) == manifest_digest(_manifest())

    def test_a_changed_result_changes_the_digest(self) -> None:
        """A sidecar must not quietly describe a different run."""
        before = manifest_digest(_manifest())
        altered = make_manifest([*_RESULTS[:3], make_result("lightgbm", 2, 9.9, 17.0)])

        assert manifest_digest(altered) != before


class TestTheObservations:
    """The aggregates a later contrast would read."""

    def test_each_arm_is_named_separately(self) -> None:
        """Two models' means must not collide on one observation name."""
        names = {o["name"] for o in benchmark_observations(_manifest())}

        assert "cleargbm.mean_fit_s" in names
        assert "lightgbm.mean_fit_s" in names

    def test_the_arms_are_not_transposed(self) -> None:
        """cleargbm is the slower arm in this fixture, and must read so."""
        values = {o["name"]: o["value"] for o in benchmark_observations(_manifest())}

        assert values["cleargbm.mean_fit_s"] > values["lightgbm.mean_fit_s"]

    def test_the_seed_count_travels_with_the_means(self) -> None:
        """A mean over eleven seeds and over one are not the same evidence.

        Read from the manifest's DECLARED seed list, which its own type
        documents as the seeds measured in execution order, rather than
        counted from the result rows. The two can disagree -- this fixture is
        an example, declaring one seed while carrying rows for two -- and the
        declaration is the manifest's own statement of what it measured.
        """
        manifest = _manifest()
        values = {o["name"]: o["value"] for o in benchmark_observations(manifest)}

        assert values["seeds"] == float(len(manifest["seeds"]))

    def test_the_headline_ratio_is_carried(self) -> None:
        """normalized_ratio is the benchmark's actual claim.

        A reader comparing two runs should see whether it moved without
        recomputing it from the per-arm means.
        """
        manifest = _manifest()
        values = {o["name"]: o["value"] for o in benchmark_observations(manifest)}

        assert values["normalized_ratio"] == summarize_gap(manifest).normalized_ratio

    def test_no_observation_name_repeats(self) -> None:
        """run_record refuses a duplicate outright, because it makes the
        pairing ambiguous. Per-seed rows therefore stay in the manifest.
        """
        names = [o["name"] for o in benchmark_observations(_manifest())]

        assert len(names) == len(set(names))


class TestTheRecord:
    """The whole record, as it lands beside the manifest."""

    def test_the_record_names_the_experiment_and_the_run(self) -> None:
        """Experiment pairs runs; label distinguishes them within it."""
        record = benchmark_run_record(_manifest(), "taiwan-median")

        assert record["experiment"] == BENCHMARK_EXPERIMENT
        assert record["label"] == "taiwan-median"

    def test_the_manifests_own_fingerprint_is_carried_through(self) -> None:
        """Recapturing would describe the machine writing the record.

        For a benchmark whose headline is a fit time, the machine that
        produced the timings is the axis that moves it most, so the
        fingerprint must be the manifest's rather than this process's.
        """
        manifest = _manifest()

        record = benchmark_run_record(manifest, "taiwan-median")

        assert record["fingerprint"] == manifest["fingerprint"]

    def test_the_payload_digest_covers_the_manifest(self) -> None:
        """The record moves when the manifest does."""
        manifest = _manifest()

        record = benchmark_run_record(manifest, "taiwan-median")

        assert record["payload_digest"] == manifest_digest(manifest)

    def test_the_record_round_trips(self) -> None:
        """It must survive the codec every other consumer reads it through."""
        record = benchmark_run_record(_manifest(), "taiwan-median")

        assert decode_run_record(encode_run_record(record)) == record

    def test_an_unlabelled_run_is_refused(self) -> None:
        """A run with no label cannot be told apart from another."""
        with pytest.raises(ValueError, match="label"):
            _ = benchmark_run_record(_manifest(), "")

    def test_a_manifest_missing_an_arm_is_refused(self) -> None:
        """A gap summary needs both arms; inventing one would be a number."""
        one_arm = make_manifest([make_result("cleargbm", 1, 2.0, 30.0)])

        with pytest.raises(ValueError):
            _ = benchmark_run_record(one_arm, "taiwan-median")
