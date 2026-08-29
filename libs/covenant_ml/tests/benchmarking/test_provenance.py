"""Tests for the one way a ClearGBM benchmark says what it ran on.

The measured gap these guard is in the module docstring: 37 of 41 published
manifests carried no environment block, and five of six entry points neither
pinned the thread count nor built a fingerprint. The assertions below are
about the SHARED builder, because the fix was to make there be one.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.comparability import IMAGE_DIGEST_ENV_VAR, NO_VALUE, find_differences
from platform_core.determinism_env import SINGLE_THREAD
from platform_core.determinism_record import UNPINNED_STACK, determinism_record
from platform_core.environment_record import PackageVersion
from platform_core.run_record import (
    NO_PAYLOAD,
    compare_run_records,
    decode_run_record,
    encode_run_record,
)
from platform_core.testing import (
    SAMPLE_HOST,
    FakeHostProbe,
    FakeVersionReader,
    sample_run_fingerprint,
)

from covenant_ml.benchmarking import _test_hooks
from covenant_ml.benchmarking.provenance import (
    BENCHMARK_DISTRIBUTIONS,
    BENCHMARK_EXPERIMENT,
    benchmark_fingerprint,
    benchmark_label,
    benchmark_run_record,
)
from covenant_ml.benchmarking.types import (
    MANIFEST_SCHEMA_VERSION,
    BenchmarkManifest,
    BenchmarkModelName,
    SeedResult,
)

_DIGEST = "sha256:" + "ab" * 32

#: A launcher that exported nothing, which is a run outside any image --
#: the ordinary case for this project. Typed rather than written inline:
#: a bare ``{}.get`` is an overloaded ``dict[Any, Any]`` method and does
#: not satisfy the reader signature.
_NO_ENV: dict[str, str] = {}

#: Versions for every distribution the axis names, so the fake never has to
#: answer a question the production reader would refuse.
_VERSIONS = dict.fromkeys(BENCHMARK_DISTRIBUTIONS, "1.0.0")

_PINNED = determinism_record("cpu", {"OMP_NUM_THREADS": SINGLE_THREAD})


#: A fingerprint an entry point already built, before any numeric library
#: loaded. The record carries it rather than rebuilding one.
_FINGERPRINT = sample_run_fingerprint(
    image_digest=_DIGEST,
    gpu_model=NO_VALUE,
    driver_version=NO_VALUE,
    determinism=determinism_record("cpu", {"OMP_NUM_THREADS": SINGLE_THREAD}),
)


def _result(model: BenchmarkModelName, canonical_s: float, mean_leaves: float) -> SeedResult:
    """Build one arm's outcome at one seed.

    Args:
        model: Which arm.
        canonical_s: Its canonical fit time.
        mean_leaves: Mean leaves per tree.

    Returns:
        The record.
    """
    return {
        "model": model,
        "seed": 42,
        "position": 0,
        "timing": {
            "canonical_s": canonical_s,
            "min_s": canonical_s - 0.1,
            "median_s": canonical_s,
            "mean_s": canonical_s,
            "max_s": canonical_s + 0.1,
            "samples_s": [canonical_s],
        },
        "quality": {
            "auc_roc": 0.68,
            "auc_pr": 0.14,
            "log_loss": 0.23,
            "brier": 0.06,
            "mean_pred": 0.065,
            "positive_rate": 0.066,
        },
        "mean_leaves": mean_leaves,
    }


def _two_arms(canonical_s: float = 8.0) -> list[SeedResult]:
    """Build the two arms this benchmark compares.

    Args:
        canonical_s: ClearGBM's fit time; LightGBM's is held at 4.0 so the
            ratios are stable across calls.

    Returns:
        Both records.
    """
    return [
        _result("cleargbm", canonical_s, 31.0),
        _result("lightgbm", 4.0, 31.0),
    ]


def make_manifest(results: list[SeedResult]) -> BenchmarkManifest:
    """Wrap records in a manifest.

    Args:
        results: Records to include.

    Returns:
        The manifest.
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "estimator": "median",
        "config": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "max_bins": 64,
            "min_data_in_leaf": 20,
            "num_leaves": 31,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "repeats": 5,
            "warmups": 2,
        },
        "dataset": {"sha256": "c" * 64, "n_rows": 78682, "n_features": 18},
        "seeds": [42],
        "results": results,
        "fingerprint": _FINGERPRINT,
    }


def _sample_probe() -> FakeHostProbe:
    """Build a probe reporting the stated machine.

    Returns:
        A probe reporting :data:`platform_core.testing.SAMPLE_HOST`.
    """
    return FakeHostProbe(
        platform=SAMPLE_HOST["platform"],
        machine=SAMPLE_HOST["machine"],
        logical_cores=SAMPLE_HOST["logical_cores"],
    )


def _stated_environment() -> Generator[None, None, None]:
    """State the machine and the versions for one test, then restore.

    A benchmark fingerprint test that read the real box would assert a
    different platform string on every developer's machine.

    Yields:
        None, for the duration of one test.
    """
    probe = _test_hooks.host_probe
    version = _test_hooks.installed_version
    _test_hooks.host_probe = _sample_probe
    _test_hooks.installed_version = FakeVersionReader(_VERSIONS)
    try:
        yield
    finally:
        _test_hooks.host_probe = probe
        _test_hooks.installed_version = version


stated_environment = pytest.fixture(_stated_environment)


class TestTheDistributionsItRecords:
    def test_it_names_the_subject_and_both_arms_it_is_measured_against(self) -> None:
        # A bump in lightgbm or xgboost moves the COMPARISON without moving
        # ClearGBM, which is exactly the difference a reader would otherwise
        # attribute to the subject.
        assert BENCHMARK_DISTRIBUTIONS == (
            "cleargbm",
            "cleargbm_rs",
            "lightgbm",
            "numpy",
            "scikit-learn",
            "xgboost",
        )


class TestTheFingerprint:
    @pytest.mark.usefixtures("stated_environment")
    def test_it_carries_the_machine(self) -> None:
        fingerprint = benchmark_fingerprint(_PINNED, {IMAGE_DIGEST_ENV_VAR: _DIGEST}.get)

        assert fingerprint["host"] == SAMPLE_HOST

    @pytest.mark.usefixtures("stated_environment")
    def test_it_carries_every_named_library(self) -> None:
        fingerprint = benchmark_fingerprint(_PINNED, {IMAGE_DIGEST_ENV_VAR: _DIGEST}.get)

        assert [p["name"] for p in fingerprint["packages"]] == list(BENCHMARK_DISTRIBUTIONS)

    @pytest.mark.usefixtures("stated_environment")
    def test_it_carries_the_image_the_launcher_named(self) -> None:
        fingerprint = benchmark_fingerprint(_PINNED, {IMAGE_DIGEST_ENV_VAR: _DIGEST}.get)

        assert fingerprint["image_digest"] == _DIGEST

    @pytest.mark.usefixtures("stated_environment")
    def test_a_benchmark_outside_any_image_records_no_card_and_no_digest(self) -> None:
        # The ordinary case for this project: run from a directory
        # environment on a workstation. Empty differs from every real value
        # rather than matching all of them.
        fingerprint = benchmark_fingerprint(_PINNED, _NO_ENV.get)

        assert fingerprint["image_digest"] == NO_VALUE
        assert fingerprint["gpu_model"] == NO_VALUE
        assert fingerprint["driver_version"] == NO_VALUE

    @pytest.mark.usefixtures("stated_environment")
    def test_it_carries_the_posture_the_entry_point_pinned(self) -> None:
        # Passed in rather than pinned here: pinning must precede the load of
        # any native numeric library, which is above this module's import.
        fingerprint = benchmark_fingerprint(_PINNED, _NO_ENV.get)

        assert fingerprint["determinism"] == _PINNED

    @pytest.mark.usefixtures("stated_environment")
    def test_a_run_that_pinned_nothing_says_so(self) -> None:
        unpinned = determinism_record(UNPINNED_STACK, {})

        fingerprint = benchmark_fingerprint(unpinned, _NO_ENV.get)

        assert fingerprint["determinism"] == unpinned


class TestWhatItLetsAReaderTellApart:
    @pytest.mark.usefixtures("stated_environment")
    def test_two_benchmarks_on_one_box_are_identical(self) -> None:
        left = benchmark_fingerprint(_PINNED, _NO_ENV.get)
        right = benchmark_fingerprint(_PINNED, _NO_ENV.get)

        assert find_differences(left, right) == ()

    def test_two_benchmarks_on_two_boxes_differ(self) -> None:
        # THE GAP THIS CLOSES. Before the host axis, a run on a 24-core
        # cluster node and one on an 8-core workstation produced identical
        # fingerprints, and the 37 manifests without an environment block
        # could not tell a reader which had happened.
        probe = _test_hooks.host_probe
        version = _test_hooks.installed_version
        _test_hooks.installed_version = FakeVersionReader(_VERSIONS)
        try:
            _test_hooks.host_probe = _sample_probe
            workstation = benchmark_fingerprint(_PINNED, _NO_ENV.get)
            _test_hooks.host_probe = lambda: FakeHostProbe(
                platform=SAMPLE_HOST["platform"],
                machine=SAMPLE_HOST["machine"],
                logical_cores=24,
            )
            node = benchmark_fingerprint(_PINNED, _NO_ENV.get)
        finally:
            _test_hooks.host_probe = probe
            _test_hooks.installed_version = version

        assert [d["axis"] for d in find_differences(workstation, node)] == ["host"]

    def test_a_library_bump_on_one_box_differs(self) -> None:
        probe = _test_hooks.host_probe
        version = _test_hooks.installed_version
        _test_hooks.host_probe = _sample_probe
        try:
            _test_hooks.installed_version = FakeVersionReader(_VERSIONS)
            before = benchmark_fingerprint(_PINNED, _NO_ENV.get)
            _test_hooks.installed_version = FakeVersionReader({**_VERSIONS, "lightgbm": "4.7.0"})
            after = benchmark_fingerprint(_PINNED, _NO_ENV.get)
        finally:
            _test_hooks.host_probe = probe
            _test_hooks.installed_version = version

        assert [d["axis"] for d in find_differences(before, after)] == ["packages"]

    def test_a_library_the_environment_lacks_is_refused_not_recorded(self) -> None:
        # Rather than written as "unknown", which is a non-empty string and
        # would compare EQUAL between two environments that each failed to
        # find it.
        probe = _test_hooks.host_probe
        version = _test_hooks.installed_version
        _test_hooks.host_probe = _sample_probe
        _test_hooks.installed_version = FakeVersionReader({"numpy": "2.3.5"})
        try:
            with pytest.raises(KeyError):
                benchmark_fingerprint(_PINNED, _NO_ENV.get)
        finally:
            _test_hooks.host_probe = probe
            _test_hooks.installed_version = version


class TestTheProductionHooks:
    def test_the_default_probe_reads_a_real_machine(self) -> None:
        record = _test_hooks.host_probe()

        assert record.logical_cores() >= 1
        assert record.machine() != ""

    def test_the_default_version_reader_reads_installed_metadata(self) -> None:
        assert _test_hooks.installed_version("pytest") != ""

    def test_the_hook_surface_is_the_declared_one(self) -> None:
        # Doubles as a shape-drift guard: a hook added without a test is a
        # seam nobody is exercising.
        assert sorted(_test_hooks.__all__) == [
            "host_probe",
            "installed_version",
            "monotonic_clock",
            "power_throttling_opt_out",
        ]


class TestThePackageEntryShape:
    @pytest.mark.usefixtures("stated_environment")
    def test_each_entry_names_a_distribution_and_its_version(self) -> None:
        fingerprint = benchmark_fingerprint(_PINNED, _NO_ENV.get)

        assert fingerprint["packages"][0] == PackageVersion(name="cleargbm", version="1.0.0")


class TestTheRecordEveryExperimentEmits:
    """The gap the fingerprint alone did not close.

    A manifest says everything about one benchmark and nothing in a form
    another experiment's records can be read beside. This project's headline
    is a comparison, and until now that comparison lived only in a shape only
    this project could parse -- so `platform_core.run_record`, which exists to
    tell whether two numbers may be subtracted, had nothing to work with.
    """

    def test_the_experiment_is_stable_across_invocations(self) -> None:
        """Two runs on different data are still the same experiment; that is
        what makes the dataset part of the label instead."""
        first = benchmark_run_record(make_manifest(_two_arms()))
        second = benchmark_run_record(make_manifest(_two_arms(canonical_s=9.0)))
        assert first["experiment"] == second["experiment"] == BENCHMARK_EXPERIMENT

    def test_the_label_separates_runs_that_are_not_the_same_run(self) -> None:
        manifest = make_manifest(_two_arms())
        other = make_manifest(_two_arms())
        other["seeds"] = [42, 43]
        assert benchmark_label(manifest) != benchmark_label(other)

    def test_the_label_names_the_dataset_by_its_bytes(self) -> None:
        """A filename is not the bytes; two datasets named alike are two
        datasets."""
        assert benchmark_label(make_manifest(_two_arms())).startswith("cccccccccccc-")

    def test_it_carries_the_fingerprint_the_entry_point_built(self) -> None:
        """Not rebuilt here: building one reads installed metadata, and this
        runs long after the thread pin."""
        record = benchmark_run_record(make_manifest(_two_arms()))
        assert record["fingerprint"] == _FINGERPRINT

    def test_the_comparison_itself_is_an_observation(self) -> None:
        """The headline is a ratio, so a record that omitted it would carry
        every input to the claim and not the claim."""
        record = benchmark_run_record(make_manifest(_two_arms()))
        names = {o["name"] for o in record["observations"]}
        assert {"raw_ratio", "leaf_ratio", "normalized_ratio"} <= names

    def test_every_arm_gets_its_own_prefixed_numbers(self) -> None:
        """`mean_fit_s` unprefixed would pair ClearGBM's number with
        LightGBM's in any contrast reading two records side by side."""
        record = benchmark_run_record(make_manifest(_two_arms()))
        names = {o["name"] for o in record["observations"]}
        assert "cleargbm.mean_fit_s" in names
        assert "lightgbm.mean_fit_s" in names
        assert "mean_fit_s" not in names

    def test_observations_are_in_canonical_order(self) -> None:
        """So two records list them one way and a reader can zip them."""
        observations = benchmark_run_record(make_manifest(_two_arms()))["observations"]
        assert [o["name"] for o in observations] == sorted(o["name"] for o in observations)

    def test_the_per_seed_detail_is_digested_rather_than_carried(self) -> None:
        """2 arms x N seeds of timing detail is the payload; no cross-
        experiment layer should read it to tell two runs apart."""
        record = benchmark_run_record(make_manifest(_two_arms()))
        assert len(record["payload_digest"]) == 64
        assert record["payload_digest"] != NO_PAYLOAD

    def test_two_runs_with_different_seed_results_digest_differently(self) -> None:
        left = benchmark_run_record(make_manifest(_two_arms()))
        right = benchmark_run_record(make_manifest(_two_arms(canonical_s=9.0)))
        assert left["payload_digest"] != right["payload_digest"]

    def test_the_same_results_digest_identically(self) -> None:
        """Which is what makes bit-identity checkable without reading them."""
        left = benchmark_run_record(make_manifest(_two_arms()))
        right = benchmark_run_record(make_manifest(_two_arms()))
        assert left["payload_digest"] == right["payload_digest"]

    def test_it_round_trips_through_the_shared_codec(self) -> None:
        """The point of emitting this shape is that a layer which knows
        nothing about boosting can read it."""
        record = benchmark_run_record(make_manifest(_two_arms()))
        assert decode_run_record(encode_run_record(record)) == record

    def test_two_records_from_one_configuration_compare(self) -> None:
        """End to end: the shared comparator subtracts them, which is the
        thing a manifest could never be handed to."""
        left = benchmark_run_record(make_manifest(_two_arms()))
        right = benchmark_run_record(make_manifest(_two_arms(canonical_s=9.0)))
        # No calibrations needed: both records carry the same fingerprint, so
        # the configurations are identical and subtraction is permitted
        # outright. That is the case this shape exists to make checkable.
        comparison = compare_run_records(left, right, ())
        # Narrowed on the discriminant the union carries, not on key
        # membership: `kind` exists so a reader never has to guess which
        # shape came back.
        if comparison["kind"] != "compared":
            raise AssertionError(f"identical configurations were not comparable: {comparison}")
        deltas = {delta["name"]: delta["difference"] for delta in comparison["deltas"]}
        assert deltas["cleargbm.mean_fit_s"] == pytest.approx(1.0)

    def test_two_records_from_different_machines_are_refused(self) -> None:
        """The reason this project needed the shape at all: its headline is a
        TIMING claim, and a fit time from a 24-core node is not one from an
        8-core workstation. Uncalibrated, the layer says so instead of
        subtracting."""
        left = benchmark_run_record(make_manifest(_two_arms()))
        elsewhere = make_manifest(_two_arms(canonical_s=9.0))
        elsewhere["fingerprint"] = sample_run_fingerprint(
            image_digest=_DIGEST,
            gpu_model=NO_VALUE,
            driver_version=NO_VALUE,
            determinism=determinism_record("cpu", {"OMP_NUM_THREADS": "24"}),
        )
        verdict = compare_run_records(left, benchmark_run_record(elsewhere), ())
        assert verdict["kind"] == "uncalibrated"
