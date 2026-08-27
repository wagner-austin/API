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
from platform_core.testing import SAMPLE_HOST, FakeHostProbe, FakeVersionReader

from covenant_ml.benchmarking import _test_hooks
from covenant_ml.benchmarking.provenance import (
    BENCHMARK_DISTRIBUTIONS,
    benchmark_fingerprint,
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
