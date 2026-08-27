"""Describing the configuration of a run that uses no GPU.

Most of this monorepo's research is not torch -- gradient boosting,
transliteration, metabolomics -- and those runs were recording nothing about
what produced their numbers. The cleargbm p6 sweeps ran on HPC3 with a
manifest carrying hyperparameters and no environment at all, so a result from
a 24-core cluster node and one from an 8-core workstation were
indistinguishable in the file.

The axis that matters for them is not the card. It is the BLAS thread count,
which `determinism_cpu` measured changing 865,498 of 16,777,216 matmul
elements between 1, 8 and 24 threads.
"""

from __future__ import annotations

from platform_core.comparability import (
    IMAGE_DIGEST_ENV_VAR,
    NO_VALUE,
    compare_configurations,
    cpu_run_fingerprint,
    image_digest_from_env,
)
from platform_core.determinism_cpu import apply_cpu_determinism
from platform_core.determinism_env import SINGLE_THREAD
from platform_core.determinism_record import UNPINNED_STACK, determinism_record
from platform_core.environment_record import PackageVersion, host_record
from platform_core.testing import SAMPLE_HOST, SAMPLE_PACKAGES

_DIGEST = "sha256:" + "ab" * 32

_NO_ENV: dict[str, str] = {}
"""A launcher that exported nothing, which is a run outside any image."""

NOT_LOADED: tuple[str, ...] = ()
"""A module table with no native numeric library in it.

Stated rather than left to `sys.modules`: `apply_cpu_determinism` refuses
once the natives are loaded, and a test whose meaning depended on what else
the pytest worker imported would be worse than no test.
"""


def _discard_env(name: str, value: str) -> None:
    """Accept a pinned variable without writing it.

    Args:
        name: The variable the pin would set.
        value: The value it would be set to.
    """


class TestImageDigestFromEnv:
    def test_it_reads_the_variable_the_launcher_exports(self) -> None:
        assert image_digest_from_env({IMAGE_DIGEST_ENV_VAR: _DIGEST}.get) == _DIGEST

    def test_an_unset_variable_reads_as_no_value(self) -> None:
        """A run out of a directory environment has no image and no digest."""
        assert image_digest_from_env(_NO_ENV.get) == NO_VALUE

    def test_an_empty_variable_reads_the_same_as_unset(self) -> None:
        """Both mean nobody told this process which image it is in."""
        assert image_digest_from_env({IMAGE_DIGEST_ENV_VAR: ""}.get) == NO_VALUE

    def test_the_variable_has_one_spelling(self) -> None:
        """A second spelling is a silent "unknown image" in whichever drifts."""
        assert IMAGE_DIGEST_ENV_VAR == "IMAGE_DIGEST"


class TestCpuRunFingerprint:
    def test_it_carries_the_image_and_the_pinned_posture(self) -> None:
        determinism = apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED)

        fingerprint = cpu_run_fingerprint(
            determinism, {IMAGE_DIGEST_ENV_VAR: _DIGEST}.get, SAMPLE_HOST, SAMPLE_PACKAGES
        )

        assert fingerprint["image_digest"] == _DIGEST
        assert fingerprint["determinism"] == determinism

    def test_the_thread_count_is_what_the_posture_records(self) -> None:
        """The axis that decides a cpu run's numbers, in the record."""
        fingerprint = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
            _NO_ENV.get,
            SAMPLE_HOST,
            SAMPLE_PACKAGES,
        )

        settings = dict(fingerprint["determinism"]["settings"])
        assert settings["OMP_NUM_THREADS"] == SINGLE_THREAD

    def test_no_card_is_recorded_as_empty_rather_than_omitted(self) -> None:
        """Empty differs from every real card; an absent axis would not."""
        fingerprint = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
            _NO_ENV.get,
            SAMPLE_HOST,
            SAMPLE_PACKAGES,
        )

        assert fingerprint["gpu_model"] == NO_VALUE
        assert fingerprint["driver_version"] == NO_VALUE
        assert sorted(fingerprint) == [
            "determinism",
            "driver_version",
            "gpu_model",
            "host",
            "image_digest",
            "packages",
        ]


class TestItComparesAgainstTheOtherPaths:
    """One rule for three research stacks, or three rules that drift."""

    def test_two_cpu_runs_in_one_image_at_one_thread_count_are_identical(self) -> None:
        left = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
            {IMAGE_DIGEST_ENV_VAR: _DIGEST}.get,
            SAMPLE_HOST,
            SAMPLE_PACKAGES,
        )
        right = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
            {IMAGE_DIGEST_ENV_VAR: _DIGEST}.get,
            SAMPLE_HOST,
            SAMPLE_PACKAGES,
        )

        assert compare_configurations(left, right, ())["kind"] == "identical"

    def test_a_different_thread_count_is_a_difference(self) -> None:
        """The 24-core node versus the 8-core one, caught rather than assumed."""
        one = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
            {IMAGE_DIGEST_ENV_VAR: _DIGEST}.get,
            SAMPLE_HOST,
            SAMPLE_PACKAGES,
        )
        eight = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, "8", NOT_LOADED),
            {IMAGE_DIGEST_ENV_VAR: _DIGEST}.get,
            SAMPLE_HOST,
            SAMPLE_PACKAGES,
        )

        verdict = compare_configurations(one, eight, ())
        assert verdict["kind"] == "uncalibrated"
        assert [d["axis"] for d in verdict["differences"]] == ["determinism"]

    def test_a_run_that_pinned_nothing_differs_from_one_that_did(self) -> None:
        pinned = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
            {IMAGE_DIGEST_ENV_VAR: _DIGEST}.get,
            SAMPLE_HOST,
            SAMPLE_PACKAGES,
        )
        unpinned = cpu_run_fingerprint(
            determinism_record(UNPINNED_STACK, {}),
            {IMAGE_DIGEST_ENV_VAR: _DIGEST}.get,
            SAMPLE_HOST,
            SAMPLE_PACKAGES,
        )

        assert compare_configurations(pinned, unpinned, ())["kind"] == "uncalibrated"

    def test_the_same_code_in_two_images_is_a_difference(self) -> None:
        """What `versions.torch` and a git commit both fail to distinguish."""
        here = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
            {IMAGE_DIGEST_ENV_VAR: _DIGEST}.get,
            SAMPLE_HOST,
            SAMPLE_PACKAGES,
        )
        there = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
            {IMAGE_DIGEST_ENV_VAR: "sha256:" + "cd" * 32}.get,
            SAMPLE_HOST,
            SAMPLE_PACKAGES,
        )

        verdict = compare_configurations(here, there, ())
        assert verdict["kind"] == "uncalibrated"
        assert [d["axis"] for d in verdict["differences"]] == ["image_digest"]

    def test_two_machines_outside_any_image_are_a_difference(self) -> None:
        """The case this module's docstring named and could not detect.

        No image, so no digest. No card, so no card or driver. The same pin
        on both, so the determinism records match. Before the host axis every
        remaining field was equal and the verdict was IDENTICAL -- which is
        the 24-core cluster node and the 8-core workstation reported as one
        configuration, exactly as the docstring above says the cleargbm p6
        sweeps recorded them.
        """
        workstation = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
            _NO_ENV.get,
            SAMPLE_HOST,
            SAMPLE_PACKAGES,
        )
        node = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
            _NO_ENV.get,
            host_record(
                platform=SAMPLE_HOST["platform"],
                machine=SAMPLE_HOST["machine"],
                logical_cores=24,
            ),
            SAMPLE_PACKAGES,
        )

        verdict = compare_configurations(workstation, node, ())
        assert verdict["kind"] == "uncalibrated"
        assert [d["axis"] for d in verdict["differences"]] == ["host"]

    def test_a_library_bump_outside_any_image_is_a_difference(self) -> None:
        """The other half: no image digest, so the packages ARE the software axis."""
        before = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
            _NO_ENV.get,
            SAMPLE_HOST,
            (PackageVersion(name="numpy", version="2.3.5"),),
        )
        after = cpu_run_fingerprint(
            apply_cpu_determinism(_discard_env, SINGLE_THREAD, NOT_LOADED),
            _NO_ENV.get,
            SAMPLE_HOST,
            (PackageVersion(name="numpy", version="2.4.0"),),
        )

        verdict = compare_configurations(before, after, ())
        assert verdict["kind"] == "uncalibrated"
        assert [d["axis"] for d in verdict["differences"]] == ["packages"]
