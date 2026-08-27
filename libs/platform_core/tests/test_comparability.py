"""Tests for the comparability verdict.

The cases below are the two failures this module exists to catch, written as
fixtures: a torch swap that changes the image digest while everything else
matches, and a card change with and without a measured offset. Each asserts
the VERDICT SHAPE, not merely that something was returned, because the whole
design claim is that a boolean would have thrown away the actionable half.
"""

from __future__ import annotations

import pytest

from platform_core.comparability import (
    COMPARABILITY_AXES,
    AxisDifference,
    Calibration,
    RunFingerprint,
    compare_configurations,
    describe_verdict,
    find_differences,
)
from platform_core.determinism_record import (
    FALSE,
    TRUE,
    UNPINNED_STACK,
    DeterminismRecord,
    determinism_record,
)
from platform_core.environment_record import HostRecord, PackageVersion, host_record
from platform_core.testing import SAMPLE_HOST, SAMPLE_PACKAGES, sample_run_fingerprint

# Spelled as a literal rather than imported: platform_core knows nothing
# about torch, and a test here reaching into platform_ml for a constant
# would invert the dependency this move exists to establish.
_TORCH = "torch"

DETERMINISTIC = determinism_record(
    _TORCH,
    {
        "deterministic_algorithms": TRUE,
        "cublas_workspace_config": ":4096:8",
        "matmul_tf32": FALSE,
        "cudnn_tf32": FALSE,
        "cudnn_deterministic": TRUE,
        "cudnn_benchmark": FALSE,
    },
)

NONDETERMINISTIC = determinism_record(UNPINNED_STACK, {})

#: A second machine. Differs from :data:`SAMPLE_HOST` only in core count,
#: which is the case the axis exists for: same OS, same architecture, same
#: libraries, and a threaded reduction that partitions differently.
OTHER_HOST = host_record(
    platform=SAMPLE_HOST["platform"],
    machine=SAMPLE_HOST["machine"],
    logical_cores=24,
)

#: The same libraries with one bumped.
OTHER_PACKAGES: tuple[PackageVersion, ...] = (PackageVersion(name="numpy", version="2.4.0"),)


def fingerprint(
    *,
    image: str = "sha256:aaaa",
    gpu: str = "NVIDIA GeForce RTX 3090 Ti",
    driver: str = "550.90.07",
    determinism: DeterminismRecord = DETERMINISTIC,
    host: HostRecord = SAMPLE_HOST,
    packages: tuple[PackageVersion, ...] = SAMPLE_PACKAGES,
) -> RunFingerprint:
    """Build a fingerprint, defaulting to the local card fully pinned."""
    return sample_run_fingerprint(
        image_digest=image,
        gpu_model=gpu,
        driver_version=driver,
        determinism=determinism,
        host=host,
        packages=packages,
    )


def test_identical_configurations_subtract() -> None:
    verdict = compare_configurations(fingerprint(), fingerprint(), ())

    assert verdict == {"kind": "identical"}
    assert describe_verdict(verdict) == "comparable: configurations identical"


def test_a_torch_swap_shows_up_as_an_image_digest_difference() -> None:
    # The real failure: published arms ran one torch, a rebuilt image ran
    # another, everything else matched, and nothing objected.
    verdict = compare_configurations(fingerprint(), fingerprint(image="sha256:bbbb"), ())

    assert verdict["kind"] == "uncalibrated"
    assert verdict["differences"] == (
        AxisDifference(axis="image_digest", left="sha256:aaaa", right="sha256:bbbb"),
    )
    assert verdict["uncalibrated"] == verdict["differences"]


def test_an_uncalibrated_card_change_names_the_axis_rather_than_refusing_silently() -> None:
    verdict = compare_configurations(fingerprint(), fingerprint(gpu="NVIDIA A100 80GB PCIe"), ())

    assert verdict["kind"] == "uncalibrated"
    assert [d["axis"] for d in verdict["uncalibrated"]] == ["gpu_model"]
    assert describe_verdict(verdict) == (
        "NOT comparable: differs on gpu_model; unmeasured: gpu_model"
    )


def test_a_measured_card_offset_makes_the_numbers_subtract() -> None:
    calibration = Calibration(
        axis="gpu_model",
        left="NVIDIA GeForce RTX 3090 Ti",
        right="NVIDIA A100 80GB PCIe",
        offset=0.31,
        measured_by="armC-s42 on both cards",
    )

    verdict = compare_configurations(
        fingerprint(), fingerprint(gpu="NVIDIA A100 80GB PCIe"), (calibration,)
    )

    assert verdict["kind"] == "offset"
    assert verdict["offset"] == pytest.approx(0.31)
    assert verdict["calibrations"] == (calibration,)
    assert describe_verdict(verdict) == ("comparable with offset +0.3100 across gpu_model")


def test_a_calibration_answers_in_the_reverse_direction_with_the_sign_flipped() -> None:
    # Measuring 3090->A100 also answers A100->3090. Requiring both directions
    # to be measured separately would double the calibration cost for nothing.
    calibration = Calibration(
        axis="gpu_model",
        left="NVIDIA GeForce RTX 3090 Ti",
        right="NVIDIA A100 80GB PCIe",
        offset=0.31,
        measured_by="armC-s42 on both cards",
    )

    verdict = compare_configurations(
        fingerprint(gpu="NVIDIA A100 80GB PCIe"), fingerprint(), (calibration,)
    )

    assert verdict["kind"] == "offset"
    assert verdict["offset"] == pytest.approx(-0.31)
    assert verdict["calibrations"][0]["measured_by"] == "armC-s42 on both cards"


def test_offsets_across_several_axes_sum() -> None:
    calibrations = (
        Calibration(
            axis="gpu_model",
            left="NVIDIA GeForce RTX 3090 Ti",
            right="NVIDIA A100 80GB PCIe",
            offset=0.31,
            measured_by="cards",
        ),
        Calibration(
            axis="driver_version",
            left="550.90.07",
            right="560.1.2",
            offset=-0.05,
            measured_by="drivers",
        ),
    )

    verdict = compare_configurations(
        fingerprint(),
        fingerprint(gpu="NVIDIA A100 80GB PCIe", driver="560.1.2"),
        calibrations,
    )

    assert verdict["kind"] == "offset"
    assert verdict["offset"] == pytest.approx(0.26)
    assert len(verdict["calibrations"]) == 2


def test_one_uncovered_axis_makes_the_whole_comparison_uncalibrated() -> None:
    # A partial correction is the dangerous outcome: it looks quantitative
    # and is missing a term. So any uncovered axis wins over any covered one.
    calibrations = (
        Calibration(
            axis="gpu_model",
            left="NVIDIA GeForce RTX 3090 Ti",
            right="NVIDIA A100 80GB PCIe",
            offset=0.31,
            measured_by="cards",
        ),
    )

    verdict = compare_configurations(
        fingerprint(),
        fingerprint(gpu="NVIDIA A100 80GB PCIe", image="sha256:bbbb"),
        calibrations,
    )

    assert verdict["kind"] == "uncalibrated"
    assert [d["axis"] for d in verdict["differences"]] == ["image_digest", "gpu_model"]
    assert [d["axis"] for d in verdict["uncalibrated"]] == ["image_digest"]


def test_determinism_settings_are_an_axis() -> None:
    # Same card, same image, one run pinned and one not. This is the second
    # failure: same-seed GPT-2 diverges without the controls, so two such runs
    # are not comparable however identical everything else is.
    verdict = compare_configurations(fingerprint(), fingerprint(determinism=NONDETERMINISTIC), ())

    assert verdict["kind"] == "uncalibrated"
    assert [d["axis"] for d in verdict["uncalibrated"]] == ["determinism"]
    assert verdict["differences"][0]["left"].startswith(f"{_TORCH}[")
    assert verdict["differences"][0]["right"] == f"{UNPINNED_STACK}[]"


def test_a_single_flipped_determinism_flag_is_still_a_difference() -> None:
    # TF32 alone changes the numbers, so a report differing only there must
    # not compare equal to one that pinned it.
    tf32_on = determinism_record(
        _TORCH,
        {
            "deterministic_algorithms": TRUE,
            "cublas_workspace_config": ":4096:8",
            "matmul_tf32": TRUE,
            "cudnn_tf32": FALSE,
            "cudnn_deterministic": TRUE,
            "cudnn_benchmark": FALSE,
        },
    )

    assert find_differences(fingerprint(), fingerprint(determinism=tf32_on)) != ()


def test_an_absent_digest_differs_from_a_known_one_rather_than_matching_anything() -> None:
    # Empty means unknown. Treating unknown as a wildcard would report a run
    # with no recorded image comparable to every run that has one.
    verdict = compare_configurations(fingerprint(image=""), fingerprint(), ())

    assert verdict["kind"] == "uncalibrated"
    assert verdict["uncalibrated"][0]["left"] == ""


def test_differences_are_reported_in_the_declared_axis_order() -> None:
    # Two verdicts over the same pair must render identically, so the order
    # is the constant's order rather than dict iteration order.
    verdict = compare_configurations(
        fingerprint(),
        fingerprint(
            image="sha256:bbbb",
            gpu="NVIDIA A100 80GB PCIe",
            driver="560.1.2",
            determinism=NONDETERMINISTIC,
            host=OTHER_HOST,
            packages=OTHER_PACKAGES,
        ),
        (),
    )

    assert verdict["kind"] == "uncalibrated"
    assert [d["axis"] for d in verdict["differences"]] == list(COMPARABILITY_AXES)


def test_two_machines_running_identical_software_are_not_identical() -> None:
    # The gap the host axis exists to close. Before it, this pair differed on
    # nothing a fingerprint recorded -- so a gradient-boosting benchmark run
    # on two different boxes compared as one configuration, and its numbers
    # subtracted without complaint.
    verdict = compare_configurations(fingerprint(), fingerprint(host=OTHER_HOST), ())

    assert verdict["kind"] == "uncalibrated"
    assert [d["axis"] for d in verdict["differences"]] == ["host"]


def test_the_host_difference_names_both_machines() -> None:
    differences = find_differences(fingerprint(), fingerprint(host=OTHER_HOST))

    assert differences == (
        AxisDifference(
            axis="host",
            left="Linux-5.14.0-x86_64-with-glibc2.34/x86_64/8",
            right="Linux-5.14.0-x86_64-with-glibc2.34/x86_64/24",
        ),
    )


def test_a_core_count_change_alone_is_a_difference() -> None:
    # Same OS, same architecture, same libraries, more cores. A threaded
    # reduction partitions by the count, so this is the axis and not a
    # cosmetic one.
    differences = find_differences(fingerprint(), fingerprint(host=OTHER_HOST))

    assert [d["axis"] for d in differences] == ["host"]


def test_a_library_bump_on_one_machine_is_a_difference() -> None:
    differences = find_differences(fingerprint(), fingerprint(packages=OTHER_PACKAGES))

    assert differences == (
        AxisDifference(axis="packages", left="numpy=2.3.5", right="numpy=2.4.0"),
    )


def test_a_measured_library_bump_can_be_calibrated_away() -> None:
    # The reason host and packages are SEPARATE axes: an offset measured for
    # a numpy bump names only the numpy values, so it applies on any machine.
    # Folded into one axis its endpoints would carry the CPU too.
    bump = Calibration(
        axis="packages",
        left="numpy=2.3.5",
        right="numpy=2.4.0",
        offset=0.02,
        measured_by="run-1 vs run-2",
    )

    verdict = compare_configurations(fingerprint(), fingerprint(packages=OTHER_PACKAGES), (bump,))

    assert verdict["kind"] == "offset"
    assert verdict["offset"] == 0.02


def test_a_calibration_on_a_different_axis_does_not_cover_the_difference() -> None:
    wrong_axis = Calibration(
        axis="driver_version",
        left="NVIDIA GeForce RTX 3090 Ti",
        right="NVIDIA A100 80GB PCIe",
        offset=0.31,
        measured_by="mislabelled",
    )

    verdict = compare_configurations(
        fingerprint(), fingerprint(gpu="NVIDIA A100 80GB PCIe"), (wrong_axis,)
    )

    assert verdict["kind"] == "uncalibrated"


def test_a_calibration_between_other_values_does_not_cover_this_pair() -> None:
    unrelated = Calibration(
        axis="gpu_model",
        left="NVIDIA A30",
        right="NVIDIA L40S",
        offset=1.0,
        measured_by="other cards",
    )

    verdict = compare_configurations(
        fingerprint(), fingerprint(gpu="NVIDIA A100 80GB PCIe"), (unrelated,)
    )

    assert verdict["kind"] == "uncalibrated"
