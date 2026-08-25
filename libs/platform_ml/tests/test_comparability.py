"""Tests for the comparability verdict.

The cases below are the two failures this module exists to catch, written as
fixtures: a torch swap that changes the image digest while everything else
matches, and a card change with and without a measured offset. Each asserts
the VERDICT SHAPE, not merely that something was returned, because the whole
design claim is that a boolean would have thrown away the actionable half.
"""

from __future__ import annotations

import pytest

from platform_ml.comparability import (
    COMPARABILITY_AXES,
    AxisDifference,
    Calibration,
    RunFingerprint,
    compare_runs,
    describe_verdict,
    find_differences,
)
from platform_ml.determinism import DeterminismReport

DETERMINISTIC = DeterminismReport(
    deterministic_algorithms=True,
    cublas_workspace_config=":4096:8",
    matmul_tf32=False,
    cudnn_tf32=False,
    cudnn_deterministic=True,
    cudnn_benchmark=False,
)

NONDETERMINISTIC = DeterminismReport(
    deterministic_algorithms=False,
    cublas_workspace_config="",
    matmul_tf32=True,
    cudnn_tf32=True,
    cudnn_deterministic=False,
    cudnn_benchmark=True,
)


def fingerprint(
    *,
    image: str = "sha256:aaaa",
    gpu: str = "NVIDIA GeForce RTX 3090 Ti",
    driver: str = "550.90.07",
    determinism: DeterminismReport = DETERMINISTIC,
) -> RunFingerprint:
    """Build a fingerprint, defaulting to the local card fully pinned."""
    return RunFingerprint(
        image_digest=image,
        gpu_model=gpu,
        driver_version=driver,
        determinism=determinism,
    )


def test_identical_configurations_subtract() -> None:
    verdict = compare_runs(fingerprint(), fingerprint(), ())

    assert verdict == {"kind": "identical"}
    assert describe_verdict(verdict) == "comparable: configurations identical"


def test_a_torch_swap_shows_up_as_an_image_digest_difference() -> None:
    # The real failure: published arms ran one torch, a rebuilt image ran
    # another, everything else matched, and nothing objected.
    verdict = compare_runs(fingerprint(), fingerprint(image="sha256:bbbb"), ())

    assert verdict["kind"] == "uncalibrated"
    assert verdict["differences"] == (
        AxisDifference(axis="image_digest", left="sha256:aaaa", right="sha256:bbbb"),
    )
    assert verdict["uncalibrated"] == verdict["differences"]


def test_an_uncalibrated_card_change_names_the_axis_rather_than_refusing_silently() -> None:
    verdict = compare_runs(fingerprint(), fingerprint(gpu="NVIDIA A100 80GB PCIe"), ())

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

    verdict = compare_runs(fingerprint(), fingerprint(gpu="NVIDIA A100 80GB PCIe"), (calibration,))

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

    verdict = compare_runs(fingerprint(gpu="NVIDIA A100 80GB PCIe"), fingerprint(), (calibration,))

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

    verdict = compare_runs(
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

    verdict = compare_runs(
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
    verdict = compare_runs(fingerprint(), fingerprint(determinism=NONDETERMINISTIC), ())

    assert verdict["kind"] == "uncalibrated"
    assert [d["axis"] for d in verdict["uncalibrated"]] == ["determinism"]
    assert "deterministic=True" in verdict["differences"][0]["left"]
    assert "deterministic=False" in verdict["differences"][0]["right"]


def test_a_single_flipped_determinism_flag_is_still_a_difference() -> None:
    # TF32 alone changes the numbers, so a report differing only there must
    # not compare equal to one that pinned it.
    tf32_on = DeterminismReport(
        deterministic_algorithms=True,
        cublas_workspace_config=":4096:8",
        matmul_tf32=True,
        cudnn_tf32=False,
        cudnn_deterministic=True,
        cudnn_benchmark=False,
    )

    assert find_differences(fingerprint(), fingerprint(determinism=tf32_on)) != ()


def test_an_absent_digest_differs_from_a_known_one_rather_than_matching_anything() -> None:
    # Empty means unknown. Treating unknown as a wildcard would report a run
    # with no recorded image comparable to every run that has one.
    verdict = compare_runs(fingerprint(image=""), fingerprint(), ())

    assert verdict["kind"] == "uncalibrated"
    assert verdict["uncalibrated"][0]["left"] == ""


def test_differences_are_reported_in_the_declared_axis_order() -> None:
    # Two verdicts over the same pair must render identically, so the order
    # is the constant's order rather than dict iteration order.
    verdict = compare_runs(
        fingerprint(),
        fingerprint(
            image="sha256:bbbb",
            gpu="NVIDIA A100 80GB PCIe",
            driver="560.1.2",
            determinism=NONDETERMINISTIC,
        ),
        (),
    )

    assert verdict["kind"] == "uncalibrated"
    assert [d["axis"] for d in verdict["differences"]] == list(COMPARABILITY_AXES)


def test_a_calibration_on_a_different_axis_does_not_cover_the_difference() -> None:
    wrong_axis = Calibration(
        axis="driver_version",
        left="NVIDIA GeForce RTX 3090 Ti",
        right="NVIDIA A100 80GB PCIe",
        offset=0.31,
        measured_by="mislabelled",
    )

    verdict = compare_runs(fingerprint(), fingerprint(gpu="NVIDIA A100 80GB PCIe"), (wrong_axis,))

    assert verdict["kind"] == "uncalibrated"


def test_a_calibration_between_other_values_does_not_cover_this_pair() -> None:
    unrelated = Calibration(
        axis="gpu_model",
        left="NVIDIA A30",
        right="NVIDIA L40S",
        offset=1.0,
        measured_by="other cards",
    )

    verdict = compare_runs(fingerprint(), fingerprint(gpu="NVIDIA A100 80GB PCIe"), (unrelated,))

    assert verdict["kind"] == "uncalibrated"
