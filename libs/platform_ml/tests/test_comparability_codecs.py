"""Round-trip and rejection tests for the comparability codecs.

Every decode is exercised against a value that should be refused, not only
against one that should pass. A decoder that accepts everything round-trips
perfectly and validates nothing.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from platform_ml.comparability import (
    Calibration,
    RunFingerprint,
    compare_runs,
    decode_calibration,
    decode_run_fingerprint,
    encode_calibration,
    encode_comparability_verdict,
    encode_run_fingerprint,
)
from platform_ml.determinism import (
    DeterminismReport,
    decode_determinism_report,
    encode_determinism_report,
)

REPORT = DeterminismReport(
    deterministic_algorithms=True,
    cublas_workspace_config=":4096:8",
    matmul_tf32=False,
    cudnn_tf32=False,
    cudnn_deterministic=True,
    cudnn_benchmark=False,
)

FINGERPRINT = RunFingerprint(
    image_digest="sha256:aaaa",
    gpu_model="NVIDIA GeForce RTX 3090 Ti",
    driver_version="550.90.07",
    determinism=REPORT,
)

CALIBRATION = Calibration(
    axis="gpu_model",
    left="NVIDIA GeForce RTX 3090 Ti",
    right="NVIDIA A100 80GB PCIe",
    offset=0.31,
    measured_by="armC-s42 on both cards",
)


def test_determinism_report_round_trips() -> None:
    assert decode_determinism_report(encode_determinism_report(REPORT)) == REPORT


def test_determinism_encode_emits_native_types_not_rendered_strings() -> None:
    # Native types are what makes the round-trip exact. A rendered "true"
    # would decode only if the reader agreed on the spelling.
    encoded = encode_determinism_report(REPORT)

    assert encoded["deterministic_algorithms"] is True
    assert encoded["cudnn_benchmark"] is False
    assert encoded["cublas_workspace_config"] == ":4096:8"


def test_determinism_decode_rejects_a_non_object() -> None:
    with pytest.raises(JSONTypeError):
        decode_determinism_report(["not", "an", "object"])


def test_determinism_decode_rejects_a_missing_field() -> None:
    encoded = encode_determinism_report(REPORT)
    del encoded["cudnn_benchmark"]

    with pytest.raises(JSONTypeError):
        decode_determinism_report(encoded)


def test_determinism_decode_rejects_a_bool_sent_as_a_string() -> None:
    encoded = encode_determinism_report(REPORT)
    encoded["matmul_tf32"] = "false"

    with pytest.raises(JSONTypeError):
        decode_determinism_report(encoded)


def test_fingerprint_round_trips_including_the_nested_report() -> None:
    assert decode_run_fingerprint(encode_run_fingerprint(FINGERPRINT)) == FINGERPRINT


def test_fingerprint_decode_rejects_a_missing_axis() -> None:
    encoded = encode_run_fingerprint(FINGERPRINT)
    del encoded["gpu_model"]

    with pytest.raises(JSONTypeError):
        decode_run_fingerprint(encoded)


def test_fingerprint_decode_rejects_a_broken_nested_report() -> None:
    # The nested decoder runs; a fingerprint cannot pass by carrying a
    # determinism block that would fail on its own.
    encoded = encode_run_fingerprint(FINGERPRINT)
    encoded["determinism"] = {"deterministic_algorithms": True}

    with pytest.raises(JSONTypeError):
        decode_run_fingerprint(encoded)


def test_fingerprint_decode_rejects_a_non_object() -> None:
    with pytest.raises(JSONTypeError):
        decode_run_fingerprint("sha256:aaaa")


def test_calibration_round_trips() -> None:
    assert decode_calibration(encode_calibration(CALIBRATION)) == CALIBRATION


def test_calibration_decode_rejects_an_unknown_axis() -> None:
    # An offset naming an axis nothing compares would never apply, and would
    # do so silently.
    encoded = encode_calibration(CALIBRATION)
    encoded["axis"] = "cuda_version"

    with pytest.raises(JSONTypeError):
        decode_calibration(encoded)


def test_calibration_decode_rejects_unrecorded_provenance() -> None:
    # An offset nobody can trace is worse than no offset: it silently moves
    # a published number and cannot be audited.
    encoded = encode_calibration(CALIBRATION)
    encoded["measured_by"] = ""

    with pytest.raises(JSONTypeError):
        decode_calibration(encoded)


def test_calibration_decode_rejects_a_non_numeric_offset() -> None:
    encoded = encode_calibration(CALIBRATION)
    encoded["offset"] = "0.31"

    with pytest.raises(JSONTypeError):
        decode_calibration(encoded)


def test_calibration_decode_rejects_a_non_object() -> None:
    with pytest.raises(JSONTypeError):
        decode_calibration(42)


def test_identical_verdict_encodes_to_its_discriminant_alone() -> None:
    verdict = compare_runs(FINGERPRINT, FINGERPRINT, ())

    assert encode_comparability_verdict(verdict) == {"kind": "identical"}


def test_offset_verdict_encodes_its_offset_and_the_measurements_applied() -> None:
    other = RunFingerprint(
        image_digest=FINGERPRINT["image_digest"],
        gpu_model="NVIDIA A100 80GB PCIe",
        driver_version=FINGERPRINT["driver_version"],
        determinism=REPORT,
    )

    encoded = encode_comparability_verdict(compare_runs(FINGERPRINT, other, (CALIBRATION,)))

    assert encoded["kind"] == "offset"
    assert encoded["offset"] == pytest.approx(0.31)
    assert encoded["differences"] == [
        {
            "axis": "gpu_model",
            "left": "NVIDIA GeForce RTX 3090 Ti",
            "right": "NVIDIA A100 80GB PCIe",
        }
    ]
    assert encoded["calibrations"] == [encode_calibration(CALIBRATION)]


def test_uncalibrated_verdict_encodes_which_axes_lack_a_measurement() -> None:
    other = RunFingerprint(
        image_digest="sha256:bbbb",
        gpu_model=FINGERPRINT["gpu_model"],
        driver_version=FINGERPRINT["driver_version"],
        determinism=REPORT,
    )

    encoded = encode_comparability_verdict(compare_runs(FINGERPRINT, other, ()))

    assert encoded["kind"] == "uncalibrated"
    assert encoded["uncalibrated"] == [
        {"axis": "image_digest", "left": "sha256:aaaa", "right": "sha256:bbbb"}
    ]
