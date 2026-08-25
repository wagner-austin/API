"""Round-trip and rejection tests for the comparability codecs.

Every decode is exercised against a value that should be refused, not only
against one that should pass. A decoder that accepts everything round-trips
perfectly and validates nothing.
"""

from __future__ import annotations

import pytest

from platform_core.comparability import (
    Calibration,
    RunFingerprint,
    compare_configurations,
    decode_calibration,
    decode_run_fingerprint,
    encode_calibration,
    encode_comparability_verdict,
    encode_run_fingerprint,
)
from platform_core.determinism_record import FALSE, TRUE, determinism_record
from platform_core.json_utils import JSONTypeError

# See test_comparability: a literal, because platform_core knows no torch.
_TORCH = "torch"

REPORT = determinism_record(
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


# The determinism record's own codec is covered in test_determinism.py,
# beside the type it encodes. This module is about the fingerprint and the
# verdict, and only exercises the record as a nested value.


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
    verdict = compare_configurations(FINGERPRINT, FINGERPRINT, ())

    assert encode_comparability_verdict(verdict) == {"kind": "identical"}


def test_offset_verdict_encodes_its_offset_and_the_measurements_applied() -> None:
    other = RunFingerprint(
        image_digest=FINGERPRINT["image_digest"],
        gpu_model="NVIDIA A100 80GB PCIe",
        driver_version=FINGERPRINT["driver_version"],
        determinism=REPORT,
    )

    verdict = compare_configurations(FINGERPRINT, other, (CALIBRATION,))
    encoded = encode_comparability_verdict(verdict)

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

    encoded = encode_comparability_verdict(compare_configurations(FINGERPRINT, other, ()))

    assert encoded["kind"] == "uncalibrated"
    assert encoded["uncalibrated"] == [
        {"axis": "image_digest", "left": "sha256:aaaa", "right": "sha256:bbbb"}
    ]
