"""Tests for the record an experiment emits, and for subtracting two of them.

Two cases carry the design. A comparison across an unbridged configuration
returns the verdict and NO numbers, because numbers beside a "not comparable"
note get used. And an observation present on only one side is reported rather
than dropped, because a metric that quietly disappears between two runs is a
finding and a comparison that omits it looks complete.
"""

from __future__ import annotations

import pytest

from platform_core.comparability import Calibration, RunFingerprint
from platform_core.determinism_record import TRUE, determinism_record
from platform_core.json_utils import JSONTypeError
from platform_core.run_record import (
    Observation,
    compare_run_records,
    decode_run_record,
    encode_run_record,
    run_record,
)
from platform_core.testing import sample_run_fingerprint

PINNED = determinism_record("torch", {"cudnn_deterministic": TRUE})
ABLATION = "wiki-corpus-extraction-ablation"


def fingerprint(*, gpu: str = "NVIDIA GeForce RTX 3090 Ti") -> RunFingerprint:
    """Build a fingerprint, defaulting to the local card fully pinned."""
    return sample_run_fingerprint(
        image_digest="sha256:aaaa",
        gpu_model=gpu,
        driver_version="591.86",
        determinism=PINNED,
    )


def test_two_runs_on_one_configuration_subtract() -> None:
    left = run_record(
        ABLATION, "armA-s42", fingerprint(), (Observation(name="acc", value=0.8380),), ""
    )
    right = run_record(
        ABLATION, "armB-s42", fingerprint(), (Observation(name="acc", value=0.7766),), ""
    )

    result = compare_run_records(left, right, ())

    assert result["kind"] == "compared"
    assert result["verdict"] == {"kind": "identical"}
    assert result["deltas"] == (
        {"name": "acc", "left": 0.8380, "right": 0.7766, "difference": pytest.approx(-0.0614)},
    )
    assert result["unmatched"] == ()


def test_an_unbridged_configuration_difference_returns_no_numbers_at_all() -> None:
    # The headline rule. Reporting deltas beside a "not comparable" verdict
    # is how a caller ends up using them.
    left = run_record(
        ABLATION, "armA-s42", fingerprint(), (Observation(name="acc", value=0.83),), ""
    )
    right = run_record(
        ABLATION,
        "armA-s42-a100",
        fingerprint(gpu="NVIDIA A100 80GB PCIe"),
        (Observation(name="acc", value=0.88),),
        "",
    )

    result = compare_run_records(left, right, ())

    assert result["kind"] == "uncalibrated"
    assert [d["axis"] for d in result["uncalibrated"]] == ["gpu_model"]
    assert "deltas" not in result


def test_a_measured_offset_is_applied_to_the_difference() -> None:
    # The payoff of a verdict over a boolean: a measured card offset turns
    # "incomparable" into a number, with the correction already subtracted.
    calibration = Calibration(
        axis="gpu_model",
        left="NVIDIA GeForce RTX 3090 Ti",
        right="NVIDIA A100 80GB PCIe",
        offset=0.02,
        measured_by="armA-s42 on both cards",
    )
    left = run_record(
        ABLATION, "armA-3090", fingerprint(), (Observation(name="acc", value=0.80),), ""
    )
    right = run_record(
        ABLATION,
        "armA-a100",
        fingerprint(gpu="NVIDIA A100 80GB PCIe"),
        (Observation(name="acc", value=0.85),),
        "",
    )

    result = compare_run_records(left, right, (calibration,))

    assert result["kind"] == "compared"
    assert result["verdict"]["kind"] == "offset"
    # 0.85 - 0.80 = 0.05 raw; 0.02 of it is the card, so 0.03 is the run.
    assert result["deltas"][0]["difference"] == pytest.approx(0.03)


def test_an_observation_only_one_run_reported_is_named_not_dropped() -> None:
    left = run_record(
        ABLATION,
        "armA",
        fingerprint(),
        (Observation(name="acc", value=0.8), Observation(name="wall_clock_sec", value=12.0)),
        "",
    )
    right = run_record(ABLATION, "armB", fingerprint(), (Observation(name="acc", value=0.7),), "")

    result = compare_run_records(left, right, ())

    assert result["kind"] == "compared"
    assert [d["name"] for d in result["deltas"]] == ["acc"]
    assert result["unmatched"] == ("wall_clock_sec",)


def test_comparing_two_different_experiments_is_refused() -> None:
    # No calibration bridges two different questions, so there is nothing to
    # return a verdict about.
    left = run_record(ABLATION, "armA", fingerprint(), (Observation(name="acc", value=0.8),), "")
    right = run_record(
        "cleargbm-benchmark", "run1", fingerprint(), (Observation(name="acc", value=0.9),), ""
    )

    with pytest.raises(ValueError, match="different experiments"):
        compare_run_records(left, right, ())


def test_observations_are_canonically_ordered() -> None:
    built = run_record(
        ABLATION,
        "armA",
        fingerprint(),
        (Observation(name="z", value=1.0), Observation(name="a", value=2.0)),
        "",
    )

    assert [o["name"] for o in built["observations"]] == ["a", "z"]


def test_a_record_round_trips_through_the_ledger() -> None:
    built = run_record(
        ABLATION,
        "armB-s42",
        fingerprint(),
        (Observation(name="acc", value=0.7766), Observation(name="wall_clock_sec", value=931.5)),
        "sha256:beef",
    )

    assert decode_run_record(encode_run_record(built)) == built


def test_a_payload_digest_survives_the_round_trip_for_bit_identity_checks() -> None:
    built = run_record(ABLATION, "armA", fingerprint(), (), "sha256:beef")

    assert decode_run_record(encode_run_record(built))["payload_digest"] == "sha256:beef"


@pytest.mark.parametrize(
    ("experiment", "label", "message"),
    [("", "armA", "experiment"), (ABLATION, "", "label")],
    ids=["unnamed-experiment", "unnamed-run"],
)
def test_a_record_that_cannot_say_what_it_is_gets_refused(
    experiment: str, label: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_record(experiment, label, fingerprint(), (), "")


def test_two_observations_of_one_name_are_refused() -> None:
    # Keeping one of the two would silently decide which number a later
    # contrast reads.
    with pytest.raises(ValueError, match="distinct names"):
        run_record(
            ABLATION,
            "armA",
            fingerprint(),
            (Observation(name="acc", value=0.8), Observation(name="acc", value=0.9)),
            "",
        )


def test_decode_rejects_an_unnamed_observation() -> None:
    encoded = encode_run_record(
        run_record(ABLATION, "armA", fingerprint(), (Observation(name="acc", value=0.8),), "")
    )
    encoded["observations"] = [{"name": "", "value": 0.8}]

    with pytest.raises(JSONTypeError, match="name"):
        decode_run_record(encoded)


def test_decode_rejects_a_non_numeric_observation_value() -> None:
    encoded = encode_run_record(
        run_record(ABLATION, "armA", fingerprint(), (Observation(name="acc", value=0.8),), "")
    )
    encoded["observations"] = [{"name": "acc", "value": "0.8"}]

    with pytest.raises(JSONTypeError):
        decode_run_record(encoded)


def test_decode_rejects_a_broken_nested_fingerprint_and_a_non_object() -> None:
    encoded = encode_run_record(run_record(ABLATION, "armA", fingerprint(), (), ""))
    encoded["fingerprint"] = {"image_digest": "sha256:aaaa"}

    with pytest.raises(JSONTypeError):
        decode_run_record(encoded)
    with pytest.raises(JSONTypeError):
        decode_run_record("armA")
