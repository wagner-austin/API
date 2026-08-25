"""Tests for the known-answer gate.

The headline case is the one that motivated the module: an image whose torch
version silently changed produces a different value, and the check catches it
in seconds rather than after a training run. The case next to it is the one
that keeps the check trustworthy: the same value under a different card is
NOT reported as a broken image.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from platform_ml.comparability import RunFingerprint
from platform_ml.determinism import FALSE, TORCH_STACK, TRUE, determinism_record
from platform_ml.known_answer import (
    KnownAnswer,
    check_known_answer,
    decode_known_answer,
    describe_known_answer_outcome,
    encode_known_answer,
)

PINNED = determinism_record(
    TORCH_STACK,
    {
        "deterministic_algorithms": TRUE,
        "cublas_workspace_config": ":4096:8",
        "matmul_tf32": FALSE,
        "cudnn_tf32": FALSE,
        "cudnn_deterministic": TRUE,
        "cudnn_benchmark": FALSE,
    },
)


def fingerprint(
    *, image: str = "sha256:aaaa", gpu: str = "NVIDIA GeForce RTX 3090 Ti"
) -> RunFingerprint:
    """Build a fingerprint, defaulting to the local card fully pinned."""
    return RunFingerprint(
        image_digest=image,
        gpu_model=gpu,
        driver_version="550.90.07",
        determinism=PINNED,
    )


ANSWER = KnownAnswer(
    label="armA-gpt2-s42-200steps",
    fingerprint=fingerprint(),
    expected=6.723808288574219,
    tolerance=0.0,
)


def test_the_same_value_under_the_same_configuration_matches() -> None:
    outcome = check_known_answer(ANSWER, fingerprint(), 6.723808288574219)

    assert outcome == {
        "kind": "matches",
        "observed": 6.723808288574219,
        "deviation": 0.0,
    }


def test_bit_exact_is_expressible_and_a_last_bit_difference_deviates() -> None:
    # Tolerance zero is the right default within one configuration once
    # determinism is pinned: same-seed runs were measured bit-identical.
    outcome = check_known_answer(ANSWER, fingerprint(), 6.723808288574220)

    assert outcome["kind"] == "deviates"
    assert outcome["deviation"] > 0.0


def test_a_silently_changed_image_is_caught_as_a_deviation() -> None:
    # The motivating failure, reduced: the digest is the same because the
    # tag was reused, and only the VALUE reveals the swap.
    outcome = check_known_answer(ANSWER, fingerprint(), 6.723791122436523)

    assert outcome["kind"] == "deviates"
    assert outcome["deviation"] == pytest.approx(1.7166137696e-05)
    assert outcome["tolerance"] == 0.0
    assert describe_known_answer_outcome(ANSWER, outcome).startswith(
        "known answer 'armA-gpt2-s42-200steps': DEVIATES"
    )


def test_a_different_card_does_not_condemn_the_image() -> None:
    # The rule that keeps the gate trustworthy. A known answer establishes
    # what ONE configuration produces; under another it has nothing to say,
    # and reporting "deviates" would train everyone to ignore the check.
    outcome = check_known_answer(ANSWER, fingerprint(gpu="NVIDIA A100 80GB PCIe"), 6.9)

    assert outcome["kind"] == "configuration_differs"
    assert [d["axis"] for d in outcome["differences"]] == ["gpu_model"]
    assert describe_known_answer_outcome(ANSWER, outcome) == (
        "known answer 'armA-gpt2-s42-200steps': does not apply, configuration differs on gpu_model"
    )


def test_configuration_is_checked_before_the_value() -> None:
    # Even an exactly-equal value reports configuration_differs, because the
    # answer was never established for that configuration. Agreement there
    # would be a coincidence the check must not launder into a confirmation.
    outcome = check_known_answer(ANSWER, fingerprint(image="sha256:bbbb"), ANSWER["expected"])

    assert outcome["kind"] == "configuration_differs"


def test_a_tolerance_admits_values_inside_it_and_rejects_the_edge_beyond() -> None:
    loose = KnownAnswer(
        label="loose",
        fingerprint=fingerprint(),
        expected=10.0,
        tolerance=0.5,
    )

    assert check_known_answer(loose, fingerprint(), 10.5)["kind"] == "matches"
    assert check_known_answer(loose, fingerprint(), 9.5)["kind"] == "matches"
    assert check_known_answer(loose, fingerprint(), 10.51)["kind"] == "deviates"


def test_a_match_still_reports_its_deviation() -> None:
    # So drift toward the edge of tolerance is visible before it crosses.
    loose = KnownAnswer(label="loose", fingerprint=fingerprint(), expected=10.0, tolerance=0.5)

    outcome = check_known_answer(loose, fingerprint(), 10.4)

    assert outcome["kind"] == "matches"
    assert outcome["deviation"] == pytest.approx(0.4)
    assert describe_known_answer_outcome(loose, outcome) == (
        "known answer 'loose': matches (deviation 0.4)"
    )


def test_known_answer_round_trips() -> None:
    assert decode_known_answer(encode_known_answer(ANSWER)) == ANSWER


def test_decode_rejects_an_unnamed_answer() -> None:
    # A failing check that cannot say what it ran is not actionable.
    encoded = encode_known_answer(ANSWER)
    encoded["label"] = ""

    with pytest.raises(JSONTypeError):
        decode_known_answer(encoded)


def test_decode_rejects_a_negative_tolerance() -> None:
    # A negative band admits no value at all, so every check would report a
    # deviation and read as a broken image rather than a broken answer.
    encoded = encode_known_answer(ANSWER)
    encoded["tolerance"] = -0.1

    with pytest.raises(JSONTypeError):
        decode_known_answer(encoded)


def test_decode_rejects_a_broken_nested_fingerprint() -> None:
    encoded = encode_known_answer(ANSWER)
    encoded["fingerprint"] = {"image_digest": "sha256:aaaa"}

    with pytest.raises(JSONTypeError):
        decode_known_answer(encoded)


def test_decode_rejects_a_missing_expected_value() -> None:
    encoded = encode_known_answer(ANSWER)
    del encoded["expected"]

    with pytest.raises(JSONTypeError):
        decode_known_answer(encoded)


def test_decode_rejects_a_non_object() -> None:
    with pytest.raises(JSONTypeError):
        decode_known_answer("armA-gpt2-s42-200steps")
