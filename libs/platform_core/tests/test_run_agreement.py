"""Tests for judging whether a set of runs computed the same numbers.

The case that carries the design is the last-bit one. Two cards that agree to
fifteen digits and differ in the sixteenth have NOT agreed, and a check built
on a tolerance would say they had. `distinct` counts values, so it cannot.
"""

from __future__ import annotations

import pytest

from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import TRUE, determinism_record
from platform_core.run_record import (
    Observation,
    RunRecord,
    agree_across_runs,
    run_record,
)

PINNED = determinism_record("torch", {"cudnn_deterministic": TRUE})
LADDER = "probe-shape-threshold"

TINY = "gpt2-tiny-L2-d128-h2-v512-len64-seed42"
SMALL = "gpt2-small-L12-d768-h12-v512-len64-seed42"

#: The gate probe's measured value on three cards, which agreed exactly.
GATE_VALUE = 6.250983715057373


def fingerprint(gpu: str) -> RunFingerprint:
    """Build a fingerprint differing from its siblings only in the card."""
    return RunFingerprint(
        image_digest="sha256:1112dbb1",
        gpu_model=gpu,
        driver_version="580.82.07",
        determinism=PINNED,
    )


def ladder(gpu: str, values: dict[str, float]) -> RunRecord:
    """Build a ladder record for one card.

    Args:
        gpu: The card the run reports.
        values: Rung label to value.

    Returns:
        The record.
    """
    return run_record(
        experiment=LADDER,
        label="probe-ladder-2xdeadbeefcafe",
        fingerprint=fingerprint(gpu),
        observations=tuple(Observation(name=name, value=value) for name, value in values.items()),
        payload_digest="",
    )


def test_three_runs_reporting_one_value_agree_exactly() -> None:
    records = tuple(
        ladder(gpu, {TINY: GATE_VALUE})
        for gpu in ("Tesla V100-PCIE-16GB", "NVIDIA A100 80GB PCIe", "NVIDIA A30")
    )

    agreement = agree_across_runs(records)

    assert agreement["runs"] == 3
    assert agreement["unmatched"] == ()
    assert agreement["shared"] == (
        {
            "name": TINY,
            "values": (GATE_VALUE, GATE_VALUE, GATE_VALUE),
            "distinct": 1,
            "spread": 0.0,
        },
    )


def test_a_last_bit_difference_is_a_disagreement_and_not_a_rounding_note() -> None:
    # The whole reason this counts values instead of comparing against a
    # tolerance. These two agree to fifteen significant digits.
    nudged = 6.250983715057374
    records = (
        ladder("Tesla V100-PCIE-16GB", {SMALL: GATE_VALUE}),
        ladder("NVIDIA A100 80GB PCIe", {SMALL: nudged}),
    )

    entry = agree_across_runs(records)["shared"][0]

    assert entry["distinct"] == 2
    assert entry["spread"] == nudged - GATE_VALUE


def test_values_appear_in_the_order_the_runs_were_given() -> None:
    # Which run is the odd one out is the finding; sorting the values would
    # destroy exactly that.
    records = (
        ladder("Tesla V100-PCIE-16GB", {SMALL: 3.0}),
        ladder("NVIDIA A100 80GB PCIe", {SMALL: 1.0}),
        ladder("NVIDIA A30", {SMALL: 3.0}),
    )

    entry = agree_across_runs(records)["shared"][0]

    assert entry["values"] == (3.0, 1.0, 3.0)
    assert entry["distinct"] == 2
    assert entry["spread"] == 2.0


def test_a_rung_only_some_runs_reported_is_named_rather_than_dropped() -> None:
    # A ladder missing a rung agrees trivially over the rungs it kept. Shared
    # results that silently omitted it would read as complete.
    records = (
        ladder("Tesla V100-PCIE-16GB", {TINY: GATE_VALUE, SMALL: 4.0}),
        ladder("NVIDIA A100 80GB PCIe", {TINY: GATE_VALUE}),
    )

    agreement = agree_across_runs(records)

    assert [entry["name"] for entry in agreement["shared"]] == [TINY]
    assert agreement["unmatched"] == (SMALL,)


def test_shared_entries_come_back_in_name_order() -> None:
    records = (
        ladder("Tesla V100-PCIE-16GB", {SMALL: 4.0, TINY: GATE_VALUE}),
        ladder("NVIDIA A30", {TINY: GATE_VALUE, SMALL: 4.0}),
    )

    agreement = agree_across_runs(records)

    assert [entry["name"] for entry in agreement["shared"]] == sorted([SMALL, TINY])


def test_the_experiment_is_carried_through() -> None:
    records = (
        ladder("NVIDIA A30", {TINY: GATE_VALUE}),
        ladder("NVIDIA A100 80GB PCIe", {TINY: GATE_VALUE}),
    )

    assert agree_across_runs(records)["experiment"] == LADDER


def test_one_run_is_refused_because_a_set_of_one_always_agrees() -> None:
    with pytest.raises(ValueError, match="at least two runs, got 1"):
        agree_across_runs((ladder("NVIDIA A30", {TINY: GATE_VALUE}),))


def test_no_runs_are_refused_for_the_same_reason() -> None:
    with pytest.raises(ValueError, match="at least two runs, got 0"):
        agree_across_runs(())


def test_two_experiments_are_refused_and_both_are_named() -> None:
    other = run_record(
        experiment="environment-known-answer",
        label=TINY,
        fingerprint=fingerprint("NVIDIA A30"),
        observations=(Observation(name="probe_loss", value=GATE_VALUE),),
        payload_digest="",
    )

    with pytest.raises(ValueError) as excinfo:
        agree_across_runs((ladder("NVIDIA A30", {TINY: GATE_VALUE}), other))

    assert "'environment-known-answer'" in str(excinfo.value)
    assert f"'{LADDER}'" in str(excinfo.value)


def test_runs_sharing_no_observation_at_all_report_every_name_unmatched() -> None:
    records = (
        ladder("Tesla V100-PCIE-16GB", {TINY: GATE_VALUE}),
        ladder("NVIDIA A30", {SMALL: 4.0}),
    )

    agreement = agree_across_runs(records)

    assert agreement["shared"] == ()
    assert agreement["unmatched"] == tuple(sorted([SMALL, TINY]))
