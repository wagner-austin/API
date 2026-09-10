"""How an arm becomes named rows, and how it comes back.

SPLIT FROM ``test_replicated_measurement`` when the 600-line ceiling caught
it, and the boundary is a role rather than a line count: everything there is
STATISTICS over an arm's replicates -- the summary, the floor, the separation
verdicts -- and everything here is the arm's TRAVEL between a live
measurement and a run record's flat mapping of name to float.

That travel now runs both ways. It went one way for as long as a record was
only ever written; a sweep that resumes from a checkpoint reads its own rows
back and rebuilds the arms its end-of-run reductions need, so the naming
convention is load-bearing in both directions and both are tested here,
beside each other, where a change to one that breaks the other is visible.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.run_record import Observation

from model_trainer.core.contracts.replicated_measurement import (
    ReplicatedGain,
    gain_observations,
    per_seed_observations,
    replicate,
    replicated_from_observations,
)


def _gain(arm: str, gains: tuple[float, ...]) -> ReplicatedGain:
    """Build a gain through the constructor, so the summary is never hand-written.

    Args:
        arm: What was measured.
        gains: One gain per seed, seeds starting at 7.

    Returns:
        The replicated gain.
    """
    return replicate(arm, [(7 + index, value) for index, value in enumerate(gains)])


class TestGainObservations:
    def test_the_spread_is_named_beside_the_mean(self) -> None:
        """Both, so two runs can be compared on their noise as well as their answer.

        A run whose spread doubled measured something different from the run
        before it, whatever its mean says.
        """
        named = gain_observations(_gain("slots-8", (0.80, 0.90, 0.85)))

        assert named == (
            {"name": "slots-8_mean", "value": pytest.approx(0.85)},
            {"name": "slots-8_spread", "value": pytest.approx(0.10)},
        )


class TestPerSeedObservations:
    def test_each_seed_is_named_with_its_own_gain(self) -> None:
        named = per_seed_observations(_gain("slots-8", (0.80, 0.90, 0.85)))

        assert named == (
            {"name": "slots-8_seed7_gain", "value": pytest.approx(0.80)},
            {"name": "slots-8_seed8_gain", "value": pytest.approx(0.90)},
            {"name": "slots-8_seed9_gain", "value": pytest.approx(0.85)},
        )

    def test_the_names_pair_across_arms_on_the_seed(self) -> None:
        """THE PROPERTY THE WHOLE FUNCTION EXISTS FOR, asserted rather than
        implied by the format string.

        Every arm of a run trains under the same seeds, so one seed's gain at
        two slot counts are two measurements of ONE draw. A later reader
        recovers that by matching the seed segment of the name; if the names
        did not agree the record would carry the numbers and still not permit
        the paired comparison, which is the state this replaced.
        """
        smaller = per_seed_observations(_gain("slots-32", (0.10, 0.20, 0.30)))
        larger = per_seed_observations(_gain("slots-128", (0.15, 0.35, 0.25)))

        def by_seed(named: tuple[Observation, ...], arm: str) -> dict[str, float]:
            return {
                observation["name"].removeprefix(f"{arm}_"): observation["value"]
                for observation in named
            }

        assert set(by_seed(smaller, "slots-32")) == set(by_seed(larger, "slots-128"))
        paired = {
            key: by_seed(larger, "slots-128")[key] - value
            for key, value in by_seed(smaller, "slots-32").items()
        }
        assert paired == {
            "seed7_gain": pytest.approx(0.05),
            "seed8_gain": pytest.approx(0.15),
            "seed9_gain": pytest.approx(-0.05),
        }

    def test_a_paired_difference_is_not_recoverable_from_mean_and_spread(self) -> None:
        """WHY mean+spread WAS NOT ENOUGH, shown rather than argued.

        These two arms have IDENTICAL means and IDENTICAL spreads, so the
        record as it stood could not tell them apart -- yet their paired
        differences are +0.2/0.0/-0.2 in one case and 0.0/0.0/0.0 in the
        other. One of those is an arm that moved every draw; the other did
        nothing at all.
        """
        base = _gain("base", (0.10, 0.20, 0.30))
        moved = _gain("moved", (0.30, 0.20, 0.10))

        assert base["mean"] == pytest.approx(moved["mean"])
        assert base["spread"] == pytest.approx(moved["spread"])

        gains = {
            observation["name"].split("_")[-2]: observation["value"]
            for observation in per_seed_observations(moved)
        }
        assert gains["seed7"] == pytest.approx(0.30)
        assert gains["seed9"] == pytest.approx(0.10)


class TestReplicatedFromObservations:
    """The inverse, which a resumed sweep needs.

    A checkpoint carries OBSERVATIONS; the end-of-run reductions take ARMS. So
    a resumed sweep has to rebuild its arms from the rows it recorded, and
    these tests are what say the rebuild is exact rather than approximate.
    """

    def test_an_arm_survives_the_round_trip_exactly(self) -> None:
        """Not merely close: every field of an arm is a function of its
        per-seed gains, so the rebuilt arm must be equal, summary numbers
        included. A resumed sweep's noise floor is computed over these."""
        original = _gain("lora-plain-n4-composed", (0.80, 0.90, 0.85))

        rebuilt = replicated_from_observations(
            per_seed_observations(original), arm=original["arm"], seeds=original["seeds"]
        )

        assert rebuilt == original

    def test_rows_belonging_to_other_arms_are_ignored(self) -> None:
        """One checkpointed cell carries several arms' rows, so the same
        observations are read once per arm."""
        wanted = _gain("lora-plain-n4-composed", (0.80, 0.90, 0.85))
        other = _gain("lora-plain-n4-alone", (0.10, 0.20, 0.30))

        rebuilt = replicated_from_observations(
            (*per_seed_observations(other), *per_seed_observations(wanted)),
            arm=wanted["arm"],
            seeds=wanted["seeds"],
        )

        assert rebuilt == wanted

    def test_the_seed_order_given_becomes_the_arms_own(self) -> None:
        """The pairing is positional, so an arm rebuilt under a different
        seed order would silently unpair from the arms it is subtracted
        against."""
        original = _gain("arm", (0.80, 0.90, 0.85))

        rebuilt = replicated_from_observations(
            per_seed_observations(original), arm="arm", seeds=(9, 8, 7)
        )

        assert rebuilt["seeds"] == (9, 8, 7)
        assert rebuilt["gains"] == (0.85, 0.90, 0.80)

    def test_a_missing_seed_row_is_refused_not_skipped(self) -> None:
        """Rebuilding over only the seeds that happen to be present would
        report a mean and a spread over fewer draws than the arm's own label
        claims -- a complete table, every number plausible, and nothing
        downstream able to see it."""
        original = _gain("arm", (0.80, 0.90, 0.85))

        with pytest.raises(AppError) as excinfo:
            replicated_from_observations(
                per_seed_observations(original), arm="arm", seeds=(7, 8, 9, 10)
            )

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_ARM_ROWS_INCOMPLETE
        assert "arm_seed10_gain" in excinfo.value.message

    def test_the_refusal_names_every_absent_row_and_the_total(self) -> None:
        """An operator told only that one row was missing fixes one thing and
        meets the next refusal."""
        with pytest.raises(AppError) as excinfo:
            replicated_from_observations((), arm="arm", seeds=(7, 8, 9))

        assert "3 of 3" in excinfo.value.message
        for seed in (7, 8, 9):
            assert f"arm_seed{seed}_gain" in excinfo.value.message

    def test_rows_from_a_differently_named_arm_are_refused(self) -> None:
        """The arm name is what pairs a run against the recorded ladder, so
        reading one arm's rows under another arm's name is exactly the
        mistake this refusal exists to catch."""
        original = _gain("lora-plain-n4-composed", (0.80, 0.90, 0.85))

        with pytest.raises(AppError) as excinfo:
            replicated_from_observations(
                per_seed_observations(original),
                arm="lora-diverse-n4-composed",
                seeds=original["seeds"],
            )

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_ARM_ROWS_INCOMPLETE
