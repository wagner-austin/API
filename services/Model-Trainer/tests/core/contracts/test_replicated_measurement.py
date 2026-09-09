"""A gain you measured once is not a gain, and this is where that is enforced.

The type exists because the first pass at the cartridge work reported
differences of 0.02 as findings and then measured its own noise at 0.02. Every
test here is about keeping that from being possible again: a single-seed gain
cannot be constructed, a floor comes from the arms actually run, and a ratio
against an arm that did not gain is refused rather than printed.
"""

from __future__ import annotations

import math

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str
from platform_core.run_record import Observation

from model_trainer.core.contracts.replicated_measurement import (
    MIN_SEEDS,
    PAIRED_ALPHA,
    ReplicatedGain,
    decode_replicated_gain,
    decode_replicated_gains,
    encode_replicated_gain,
    gain_observations,
    noise_floor,
    paired_separation,
    per_seed_observations,
    replicate,
    retention,
    separates,
)


def _gain(arm: str, gains: tuple[float, ...]) -> ReplicatedGain:
    """Build a gain through the constructor, so the summary is never hand-written."""
    return replicate(arm, [(7 + index, value) for index, value in enumerate(gains)])


class TestReplicate:
    def test_it_summarises_every_replicate(self) -> None:
        measured = replicate("slots-8", [(7, 0.80), (8, 0.90), (9, 0.85)])

        assert measured == {
            "arm": "slots-8",
            "seeds": (7, 8, 9),
            "gains": (0.80, 0.90, 0.85),
            "mean": pytest.approx(0.85),
            "spread": pytest.approx(0.10),
        }

    def test_it_keeps_the_replicates_beside_the_summary(self) -> None:
        """The individual gains survive, so a reader is never left with a mean.

        A spread of 0.10 reads very differently for (0.80, 0.85, 0.90) than for
        (0.80, 0.90, 0.90), and only the replicates distinguish them.
        """
        measured = replicate("slots-8", [(7, 0.80), (8, 0.90), (9, 0.90)])

        assert measured["gains"] == (0.80, 0.90, 0.90)
        assert measured["seeds"] == (7, 8, 9)

    def test_one_seed_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            replicate("slots-8", [(7, 0.80)])

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_MEASUREMENT_UNREPLICATED
        assert "1 time(s)" in excinfo.value.message

    def test_the_bar_is_three_and_two_does_not_clear_it(self) -> None:
        """Two was the original bar, and was measured to be too few.

        At two replicates the spread is one subtraction: the same gpt2 sweep
        reported a floor of 0.0180 across three seeds and 0.0307 across two,
        and the difference flipped a verdict. Pinned as a test because the
        constant is the whole enforcement, and lowering it back would
        otherwise be a one-character change nothing objected to.
        """
        assert MIN_SEEDS == 3

        with pytest.raises(AppError) as excinfo:
            replicate("slots-8", [(7, 0.80), (8, 0.90)])

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_MEASUREMENT_UNREPLICATED


class TestNoiseFloor:
    def test_it_takes_the_largest_spread(self) -> None:
        """Largest, not mean: a floor exists to be cleared.

        Underestimating it licenses claims the noise could have produced;
        overestimating it only withholds claims, which is the safe direction.
        """
        floor = noise_floor(
            [
                _gain("a", (0.80, 0.81, 0.82)),
                _gain("b", (0.50, 0.60, 0.70)),
                _gain("c", (0.90, 0.90, 0.91)),
            ]
        )

        assert floor == pytest.approx(0.20)

    def test_no_arms_means_no_evidence_of_noise(self) -> None:
        assert noise_floor([]) == 0.0


class TestSeparates:
    def test_a_difference_over_the_floor_separates(self) -> None:
        verdict = separates(
            _gain("big", (0.90, 0.91, 0.92)), _gain("small", (0.50, 0.51, 0.52)), floor=0.05
        )

        assert verdict == {
            "first": "big",
            "second": "small",
            "difference": pytest.approx(0.40),
            "floor": 0.05,
            "separated": True,
        }

    def test_a_difference_under_the_floor_does_not(self) -> None:
        verdict = separates(
            _gain("big", (0.90, 0.91, 0.92)), _gain("small", (0.89, 0.90, 0.91)), floor=0.05
        )

        assert verdict["separated"] is False
        assert verdict["difference"] == pytest.approx(0.01)

    def test_the_sign_survives(self) -> None:
        """A negative difference is a real answer, not a swapped argument.

        The composed arm scoring below the arm it contains is exactly what the
        composition measurement found, so the direction has to come through.
        """
        verdict = separates(
            _gain("composed", (0.50, 0.51, 0.52)), _gain("alone", (0.90, 0.91, 0.92)), floor=0.05
        )

        assert verdict["difference"] == pytest.approx(-0.40)
        assert verdict["separated"] is True

    def test_a_difference_exactly_at_the_floor_does_not_separate(self) -> None:
        """The boundary is exclusive, so a tie with the noise is not a finding."""
        verdict = separates(
            _gain("big", (0.60, 0.60, 0.60)), _gain("small", (0.50, 0.50, 0.50)), floor=0.10
        )

        assert verdict["separated"] is False


class TestPairedSeparation:
    """The verdict that reads the seed pairing the range statistic throws away."""

    def test_a_consistent_step_clears_the_paired_test_inside_its_own_range_floor(
        self,
    ) -> None:
        """THE DISAGREEMENT THAT MOTIVATES THE WHOLE ADDITION.

        Both arms wander over 0.20 across seeds, so the range floor is 0.20
        and a gap of 0.05 cannot clear it. But the gap is +0.05 on EVERY
        draw -- the arms wander together -- so the paired differences have no
        spread at all and the step is real. On the `gpt2-wiki` plan four 4x
        steps were in exactly this state, called saturated by the floor and
        overturned by a paired test nothing in the tree computed.
        """
        smaller = _gain("slots-32", (0.10, 0.30, 0.20))
        larger = _gain("slots-128", (0.15, 0.35, 0.25))
        floor = noise_floor([smaller, larger])

        assert separates(larger, smaller, floor=floor)["separated"] is False

        paired = paired_separation(larger, smaller)

        assert paired["significant"] is True
        assert paired["sample_sd"] == pytest.approx(0.0)
        assert paired["mean_difference"] == pytest.approx(0.05)

    def test_arms_that_cross_do_not_clear_it(self) -> None:
        """The mirror case: identical means, and per-seed differences that
        disagree in sign. 4.302653 is the two-sided t at 2 df and alpha 0.05,
        a table value rather than this module's own output.
        """
        base = _gain("base", (0.10, 0.20, 0.30))
        crossed = _gain("crossed", (0.30, 0.20, 0.10))

        paired = paired_separation(crossed, base)

        assert paired["mean_difference"] == pytest.approx(0.0)
        assert paired["sample_sd"] == pytest.approx(0.2)
        assert paired["minimum_detectable_effect"] == pytest.approx(
            4.302653 * 0.2 / math.sqrt(3), abs=1e-6
        )
        assert paired["significant"] is False

    def test_the_mean_difference_is_the_difference_of_the_means(self) -> None:
        """True by algebra, asserted because a reader comparing this field to
        `Separation.difference` needs to know they are the same number and
        that only the SPREAD differs between the two verdicts.
        """
        smaller = _gain("slots-32", (0.10, 0.30, 0.20))
        larger = _gain("slots-128", (0.40, 0.20, 0.60))

        paired = paired_separation(larger, smaller)

        assert paired["mean_difference"] == pytest.approx(larger["mean"] - smaller["mean"])

    def test_the_sd_is_of_the_differences_and_not_of_either_arm(self) -> None:
        """The field most likely to be misread as an arm's spread.

        The per-seed differences are +0.30, -0.10, +0.40, whose sample sd is
        ``sqrt(0.07)``. Neither arm's spread is that number and neither is the
        larger of the two, so a reader who took this field for a range would
        be reading something no arm produced.
        """
        smaller = _gain("slots-32", (0.10, 0.30, 0.20))
        larger = _gain("slots-128", (0.40, 0.20, 0.60))

        paired = paired_separation(larger, smaller)

        assert paired["sample_sd"] == pytest.approx(math.sqrt(0.07), abs=1e-9)
        assert smaller["spread"] == pytest.approx(0.2)
        assert larger["spread"] == pytest.approx(0.4)

    def test_it_carries_what_made_the_pairing_legal(self) -> None:
        smaller = _gain("slots-32", (0.10, 0.30, 0.20))
        larger = _gain("slots-128", (0.15, 0.35, 0.25))

        paired = paired_separation(larger, smaller)

        assert paired["first"] == "slots-128"
        assert paired["second"] == "slots-32"
        assert paired["seeds"] == (7, 8, 9)
        assert paired["replicates"] == 3
        assert paired["alpha"] == PAIRED_ALPHA

    def test_arms_drawn_under_different_seeds_are_refused(self) -> None:
        """REFUSED RATHER THAN DEGRADED. Subtracting position by position
        would produce a number shaped exactly like a paired difference and
        built from unrelated draws.
        """
        smaller = replicate("slots-32", [(7, 0.10), (8, 0.30), (9, 0.20)])
        larger = replicate("slots-128", [(1, 0.15), (2, 0.35), (3, 0.25)])

        with pytest.raises(AppError) as excinfo:
            paired_separation(larger, smaller)

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_ARMS_UNPAIRABLE

    def test_the_same_seeds_in_a_different_order_are_refused_too(self) -> None:
        """The subtler half, and the one a membership check would miss: the
        seeds match as a set and index 0 is seed 7 on one side and seed 9 on
        the other.
        """
        smaller = replicate("slots-32", [(7, 0.10), (8, 0.30), (9, 0.20)])
        larger = replicate("slots-128", [(9, 0.25), (8, 0.35), (7, 0.15)])

        with pytest.raises(AppError) as excinfo:
            paired_separation(larger, smaller)

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_ARMS_UNPAIRABLE

    def test_too_few_paired_differences_are_refused(self) -> None:
        """The guard :func:`replicate` cannot supply, because this function
        does not go through it.

        ``replicate`` refuses fewer than MIN_SEEDS results, so no gain BUILT
        BY IT can reach here short. But ``paired_separation`` is public and
        takes the TypedDicts, so a caller assembling them by hand reaches
        ``stdev`` directly -- and at one difference that is a bare
        ``statistics.StatisticsError`` with no code and nothing a caller can
        act on. Refused with this package's own code instead, at the same
        floor ``replicate`` enforces, because a sample sd from one draw is a
        range estimate and the MDE computed from it means nothing.
        """
        smaller = ReplicatedGain(arm="slots-32", seeds=(7,), gains=(0.10,), mean=0.10, spread=0.0)
        larger = ReplicatedGain(arm="slots-128", seeds=(7,), gains=(0.30,), mean=0.30, spread=0.0)

        with pytest.raises(AppError) as excinfo:
            paired_separation(larger, smaller)

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_MEASUREMENT_UNREPLICATED
        assert str(MIN_SEEDS) in excinfo.value.message

    def test_the_floor_is_the_one_replicate_enforces(self) -> None:
        """One below MIN_SEEDS refuses and MIN_SEEDS itself does not, so the
        two entry points cannot drift to different floors.
        """
        seeds = tuple(range(7, 7 + MIN_SEEDS))
        short = tuple(range(7, 7 + MIN_SEEDS - 1))

        with pytest.raises(AppError):
            paired_separation(
                ReplicatedGain(
                    arm="a", seeds=short, gains=(0.3,) * len(short), mean=0.3, spread=0.0
                ),
                ReplicatedGain(
                    arm="b", seeds=short, gains=(0.1,) * len(short), mean=0.1, spread=0.0
                ),
            )

        verdict = paired_separation(
            replicate("a", [(seed, 0.30 + index * 0.01) for index, seed in enumerate(seeds)]),
            replicate("b", [(seed, 0.10 + index * 0.01) for index, seed in enumerate(seeds)]),
        )

        assert verdict["replicates"] == MIN_SEEDS

    def test_a_narrower_alpha_raises_the_bar(self) -> None:
        """Alpha is a parameter with a default, not a constant baked into the
        arithmetic, so a caller reporting at another level gets an MDE for the
        test they ran rather than for this module's default.
        """
        smaller = _gain("slots-32", (0.10, 0.30, 0.20))
        larger = _gain("slots-128", (0.40, 0.20, 0.60))

        assert (
            paired_separation(larger, smaller, alpha=0.01)["minimum_detectable_effect"]
            > paired_separation(larger, smaller, alpha=0.05)["minimum_detectable_effect"]
        )


class TestRetention:
    def test_it_is_the_fraction_that_survived(self) -> None:
        kept = retention(_gain("alone", (0.90, 0.90, 0.90)), _gain("composed", (0.54, 0.54, 0.54)))

        assert kept == pytest.approx(0.60)

    def test_a_fraction_of_a_non_gain_is_refused(self) -> None:
        """A cartridge that did not help has nothing for another to have kept.

        Without this, an arm at -0.01 alone and -0.02 composed reports 200%
        retention -- a number that reads as "kept more than all of it" and
        means the opposite.
        """
        with pytest.raises(AppError) as excinfo:
            retention(
                _gain("alone", (-0.01, -0.01, -0.01)), _gain("composed", (-0.02, -0.02, -0.02))
            )

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_MEASUREMENT_UNREPLICATED
        assert "-0.0100" in excinfo.value.message

    def test_exactly_zero_is_refused_too(self) -> None:
        """Zero is refused as firmly as negative: the ratio is undefined, not large."""
        with pytest.raises(AppError) as excinfo:
            retention(_gain("alone", (0.0, 0.0, 0.0)), _gain("composed", (0.5, 0.5, 0.5)))

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_MEASUREMENT_UNREPLICATED


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


class TestRoundTrip:
    def test_a_gain_survives_encoding(self) -> None:
        original = _gain("slots-128", (0.90, 0.91, 0.93))

        restored = decode_replicated_gain(
            load_json_str(dump_json_str(encode_replicated_gain(original)))
        )

        assert restored == original

    def test_a_list_of_gains_survives_encoding(self) -> None:
        originals = [_gain("slots-2", (0.7, 0.71, 0.72)), _gain("slots-8", (0.8, 0.81, 0.82))]

        restored = decode_replicated_gains(
            load_json_str(dump_json_str([encode_replicated_gain(g) for g in originals]))
        )

        assert restored == originals


class TestDecodeRefusals:
    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_replicated_gain(["slots-8"])

    def test_a_non_list_of_gains_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON array"):
            decode_replicated_gains({"arm": "slots-8"})

    def test_mismatched_seeds_and_gains_are_refused(self) -> None:
        """They are positionally matched, so a mismatch attributes nothing.

        Zipping to the shorter would discard the remainder without saying so,
        and the record would look complete.
        """
        encoded = encode_replicated_gain(_gain("slots-8", (0.80, 0.90, 0.85)))
        encoded["seeds"] = [7, 8]

        with pytest.raises(JSONTypeError, match="2 seeds and 3 gains"):
            decode_replicated_gain(encoded)

    def test_a_missing_field_is_refused(self) -> None:
        encoded = encode_replicated_gain(_gain("slots-8", (0.80, 0.90, 0.85)))
        del encoded["mean"]

        with pytest.raises(JSONTypeError):
            decode_replicated_gain(encoded)

    def test_a_mistyped_seed_is_refused(self) -> None:
        encoded = encode_replicated_gain(_gain("slots-8", (0.80, 0.90, 0.85)))
        encoded["seeds"] = [7, 8, "nine"]

        with pytest.raises(JSONTypeError):
            decode_replicated_gain(encoded)

    def test_a_mistyped_gain_is_refused(self) -> None:
        encoded = encode_replicated_gain(_gain("slots-8", (0.80, 0.90, 0.85)))
        encoded["gains"] = [0.80, 0.90, "high"]

        with pytest.raises(JSONTypeError):
            decode_replicated_gain(encoded)
