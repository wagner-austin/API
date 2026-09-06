"""A gain you measured once is not a gain, and this is where that is enforced.

The type exists because the first pass at the cartridge work reported
differences of 0.02 as findings and then measured its own noise at 0.02. Every
test here is about keeping that from being possible again: a single-seed gain
cannot be constructed, a floor comes from the arms actually run, and a ratio
against an arm that did not gain is refused rather than printed.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str
from platform_core.run_record import Observation

from model_trainer.core.contracts.replicated_measurement import (
    MIN_SEEDS,
    ReplicatedGain,
    decode_replicated_gain,
    decode_replicated_gains,
    encode_replicated_gain,
    gain_observations,
    noise_floor,
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
