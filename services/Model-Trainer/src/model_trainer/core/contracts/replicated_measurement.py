"""A measured quantity, its replicates, and the difference that means nothing.

WHY THIS TYPE EXISTS RATHER THAN A FLOAT. The cartridge work spent its first
pass reporting single numbers, and two of the conclusions it drew from them
were differences smaller than the measurement's own run-to-run spread. The
sweep reported 512 slots at +0.9337 in one run and +0.9140 in the next under a
byte-identical configuration; anything read off a gap of that size was reading
noise. Nothing in the code made that mistake hard, because a gain was a
``float`` and two floats always subtract.

So a gain is a :class:`ReplicatedGain` here, and one cannot be built from
fewer than :data:`MIN_SEEDS` replicates -- :func:`replicate` refuses. The
refusal is the point: a single-seed number is not a weaker measurement, it is
an unfalsifiable one, and the type system is where that belongs rather than a
review comment.

WHY THE FLOOR IS MEASURED AND NOT ASSUMED. :func:`noise_floor` takes the
largest spread across the arms actually run, so the threshold comes from the
same hardware, stack and corpus as the claim it gates. A constant would be a
guess that goes stale the first time anything underneath it changes.

WHAT THE SPREAD IS, ONCE THE RUN IS ACTUALLY REPRODUCIBLE. Re-running an
identical configuration now produces a BIT-IDENTICAL record -- two runs of
``gpt2-wiki`` in separate processes, 2026-09-03, agreed on every one of its
28 observations exactly. So the spread across seeds is not contaminated by
anything the stack failed to pin: it measures one thing, how much the answer
depended on which cartridge happened to be drawn.

That was not true when this module was written. Training drew dropout from a
process-wide generator nothing seeded, so repeated runs differed and the
spread was part seed and part accident -- 0.0116 between repeats against
0.0180 between seeds, at 128 slots. The fix is in
:func:`~model_trainer.core.services.model.cartridge_measurement.train_cartridge`.
It matters here because a floor built from a contaminated spread is too
large, and a floor that is too large does not make a claim conservative: it
invents plateaus. Removing that contamination moved the ``gpt2-wiki`` sweep
floor from 0.0342 to 0.0202 and turned two "saturated" steps back into real
ones.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_float,
    narrow_json_to_int,
    require_float,
    require_list,
    require_str,
)
from platform_core.run_record import Observation
from typing_extensions import TypedDict

#: Fewest replicates a gain may be built from.
#:
#: THIS WAS TWO, AND TWO WAS MEASURED TO BE TOO FEW. At two replicates a
#: spread is a single ``|a - b|``, which is a range estimate from one draw:
#: the same gpt2 sweep reported a floor of 0.0180 across three seeds at one
#: slot count and 0.0307 across two seeds at five, and the difference flipped
#: whether 32 slots separated from 128. A threshold that swings by 70% with
#: the replicate count is not a threshold.
#:
#: Three is not a cure for that -- no small number is -- but it is the point
#: where the spread stops being one subtraction, and it costs one more arm.
#: The measurement that raised this constant is in
#: :mod:`model_trainer.core.services.model.cartridge_measurement`.
MIN_SEEDS = 3


class ReplicatedGain(TypedDict):
    """One arm, measured several times.

    Attributes:
        arm: What was measured, e.g. ``"slots-128"`` or ``"compose-me-civic"``.
            Stable across runs, because it is what pairs two runs' numbers.
        seeds: The initialisation seeds used, in the order run.
        gains: Each seed's gain, positionally matched to ``seeds``.
        mean: Mean of ``gains``. The number a report shows.
        spread: ``max - min``. The number a report must show BESIDE the mean,
            because it is what says whether the mean is worth reading.
    """

    arm: str
    seeds: tuple[int, ...]
    gains: tuple[float, ...]
    mean: float
    spread: float


class Separation(TypedDict):
    """Whether two arms actually differ, and by how much against what.

    Carries the numbers rather than only the verdict. A bare boolean tells a
    reader that two arms did not separate and leaves them unable to see
    whether it was close, which is exactly the question they will ask next.

    Attributes:
        first: Name of the arm subtracted from.
        second: Name of the arm subtracted.
        difference: ``first.mean - second.mean``. Signed, so the direction
            survives.
        floor: The noise floor it was judged against.
        separated: Whether ``abs(difference)`` exceeds ``floor``.
    """

    first: str
    second: str
    difference: float
    floor: float
    separated: bool


def replicate(arm: str, results: Sequence[tuple[int, float]]) -> ReplicatedGain:
    """Reduce one arm's replicates to a gain that can be reported.

    Args:
        arm: What was measured.
        results: ``(seed, gain)`` pairs, in the order run.

    Returns:
        The replicated gain, carrying every input alongside the summary so a
        reader can see the replicates rather than trust the mean.

    Raises:
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` when fewer than
            :data:`MIN_SEEDS` results are supplied. A one-seed gain has no
            spread, so nothing downstream could tell it from noise -- and the
            arms that produced it are the expensive part, which is precisely
            why the refusal has to be here rather than at reporting time.
    """
    if len(results) < MIN_SEEDS:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_MEASUREMENT_UNREPLICATED,
            (
                f"arm {arm!r} was measured {len(results)} time(s), and a gain needs at "
                f"least {MIN_SEEDS} to have a spread; a single run reports a number "
                f"that nothing can distinguish from the stack's own noise"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_MEASUREMENT_UNREPLICATED),
        )
    gains = tuple(gain for _seed, gain in results)
    return ReplicatedGain(
        arm=arm,
        seeds=tuple(seed for seed, _gain in results),
        gains=gains,
        mean=sum(gains) / len(gains),
        spread=max(gains) - min(gains),
    )


def noise_floor(measurements: Sequence[ReplicatedGain]) -> float:
    """The largest spread any arm produced.

    The largest is taken rather than the mean because a floor exists to be
    cleared: underestimating it licenses claims the noise could have produced.

    PASS ARMS OF ONE KIND. The caller chooses what goes in, and the choice is
    load-bearing -- every arm's spread has to be an estimate of the SAME
    underlying noise for the maximum to mean anything. A composed arm trains
    two cartridges and runs a prefix twice as long, so it varies more than a
    sweep point for reasons that say nothing about the sweep. On the
    ``gpt2-wiki`` plan the sweep's own floor is 0.0202 and the composition
    pair's is 0.0532; folding them together would bury a measured sweep step
    of +0.0278. Importing another kind's noise does not make a claim
    conservative, it makes it wrong in a direction nobody can predict.

    Args:
        measurements: The arms this run measured, all of one kind.

    Returns:
        The floor. Zero when nothing was measured, which is honest: with no
        replicates there is no evidence of noise, and :func:`separates` will
        then separate everything, which is what a caller passing an empty
        sequence has asked for.
    """
    return max((measurement["spread"] for measurement in measurements), default=0.0)


def separates(first: ReplicatedGain, second: ReplicatedGain, *, floor: float) -> Separation:
    """Judge whether two arms differ by more than the noise between them.

    Args:
        first: The arm subtracted from.
        second: The arm subtracted.
        floor: The noise floor, from :func:`noise_floor`.

    Returns:
        The separation, carrying the difference and the floor it was judged
        against as well as the verdict.
    """
    difference = first["mean"] - second["mean"]
    return Separation(
        first=first["arm"],
        second=second["arm"],
        difference=difference,
        floor=floor,
        separated=abs(difference) > floor,
    )


def retention(alone: ReplicatedGain, combined: ReplicatedGain) -> float:
    """What fraction of one arm's gain survived in another.

    The number the composition question is asked in: a cartridge worth
    ``+0.9104`` on its own and ``+0.5395`` once an unrelated one is
    concatenated in front of the same base retained 59%.

    Args:
        alone: The arm measured by itself.
        combined: The arm it was measured inside.

    Returns:
        ``combined.mean / alone.mean``.

    Raises:
        AppError: With ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` when ``alone``
            did not improve on its base. A fraction of a non-gain is not a
            small retention, it is a ratio whose sign and size both mean
            nothing -- a cartridge that scored ``-0.01`` alone and ``-0.02``
            combined would report 200% retention, and reporting that as
            "retained more than all of it" is the failure this refuses.
    """
    if alone["mean"] <= 0.0:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_MEASUREMENT_UNREPLICATED,
            (
                f"arm {alone['arm']!r} has a mean gain of {alone['mean']:+.4f}, so it did "
                f"not improve on its base and there is nothing for {combined['arm']!r} to "
                f"have retained; a ratio against it would be a number with no reading"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_MEASUREMENT_UNREPLICATED),
        )
    return combined["mean"] / alone["mean"]


def per_seed_observations(measurement: ReplicatedGain) -> tuple[Observation, ...]:
    """Name every seed's OWN gain, so the arms stay paired in the record.

    THE MEAN AND THE SPREAD LOSE THE PAIRING, and the pairing is the evidence.
    Every arm of a run trains under the SAME seeds, so seed 7's 32-slot gain
    and seed 7's 128-slot gain are two measurements of one draw. Comparing two
    arms through their means alone throws that away and asks a much weaker
    question -- and a spread cannot give it back, because a range carries no
    information about which replicate produced which end of it.

    It matters most exactly where the run is least conclusive. On the
    ``gpt2-wiki`` plan the top two 4x steps sat inside the spread at three
    seeds; whether they are real is a question about paired differences, and
    the record as it stood could not be asked it.

    A RANGE ALSO SCALES WITH THE SEED COUNT -- sigma*d2(n), d2(3)=1.69 against
    d2(9)=2.97 -- so a plan cannot be made more conclusive simply by adding
    seeds and re-reading its floor. Per-seed numbers are what let a later
    reader compute a statistic that does not move with n.

    Emitted as named scalars rather than as a nested array because that is
    what :class:`~platform_core.run_record.RunRecord` carries: one flat
    mapping of name to float, sorted at construction. ``arm_seed7_gain`` is
    both a legal observation name and a key a reader can pair across arms.

    Args:
        measurement: The arm to name.

    Returns:
        One observation per seed, named ``<arm>_seed<seed>_gain``.
    """
    return tuple(
        Observation(name=f"{measurement['arm']}_seed{seed}_gain", value=gain)
        for seed, gain in zip(measurement["seeds"], measurement["gains"], strict=True)
    )


def gain_observations(measurement: ReplicatedGain) -> tuple[Observation, ...]:
    """Name a gain's numbers for a run record.

    The spread is emitted as its own observation rather than folded into a
    note, so that two runs of one experiment can be compared on their NOISE as
    well as their answer. A run whose spread doubled measured something
    different from the run before it, whatever its mean says.

    Args:
        measurement: The arm to name.

    Returns:
        The mean and the spread, as named scalars.
    """
    return (
        Observation(name=f"{measurement['arm']}_mean", value=measurement["mean"]),
        Observation(name=f"{measurement['arm']}_spread", value=measurement["spread"]),
    )


def encode_replicated_gain(measurement: ReplicatedGain) -> JSONObject:
    """Encode a replicated gain.

    Args:
        measurement: The gain to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "arm": measurement["arm"],
        "seeds": list(measurement["seeds"]),
        "gains": list(measurement["gains"]),
        "mean": measurement["mean"],
        "spread": measurement["spread"],
    }


def decode_replicated_gain(value: JSONValue) -> ReplicatedGain:
    """Decode and validate a replicated gain.

    The seed and gain lists are checked for equal length. They are positionally
    matched, so a record where they disagree describes replicates nobody can
    attribute to a seed, and silently zipping to the shorter would discard the
    remainder without saying so.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated gain.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or the seed and gain lists differ in length.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"replicated gain must be a JSON object, got {type(value).__name__}")
    seeds = tuple(narrow_json_to_int(entry) for entry in require_list(value, "seeds"))
    gains = tuple(narrow_json_to_float(entry) for entry in require_list(value, "gains"))
    if len(seeds) != len(gains):
        raise JSONTypeError(
            f"replicated gain carries {len(seeds)} seeds and {len(gains)} gains; "
            f"they are positionally matched, so neither can be attributed"
        )
    return ReplicatedGain(
        arm=require_str(value, "arm"),
        seeds=seeds,
        gains=gains,
        mean=require_float(value, "mean"),
        spread=require_float(value, "spread"),
    )


def decode_replicated_gains(value: JSONValue) -> list[ReplicatedGain]:
    """Decode a list of replicated gains.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated gains, in the order read.

    Raises:
        JSONTypeError: If the value is not a list, or any entry is invalid.
    """
    if not isinstance(value, list):
        raise JSONTypeError(f"replicated gains must be a JSON array, got {type(value).__name__}")
    return [decode_replicated_gain(entry) for entry in value]


__all__ = [
    "MIN_SEEDS",
    "ReplicatedGain",
    "Separation",
    "decode_replicated_gain",
    "decode_replicated_gains",
    "encode_replicated_gain",
    "gain_observations",
    "noise_floor",
    "per_seed_observations",
    "replicate",
    "retention",
    "separates",
]
