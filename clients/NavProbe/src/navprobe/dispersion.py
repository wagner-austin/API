"""Measure how far apart repeated rollouts end up, not merely whether they differ.

A digest comparison is a leading indicator: it fails at the first differing bit,
long before anyone could see the difference. That is what makes it useful, and
also what makes it insufficient on its own — "twelve runs produced twelve
digests" does not say whether the twelve outcomes are a nanometre apart or half
a metre.

This module answers the second question. It drives the same configuration
repeatedly, keeps each rollout's final observation, and reports the element-wise
spread across them.

The spread is reported in the observation's own units and the instrument does
not name them. For a state observation they are metres; for a rendered one they
are depth units or packed colour. Which of those it is depends on the adapter
that produced them, and this layer has never known that.
"""

from __future__ import annotations

from collections.abc import Sequence

from navprobe import NavProbeError
from navprobe.canonical import require_encodable
from navprobe.experiment import MINIMUM_REPETITIONS, SimulatorFactoryProtocol
from navprobe.records import DispersionRecord
from navprobe.rollout import SimulatorProtocol


class DispersionError(NavProbeError):
    """A dispersion measurement could not be made.

    Args:
        code: Stable identifier in the ``NP-DISP-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


def final_observation(simulator: SimulatorProtocol, seed: int, step_count: int) -> Sequence[float]:
    """Drive a simulator and return only its last observation.

    Args:
        simulator: The simulator to drive.
        seed: The seed to pin before the first step.
        step_count: Number of steps to take.

    Returns:
        The observation after the final step.

    Raises:
        DispersionError: When ``step_count`` is below one. A rollout of no steps
            produces no observation, and there would be nothing to disperse.
        CanonicalEncodingError: When the final observation contains NaN. A NaN
            spread compares false against every threshold, so it would read as a
            pass rather than as the failure it is.
    """
    if step_count < 1:
        raise DispersionError(
            "NP-DISP-001",
            f"step_count must be one or greater, got {step_count}; "
            "a rollout with no steps produces no observation to compare",
        )
    simulator.reset(seed)
    observation: Sequence[float] = ()
    for _ in range(step_count):
        observation = simulator.advance()
    require_encodable(observation)
    return observation


def measure_dispersion(
    simulator_factory: SimulatorFactoryProtocol,
    seed: int,
    step_count: int,
    repetitions: int,
) -> DispersionRecord:
    """Measure the spread of final observations across repeated rollouts.

    Each repetition gets a freshly constructed simulator, for the same reason a
    trial does: reusing one instance would measure whether ``reset`` restores
    state rather than whether the configuration reproduces.

    Args:
        simulator_factory: Builds a freshly constructed simulator per repetition.
        seed: The seed every repetition is pinned to.
        step_count: Steps per repetition.
        repetitions: Number of rollouts to compare.

    Returns:
        The dispersion record.

    Raises:
        DispersionError: When ``repetitions`` is below
            :data:`navprobe.experiment.MINIMUM_REPETITIONS`, when ``step_count``
            is below one, when the simulator observes nothing, or when two
            repetitions produce observations of different lengths.
        RolloutError: When a simulator reports an unusable world count.
        CanonicalEncodingError: When an observation cannot be encoded.
    """
    if repetitions < MINIMUM_REPETITIONS:
        raise DispersionError(
            "NP-DISP-002",
            f"dispersion needs at least {MINIMUM_REPETITIONS} repetitions to have a "
            f"spread, got {repetitions}",
        )
    observations = [
        list(final_observation(simulator_factory(), seed, step_count)) for _ in range(repetitions)
    ]
    length = len(observations[0])
    if length < 1:
        raise DispersionError(
            "NP-DISP-004",
            "the simulator produced an empty observation; a rollout that observes "
            "nothing would report a spread of zero over no evidence",
        )
    for index, observation in enumerate(observations):
        if len(observation) != length:
            raise DispersionError(
                "NP-DISP-003",
                f"repetition {index} produced {len(observation)} values but repetition 0 "
                f"produced {length}; observations of different shapes cannot be compared",
            )
    spreads = [
        max(observation[index] for observation in observations)
        - min(observation[index] for observation in observations)
        for index in range(length)
    ]
    return DispersionRecord(
        repetitions=repetitions,
        observation_length=length,
        max_spread=max(spreads),
        mean_spread=sum(spreads) / len(spreads),
    )


__all__ = ["DispersionError", "final_observation", "measure_dispersion"]
