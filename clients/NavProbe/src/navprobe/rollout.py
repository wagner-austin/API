"""Drive a simulator to a run record.

The simulator is injected as a :class:`SimulatorProtocol` rather than imported.
That is what lets the whole instrument be exercised against real deterministic
and real non-deterministic implementations in the test suite: a determinism
instrument validated against a mock would only establish that the mock is
deterministic.

:class:`SimulatorProtocol` is this package's own port, not a mirror of any
simulator vendor's surface. A driven adapter converts a concrete simulator to
it, which is where a vendor signature is matched exactly and where array types
are flattened to floats.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from navprobe import NavProbeError
from navprobe.digest import digest_run, digest_step
from navprobe.records import RunRecord, RunSpec, StepRecord


class RolloutError(NavProbeError):
    """A rollout could not be completed.

    Args:
        code: Stable identifier in the ``NP-ROLLOUT-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


class SimulatorProtocol(Protocol):
    """The minimal simulator surface a probe rollout drives."""

    @property
    def world_count(self) -> int:
        """Number of parallel worlds this simulator is configured with.

        Returns:
            The world count, which is one or greater.
        """
        ...

    def reset(self, seed: int) -> None:
        """Return the simulator to its initial state under a pinned seed.

        Args:
            seed: The seed to pin.
        """
        ...

    def advance(self) -> Sequence[float]:
        """Advance one step and return the resulting observation.

        Returns:
            The observation flattened to floats, in a stable element order.
            Element order is part of the contract: a simulator that reordered
            its output between runs would register as non-determinism.
        """
        ...


def roll_out(simulator: SimulatorProtocol, label: str, seed: int, step_count: int) -> RunRecord:
    """Run a simulator for a fixed number of steps and digest every step.

    Args:
        simulator: The simulator to drive.
        label: Name of the experimental condition, recorded in the spec.
        seed: The seed to pin before the first step.
        step_count: Number of steps to take. Zero is permitted and yields a
            record with no steps, which is the correct base case rather than an
            error.

    Returns:
        The run record, carrying the spec, every step digest, and the folded
        run digest.

    Raises:
        RolloutError: When ``step_count`` is negative, or when the simulator
            reports a world count below one.
        CanonicalEncodingError: When an observation cannot be canonically
            encoded, which includes any observation containing NaN.
    """
    if step_count < 0:
        raise RolloutError(
            "NP-ROLLOUT-001", f"step_count must be zero or greater, got {step_count}"
        )
    world_count = simulator.world_count
    if world_count < 1:
        raise RolloutError(
            "NP-ROLLOUT-002", f"simulator reports world_count {world_count}; must be one or greater"
        )

    simulator.reset(seed)
    steps: list[StepRecord] = []
    for step_index in range(step_count):
        observation = simulator.advance()
        steps.append(StepRecord(step_index=step_index, digest=digest_step(step_index, observation)))

    spec = RunSpec(label=label, seed=seed, step_count=step_count, world_count=world_count)
    return RunRecord(
        spec=spec,
        steps=tuple(steps),
        digest=digest_run([step["digest"] for step in steps]),
    )


__all__ = ["RolloutError", "SimulatorProtocol", "roll_out"]
