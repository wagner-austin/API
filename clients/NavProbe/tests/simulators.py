"""Real simulator implementations the suite drives.

These are not mocks. A mock records that it was called and returns what the
test told it to, which would prove only that the test can be satisfied. The
instrument under test decides whether a simulator is deterministic, so the
suite needs simulators that genuinely are and genuinely are not, and lets the
instrument reach its own verdict about each.

Every implementation here satisfies :class:`navprobe.rollout.SimulatorProtocol`
structurally. None of them inherits from it: structural conformance is what a
Protocol is for, and inheriting would let a signature drift in the Protocol go
unnoticed here.

Divergence is a construction-time offset rather than a counter mutated on
``reset``. A simulator whose behaviour depends on how many times it has been
reset is a third thing under test, and it made two rollouts of "the same"
simulator mean something different depending on call order.

The factories that build these live in :mod:`tests.factories`.
"""

from __future__ import annotations

import math


class LinearSimulator:
    """A fully deterministic simulator.

    The observation is a pure function of the seed and the step counter, so two
    rollouts at one seed must agree bit for bit. This is the positive control:
    if the instrument reports divergence here, the instrument is wrong.

    Args:
        world_count: Number of parallel worlds to report and to emit values
            for.
    """

    def __init__(self, world_count: int) -> None:
        self._world_count = world_count
        self._seed = 0
        self._step_index = 0

    @property
    def world_count(self) -> int:
        """Number of parallel worlds.

        Returns:
            The configured world count.
        """
        return self._world_count

    def reset(self, seed: int) -> None:
        """Pin the seed and return the step counter to zero.

        Args:
            seed: The seed to pin.
        """
        self._seed = seed
        self._step_index = 0

    def advance(self) -> tuple[float, ...]:
        """Advance one step.

        Returns:
            One float per world, each a pure function of seed, step, and world
            index.
        """
        values = tuple(
            float(self._seed * 1000 + self._step_index * 10 + world)
            for world in range(self._world_count)
        )
        self._step_index += 1
        return values


class DriftingSimulator:
    """A simulator that departs from its peers after a fixed number of steps.

    Models the accumulate-then-diverge failure mode the probe exists to detect:
    identical early behaviour followed by a departure at a specific step. The
    departure is a fixed offset supplied at construction rather than randomness,
    so a test can assert exactly which step diverges rather than that something
    somewhere differed.

    Two instances built with the same ``offset`` agree completely; instances
    built with different offsets agree up to ``diverge_at_step`` and disagree
    from there on.

    Args:
        world_count: Number of parallel worlds.
        diverge_at_step: Step index from which ``offset`` is applied.
        offset: Amount added to every observation once ``diverge_at_step`` is
            reached. Zero reproduces the reference behaviour.
    """

    def __init__(self, world_count: int, diverge_at_step: int, offset: int) -> None:
        self._world_count = world_count
        self._diverge_at_step = diverge_at_step
        self._offset = offset
        self._seed = 0
        self._step_index = 0

    @property
    def world_count(self) -> int:
        """Number of parallel worlds.

        Returns:
            The configured world count.
        """
        return self._world_count

    def reset(self, seed: int) -> None:
        """Pin the seed and return the step counter to zero.

        Args:
            seed: The seed to pin. Recorded but deliberately unused: this
                simulator's divergence is a property of the instance, not the
                seed, which is what makes it a same-seed non-determinism case.
        """
        self._seed = seed
        self._step_index = 0

    def advance(self) -> tuple[float, ...]:
        """Advance one step.

        Returns:
            One float per world. Identical across instances until
            ``diverge_at_step``, and shifted by ``offset`` from then on.
        """
        applied = self._offset if self._step_index >= self._diverge_at_step else 0
        values = tuple(
            float(self._step_index * 10 + world + applied) for world in range(self._world_count)
        )
        self._step_index += 1
        return values


class NaNSimulator:
    """A simulator whose observation contains NaN at a chosen step.

    Exists because NaN is the one value the canonical encoder refuses, and the
    refusal has to be reachable from a rollout rather than only from a direct
    call to the encoder.

    Args:
        world_count: Number of parallel worlds.
        nan_at_step: Step index whose observation contains NaN.
    """

    def __init__(self, world_count: int, nan_at_step: int) -> None:
        self._world_count = world_count
        self._nan_at_step = nan_at_step
        self._seed = 0
        self._step_index = 0

    @property
    def world_count(self) -> int:
        """Number of parallel worlds.

        Returns:
            The configured world count.
        """
        return self._world_count

    def reset(self, seed: int) -> None:
        """Pin the seed and return the step counter to zero.

        Args:
            seed: The seed to pin.
        """
        self._seed = seed
        self._step_index = 0

    def advance(self) -> tuple[float, ...]:
        """Advance one step.

        Returns:
            One float per world, all NaN at ``nan_at_step`` and finite
            elsewhere.
        """
        emit_nan = self._step_index == self._nan_at_step
        values = tuple(
            math.nan if emit_nan else float(self._step_index * 10 + world)
            for world in range(self._world_count)
        )
        self._step_index += 1
        return values


class EmptyWorldSimulator:
    """A simulator reporting a world count below one.

    A simulator configured with no worlds produces no observations, so a
    rollout against it would report perfect determinism over zero evidence.
    The instrument rejects it, and this makes that rejection reachable.

    Args:
        world_count: The invalid world count to report.
    """

    def __init__(self, world_count: int) -> None:
        self._world_count = world_count
        self._seed = 0

    @property
    def world_count(self) -> int:
        """Number of parallel worlds.

        Returns:
            The configured world count, which callers expect to be invalid.
        """
        return self._world_count

    def reset(self, seed: int) -> None:
        """Pin the seed.

        Args:
            seed: The seed to pin.
        """
        self._seed = seed

    def advance(self) -> tuple[float, ...]:
        """Advance one step.

        Returns:
            An empty observation. Never reached in practice: the rollout
            rejects this simulator on its world count before stepping.
        """
        return ()


__all__ = [
    "DriftingSimulator",
    "EmptyWorldSimulator",
    "LinearSimulator",
    "NaNSimulator",
]
