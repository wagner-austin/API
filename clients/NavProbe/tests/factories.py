"""Simulator factories the experiment layer is driven with.

Every factory here satisfies
:class:`navprobe.experiment.SimulatorFactoryProtocol` structurally, and none
inherits from it: structural conformance is what a Protocol is for, and
inheriting would let a signature drift in the Protocol go unnoticed here.

Each factory builds a freshly constructed simulator per call, which is the
contract a trial depends on. The ones that record ``built`` do so because a test
has to be able to assert that the factory was called once per repetition rather
than once per trial — reusing one instance would measure whether ``reset``
restores state, a weaker question wearing the same name.

The simulators themselves live in :mod:`tests.simulators`.
"""

from __future__ import annotations

from tests.simulators import (
    DriftingSimulator,
    EmptyWorldSimulator,
    LinearSimulator,
    NaNSimulator,
)


class LinearSimulatorFactory:
    """Builds fully deterministic simulators.

    Every instance is identically configured, so a trial driven by this factory
    must be reported as deterministic. This is the experiment layer's positive
    control.

    Args:
        world_count: World count for every simulator built.
    """

    def __init__(self, world_count: int) -> None:
        self._world_count = world_count
        self.built = 0

    def __call__(self) -> LinearSimulator:
        """Build a simulator.

        Returns:
            A freshly constructed deterministic simulator.
        """
        self.built += 1
        return LinearSimulator(world_count=self._world_count)


class DriftingSimulatorFactory:
    """Builds simulators that disagree with the first one built.

    Each instance receives its position in the build order as its offset, so
    the first simulator is the reference and every later one departs from it at
    ``diverge_at_step``. That models a simulator whose output depends on
    something outside the seed — process state, device state, kernel scheduling
    — which is precisely what a determinism trial exists to detect.

    Args:
        world_count: World count for every simulator built.
        diverge_at_step: Step index from which the offset is applied.
    """

    def __init__(self, world_count: int, diverge_at_step: int) -> None:
        self._world_count = world_count
        self._diverge_at_step = diverge_at_step
        self.built = 0

    def __call__(self) -> DriftingSimulator:
        """Build a simulator offset by its position in the build order.

        Returns:
            A freshly constructed simulator whose offset is the number of
            simulators built before it.
        """
        offset = self.built
        self.built += 1
        return DriftingSimulator(
            world_count=self._world_count,
            diverge_at_step=self._diverge_at_step,
            offset=offset,
        )


class NaNSimulatorFactory:
    """Builds simulators whose observation contains NaN at a chosen step.

    Makes the canonical encoder's NaN refusal reachable from the experiment
    layer, so a trial is shown to fail on it rather than record the repetition
    as merely divergent.

    Args:
        world_count: World count for every simulator built.
        nan_at_step: Step index whose observation contains NaN.
    """

    def __init__(self, world_count: int, nan_at_step: int) -> None:
        self._world_count = world_count
        self._nan_at_step = nan_at_step

    def __call__(self) -> NaNSimulator:
        """Build a simulator.

        Returns:
            A freshly constructed simulator that emits NaN at ``nan_at_step``.
        """
        return NaNSimulator(world_count=self._world_count, nan_at_step=self._nan_at_step)


class WideningSimulatorFactory:
    """Builds simulators one world wider on each call.

    A factory is only obliged to return a fresh simulator, not an identically
    shaped one, so a mis-parametrised factory can hand successive repetitions
    different observation widths. Comparing those element-wise would silently
    compare position three of one rollout against position three of a differently
    shaped one, so the dispersion layer refuses them — and this makes that
    refusal reachable.

    Args:
        first_world_count: World count of the first simulator built.
    """

    def __init__(self, first_world_count: int) -> None:
        self._first_world_count = first_world_count
        self.built = 0

    def __call__(self) -> LinearSimulator:
        """Build a simulator one world wider than the last.

        Returns:
            A freshly constructed simulator whose world count grows with the
            build order.
        """
        world_count = self._first_world_count + self.built
        self.built += 1
        return LinearSimulator(world_count=world_count)


class EmptyWorldSimulatorFactory:
    """Builds simulators reporting an unusable world count.

    Makes the rollout's world-count rejection reachable from the experiment
    layer, so a trial is shown to propagate it rather than swallow it.

    Args:
        world_count: The invalid world count every built simulator reports.
    """

    def __init__(self, world_count: int) -> None:
        self._world_count = world_count

    def __call__(self) -> EmptyWorldSimulator:
        """Build a simulator.

        Returns:
            A freshly constructed simulator with the configured world count.
        """
        return EmptyWorldSimulator(world_count=self._world_count)


__all__ = [
    "DriftingSimulatorFactory",
    "EmptyWorldSimulatorFactory",
    "LinearSimulatorFactory",
    "NaNSimulatorFactory",
    "WideningSimulatorFactory",
]
