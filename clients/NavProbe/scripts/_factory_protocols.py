"""Protocols for the adapter factories the measurement scripts load.

Split out of :mod:`scripts._test_hooks` when that module crossed its size
ceiling. The split is by role -- everything here declares the *shape of an
adapter factory*, and nothing here is rebindable state.

That distinction is the whole reason the split is safe. Tests install fakes by
rebinding attributes on :mod:`scripts._test_hooks`, so the hook VARIABLES must
stay there: a script importing a hook by value from a submodule would hold a
reference the rebinding never reaches, and the DI seam would silently stop
working while every test still passed. Types are not rebound, so moving the
Protocol declarations costs nothing and :mod:`scripts._test_hooks` re-exports
them so existing imports are unaffected.
"""

from __future__ import annotations

from typing import Protocol

from navprobe.experiment import SimulatorFactoryProtocol
from navprobe.rollout import SimulatorProtocol


class StateFactoryConstructorProtocol(Protocol):
    """Construct a MuJoCo-Warp state simulator factory for one scene."""

    def __call__(
        self,
        model_xml: str,
        world_count: int,
        perturbation: float,
        constraint_capacity: int,
        linesearch_block_dim: int | None = None,
    ) -> SimulatorFactoryProtocol:
        """Build the factory.

        Args:
            model_xml: The compiled scene's MJCF document.
            world_count: Parallel worlds each simulator carries.
            perturbation: Half-width of the seed-driven initial offset range.
            constraint_capacity: Upper bound on constraints, contacts and
                Jacobian non-zeros the allocation reserves.
            linesearch_block_dim: CUDA block size to pin the iterative
                line-search kernel to, or ``None`` for the vendor default.
                Optional so the sweeps that predate the block-size finding call
                this unchanged; declared because leaving it out would mean a
                script could not pin the one setting that decides whether a
                coupled-body scene reproduces.

        Returns:
            A factory producing freshly constructed simulators for that scene.
        """
        ...


class LoadStateFactoryProtocol(Protocol):
    """Load the MuJoCo-Warp state adapter, after Warp is initialised."""

    def __call__(self) -> StateFactoryConstructorProtocol:
        """Return the adapter's factory constructor.

        Returns:
            The constructor, typed by this declaration rather than by the
            import it comes from.
        """
        ...


class WitnessSimulatorProtocol(SimulatorProtocol, Protocol):
    """A simulator that also reports whether its scene is still interacting.

    Declared here rather than widening
    :class:`navprobe.rollout.SimulatorProtocol` because a contact is a MuJoCo
    notion and the rollout layer is vendor-agnostic on purpose: widening it
    would oblige every simulator, including the in-repo ones a trial is
    positive-controlled against, to answer a question only a physics engine
    can. The vendor adapter already provides this; this declaration is what
    lets a script depend on it without reaching past the hook boundary.
    """

    def contact_count(self) -> int:
        """Report contacts produced by the most recent step.

        Returns:
            The active contact count across the batch.
        """
        ...


class WitnessFactoryProtocol(Protocol):
    """Builds witness-capable simulators."""

    def __call__(self) -> WitnessSimulatorProtocol:
        """Construct a simulator.

        Returns:
            A simulator in its initial state, not yet reset to any seed.
        """
        ...


class WitnessFactoryConstructorProtocol(Protocol):
    """Construct a witness-capable simulator factory for one scene."""

    def __call__(
        self,
        model_xml: str,
        world_count: int,
        perturbation: float,
        constraint_capacity: int,
        linesearch_block_dim: int | None = None,
    ) -> WitnessFactoryProtocol:
        """Build the factory.

        Args:
            model_xml: The scene's MJCF document.
            world_count: Parallel worlds each simulator carries.
            perturbation: Half-width of the seed-driven initial offset range.
            constraint_capacity: Upper bound on constraints, contacts and
                Jacobian non-zeros the allocation reserves.
            linesearch_block_dim: CUDA block size to pin the iterative
                line-search kernel to, or ``None`` for the vendor default.

        Returns:
            A factory producing freshly constructed simulators for that scene.
        """
        ...


class LoadWitnessFactoryProtocol(Protocol):
    """Load the state adapter typed to expose its liveness witness."""

    def __call__(self) -> WitnessFactoryConstructorProtocol:
        """Return the adapter's factory constructor.

        Returns:
            The constructor, typed by this declaration rather than by the
            import it comes from.
        """
        ...


__all__ = [
    "LoadStateFactoryProtocol",
    "LoadWitnessFactoryProtocol",
    "StateFactoryConstructorProtocol",
    "WitnessFactoryConstructorProtocol",
    "WitnessFactoryProtocol",
    "WitnessSimulatorProtocol",
]
