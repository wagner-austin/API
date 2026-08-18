"""Typed boundary onto MJX, and onto the JAX transforms applied to it.

``mujoco`` ships no ``py.typed`` marker and neither does its MJX submodule, so
importing them directly would pull untyped modules into a strictly typed
package. Every symbol this package needs is therefore declared as a Protocol and
bound by assigning the imported module straight to that Protocol, which is where
the type comes from. Nothing outside these binding modules touches a vendor name.

Two neighbours own what this module does not. Compiling MJCF is the same call
for every backend and lives in :mod:`navprobe.adapters.mujoco_bindings`; JAX's
arrays and its NumPy surface live in :mod:`navprobe.adapters.jax_bindings`.

The ``vmap`` and ``jit`` Protocols *do* live here, despite being JAX's, because
they are declared over MJX's step and state types. They describe how this
package transforms MJX functions rather than JAX's general surface, so an MJX
signature change is what would invalidate them.

The Protocols are deliberately minimal: they declare the surface the adapter
actually calls, and no more. A Protocol that guessed at the rest of MJX would be
a second, unverified copy of someone else's API. What is declared here was read
off the installed package rather than remembered, and
``tests/adapters/test_mjx_bindings.py`` re-reads it on every run so a vendor
signature change fails the suite instead of failing a measurement.
"""

from __future__ import annotations

from typing import Protocol

from navprobe.adapters.jax_bindings import BatchedArrayProtocol, FlatArrayProtocol
from navprobe.adapters.mujoco_bindings import MjModelProtocol

#: Import path of the MJX submodule.
MJX_MODULE = "mujoco.mjx"

#: Import path of JAX.
JAX_MODULE = "jax"


class MjxModelProtocol(Protocol):
    """A device-resident model.

    Attributes:
        nq: Number of generalised position coordinates, mirroring the compiled
            model it was placed from.
    """

    nq: int


class BatchedMjxDataProtocol(Protocol):
    """Simulation state batched across parallel worlds."""

    @property
    def qpos(self) -> BatchedArrayProtocol:
        """Generalised positions, one row per world.

        Returns:
            The batched position array.
        """
        ...


class MjxDataProtocol(Protocol):
    """Simulation state for a single world."""

    @property
    def qpos(self) -> FlatArrayProtocol:
        """Generalised positions.

        Returns:
            The position array.
        """
        ...

    def replace(self, *, qpos: FlatArrayProtocol) -> MjxDataProtocol:
        """Return a copy carrying different generalised positions.

        Args:
            qpos: The positions the copy should carry.

        Returns:
            The updated state.
        """
        ...


class PutModelProtocol(Protocol):
    """``mjx.put_model``."""

    def __call__(self, m: MjModelProtocol) -> MjxModelProtocol:
        """Place a compiled model on the device.

        Args:
            m: The compiled model.

        Returns:
            The device-resident model.
        """
        ...


class MakeDataProtocol(Protocol):
    """``mjx.make_data``."""

    def __call__(self, m: MjxModelProtocol) -> MjxDataProtocol:
        """Allocate simulation state for a model.

        Args:
            m: The device-resident model.

        Returns:
            State in the model's initial configuration.
        """
        ...


class StepProtocol(Protocol):
    """``mjx.step``, for one world."""

    def __call__(self, m: MjxModelProtocol, d: MjxDataProtocol) -> MjxDataProtocol:
        """Advance one simulation step.

        Args:
            m: The device-resident model.
            d: The current state.

        Returns:
            The state after one step.
        """
        ...


class BatchedStepProtocol(Protocol):
    """``mjx.step`` after ``vmap`` and ``jit``, for every world at once."""

    def __call__(self, m: MjxModelProtocol, d: BatchedMjxDataProtocol) -> BatchedMjxDataProtocol:
        """Advance every world one simulation step.

        Args:
            m: The device-resident model, shared across worlds.
            d: The batched state.

        Returns:
            The batched state after one step.
        """
        ...


class MjxModuleProtocol(Protocol):
    """The MJX module.

    Attributes:
        put_model: Places a compiled model on the device.
        make_data: Allocates simulation state.
        step: Advances one world one step.
    """

    put_model: PutModelProtocol
    make_data: MakeDataProtocol
    step: StepProtocol


class StateBuilderProtocol(Protocol):
    """Builds one world's state from that world's generalised positions."""

    def __call__(self, qpos: FlatArrayProtocol) -> MjxDataProtocol:
        """Build a single world's state.

        Args:
            qpos: That world's generalised positions.

        Returns:
            The state for that world.
        """
        ...


class BatchedStateBuilderProtocol(Protocol):
    """Builds every world's state at once, after ``vmap``."""

    def __call__(self, qpos: BatchedArrayProtocol) -> BatchedMjxDataProtocol:
        """Build the batched state.

        Args:
            qpos: Generalised positions, one row per world.

        Returns:
            The batched state, with every pytree leaf carrying a batch axis.
        """
        ...


class StepVmapProtocol(Protocol):
    """``jax.vmap`` as used to batch the step function."""

    def __call__(self, fun: StepProtocol, *, in_axes: tuple[None, int]) -> BatchedStepProtocol:
        """Vectorise the step function over its second argument.

        Args:
            fun: The single-world step function.
            in_axes: Which axis of each argument to map over. ``None`` for the
                model, because one model is shared by every world.

        Returns:
            The batched step function.
        """
        ...


class StateVmapProtocol(Protocol):
    """``jax.vmap`` as used to batch state construction.

    Batching by replacing one field of an unbatched state does not work: MJX's
    data type is a JAX pytree, and ``vmap`` requires *every* leaf to carry a
    batch axis. Vectorising the construction is what gives the untouched leaves
    theirs, because ``vmap`` broadcasts the values a mapped function closes over.
    """

    def __call__(self, fun: StateBuilderProtocol) -> BatchedStateBuilderProtocol:
        """Vectorise state construction over its only argument.

        Args:
            fun: The single-world state builder.

        Returns:
            The batched state builder.
        """
        ...


class JitProtocol(Protocol):
    """``jax.jit``."""

    def __call__(self, fun: BatchedStepProtocol) -> BatchedStepProtocol:
        """Compile the batched step function.

        Compilation is not an optimisation here. The compiled batched kernel is
        the execution path whose reproducibility is under test, so a probe that
        skipped it would measure something the user never runs.

        Args:
            fun: The batched step function.

        Returns:
            The compiled batched step function.
        """
        ...


class JaxStepTransformsProtocol(Protocol):
    """JAX's transforms, typed for the step function.

    Attributes:
        vmap: Vectorising transform, over model and state.
        jit: Compiling transform.
    """

    vmap: StepVmapProtocol
    jit: JitProtocol


class JaxStateTransformsProtocol(Protocol):
    """JAX's transforms, typed for state construction.

    A second view of the same module. ``jax.vmap`` is genuinely polymorphic —
    it takes any callable and returns its batched form — and this package calls
    it two ways. Each Protocol declares one of them precisely rather than one
    Protocol declaring both loosely, because a signature loose enough to cover
    both would not check either.

    Attributes:
        vmap: Vectorising transform, over generalised positions.
    """

    vmap: StateVmapProtocol


def load_mjx() -> MjxModuleProtocol:
    """Load the MJX module behind its Protocol.

    Returns:
        The module, typed by the annotation rather than by the import.
    """
    module: MjxModuleProtocol = __import__(MJX_MODULE, fromlist=["put_model", "make_data", "step"])
    return module


def load_jax_step_transforms() -> JaxStepTransformsProtocol:
    """Load JAX's transforms typed for batching the step function.

    Returns:
        The module, typed by the annotation rather than by the import.
    """
    module: JaxStepTransformsProtocol = __import__(JAX_MODULE, fromlist=["jit", "vmap"])
    return module


def load_jax_state_transforms() -> JaxStateTransformsProtocol:
    """Load JAX's transforms typed for batching state construction.

    Returns:
        The module, typed by the annotation rather than by the import.
    """
    module: JaxStateTransformsProtocol = __import__(JAX_MODULE, fromlist=["vmap"])
    return module


__all__ = [
    "JAX_MODULE",
    "MJX_MODULE",
    "BatchedMjxDataProtocol",
    "BatchedStateBuilderProtocol",
    "BatchedStepProtocol",
    "JaxStateTransformsProtocol",
    "JaxStepTransformsProtocol",
    "JitProtocol",
    "MakeDataProtocol",
    "MjxDataProtocol",
    "MjxModelProtocol",
    "MjxModuleProtocol",
    "PutModelProtocol",
    "StateBuilderProtocol",
    "StateVmapProtocol",
    "StepProtocol",
    "StepVmapProtocol",
    "load_jax_state_transforms",
    "load_jax_step_transforms",
    "load_mjx",
]
