"""Typed boundary onto MuJoCo-Warp and its batch renderer.

Neither ``warp`` nor ``mujoco_warp`` ships a ``py.typed`` marker, so both are
bound the way every vendor in this package is: every symbol used is declared as
a Protocol, and the module is typed by the annotation it is assigned to rather
than by the import. Compiling MJCF is shared with the MJX adapter and lives in
:mod:`navprobe.adapters.mujoco_bindings`.

This is a *separate* boundary from the MJX one, not an extension of it.
MuJoCo-Warp is not JAX: ``step`` mutates its data in place and returns nothing,
``nworld`` is a parameter of ``make_data`` rather than a ``vmap`` axis, and there
is no pytree. Sharing Protocols between the two would have meant declaring a
surface neither vendor has.

The rendered output does not come back from ``render``. It is written into the
render context, which owns the pixel buffers — so the context is both an input
and the place the observation is read from.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from navprobe.adapters.mujoco_bindings import MjModelProtocol

#: Import path of the MuJoCo-Warp module.
MJWARP_MODULE = "mujoco_warp"

#: Import path of NumPy, used only to build the host array a device write takes.
NUMPY_MODULE = "numpy"

#: NumPy dtype the device's generalised-position buffer expects.
POSITION_DTYPE = "float32"


class HostArrayProtocol(Protocol):
    """A host-side array of one row per parallel world."""

    def tolist(self) -> list[list[float]]:
        """Copy the array to Python numbers.

        Returns:
            One list per world, in world order.
        """
        ...


class DeviceArrayProtocol(Protocol):
    """A device-resident array of one row per parallel world."""

    def numpy(self) -> HostArrayProtocol:
        """Copy the array back to the host.

        Returns:
            The host-side copy.
        """
        ...

    def assign(self, src: HostArrayProtocol) -> None:
        """Overwrite the device array from a host array.

        Args:
            src: Host array of matching shape and dtype.
        """
        ...


class NumpyModuleProtocol(Protocol):
    """The single NumPy entry point this adapter needs."""

    def array(self, object: Sequence[Sequence[float]], dtype: str) -> HostArrayProtocol:
        """Build a host array from nested Python numbers.

        Args:
            object: One sequence of values per world.
            dtype: NumPy dtype name the device buffer expects.

        Returns:
            The host array.
        """
        ...


class MjWarpModelProtocol(Protocol):
    """A device-resident MuJoCo-Warp model.

    ``nq`` is declared even though the adapter reads its coordinate count from
    the compiled host model instead. An empty Protocol is structurally satisfied
    by every object, which would make ``put_model``'s declared return type say
    nothing at all; one real field is what gives it content and gives the drift
    test something to assert.

    Attributes:
        nq: Number of generalised position coordinates.
    """

    nq: int


class MjWarpDataProtocol(Protocol):
    """Device-resident simulation state for every parallel world.

    Attributes:
        qpos: Generalised positions, one row per world.
    """

    qpos: DeviceArrayProtocol


class RenderContextProtocol(Protocol):
    """Buffers and acceleration structures the batch renderer writes into.

    Attributes:
        rgb_data: Packed RGBA pixels, one row per world.
        depth_data: Depth values, one row per world.
    """

    rgb_data: DeviceArrayProtocol
    depth_data: DeviceArrayProtocol


class PutModelProtocol(Protocol):
    """``mujoco_warp.put_model``."""

    def __call__(self, mjm: MjModelProtocol) -> MjWarpModelProtocol:
        """Place a compiled model on the device.

        Args:
            mjm: The compiled model.

        Returns:
            The device-resident model.
        """
        ...


class MakeDataProtocol(Protocol):
    """``mujoco_warp.make_data``."""

    def __call__(
        self,
        mjm: MjModelProtocol,
        nworld: int,
        njmax: int | None = None,
        nconmax: int | None = None,
        naconmax: int | None = None,
    ) -> MjWarpDataProtocol:
        """Allocate simulation state for a batch of worlds.

        The three capacity arguments default to ``None``, matching the vendor,
        which then sizes them from the model. That default is too small for
        contact-rich scenes, and overflowing it prints ``nefc overflow`` and
        continues — so a probe that left them unset would measure a silently
        truncated solve and report it as a determinism result.

        Args:
            mjm: The compiled model.
            nworld: Number of parallel worlds to allocate.
            njmax: Upper bound on constraints, or ``None`` for the vendor's.
            nconmax: Upper bound on contacts, or ``None`` for the vendor's.
            naconmax: Upper bound on active contacts, or ``None`` for the
                vendor's.

        Returns:
            State in the model's initial configuration, for every world.
        """
        ...


class StepProtocol(Protocol):
    """``mujoco_warp.step``, which mutates rather than returns."""

    def __call__(self, m: MjWarpModelProtocol, d: MjWarpDataProtocol) -> None:
        """Advance every world one simulation step, in place.

        Args:
            m: The device-resident model.
            d: The state to advance. Modified.
        """
        ...


class RenderProtocol(Protocol):
    """``mujoco_warp.render``, which writes into the render context."""

    def __call__(
        self, m: MjWarpModelProtocol, d: MjWarpDataProtocol, rc: RenderContextProtocol
    ) -> None:
        """Render every world's cameras into the context's buffers.

        Args:
            m: The device-resident model.
            d: The current state.
            rc: The render context. Its pixel buffers are overwritten.
        """
        ...


class CreateRenderContextProtocol(Protocol):
    """``mujoco_warp.create_render_context``."""

    def __call__(
        self,
        mjm: MjModelProtocol,
        nworld: int,
        cam_res: tuple[int, int],
        render_rgb: bool,
        render_depth: bool,
    ) -> RenderContextProtocol:
        """Build a render context and allocate its buffers.

        Only the parameters this adapter sets are declared. The vendor accepts
        many more, all of which affect the rendered image; leaving them at their
        defaults is a deliberate choice recorded here rather than an oversight,
        because a determinism measurement needs the configuration pinned and
        every declared parameter is one the probe is answerable for.

        Args:
            mjm: The compiled model.
            nworld: Number of parallel worlds. Fixed at context creation.
            cam_res: Camera resolution as ``(width, height)``.
            render_rgb: Whether to render colour.
            render_depth: Whether to render depth.

        Returns:
            The render context.
        """
        ...


class MjWarpModuleProtocol(Protocol):
    """The MuJoCo-Warp module.

    Attributes:
        put_model: Places a compiled model on the device.
        make_data: Allocates batched simulation state.
        step: Advances every world one step, in place.
        render: Renders every world into a context.
        create_render_context: Builds a render context.
    """

    put_model: PutModelProtocol
    make_data: MakeDataProtocol
    step: StepProtocol
    render: RenderProtocol
    create_render_context: CreateRenderContextProtocol


def load_mjwarp() -> MjWarpModuleProtocol:
    """Load MuJoCo-Warp behind its Protocol.

    Returns:
        The module, typed by the annotation rather than by the import.
    """
    module: MjWarpModuleProtocol = __import__(
        MJWARP_MODULE,
        fromlist=["put_model", "make_data", "step", "render", "create_render_context"],
    )
    return module


def load_numpy() -> NumpyModuleProtocol:
    """Load NumPy behind its Protocol.

    NumPy does ship type information, but it is bound the same way as the
    untyped vendors so that the one function this adapter calls is declared
    with the signature it is called with, rather than inherited from stubs whose
    ``tolist`` returns an untyped value.

    Returns:
        The module, typed by the annotation rather than by the import.
    """
    module: NumpyModuleProtocol = __import__(NUMPY_MODULE, fromlist=["array"])
    return module


__all__ = [
    "MJWARP_MODULE",
    "NUMPY_MODULE",
    "POSITION_DTYPE",
    "CreateRenderContextProtocol",
    "DeviceArrayProtocol",
    "HostArrayProtocol",
    "MakeDataProtocol",
    "MjWarpDataProtocol",
    "MjWarpModelProtocol",
    "MjWarpModuleProtocol",
    "NumpyModuleProtocol",
    "PutModelProtocol",
    "RenderContextProtocol",
    "RenderProtocol",
    "StepProtocol",
    "load_mjwarp",
    "load_numpy",
]
