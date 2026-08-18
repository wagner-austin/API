"""Drive the MuJoCo-Warp batch renderer through the probe's simulator port.

This is the adapter the project was built for. The published determinism
measurements for GPU-batched simulators were taken with rendering disabled, and
the same papers name perception and sensor rendering as uncovered — so the
rendered observation stream, which is what a navigation policy actually consumes,
has never been checked. Rendering here is a raycaster over a per-step
bounding-volume hierarchy: a different numerical path from the contact solver
that was measured, on which the published result therefore says nothing.

The observation is the rendered pixels, not the physics state. That is the whole
point of this adapter: :mod:`navprobe.adapters.mjx` already measures the solver,
and a probe that rendered and then digested ``qpos`` would be measuring the
solver again with extra steps.

``reset`` reallocates state rather than rewriting fields. A determinism probe
whose reset left part of the previous rollout in place would report agreement
that belonged to the leftovers, so the expensive-but-total option is the correct
one.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal

from navprobe import NavProbeError
from navprobe.adapters.mjx_warp_bindings import (
    POSITION_DTYPE,
    MjWarpDataProtocol,
    MjWarpModelProtocol,
    MjWarpModuleProtocol,
    NumpyModuleProtocol,
    RenderContextProtocol,
    load_mjwarp,
    load_numpy,
)
from navprobe.adapters.mujoco_bindings import MjModelProtocol, load_mujoco

#: Index of the generalised coordinate the seed perturbs. Position zero is a
#: translational degree of freedom for the models this probe ships, so
#: perturbing it moves the body without producing an invalid quaternion.
PERTURBED_COORDINATE = 0


class MjWarpAdapterError(NavProbeError):
    """MuJoCo-Warp could not be driven as the probe requires.

    Args:
        code: Stable identifier in the ``NP-WARP-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


class MjWarpRenderSimulator:
    """One MuJoCo-Warp model, stepped and rendered across parallel worlds.

    Satisfies :class:`navprobe.rollout.SimulatorProtocol` structurally. Each
    instance owns its own state and its own render context: ``render`` writes
    into the context's buffers, so two simulators sharing one would overwrite
    each other's observation between the render and the read.

    Args:
        mjwarp: The MuJoCo-Warp module.
        numpy: NumPy, used to build the host array a device write takes.
        mjm: The compiled model, kept because ``reset`` reallocates from it.
        model: The device-resident model.
        world_count: Number of parallel worlds.
        camera_resolution: Camera resolution as ``(width, height)``.
        channel: Which rendered channel the observation carries.
        perturbation: Half-width of the seed-driven initial offset range.

    Raises:
        MjWarpAdapterError: When ``world_count`` is below one, ``perturbation``
            is not positive, or either camera dimension is below one.
    """

    def __init__(
        self,
        mjwarp: MjWarpModuleProtocol,
        numpy: NumpyModuleProtocol,
        mjm: MjModelProtocol,
        model: MjWarpModelProtocol,
        world_count: int,
        camera_resolution: tuple[int, int],
        channel: Literal["rgb", "depth", "both"],
        perturbation: float,
    ) -> None:
        if world_count < 1:
            raise MjWarpAdapterError(
                "NP-WARP-001", f"world_count must be one or greater, got {world_count}"
            )
        if perturbation <= 0.0:
            raise MjWarpAdapterError(
                "NP-WARP-002",
                f"perturbation must be positive, got {perturbation}; a zero range would "
                "make every world identical and the batch would carry no information",
            )
        width, height = camera_resolution
        if width < 1 or height < 1:
            raise MjWarpAdapterError(
                "NP-WARP-003",
                f"camera resolution must be at least one pixel in each dimension, "
                f"got {width}x{height}; a zero-area camera renders no observation",
            )
        self._mjwarp = mjwarp
        self._numpy = numpy
        self._mjm = mjm
        self._model = model
        self._world_count = world_count
        self._channel = channel
        self._perturbation = perturbation
        self._context = mjwarp.create_render_context(
            mjm=mjm,
            nworld=world_count,
            cam_res=camera_resolution,
            render_rgb=channel in {"rgb", "both"},
            render_depth=channel in {"depth", "both"},
        )
        self._data = self._build_data(0)

    @property
    def world_count(self) -> int:
        """Number of parallel worlds this simulator steps and renders at once.

        Returns:
            The configured world count.
        """
        return self._world_count

    def reset(self, seed: int) -> None:
        """Reallocate state from a pinned seed.

        Args:
            seed: The seed to pin. It determines every world's initial offset,
                so two simulators reset to one seed start from identical state.
        """
        self._data = self._build_data(seed)

    def advance(self) -> Sequence[float]:
        """Advance one step and render every world.

        Returns:
            The rendered observation, flattened world-major. When both channels
            are selected each world contributes its colour values followed by
            its depth values, so the two are distinguishable by position.
        """
        self._mjwarp.step(m=self._model, d=self._data)
        self._mjwarp.render(m=self._model, d=self._data, rc=self._context)
        return self._read_observation(self._context)

    def _read_observation(self, context: RenderContextProtocol) -> list[float]:
        """Copy the rendered buffers back to the host.

        Args:
            context: The context the render wrote into.

        Returns:
            The observation, flattened world-major.
        """
        if self._channel == "rgb":
            return self._flatten(context.rgb_data.numpy().tolist())
        if self._channel == "depth":
            return self._flatten(context.depth_data.numpy().tolist())
        colour = context.rgb_data.numpy().tolist()
        depth = context.depth_data.numpy().tolist()
        return self._flatten([c + d for c, d in zip(colour, depth, strict=True)])

    @staticmethod
    def _flatten(rows: list[list[float]]) -> list[float]:
        """Flatten per-world rows into one observation.

        Args:
            rows: One row of values per world.

        Returns:
            Every value, world-major. Converted to ``float`` because the colour
            buffer is unsigned integers, and the digest is defined over floats.
        """
        return [float(value) for row in rows for value in row]

    def _build_data(self, seed: int) -> MjWarpDataProtocol:
        """Allocate state and perturb each world's initial position.

        Args:
            seed: The seed determining each world's offset.

        Returns:
            Freshly allocated state with the seed's offsets applied.
        """
        data = self._mjwarp.make_data(mjm=self._mjm, nworld=self._world_count)
        generator = random.Random(seed)
        rows = data.qpos.numpy().tolist()
        for row in rows:
            row[PERTURBED_COORDINATE] = row[PERTURBED_COORDINATE] + generator.uniform(
                -self._perturbation, self._perturbation
            )
        data.qpos.assign(self._numpy.array(rows, dtype=POSITION_DTYPE))
        return data


class MjWarpRenderSimulatorFactory:
    """Builds rendering simulators that share one compiled model.

    Satisfies :class:`navprobe.experiment.SimulatorFactoryProtocol`
    structurally. Compiling MJCF and placing the model happen once; each call
    returns a simulator with its own state and its own render context.

    Args:
        model_xml: The MJCF document to compile.
        world_count: Number of parallel worlds each simulator renders.
        camera_resolution: Camera resolution as ``(width, height)``.
        channel: Which rendered channel the observation carries.
        perturbation: Half-width of the seed-driven initial offset range.
    """

    def __init__(
        self,
        model_xml: str,
        world_count: int,
        camera_resolution: tuple[int, int],
        channel: Literal["rgb", "depth", "both"],
        perturbation: float,
    ) -> None:
        self._mjwarp = load_mjwarp()
        self._numpy = load_numpy()
        self._mjm: MjModelProtocol = load_mujoco().MjModel.from_xml_string(model_xml)
        self._model = self._mjwarp.put_model(mjm=self._mjm)
        self._world_count = world_count
        self._camera_resolution = camera_resolution
        self._channel = channel
        self._perturbation = perturbation
        self.built = 0

    @property
    def coordinate_count(self) -> int:
        """Generalised coordinates the compiled model carries.

        Returns:
            The model's ``nq``.
        """
        return self._mjm.nq

    def __call__(self) -> MjWarpRenderSimulator:
        """Build a simulator sharing this factory's compiled model.

        Returns:
            A simulator with its own state and render context.

        Raises:
            MjWarpAdapterError: When this factory's configuration is unusable.
        """
        self.built += 1
        return MjWarpRenderSimulator(
            mjwarp=self._mjwarp,
            numpy=self._numpy,
            mjm=self._mjm,
            model=self._model,
            world_count=self._world_count,
            camera_resolution=self._camera_resolution,
            channel=self._channel,
            perturbation=self._perturbation,
        )


__all__ = [
    "PERTURBED_COORDINATE",
    "MjWarpAdapterError",
    "MjWarpRenderSimulator",
    "MjWarpRenderSimulatorFactory",
]
