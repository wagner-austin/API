"""Drive MuJoCo-Warp's solver through the probe's simulator port.

The sibling of :mod:`navprobe.adapters.mjx_warp_render`, observing the physics
state instead of the rendered pixels. Both drive the same vendor through the
same bindings; they differ only in what they consider an observation, and that
difference is the whole point of having two.

Rendering depends on the state, so a rendered rollout that fails to reproduce
could be either the raycaster or the solver. Measuring the solver on its own is
what separates them — and it is what the package's central determinism findings
are measured with, so a rendered-only adapter would leave the headline result
unreproducible from this code.

``reset`` reallocates state rather than rewriting fields, for the same reason the
render adapter does: a probe whose reset left part of the previous rollout in
place would report agreement belonging to the leftovers.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from navprobe import NavProbeError
from navprobe.adapters.mjx_warp_bindings import (
    POSITION_DTYPE,
    MjWarpDataProtocol,
    MjWarpModelProtocol,
    MjWarpModuleProtocol,
    NumpyModuleProtocol,
    load_mjwarp,
    load_numpy,
)
from navprobe.adapters.mujoco_bindings import MjModelProtocol, load_mujoco

#: Index of the generalised coordinate the seed perturbs. Position zero is a
#: translational degree of freedom for the scenes this probe builds, so
#: perturbing it moves the body without producing an invalid quaternion.
PERTURBED_COORDINATE = 0

#: Index of the contact counter within ``Data.nacon``. MuJoCo-Warp declares it
#: as a one-element array shared by the whole batch rather than one entry per
#: world, so there is exactly one position to read and it is never per-world.
CONTACT_COUNT_INDEX = 0


class MjWarpStateAdapterError(NavProbeError):
    """MuJoCo-Warp's solver could not be driven as the probe requires.

    Args:
        code: Stable identifier in the ``NP-WSTATE-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


class MjWarpStateSimulator:
    """One MuJoCo-Warp model, stepped across parallel worlds.

    Satisfies :class:`navprobe.rollout.SimulatorProtocol` structurally.

    Args:
        mjwarp: The MuJoCo-Warp module.
        numpy: Builds the host array a device write takes.
        mjm: The compiled model, kept because ``reset`` reallocates from it.
        model: The device-resident model.
        world_count: Number of parallel worlds.
        perturbation: Half-width of the seed-driven initial offset range.
        constraint_capacity: Upper bound on constraints, contacts and Jacobian
            non-zeros the allocation reserves. Contact-rich scenes overflow the
            vendor's defaults, and an overflow is reported as a warning rather
            than an error — so a probe that left it to the default would measure
            a silently truncated solve.

    Raises:
        MjWarpStateAdapterError: When ``world_count`` is below one,
            ``perturbation`` is not positive, or ``constraint_capacity`` is
            below one.
    """

    def __init__(
        self,
        mjwarp: MjWarpModuleProtocol,
        numpy: NumpyModuleProtocol,
        mjm: MjModelProtocol,
        model: MjWarpModelProtocol,
        world_count: int,
        perturbation: float,
        constraint_capacity: int,
    ) -> None:
        if world_count < 1:
            raise MjWarpStateAdapterError(
                "NP-WSTATE-001", f"world_count must be one or greater, got {world_count}"
            )
        if perturbation <= 0.0:
            raise MjWarpStateAdapterError(
                "NP-WSTATE-002",
                f"perturbation must be positive, got {perturbation}; a zero range would "
                "make every world identical and the batch would carry no information",
            )
        if constraint_capacity < 1:
            raise MjWarpStateAdapterError(
                "NP-WSTATE-003",
                f"constraint_capacity must be one or greater, got {constraint_capacity}",
            )
        self._mjwarp = mjwarp
        self._numpy = numpy
        self._mjm = mjm
        self._model = model
        self._world_count = world_count
        self._perturbation = perturbation
        self._constraint_capacity = constraint_capacity
        self._data = self._build_data(0)

    @property
    def world_count(self) -> int:
        """Number of parallel worlds this simulator steps at once.

        Returns:
            The configured world count.
        """
        return self._world_count

    def reset(self, seed: int) -> None:
        """Reallocate state from a pinned seed.

        Args:
            seed: The seed to pin. It determines every world's initial offset.
        """
        self._data = self._build_data(seed)

    def advance(self) -> Sequence[float]:
        """Advance every world one step.

        Returns:
            Every world's generalised positions, flattened world-major. Order is
            part of the contract: a reordering between runs would register as
            non-determinism, which is the signal being measured.
        """
        self._mjwarp.step(m=self._model, d=self._data)
        return [value for world in self._data.qpos.numpy().tolist() for value in world]

    def contact_count(self) -> int:
        """Report how many contacts the last step produced across the batch.

        A liveness witness, not an observation. It is deliberately NOT part of
        :class:`navprobe.rollout.SimulatorProtocol`: a contact is a MuJoCo
        notion, and the rollout layer is vendor-agnostic on purpose. It lives
        here, on the vendor boundary, because this is the only layer entitled
        to know what a contact is.

        It exists because a determinism verdict compares repetitions against
        each other and never against the physics. A mode that silently stops
        generating contacts produces identical rollouts and scores
        ``deterministic: true`` -- measured on 2026-08-30, where every geometry
        pair routing to MuJoCo-Warp's convex narrowphase returned zero contacts
        under ``RUN_TO_RUN`` while reproducing bit for bit. Reading this beside
        the verdict is what separates "reproducible" from "reproducibly inert".

        Returns:
            The active contact count reported by the last :meth:`advance`, or
            by construction if no step has been taken yet.
        """
        return self._data.nacon.numpy().tolist()[CONTACT_COUNT_INDEX]

    def _build_data(self, seed: int) -> MjWarpDataProtocol:
        """Allocate state and perturb each world's initial position.

        Args:
            seed: The seed determining each world's offset.

        Returns:
            Freshly allocated state with the seed's offsets applied.
        """
        data = self._mjwarp.make_data(
            mjm=self._mjm,
            nworld=self._world_count,
            njmax=self._constraint_capacity,
            nconmax=self._constraint_capacity,
            naconmax=self._constraint_capacity,
        )
        generator = random.Random(seed)
        rows = data.qpos.numpy().tolist()
        for row in rows:
            row[PERTURBED_COORDINATE] = row[PERTURBED_COORDINATE] + generator.uniform(
                -self._perturbation, self._perturbation
            )
        data.qpos.assign(self._numpy.array(rows, dtype=POSITION_DTYPE))
        return data


class MjWarpStateSimulatorFactory:
    """Builds state simulators that share one compiled model.

    Satisfies :class:`navprobe.experiment.SimulatorFactoryProtocol` structurally,
    and :class:`navprobe.sweep.SimulatorFactoryBuilderProtocol` when partially
    applied by :func:`build_state_factory`.

    Args:
        model_xml: The MJCF document to compile.
        world_count: Number of parallel worlds each simulator steps.
        perturbation: Half-width of the seed-driven initial offset range.
        constraint_capacity: Upper bound the allocation reserves.
        linesearch_block_dim: CUDA block size to pin the iterative line-search
            kernel to, or ``None`` to leave the vendor's default in place. It
            is a constructor argument rather than something a caller reaches in
            and sets, because it must be applied between ``put_model`` and the
            first step, and because it changes a determinism verdict: at the
            vendor default of 32 a five-body touching row never reproduces, and
            at 64 it usually does.

    Raises:
        MjWarpStateAdapterError: When ``linesearch_block_dim`` is below one.
            Rejected here rather than passed through, because the vendor would
            take a nonsensical block size and fail during codegen, long after
            the value that caused it left the call site.
    """

    def __init__(
        self,
        model_xml: str,
        world_count: int,
        perturbation: float,
        constraint_capacity: int,
        linesearch_block_dim: int | None = None,
    ) -> None:
        if linesearch_block_dim is not None and linesearch_block_dim < 1:
            raise MjWarpStateAdapterError(
                "NP-WSTATE-004",
                f"linesearch_block_dim must be one or greater, got {linesearch_block_dim}",
            )
        self._mjwarp = load_mjwarp()
        self._numpy = load_numpy()
        self._mjm: MjModelProtocol = load_mujoco().MjModel.from_xml_string(model_xml)
        self._model = self._mjwarp.put_model(mjm=self._mjm)
        if linesearch_block_dim is not None:
            self._model.block_dim.linesearch_iterative = linesearch_block_dim
        self._world_count = world_count
        self._perturbation = perturbation
        self._constraint_capacity = constraint_capacity
        self.built = 0

    @property
    def coordinate_count(self) -> int:
        """Generalised coordinates the compiled model carries.

        Returns:
            The model's ``nq``.
        """
        return self._mjm.nq

    def __call__(self) -> MjWarpStateSimulator:
        """Build a simulator sharing this factory's compiled model.

        Returns:
            A simulator with its own state.

        Raises:
            MjWarpStateAdapterError: When this factory's configuration is
                unusable.
        """
        self.built += 1
        return MjWarpStateSimulator(
            mjwarp=self._mjwarp,
            numpy=self._numpy,
            mjm=self._mjm,
            model=self._model,
            world_count=self._world_count,
            perturbation=self._perturbation,
            constraint_capacity=self._constraint_capacity,
        )


__all__ = [
    "PERTURBED_COORDINATE",
    "MjWarpStateAdapterError",
    "MjWarpStateSimulator",
    "MjWarpStateSimulatorFactory",
]
