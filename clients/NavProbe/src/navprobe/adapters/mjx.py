"""Drive MJX through the probe's simulator port.

MJX is the subject this instrument was built for. Its physics step is published
as reproducible under batched GPU execution; that measurement was taken with
rendering disabled, and nothing joins it to the rendered observation stream a
navigation policy would actually consume. Measuring either requires driving MJX
the way a user does — compiled, batched, one model shared across ``nworld``
worlds — and digesting what comes out.

Two design points are load-bearing:

* The batched, ``jit``-compiled kernel is the execution path under test, so this
  adapter compiles. A probe that stepped one world at a time in eager mode would
  reproduce perfectly and tell you nothing about the path anyone runs.
* Initial conditions come from :mod:`random` seeded per rollout, not from JAX's
  key machinery. The probe's premise is that a fixed seed reproduces across
  processes, so the perturbation itself must be reproducible by construction and
  not by assumption about a library.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from navprobe import NavProbeError
from navprobe.adapters.jax_bindings import (
    FlatArrayProtocol,
    JaxNumpyModuleProtocol,
    load_jax_numpy,
)
from navprobe.adapters.mjx_bindings import (
    BatchedMjxDataProtocol,
    BatchedStateBuilderProtocol,
    BatchedStepProtocol,
    JaxStateTransformsProtocol,
    JaxStepTransformsProtocol,
    MjxDataProtocol,
    MjxModelProtocol,
    MjxModuleProtocol,
    load_jax_state_transforms,
    load_jax_step_transforms,
    load_mjx,
)
from navprobe.adapters.mujoco_bindings import MujocoModuleProtocol, load_mujoco

#: Index of the generalised coordinate the seed perturbs. Position zero is a
#: translational degree of freedom for every model this probe ships, so
#: perturbing it moves the body without producing an invalid quaternion.
PERTURBED_COORDINATE = 0


class MjxAdapterError(NavProbeError):
    """MJX could not be driven as the probe requires.

    Args:
        code: Stable identifier in the ``NP-MJX-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


class MjxSimulator:
    """One MJX model, compiled and batched across parallel worlds.

    Satisfies :class:`navprobe.rollout.SimulatorProtocol` structurally. It is
    constructed from an already-placed model and an already-compiled step
    function so that the expensive work — compiling MJCF, moving the model to
    the device, tracing the kernel — happens once per factory rather than once
    per repetition. A repetition still gets its own instance and its own state,
    which is the isolation a trial depends on.

    Args:
        model: The device-resident model, shared with every other simulator
            built by the same factory.
        base_positions: The model's initial generalised positions, used as the
            template every world is built from.
        build_batched_state: The vmapped state builder.
        batched_step: The compiled, batched step function.
        jax_numpy: Converts host values to a device array.
        world_count: Number of parallel worlds. Fixed at construction because
            the compiled kernel is traced for one batch width.
        perturbation: Half-width of the uniform range the seed draws each
            world's initial offset from.

    Raises:
        MjxAdapterError: When ``world_count`` is below one, or ``perturbation``
            is not positive.
    """

    def __init__(
        self,
        model: MjxModelProtocol,
        base_positions: Sequence[float],
        build_batched_state: BatchedStateBuilderProtocol,
        batched_step: BatchedStepProtocol,
        jax_numpy: JaxNumpyModuleProtocol,
        world_count: int,
        perturbation: float,
    ) -> None:
        if world_count < 1:
            raise MjxAdapterError(
                "NP-MJX-001", f"world_count must be one or greater, got {world_count}"
            )
        if perturbation <= 0.0:
            raise MjxAdapterError(
                "NP-MJX-002",
                f"perturbation must be positive, got {perturbation}; a zero range would "
                "make every world identical and the batch would carry no information",
            )
        self._model = model
        self._base_positions = list(base_positions)
        self._build_batched_state = build_batched_state
        self._batched_step = batched_step
        self._jax_numpy = jax_numpy
        self._world_count = world_count
        self._perturbation = perturbation
        self._state = self._build_state(0)

    @property
    def world_count(self) -> int:
        """Number of parallel worlds this simulator steps at once.

        Returns:
            The configured world count.
        """
        return self._world_count

    def reset(self, seed: int) -> None:
        """Rebuild the batched state from a pinned seed.

        Args:
            seed: The seed to pin. It determines every world's initial offset,
                so two simulators reset to one seed start from identical state.
        """
        self._state = self._build_state(seed)

    def advance(self) -> Sequence[float]:
        """Advance every world one compiled, batched step.

        Returns:
            Every world's generalised positions, flattened world-major. Order is
            part of the contract: a reordering between runs would register as
            non-determinism, which is exactly the signal being measured.
        """
        self._state = self._batched_step(self._model, self._state)
        return [value for world in self._state.qpos.tolist() for value in world]

    def _build_state(self, seed: int) -> BatchedMjxDataProtocol:
        """Build the batched initial state for a seed.

        Args:
            seed: The seed determining each world's offset.

        Returns:
            The batched state, one row per world.
        """
        generator = random.Random(seed)
        rows: list[list[float]] = []
        for _ in range(self._world_count):
            offset = generator.uniform(-self._perturbation, self._perturbation)
            row = list(self._base_positions)
            row[PERTURBED_COORDINATE] = row[PERTURBED_COORDINATE] + offset
            rows.append(row)
        return self._build_batched_state(self._jax_numpy.asarray(rows))


class MjxSimulatorFactory:
    """Builds MJX simulators that share one compiled kernel.

    Satisfies :class:`navprobe.experiment.SimulatorFactoryProtocol`
    structurally. Compiling MJCF, placing the model, and tracing the batched
    kernel all happen once, here; each call returns a simulator with its own
    state. That split is what makes a trial affordable — a factory that
    recompiled per repetition would spend its whole runtime in the tracer — and
    it is also the honest arrangement, because sharing a compiled kernel across
    repetitions is precisely what a user does.

    Args:
        model_xml: The MJCF document to compile.
        world_count: Number of parallel worlds each simulator steps.
        perturbation: Half-width of the seed-driven initial offset range.
    """

    def __init__(self, model_xml: str, world_count: int, perturbation: float) -> None:
        mujoco: MujocoModuleProtocol = load_mujoco()
        mjx: MjxModuleProtocol = load_mjx()
        step_transforms: JaxStepTransformsProtocol = load_jax_step_transforms()
        state_transforms: JaxStateTransformsProtocol = load_jax_state_transforms()
        self._jax_numpy: JaxNumpyModuleProtocol = load_jax_numpy()
        self._model = mjx.put_model(mujoco.MjModel.from_xml_string(model_xml))
        base_data = mjx.make_data(self._model)
        self._base_positions = base_data.qpos.tolist()

        def build_one(qpos: FlatArrayProtocol) -> MjxDataProtocol:
            """Build one world's state from its generalised positions.

            Args:
                qpos: That world's generalised positions.

            Returns:
                The state for that world.
            """
            return base_data.replace(qpos=qpos)

        self._build_batched_state = state_transforms.vmap(build_one)
        self._batched_step = step_transforms.jit(step_transforms.vmap(mjx.step, in_axes=(None, 0)))
        self._world_count = world_count
        self._perturbation = perturbation
        self.built = 0

    @property
    def coordinate_count(self) -> int:
        """Generalised coordinates the compiled model carries.

        Returns:
            The model's ``nq``, which is the width of one world's observation.
        """
        return self._model.nq

    def __call__(self) -> MjxSimulator:
        """Build a simulator sharing this factory's compiled kernel.

        Returns:
            A simulator with its own batched state.

        Raises:
            MjxAdapterError: When this factory's world count or perturbation is
                unusable.
        """
        self.built += 1
        return MjxSimulator(
            model=self._model,
            base_positions=self._base_positions,
            build_batched_state=self._build_batched_state,
            batched_step=self._batched_step,
            jax_numpy=self._jax_numpy,
            world_count=self._world_count,
            perturbation=self._perturbation,
        )


__all__ = [
    "PERTURBED_COORDINATE",
    "MjxAdapterError",
    "MjxSimulator",
    "MjxSimulatorFactory",
]
