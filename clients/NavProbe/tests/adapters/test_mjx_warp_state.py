"""Tests for the MuJoCo-Warp state adapter, driven against the real solver.

This is the adapter the package's central determinism findings are measured
with, so it is exercised against real MuJoCo-Warp with real contacts rather
than a stand-in.

Scenes come from :mod:`navprobe.scenes` rather than inline MJCF, which is the
same discipline the sweeps follow: a test that built its own scene would be
measuring a different one from the results.
"""

from __future__ import annotations

import pytest

from navprobe.adapters.mjx_warp_state import (
    MjWarpStateAdapterError,
    MjWarpStateSimulatorFactory,
)
from navprobe.canonical import encode_row
from navprobe.experiment import ProbeService
from navprobe.records import TrialSpec
from navprobe.scenes import build_scene, row_scene
from tests.adapters.models import FREE_JOINT_COORDINATE_COUNT

#: Worlds each simulator steps.
WORLD_COUNT = 2

#: Half-width of the seed-driven initial offset.
PERTURBATION = 0.01

#: Constraint headroom, generous enough that these scenes never overflow it.
CAPACITY = 4096

#: A row whose spheres never touch each other, so every contact is to the floor.
SEPARATED_ROW = row_scene(4, 0.070, 0.03, 0.005)

#: What ``mujoco-warp 3.11.0`` ships as the iterative line-search block size.
#: Pinned as a literal so a vendor bump that moves it fails this test loudly
#: rather than silently re-baselining every determinism figure measured under it.
VENDOR_DEFAULT_LINESEARCH_BLOCK_DIM = 32


def _factory(
    body_count: int = 4, spacing: float = 0.070, world_count: int = WORLD_COUNT
) -> MjWarpStateSimulatorFactory:
    """Build a state factory over a single-row scene.

    Args:
        body_count: Spheres in the row.
        spacing: Centre-to-centre distance.
        world_count: Parallel worlds.

    Returns:
        The factory.
    """
    return MjWarpStateSimulatorFactory(
        model_xml=build_scene(row_scene(body_count, spacing, 0.03, 0.005)),
        world_count=world_count,
        perturbation=PERTURBATION,
        constraint_capacity=CAPACITY,
    )


class TestFactory:
    """Tests for :class:`MjWarpStateSimulatorFactory`."""

    def test_reports_the_compiled_coordinate_count(self) -> None:
        """Four free bodies compile to four free joints' worth of coordinates."""
        assert _factory().coordinate_count == 4 * FREE_JOINT_COORDINATE_COUNT

    def test_builds_a_simulator_carrying_the_world_count(self) -> None:
        """Batch width is fixed by the factory and carried by what it builds."""
        assert _factory()().world_count == WORLD_COUNT

    def test_counts_the_simulators_it_builds(self) -> None:
        """A trial's fresh-simulator contract is observable."""
        factory = _factory()
        factory()
        factory()
        assert factory.built == 2

    def test_rejects_a_world_count_below_one(self) -> None:
        """A batch of no worlds produces no observations."""
        with pytest.raises(MjWarpStateAdapterError) as caught:
            _factory(world_count=0)()
        assert caught.value.code == "NP-WSTATE-001"

    def test_rejects_a_non_positive_perturbation(self) -> None:
        """A zero range makes every world identical."""
        factory = MjWarpStateSimulatorFactory(
            model_xml=build_scene(SEPARATED_ROW),
            world_count=WORLD_COUNT,
            perturbation=0.0,
            constraint_capacity=CAPACITY,
        )
        with pytest.raises(MjWarpStateAdapterError) as caught:
            factory()
        assert caught.value.code == "NP-WSTATE-002"

    def test_rejects_a_capacity_below_one(self) -> None:
        """A solve with no room for constraints resolves nothing."""
        factory = MjWarpStateSimulatorFactory(
            model_xml=build_scene(SEPARATED_ROW),
            world_count=WORLD_COUNT,
            perturbation=PERTURBATION,
            constraint_capacity=0,
        )
        with pytest.raises(MjWarpStateAdapterError) as caught:
            factory()
        assert caught.value.code == "NP-WSTATE-003"

    def test_pins_the_linesearch_block_size_when_given_one(self) -> None:
        """The pin reaches the device model, which is where codegen reads it.

        Asserted against the real vendor object rather than a stand-in: the
        point of pinning is that MuJoCo-Warp sees the value, and a fake would
        agree with the adapter while telling us nothing about the vendor.
        """
        factory = MjWarpStateSimulatorFactory(
            model_xml=build_scene(SEPARATED_ROW),
            world_count=WORLD_COUNT,
            perturbation=PERTURBATION,
            constraint_capacity=CAPACITY,
            linesearch_block_dim=64,
        )
        assert factory._model.block_dim.linesearch_iterative == 64

    def test_leaves_the_vendor_default_alone_when_not_given_one(self) -> None:
        """Omitting the pin must not silently impose a value.

        The default is the condition every published measurement before
        2026-08-25 was taken under, so a factory that quietly moved it would
        invalidate the comparison rather than extend it.
        """
        factory = MjWarpStateSimulatorFactory(
            model_xml=build_scene(SEPARATED_ROW),
            world_count=WORLD_COUNT,
            perturbation=PERTURBATION,
            constraint_capacity=CAPACITY,
        )
        assert factory._model.block_dim.linesearch_iterative == VENDOR_DEFAULT_LINESEARCH_BLOCK_DIM

    def test_rejects_a_linesearch_block_size_below_one(self) -> None:
        """A block of no threads is rejected here, not during codegen."""
        with pytest.raises(MjWarpStateAdapterError) as caught:
            MjWarpStateSimulatorFactory(
                model_xml=build_scene(SEPARATED_ROW),
                world_count=WORLD_COUNT,
                perturbation=PERTURBATION,
                constraint_capacity=CAPACITY,
                linesearch_block_dim=0,
            )
        assert caught.value.code == "NP-WSTATE-004"


class TestObservation:
    """What the adapter emits."""

    def test_observes_every_world_s_coordinates(self) -> None:
        """One observation carries every world's positions, world-major."""
        simulator = _factory()()
        simulator.reset(7)
        assert len(simulator.advance()) == WORLD_COUNT * 4 * FREE_JOINT_COORDINATE_COUNT

    def test_the_observation_is_canonically_encodable(self) -> None:
        """What the solver emits survives the encoder the digest is built on."""
        simulator = _factory()()
        simulator.reset(7)
        observation = simulator.advance()
        width = WORLD_COUNT * 4 * FREE_JOINT_COORDINATE_COUNT
        assert len(encode_row(observation)) == 4 + 8 * width

    def test_the_state_advances(self) -> None:
        """Two consecutive steps differ, so the solver is running."""
        simulator = _factory()()
        simulator.reset(7)
        assert simulator.advance() != simulator.advance()

    def test_reset_returns_to_the_same_starting_point(self) -> None:
        """Resetting to one seed reproduces that seed's first observation."""
        simulator = _factory()()
        simulator.reset(7)
        first = simulator.advance()
        simulator.reset(7)
        assert simulator.advance() == first

    def test_different_seeds_start_differently(self) -> None:
        """The seed reaches the solver rather than only the record."""
        left, right = _factory()(), _factory()()
        left.reset(7)
        right.reset(8)
        assert left.advance() != right.advance()


class TestSeparatedRowReproduces:
    """A row whose bodies touch only the floor reproduces on any device.

    This is the condition the package's finding says is safe, so it is the one
    that can be asserted unconditionally: it holds on the CPU device and on the
    GPU alike, which is exactly what makes it a usable regression test.
    """

    def test_a_separated_row_is_deterministic(self) -> None:
        """Repetitions of a floor-contact-only scene agree bit for bit."""
        record = ProbeService(_factory(body_count=4, spacing=0.070)).run_trial(
            TrialSpec(seed=7, step_count=40, repetitions=3)
        )
        assert record["deterministic"] is True
        assert record["first_divergent_step"] is None

    def test_a_wider_separated_row_is_also_deterministic(self) -> None:
        """The property holds as the scene grows, while contacts stay uncoupled."""
        record = ProbeService(_factory(body_count=8, spacing=0.070)).run_trial(
            TrialSpec(seed=7, step_count=40, repetitions=3)
        )
        assert record["deterministic"] is True
