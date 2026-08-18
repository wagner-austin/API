"""Tests for the MuJoCo-Warp render adapter, driven against the real renderer.

Nothing is substituted. The whole question this adapter exists to answer is
whether a real raycaster over a real bounding-volume hierarchy produces the same
pixels twice, and a renderer stood in for would answer nothing.

Factories are shared where a test does not need a fresh one, and built on first
use rather than at import: compiling MJCF, placing the model and compiling the
Warp kernels are the expensive part, and acquiring a CUDA context at import time
would do it in every xdist worker during collection.
"""

from __future__ import annotations

import pytest

from navprobe.adapters.mjx_warp_render import MjWarpAdapterError, MjWarpRenderSimulatorFactory
from navprobe.canonical import encode_row
from navprobe.experiment import ProbeService
from navprobe.records import TrialSpec
from tests.adapters.models import (
    FREE_JOINT_COORDINATE_COUNT,
    RENDER_PIXEL_COUNT,
    RENDER_RESOLUTION,
    RENDERABLE_BALL_XML,
)

#: Worlds every shared factory renders.
WORLD_COUNT = 2

#: Half-width of the seed-driven initial offset.
PERTURBATION = 0.05


def _factory(channel: str, world_count: int = WORLD_COUNT) -> MjWarpRenderSimulatorFactory:
    """Build a render factory.

    Args:
        channel: Which rendered channel the observation carries.
        world_count: Number of parallel worlds.

    Returns:
        The factory.
    """
    if channel == "rgb":
        return MjWarpRenderSimulatorFactory(
            model_xml=RENDERABLE_BALL_XML,
            world_count=world_count,
            camera_resolution=RENDER_RESOLUTION,
            channel="rgb",
            perturbation=PERTURBATION,
        )
    if channel == "depth":
        return MjWarpRenderSimulatorFactory(
            model_xml=RENDERABLE_BALL_XML,
            world_count=world_count,
            camera_resolution=RENDER_RESOLUTION,
            channel="depth",
            perturbation=PERTURBATION,
        )
    return MjWarpRenderSimulatorFactory(
        model_xml=RENDERABLE_BALL_XML,
        world_count=world_count,
        camera_resolution=RENDER_RESOLUTION,
        channel="both",
        perturbation=PERTURBATION,
    )


#: Factories built on first use and reused within a worker process.
_SHARED: dict[str, MjWarpRenderSimulatorFactory] = {}


def _shared(channel: str) -> MjWarpRenderSimulatorFactory:
    """Return the shared factory for a channel, building it on first use.

    Deliberately not built at import. Constructing a factory compiles MJCF,
    places the model, and acquires a CUDA context — and module-scope code runs
    during *collection*, in every xdist worker, so an import-time factory means
    every worker races to acquire a primary context whether or not it will run
    a GPU test. Enough simultaneous acquisitions fail outright, which surfaces
    as a collection error rather than as a test failure.

    Args:
        channel: Which rendered channel the factory's simulators carry.

    Returns:
        The factory for that channel, one per channel per process.
    """
    if channel not in _SHARED:
        _SHARED[channel] = _factory(channel)
    return _SHARED[channel]


class TestFactory:
    """Tests for :class:`MjWarpRenderSimulatorFactory`."""

    def test_compiles_the_model_and_reports_its_coordinate_count(self) -> None:
        """The factory exposes the compiled model's coordinate count."""
        assert _shared("both").coordinate_count == FREE_JOINT_COORDINATE_COUNT

    def test_builds_a_simulator_carrying_the_configured_world_count(self) -> None:
        """Batch width is fixed by the factory, as the render context requires."""
        assert _shared("both")().world_count == WORLD_COUNT

    def test_counts_the_simulators_it_builds(self) -> None:
        """A trial's fresh-simulator contract is observable from the factory."""
        factory = _factory("depth")
        factory()
        factory()
        assert factory.built == 2

    def test_rejects_a_world_count_below_one(self) -> None:
        """A batch of no worlds renders no observation."""
        with pytest.raises(MjWarpAdapterError) as caught:
            _factory("depth", world_count=0)()
        assert caught.value.code == "NP-WARP-001"

    def test_rejects_a_non_positive_perturbation(self) -> None:
        """A zero range makes every world identical, so the batch says nothing."""
        factory = MjWarpRenderSimulatorFactory(
            model_xml=RENDERABLE_BALL_XML,
            world_count=WORLD_COUNT,
            camera_resolution=RENDER_RESOLUTION,
            channel="depth",
            perturbation=0.0,
        )
        with pytest.raises(MjWarpAdapterError) as caught:
            factory()
        assert caught.value.code == "NP-WARP-002"

    def test_rejects_a_zero_area_camera(self) -> None:
        """A camera with no pixels renders nothing to compare."""
        factory = MjWarpRenderSimulatorFactory(
            model_xml=RENDERABLE_BALL_XML,
            world_count=WORLD_COUNT,
            camera_resolution=(0, 32),
            channel="depth",
            perturbation=PERTURBATION,
        )
        with pytest.raises(MjWarpAdapterError) as caught:
            factory()
        assert caught.value.code == "NP-WARP-003"

    def test_rejects_a_zero_height_camera(self) -> None:
        """Both dimensions are checked, not only the first."""
        factory = MjWarpRenderSimulatorFactory(
            model_xml=RENDERABLE_BALL_XML,
            world_count=WORLD_COUNT,
            camera_resolution=(32, 0),
            channel="depth",
            perturbation=PERTURBATION,
        )
        with pytest.raises(MjWarpAdapterError) as caught:
            factory()
        assert caught.value.code == "NP-WARP-003"


class TestObservationShape:
    """What the adapter emits, per channel."""

    def test_depth_channel_carries_one_value_per_pixel_per_world(self) -> None:
        """Depth alone is width times height values for each world."""
        simulator = _factory("depth")()
        simulator.reset(7)
        assert len(simulator.advance()) == WORLD_COUNT * RENDER_PIXEL_COUNT

    def test_rgb_channel_carries_one_value_per_pixel_per_world(self) -> None:
        """Colour is one packed value per pixel, not one per component."""
        simulator = _factory("rgb")()
        simulator.reset(7)
        assert len(simulator.advance()) == WORLD_COUNT * RENDER_PIXEL_COUNT

    def test_both_channels_carry_the_sum_of_the_two(self) -> None:
        """Selecting both concatenates them per world."""
        simulator = _shared("both")()
        simulator.reset(7)
        assert len(simulator.advance()) == WORLD_COUNT * RENDER_PIXEL_COUNT * 2

    def test_the_observation_is_canonically_encodable(self) -> None:
        """Packed colour survives the encoder the digest is built on.

        Colour comes back as unsigned 32-bit integers. Widening those to the
        encoder's binary64 is exact, so this asserts the adapter hands over
        something the instrument can digest rather than assuming it.
        """
        simulator = _factory("rgb")()
        simulator.reset(7)
        observation = simulator.advance()
        expected = 4 + 8 * WORLD_COUNT * RENDER_PIXEL_COUNT
        assert len(encode_row(observation)) == expected


class TestRenderedContent:
    """The pixels are a rendering, and they move."""

    def test_the_rendered_image_changes_as_the_scene_moves(self) -> None:
        """Two consecutive steps render differently.

        A renderer returning one fixed frame would be reported as perfectly
        deterministic, which is exactly the false pass this rules out.
        """
        simulator = _factory("depth")()
        simulator.reset(7)
        assert simulator.advance() != simulator.advance()

    def test_worlds_within_a_batch_render_differently(self) -> None:
        """The batch is not a set of clones, so inter-world variance is real."""
        simulator = _factory("depth")()
        simulator.reset(7)
        observation = simulator.advance()
        assert list(observation[:RENDER_PIXEL_COUNT]) != list(
            observation[RENDER_PIXEL_COUNT : RENDER_PIXEL_COUNT * 2]
        )

    def test_the_image_is_not_uniform(self) -> None:
        """More than one distinct depth value appears, so geometry was hit.

        A camera pointed at nothing renders a constant, and a constant image
        would reproduce perfectly while measuring no geometry at all.
        """
        simulator = _factory("depth")()
        simulator.reset(7)
        first_world = simulator.advance()[:RENDER_PIXEL_COUNT]
        assert len(set(first_world)) > 1


class TestSeeding:
    """The seed determines initial conditions, reproducibly."""

    def test_one_seed_gives_two_simulators_the_same_first_frame(self) -> None:
        """Independently built simulators agree when reset to one seed."""
        left, right = _shared("both")(), _shared("both")()
        left.reset(7)
        right.reset(7)
        assert left.advance() == right.advance()

    def test_different_seeds_render_differently(self) -> None:
        """The seed reaches the rendered image, not just the recorded spec."""
        left, right = _shared("both")(), _shared("both")()
        left.reset(7)
        right.reset(8)
        assert left.advance() != right.advance()

    def test_reset_returns_to_the_same_frame(self) -> None:
        """Reset reallocates state rather than continuing from where it was."""
        simulator = _shared("both")()
        simulator.reset(7)
        first = simulator.advance()
        simulator.reset(7)
        assert simulator.advance() == first


class TestRenderedTrial:
    """The whole instrument, driven against the batch renderer.

    This is the measurement the package was built to take: the published
    determinism results for GPU-batched simulators were taken with rendering
    disabled, so the rendered observation stream a policy actually consumes had
    never been checked.
    """

    def test_the_rendered_stream_is_reproducible_across_repetitions(self) -> None:
        """Three independent rendered rollouts at one seed agree bit for bit."""
        record = ProbeService(_shared("both")).run_trial(
            TrialSpec(seed=7, step_count=10, repetitions=3)
        )
        assert record["deterministic"] is True
        assert record["first_divergent_step"] is None

    def test_depth_alone_is_reproducible(self) -> None:
        """The depth channel reproduces when measured on its own."""
        record = ProbeService(_factory("depth")).run_trial(
            TrialSpec(seed=7, step_count=10, repetitions=3)
        )
        assert record["deterministic"] is True

    def test_colour_alone_is_reproducible(self) -> None:
        """The colour channel reproduces when measured on its own.

        Asserted separately from depth so a future divergence names a channel
        rather than only a step.
        """
        record = ProbeService(_factory("rgb")).run_trial(
            TrialSpec(seed=7, step_count=10, repetitions=3)
        )
        assert record["deterministic"] is True

    def test_batch_width_changes_the_rendered_digest(self) -> None:
        """Rendering more worlds is a different measurement, not the same one."""
        spec = TrialSpec(seed=7, step_count=5, repetitions=2)
        narrow = ProbeService(_factory("depth", world_count=1)).run_trial(spec)
        wide = ProbeService(_factory("depth", world_count=4)).run_trial(spec)
        assert narrow["reference_digest"] != wide["reference_digest"]
