"""Tests for the MJX adapter, driven against the real simulator.

Nothing here substitutes for MJX. The adapter's entire job is to convert a real
vendor's real output into the probe's port, so an adapter tested against a
stand-in would establish only that the stand-in was converted.

Factories are built at module scope where a test does not need a fresh one:
compiling MJCF, placing the model, and tracing the batched kernel are the
expensive part, and they are exactly the work the factory exists to do once.
"""

from __future__ import annotations

import pytest

from navprobe.adapters.mjx import MjxAdapterError, MjxSimulatorFactory
from navprobe.canonical import encode_row
from navprobe.experiment import ProbeService
from navprobe.records import TrialSpec
from tests.adapters.models import FALLING_BALL_XML, FREE_JOINT_COORDINATE_COUNT

#: Worlds every shared factory batches.
WORLD_COUNT = 4

#: Half-width of the seed-driven initial offset.
PERTURBATION = 0.05

#: Shared across tests that only read from it.
FACTORY = MjxSimulatorFactory(
    model_xml=FALLING_BALL_XML, world_count=WORLD_COUNT, perturbation=PERTURBATION
)


class TestFactory:
    """Tests for :class:`MjxSimulatorFactory`."""

    def test_compiles_the_model_and_reports_its_coordinate_count(self) -> None:
        """The factory exposes the width of one world's observation."""
        assert FACTORY.coordinate_count == FREE_JOINT_COORDINATE_COUNT

    def test_builds_a_simulator_carrying_the_configured_world_count(self) -> None:
        """Batch width is fixed by the factory, as the compiled kernel requires."""
        assert FACTORY().world_count == WORLD_COUNT

    def test_counts_the_simulators_it_builds(self) -> None:
        """A trial's fresh-simulator contract is observable from the factory."""
        factory = MjxSimulatorFactory(
            model_xml=FALLING_BALL_XML, world_count=2, perturbation=PERTURBATION
        )
        factory()
        factory()
        assert factory.built == 2

    def test_rejects_a_world_count_below_one(self) -> None:
        """A batch of no worlds produces no observations."""
        factory = MjxSimulatorFactory(
            model_xml=FALLING_BALL_XML, world_count=0, perturbation=PERTURBATION
        )
        with pytest.raises(MjxAdapterError) as caught:
            factory()
        assert caught.value.code == "NP-MJX-001"

    def test_rejects_a_non_positive_perturbation(self) -> None:
        """A zero range makes every world identical, so the batch says nothing."""
        factory = MjxSimulatorFactory(
            model_xml=FALLING_BALL_XML, world_count=WORLD_COUNT, perturbation=0.0
        )
        with pytest.raises(MjxAdapterError) as caught:
            factory()
        assert caught.value.code == "NP-MJX-002"


class TestObservation:
    """The shape and content of what the adapter emits."""

    def test_flattens_every_world_into_one_observation(self) -> None:
        """One observation carries every world's coordinates, world-major."""
        simulator = FACTORY()
        simulator.reset(7)
        assert len(simulator.advance()) == WORLD_COUNT * FREE_JOINT_COORDINATE_COUNT

    def test_observations_are_canonically_encodable(self) -> None:
        """What MJX emits survives the encoder the digest is built on.

        MJX returns single-precision values; the probe digests double-precision
        bytes. Widening is exact, so this asserts the adapter hands over
        something the instrument can actually digest rather than assuming it.
        """
        simulator = FACTORY()
        simulator.reset(7)
        observation = simulator.advance()
        expected = 4 + 8 * WORLD_COUNT * FREE_JOINT_COORDINATE_COUNT
        assert len(encode_row(observation)) == expected

    def test_the_state_advances(self) -> None:
        """Two consecutive steps differ, so the simulation is running.

        A simulator that returned its initial state forever would be reported as
        perfectly deterministic, which is the failure this rules out.
        """
        simulator = FACTORY()
        simulator.reset(7)
        assert simulator.advance() != simulator.advance()

    def test_reset_returns_to_the_same_starting_point(self) -> None:
        """Resetting to one seed reproduces that seed's first observation."""
        simulator = FACTORY()
        simulator.reset(7)
        first = simulator.advance()
        simulator.reset(7)
        assert simulator.advance() == first


class TestSeeding:
    """The seed determines the initial conditions, reproducibly."""

    def test_one_seed_gives_two_simulators_the_same_start(self) -> None:
        """Independently built simulators agree when reset to one seed."""
        left, right = FACTORY(), FACTORY()
        left.reset(7)
        right.reset(7)
        assert left.advance() == right.advance()

    def test_different_seeds_give_different_starts(self) -> None:
        """The seed reaches the simulation, rather than being recorded and ignored."""
        left, right = FACTORY(), FACTORY()
        left.reset(7)
        right.reset(8)
        assert left.advance() != right.advance()

    def test_worlds_within_a_batch_differ(self) -> None:
        """The perturbation spreads the batch, so parallel worlds are not clones.

        Identical worlds would make inter-world variability trivially zero and
        the batched measurement meaningless.
        """
        simulator = FACTORY()
        simulator.reset(7)
        observation = simulator.advance()
        first_world = observation[:FREE_JOINT_COORDINATE_COUNT]
        second_world = observation[FREE_JOINT_COORDINATE_COUNT : FREE_JOINT_COORDINATE_COUNT * 2]
        assert list(first_world) != list(second_world)


class TestTrialAgainstMjx:
    """The whole instrument, driven against MJX.

    This is the measurement the package exists to take. It is an integration
    test in the strict sense: MJCF compiled, model placed on the device, kernel
    traced and compiled, four worlds stepped in a batch, every step digested,
    repetitions compared. Nothing is substituted at any layer.
    """

    def test_mjx_reports_deterministic_across_repetitions(self) -> None:
        """Three independent rollouts at one seed agree bit for bit."""
        record = ProbeService(FACTORY).run_trial(TrialSpec(seed=7, step_count=10, repetitions=3))
        assert record["deterministic"] is True
        assert record["first_divergent_step"] is None

    def test_the_trial_records_the_batch_width(self) -> None:
        """Batch width is carried, which is what makes a sweep readable."""
        record = ProbeService(FACTORY).run_trial(TrialSpec(seed=7, step_count=4, repetitions=2))
        assert record["world_count"] == WORLD_COUNT

    def test_batch_width_changes_the_reference_digest(self) -> None:
        """A different batch width is a different measurement, not the same one.

        This is the axis a determinism sweep varies, so the digest has to move
        with it or the sweep would report identical results for every width.
        """
        narrow = MjxSimulatorFactory(
            model_xml=FALLING_BALL_XML, world_count=2, perturbation=PERTURBATION
        )
        spec = TrialSpec(seed=7, step_count=4, repetitions=2)
        wide_digest = ProbeService(FACTORY).run_trial(spec)["reference_digest"]
        assert ProbeService(narrow).run_trial(spec)["reference_digest"] != wide_digest

    def test_a_different_seed_changes_the_reference_digest(self) -> None:
        """The trial seed reaches MJX's initial conditions."""
        service = ProbeService(FACTORY)
        first = service.run_trial(TrialSpec(seed=7, step_count=4, repetitions=2))
        second = service.run_trial(TrialSpec(seed=8, step_count=4, repetitions=2))
        assert first["reference_digest"] != second["reference_digest"]
