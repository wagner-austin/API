"""End-to-end pipeline tests: every layer, in sequence, against a real simulator.

Each layer of this package has its own tests, and every adapter is exercised
against a real vendor. Neither of those catches a break *between* layers — a
record that encodes fine and decodes to something the next stage rejects, or a
sweep whose entries lose a field on the way to disk. Unit coverage says each
link holds; only driving the chain says the chain does.

So these tests take the long way round on purpose. A scene is built from a
specification, compiled by MuJoCo, driven through the MuJoCo-Warp solver,
digested into a trial, recorded to disk, reloaded by path, and compared — with
assertions at the far end about values that entered at the near end.

The scenes are deliberately below the coupled-body threshold, with contacts only
to the floor. That condition reproduces on CPU and GPU alike, so these tests
assert exact agreement without depending on which device the suite runs on.
"""

from __future__ import annotations

from pathlib import Path

from navprobe.adapters.mjx_warp_state import MjWarpStateSimulatorFactory
from navprobe.codecs.dispersion import decode_dispersion_record, encode_dispersion_record
from navprobe.codecs.divergence import decode_divergence_record, encode_divergence_record
from navprobe.codecs.scene import decode_scene_spec, encode_scene_spec
from navprobe.codecs.sweep import decode_sweep, encode_sweep
from navprobe.crossprocess import compare_recordings, record_trial, trial_record_path
from navprobe.dispersion import final_observation, measure_dispersion
from navprobe.divergence import compare_observations, measure_divergence
from navprobe.experiment import ProbeService, SimulatorFactoryProtocol
from navprobe.records import ObservationRecord, SceneSpec, TrialSpec
from navprobe.scenes import bodies_touch, build_scene, row_scene
from navprobe.storage import load_observation_record, load_trial_record, save_observation_record
from navprobe.sweep import first_irreproducible, run_scene_sweep

#: Contacts to the floor only, so the trial reproduces on any device.
SEPARATED_SPACING = 0.070

#: Sphere radius; with the spacing above, neighbours never touch.
RADIUS = 0.03

#: Simulation timestep.
TIMESTEP = 0.005

#: Constraint headroom these scenes never approach.
CAPACITY = 4096

#: Small enough to keep the suite quick, long enough to resolve contact.
TRIAL = TrialSpec(seed=7, step_count=30, repetitions=2)

#: Parallel worlds every simulator carries.
WORLD_COUNT = 2


def _scene(body_count: int) -> SceneSpec:
    """Build a separated single-row scene.

    Args:
        body_count: Spheres in the row.

    Returns:
        The specification.
    """
    return row_scene(body_count, SEPARATED_SPACING, RADIUS, TIMESTEP)


def _build(model_xml: str, world_count: int) -> SimulatorFactoryProtocol:
    """Build a Warp state factory for a compiled scene.

    Satisfies :class:`navprobe.sweep.SimulatorFactoryBuilderProtocol`.

    Args:
        model_xml: The compiled scene.
        world_count: Parallel worlds.

    Returns:
        The factory.
    """
    return MjWarpStateSimulatorFactory(
        model_xml=model_xml,
        world_count=world_count,
        perturbation=0.01,
        constraint_capacity=CAPACITY,
    )


def _factory(body_count: int = 3, perturbation: float = 0.01) -> MjWarpStateSimulatorFactory:
    """Build a factory over a separated scene.

    Args:
        body_count: Spheres in the row.
        perturbation: Half-width of the seed-driven initial offset. Two
            factories differing only here produce observations of the same
            width from genuinely different initial conditions, which is what a
            divergence measurement needs.

    Returns:
        The factory.
    """
    return MjWarpStateSimulatorFactory(
        model_xml=build_scene(_scene(body_count)),
        world_count=WORLD_COUNT,
        perturbation=perturbation,
        constraint_capacity=CAPACITY,
    )


class TestSceneToVerdict:
    """Specification through compilation, simulation, and verdict."""

    def test_a_specification_drives_a_real_simulator_to_a_verdict(self) -> None:
        """The near end is five numbers; the far end is a determinism verdict."""
        record = ProbeService(_factory()).run_trial(TRIAL)
        assert record["deterministic"] is True
        assert record["first_divergent_step"] is None

    def test_the_verdict_carries_the_trial_it_came_from(self) -> None:
        """The design survives the whole chain unchanged."""
        assert ProbeService(_factory()).run_trial(TRIAL)["spec"] == TRIAL

    def test_the_scene_geometry_reaches_the_simulation(self) -> None:
        """A wider scene observes more, so the spec is not decorative."""
        narrow = ProbeService(_factory(body_count=2)).run_trial(TRIAL)
        wide = ProbeService(_factory(body_count=4)).run_trial(TRIAL)
        assert narrow["reference_digest"] != wide["reference_digest"]


class TestRecordAndReload:
    """A trial persisted, reloaded by path, and compared."""

    def test_a_recorded_trial_reloads_identically(self, tmp_path: Path) -> None:
        """Everything the verdict carries survives the round trip to disk."""
        returned = record_trial(ProbeService(_factory()), tmp_path, TRIAL)
        assert load_trial_record(trial_record_path(tmp_path)) == returned

    def test_two_recordings_of_one_scene_agree_through_files(self, tmp_path: Path) -> None:
        """The comparison side never sees a simulator, only the recordings."""
        left, right = tmp_path / "left", tmp_path / "right"
        record_trial(ProbeService(_factory()), left, TRIAL)
        record_trial(ProbeService(_factory()), right, TRIAL)
        assert compare_recordings(left, right, 0)["digests_match"] is True

    def test_a_different_scene_is_caught_through_files_alone(self, tmp_path: Path) -> None:
        """Two recordings of genuinely different scenes do not compare equal.

        The negative control for the row above. Without it, that test would pass
        for a comparison that reported agreement unconditionally.
        """
        left, right = tmp_path / "left", tmp_path / "right"
        record_trial(ProbeService(_factory(body_count=2)), left, TRIAL)
        record_trial(ProbeService(_factory(body_count=4)), right, TRIAL)
        assert compare_recordings(left, right, 0)["digests_match"] is False


class TestSweepThroughCodec:
    """A sweep run, encoded, and decoded back."""

    def test_a_sweep_over_real_scenes_reproduces_throughout(self) -> None:
        """Separated rows reproduce at every size, which is the known-safe shape."""
        entries = run_scene_sweep(_build, tuple(_scene(n) for n in (1, 2, 4)), TRIAL, WORLD_COUNT)
        assert [entry["trial"]["deterministic"] for entry in entries] == [True, True, True]
        assert first_irreproducible(entries) is None

    def test_a_sweep_survives_encoding_and_decoding(self) -> None:
        """Scene and verdict both cross the codec intact.

        The sweep row embeds a whole scene beside a whole trial, so this is the
        one place a field could be lost between two records rather than within
        one.
        """
        entries = run_scene_sweep(_build, tuple(_scene(n) for n in (1, 2)), TRIAL, WORLD_COUNT)
        assert decode_sweep(encode_sweep(entries)) == entries

    def test_a_decoded_sweep_still_answers_the_boundary_question(self) -> None:
        """A sweep read back from text is still usable, not just equal."""
        entries = run_scene_sweep(_build, tuple(_scene(n) for n in (1, 2)), TRIAL, WORLD_COUNT)
        assert first_irreproducible(decode_sweep(encode_sweep(entries))) is None

    def test_a_scene_round_trips_and_rebuilds_the_same_model(self) -> None:
        """A cited scene rebuilds byte-identical MJCF, which is the point of it."""
        spec = _scene(3)
        assert build_scene(decode_scene_spec(encode_scene_spec(spec))) == build_scene(spec)


class TestMagnitudesThroughCodec:
    """Dispersion and divergence, measured on a real simulator and persisted."""

    def test_a_reproducing_scene_disperses_by_exactly_zero(self) -> None:
        """The safe condition has no spread at all, not merely a small one."""
        record = measure_dispersion(_factory(), TRIAL["seed"], TRIAL["step_count"], 2)
        assert record["max_spread"] == 0.0

    def test_a_dispersion_record_survives_the_codec(self) -> None:
        """The magnitude that decides a finding crosses the codec exactly."""
        record = measure_dispersion(_factory(), TRIAL["seed"], TRIAL["step_count"], 2)
        assert decode_dispersion_record(encode_dispersion_record(record)) == record

    def test_two_identical_configurations_do_not_diverge(self) -> None:
        """Divergence against an identical configuration is zero throughout."""
        record = measure_divergence(_factory(), _factory(), TRIAL["seed"], TRIAL["step_count"])
        assert record["differing_elements"] == 0

    def test_two_different_configurations_do_diverge(self) -> None:
        """A genuinely different configuration is detected, not reported as zero.

        Same scene and so the same observation width — divergence refuses
        mismatched widths — but a different initial perturbation, so the two
        rollouts start apart and end apart. This is the negative control for the
        zero above.
        """
        record = measure_divergence(
            _factory(perturbation=0.01),
            _factory(perturbation=0.02),
            TRIAL["seed"],
            TRIAL["step_count"],
        )
        assert record["differing_elements"] > 0
        assert record["max_absolute_difference"] > 0.0

    def test_a_divergence_record_survives_the_codec(self) -> None:
        """The cross-environment magnitude crosses the codec exactly."""
        record = measure_divergence(_factory(), _factory(), TRIAL["seed"], TRIAL["step_count"])
        assert decode_divergence_record(encode_divergence_record(record)) == record


class TestObservationExchange:
    """The path that exists for environments which cannot share a process."""

    def test_an_observation_survives_disk_and_still_compares(self, tmp_path: Path) -> None:
        """Save, reload, compare — the exact route a cross-device measurement takes.

        The comparison is made against the reloaded values rather than the live
        ones, so a codec that rounded would show up here as a spurious
        difference.
        """
        values = tuple(final_observation(_factory()(), TRIAL["seed"], TRIAL["step_count"]))
        destination = tmp_path / "obs.txt"
        save_observation_record(
            destination,
            ObservationRecord(
                label="pipeline",
                seed=TRIAL["seed"],
                step_count=TRIAL["step_count"],
                values=values,
            ),
        )
        reloaded = load_observation_record(destination)
        assert compare_observations(values, reloaded["values"])["differing_elements"] == 0

    def test_the_reloaded_observation_keeps_its_width(self, tmp_path: Path) -> None:
        """Element count survives, so a comparison cannot silently truncate."""
        values = tuple(final_observation(_factory()(), TRIAL["seed"], TRIAL["step_count"]))
        destination = tmp_path / "obs.txt"
        save_observation_record(
            destination,
            ObservationRecord(
                label="pipeline",
                seed=TRIAL["seed"],
                step_count=TRIAL["step_count"],
                values=values,
            ),
        )
        assert len(load_observation_record(destination)["values"]) == len(values)


class TestSceneInvariantsHoldEndToEnd:
    """The scene predicates the findings rest on, checked on the real thing."""

    def test_the_pipeline_scenes_are_genuinely_separated(self) -> None:
        """These tests assert exact agreement, which requires the safe condition.

        If the spacing ever dropped below one diameter these scenes would become
        the irreproducible kind, and the assertions above would start failing on
        GPU for a reason that had nothing to do with the pipeline.
        """
        assert bodies_touch(_scene(4)) is False
