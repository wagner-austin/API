"""Tests for running one trial design across a family of scenes.

The builder is injected, so these exercise the sweep against in-repo simulators
rather than a vendor. That is the point of the injection: the sweep's own logic —
holding the trial fixed, preserving order, locating the boundary — is what is
under test here, and it is the same logic that runs against MJX.
"""

from __future__ import annotations

import pytest

from navprobe.experiment import SimulatorFactoryProtocol
from navprobe.records import SceneSpec, TrialSpec
from navprobe.scenes import SceneError, row_scene
from navprobe.sweep import SweepError, first_irreproducible, run_scene_sweep
from tests.factories import DriftingSimulatorFactory, LinearSimulatorFactory

#: The trial design every sweep here applies.
TRIAL = TrialSpec(seed=7, step_count=4, repetitions=2)


class RecordingBuilder:
    """Builds deterministic factories and records the scenes it was given.

    A real implementation of :class:`navprobe.sweep.SimulatorFactoryBuilderProtocol`
    that compiles nothing, so the sweep's ordering and pass-through can be
    checked without a vendor in the way.
    """

    def __init__(self, world_count_seen: list[int] | None = None) -> None:
        self.documents: list[str] = []
        self.world_counts: list[int] = world_count_seen if world_count_seen is not None else []

    def __call__(self, model_xml: str, world_count: int) -> SimulatorFactoryProtocol:
        """Record the scene and return a deterministic factory.

        Args:
            model_xml: The compiled scene.
            world_count: Parallel worlds requested.

        Returns:
            A factory producing reproducible simulators.
        """
        self.documents.append(model_xml)
        self.world_counts.append(world_count)
        return LinearSimulatorFactory(world_count=world_count)


class DivergingAboveSizeBuilder:
    """Builds factories that reproduce below a body count and not above it.

    Models the shape of the real finding — a threshold in the scene rather than
    in the trial — so the sweep's boundary reporting is tested against a
    boundary that actually exists.

    Args:
        threshold: The body count at and above which scenes stop reproducing.
    """

    def __init__(self, threshold: int) -> None:
        self._threshold = threshold
        self.body_counts: list[int] = []

    def __call__(self, model_xml: str, world_count: int) -> SimulatorFactoryProtocol:
        """Return a reproducing or diverging factory according to the scene size.

        Args:
            model_xml: The compiled scene, counted for its bodies.
            world_count: Parallel worlds requested.

        Returns:
            A factory whose reproducibility depends on the scene's body count.
        """
        body_count = model_xml.count("<freejoint/>")
        self.body_counts.append(body_count)
        if body_count >= self._threshold:
            return DriftingSimulatorFactory(world_count=world_count, diverge_at_step=1)
        return LinearSimulatorFactory(world_count=world_count)


def _scenes(*body_counts: int) -> tuple[SceneSpec, ...]:
    """Build a family of single-row scenes.

    Args:
        body_counts: One body count per scene.

    Returns:
        The scenes, in the order given.
    """
    return tuple(row_scene(count, 0.055, 0.03, 0.005) for count in body_counts)


class TestRunSceneSweep:
    """Tests for :func:`run_scene_sweep`."""

    def test_produces_one_entry_per_scene(self) -> None:
        """Every scene asked for appears in the result."""
        entries = run_scene_sweep(RecordingBuilder(), _scenes(1, 2, 3), TRIAL, 2)
        assert len(entries) == 3

    def test_preserves_scene_order(self) -> None:
        """Entries come back in sweep order, so a boundary is readable."""
        entries = run_scene_sweep(RecordingBuilder(), _scenes(1, 2, 3), TRIAL, 2)
        assert [entry["scene"]["body_count"] for entry in entries] == [1, 2, 3]

    def test_each_entry_carries_the_scene_it_came_from(self) -> None:
        """A result is readable without the call site that produced it."""
        scenes = _scenes(5)
        entries = run_scene_sweep(RecordingBuilder(), scenes, TRIAL, 2)
        assert entries[0]["scene"] == scenes[0]

    def test_compiles_a_different_document_per_scene(self) -> None:
        """Each scene is built, rather than one being reused."""
        builder = RecordingBuilder()
        run_scene_sweep(builder, _scenes(1, 2, 3), TRIAL, 2)
        assert len(set(builder.documents)) == 3

    def test_passes_the_world_count_to_every_build(self) -> None:
        """Batch width is a sweep-level constant, not a per-scene one."""
        builder = RecordingBuilder()
        run_scene_sweep(builder, _scenes(1, 2, 3), TRIAL, 4)
        assert builder.world_counts == [4, 4, 4]

    def test_applies_the_same_trial_to_every_scene(self) -> None:
        """Holding the trial fixed is what makes the scene the only variable."""
        entries = run_scene_sweep(RecordingBuilder(), _scenes(1, 2, 3), TRIAL, 2)
        assert [entry["trial"]["spec"] for entry in entries] == [TRIAL, TRIAL, TRIAL]

    def test_reports_a_deterministic_family_as_deterministic(self) -> None:
        """The positive control passes at every scene."""
        entries = run_scene_sweep(RecordingBuilder(), _scenes(1, 2, 3), TRIAL, 2)
        assert [entry["trial"]["deterministic"] for entry in entries] == [True, True, True]

    def test_reports_a_threshold_where_one_exists(self) -> None:
        """A family that stops reproducing above a size is reported that way."""
        entries = run_scene_sweep(DivergingAboveSizeBuilder(3), _scenes(1, 2, 3, 4), TRIAL, 2)
        assert [entry["trial"]["deterministic"] for entry in entries] == [
            True,
            True,
            False,
            False,
        ]

    def test_rejects_an_empty_family(self) -> None:
        """A sweep over no scenes measures nothing."""
        with pytest.raises(SweepError) as caught:
            run_scene_sweep(RecordingBuilder(), (), TRIAL, 2)
        assert caught.value.code == "NP-SWEEP-001"

    def test_rejects_a_world_count_below_one(self) -> None:
        """A batch of no worlds produces no observations."""
        with pytest.raises(SweepError) as caught:
            run_scene_sweep(RecordingBuilder(), _scenes(1), TRIAL, 0)
        assert caught.value.code == "NP-SWEEP-002"

    def test_propagates_an_unbuildable_scene(self) -> None:
        """A bad scene fails the sweep rather than being skipped."""
        bad = SceneSpec(body_count=0, lattice_width=1, spacing=0.055, radius=0.03, timestep=0.005)
        with pytest.raises(SceneError) as caught:
            run_scene_sweep(RecordingBuilder(), (bad,), TRIAL, 2)
        assert caught.value.code == "NP-SCENE-001"


class TestFirstIrreproducible:
    """Tests for :func:`first_irreproducible`."""

    def test_returns_none_when_every_scene_reproduces(self) -> None:
        """Agreement throughout is reported as an absence."""
        entries = run_scene_sweep(RecordingBuilder(), _scenes(1, 2, 3), TRIAL, 2)
        assert first_irreproducible(entries) is None

    def test_returns_the_first_failing_scene(self) -> None:
        """The boundary is the first failure, not any failure."""
        entries = run_scene_sweep(DivergingAboveSizeBuilder(3), _scenes(1, 2, 3, 4), TRIAL, 2)
        boundary = first_irreproducible(entries)
        if boundary is None:
            raise AssertionError("expected a boundary in a family that stops reproducing")
        assert boundary["scene"]["body_count"] == 3

    def test_returns_the_first_of_several_failures(self) -> None:
        """A family that fails everywhere reports its first scene."""
        entries = run_scene_sweep(DivergingAboveSizeBuilder(1), _scenes(2, 3), TRIAL, 2)
        boundary = first_irreproducible(entries)
        if boundary is None:
            raise AssertionError("expected a boundary in a family that never reproduces")
        assert boundary["scene"]["body_count"] == 2
