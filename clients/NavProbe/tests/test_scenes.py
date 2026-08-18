"""Tests for parametrised scene construction.

The scenes here are the ones every published determinism result was measured on,
so what these tests pin is not "the XML is well-formed" but "the geometry is the
geometry the finding refers to" — spacing decides whether bodies touch, lattice
width decides whether they stack, and both of those decided a result.

The built MJCF is also compiled by the real MuJoCo compiler rather than only
string-matched, because a scene that reads correctly and does not compile is
still not a scene.
"""

from __future__ import annotations

import pytest

from navprobe.adapters.mujoco_bindings import load_mujoco
from navprobe.records import SceneSpec
from navprobe.scenes import (
    DROP_HEIGHT,
    LAYER_HEIGHT,
    SceneError,
    bodies_touch,
    build_scene,
    layer_count,
    require_scene,
    row_scene,
)

#: Coordinates a free joint contributes per body.
FREE_JOINT_COORDINATES = 7


def _spec(
    body_count: int = 4,
    lattice_width: int = 4,
    spacing: float = 0.07,
    radius: float = 0.03,
    timestep: float = 0.005,
) -> SceneSpec:
    """Build a scene specification.

    Args:
        body_count: Number of spheres.
        lattice_width: Columns per row.
        spacing: Centre-to-centre distance.
        radius: Sphere radius.
        timestep: Simulation timestep.

    Returns:
        The specification.
    """
    return SceneSpec(
        body_count=body_count,
        lattice_width=lattice_width,
        spacing=spacing,
        radius=radius,
        timestep=timestep,
    )


class TestRequireScene:
    """Tests for :func:`require_scene`."""

    def test_returns_a_valid_specification_unchanged(self) -> None:
        """Validation is a gate, not a transform."""
        spec = _spec()
        assert require_scene(spec) == spec

    def test_rejects_no_bodies(self) -> None:
        """A scene with no spheres observes nothing."""
        with pytest.raises(SceneError) as caught:
            require_scene(_spec(body_count=0))
        assert caught.value.code == "NP-SCENE-001"

    def test_rejects_a_zero_lattice_width(self) -> None:
        """A lattice needs at least one column to place bodies in."""
        with pytest.raises(SceneError) as caught:
            require_scene(_spec(lattice_width=0))
        assert caught.value.code == "NP-SCENE-002"

    def test_rejects_a_non_positive_spacing(self) -> None:
        """Zero spacing puts every body at one point."""
        with pytest.raises(SceneError) as caught:
            require_scene(_spec(spacing=0.0))
        assert caught.value.code == "NP-SCENE-003"

    def test_rejects_a_non_positive_radius(self) -> None:
        """A sphere of no size has no contacts."""
        with pytest.raises(SceneError) as caught:
            require_scene(_spec(radius=0.0))
        assert caught.value.code == "NP-SCENE-004"

    def test_rejects_a_non_positive_timestep(self) -> None:
        """A timestep of zero never advances."""
        with pytest.raises(SceneError) as caught:
            require_scene(_spec(timestep=0.0))
        assert caught.value.code == "NP-SCENE-005"


class TestBodiesTouch:
    """Tests for :func:`bodies_touch`, the variable that decided a finding."""

    def test_spacing_below_one_diameter_touches(self) -> None:
        """Spheres closer than their diameter overlap at rest."""
        assert bodies_touch(_spec(spacing=0.055, radius=0.03)) is True

    def test_spacing_above_one_diameter_does_not(self) -> None:
        """Spheres further apart than their diameter contact only the floor."""
        assert bodies_touch(_spec(spacing=0.070, radius=0.03)) is False

    def test_spacing_of_exactly_one_diameter_does_not(self) -> None:
        """Touching is strict: exactly one diameter is not overlap."""
        assert bodies_touch(_spec(spacing=0.060, radius=0.03)) is False


class TestLayerCount:
    """Tests for :func:`layer_count`."""

    def test_a_full_single_layer(self) -> None:
        """A lattice exactly filling its grid occupies one layer."""
        assert layer_count(_spec(body_count=16, lattice_width=4)) == 1

    def test_one_body_over_starts_a_second_layer(self) -> None:
        """The boundary is where a grid overflows."""
        assert layer_count(_spec(body_count=17, lattice_width=4)) == 2

    def test_a_row_never_stacks(self) -> None:
        """A lattice as wide as its body count is a single row."""
        assert layer_count(row_scene(32, 0.055, 0.03, 0.005)) == 1


class TestRowScene:
    """Tests for :func:`row_scene`."""

    def test_width_matches_the_body_count(self) -> None:
        """A row puts every body in one line, which is what makes it a row."""
        spec = row_scene(9, 0.055, 0.03, 0.005)
        assert spec["lattice_width"] == spec["body_count"] == 9

    def test_carries_the_geometry_it_was_given(self) -> None:
        """The remaining fields pass through unchanged."""
        spec = row_scene(9, 0.055, 0.03, 0.005)
        assert (spec["spacing"], spec["radius"], spec["timestep"]) == (0.055, 0.03, 0.005)


class TestBuildScene:
    """The built MJCF, checked by compiling it."""

    def test_compiles_and_carries_one_free_joint_per_body(self) -> None:
        """MuJoCo accepts the document and finds the bodies asked for."""
        model = load_mujoco().MjModel.from_xml_string(xml=build_scene(_spec(body_count=4)))
        assert model.nq == 4 * FREE_JOINT_COORDINATES

    def test_body_count_reaches_the_compiled_model(self) -> None:
        """A different body count compiles to a different model size."""
        model = load_mujoco().MjModel.from_xml_string(xml=build_scene(_spec(body_count=9)))
        assert model.nq == 9 * FREE_JOINT_COORDINATES

    def test_a_row_compiles(self) -> None:
        """The single-row arrangement is buildable at the sizes swept."""
        model = load_mujoco().MjModel.from_xml_string(
            xml=build_scene(row_scene(32, 0.055, 0.03, 0.005))
        )
        assert model.nq == 32 * FREE_JOINT_COORDINATES

    def test_places_the_first_layer_at_the_drop_height(self) -> None:
        """Bodies start above the floor so they fall onto it."""
        assert f'{DROP_HEIGHT:.6f}"' in build_scene(_spec(body_count=4, lattice_width=4))

    def test_places_a_second_layer_above_the_first(self) -> None:
        """An overflowing lattice stacks by exactly one layer height."""
        scene = build_scene(_spec(body_count=17, lattice_width=4))
        assert f'{DROP_HEIGHT + LAYER_HEIGHT:.6f}"' in scene

    def test_a_single_layer_scene_has_only_one_height(self) -> None:
        """A row must never place a body at the second layer's height."""
        scene = build_scene(row_scene(16, 0.055, 0.03, 0.005))
        assert f'{DROP_HEIGHT + LAYER_HEIGHT:.6f}"' not in scene

    def test_the_timestep_reaches_the_document(self) -> None:
        """The integration step is part of the scene, not a caller default."""
        assert 'timestep="0.002500"' in build_scene(_spec(timestep=0.0025))

    def test_validates_before_building(self) -> None:
        """An unbuildable specification fails here rather than inside MuJoCo."""
        with pytest.raises(SceneError) as caught:
            build_scene(_spec(body_count=0))
        assert caught.value.code == "NP-SCENE-001"
