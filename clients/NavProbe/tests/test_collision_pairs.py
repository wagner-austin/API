"""Tests for the collision-pair family.

What these pin is not "the XML is well-formed" but the two properties the
convex-narrowphase finding rests on: that each pair is the geometry pairing it
claims to be, and that each one **starts in contact**.

The second matters more than it looks. A pair whose bodies never touch reports
zero contacts in every determinism mode, which reads exactly like a mode that
dropped them. Two hand-built scenes did precisely that during the 2026-08-30
work -- a primitive sphere left floating above the box, and a mesh pair placed
by arithmetic on bounding radii that the coarse hulls do not reach -- and both
were caught by a control rather than by inspection. These tests are that
control, made permanent.
"""

from __future__ import annotations

import pytest

from navprobe.adapters.mujoco_bindings import load_mujoco
from navprobe.collision_pairs import (
    COLLISION_PAIRS,
    CONVEX_PAIRS,
    CollisionPairError,
    build_pair,
    require_pair,
    start_height,
)

#: Coordinates a free joint contributes: three translations and a quaternion.
FREE_JOINT_COORDINATES = 7


def _contacts_at_rest(pair: str) -> int:
    """Compile a pair and ask MuJoCo how many contacts it starts with.

    Args:
        pair: The pair name.

    Returns:
        The contact count in the initial state.
    """
    mujoco = load_mujoco()
    model = mujoco.MjModel.from_xml_string(xml=build_pair(pair))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data.ncon


class TestRequirePair:
    """Validation of a pair name."""

    def test_returns_a_known_pair_unchanged(self) -> None:
        """A valid name passes through so callers can validate inline."""
        assert require_pair("box_box") == "box_box"

    def test_rejects_an_unknown_pair(self) -> None:
        """An unknown name fails with a stable code rather than a KeyError."""
        with pytest.raises(CollisionPairError) as caught:
            require_pair("sphere_capsule")
        assert caught.value.code == "NP-PAIR-001"


class TestStartHeight:
    """The per-pair drop heights."""

    def test_reports_the_declared_height(self) -> None:
        """The height is the value the module publishes for that pair."""
        assert start_height("box_box") == 0.58

    def test_heights_are_not_shared_across_pairs(self) -> None:
        """A single shared height is what left one pair floating.

        The builtin mesh sphere's bounding radius is near 1.5 while a primitive
        sphere here is 0.3, so one height cannot seat both.
        """
        assert start_height("mesh_box") != start_height("sphere_box")

    def test_rejects_an_unknown_pair(self) -> None:
        """Asking for an unbuildable pair's height fails the same way."""
        with pytest.raises(CollisionPairError) as caught:
            start_height("sphere_capsule")
        assert caught.value.code == "NP-PAIR-001"


class TestBuildPair:
    """The built MJCF, checked by compiling it."""

    def test_rejects_an_unknown_pair(self) -> None:
        """Building an unknown pair fails before MuJoCo sees anything."""
        with pytest.raises(CollisionPairError) as caught:
            build_pair("sphere_capsule")
        assert caught.value.code == "NP-PAIR-001"

    @pytest.mark.parametrize("pair", COLLISION_PAIRS)
    def test_every_pair_compiles_with_one_free_body(self, pair: str) -> None:
        """Each pair is one falling body against a static geom."""
        model = load_mujoco().MjModel.from_xml_string(xml=build_pair(pair))
        assert model.nq == FREE_JOINT_COORDINATES

    @pytest.mark.parametrize("pair", COLLISION_PAIRS)
    def test_every_pair_starts_in_contact(self, pair: str) -> None:
        """MuJoCo finds contacts in the initial state of every pair.

        The guard the whole family depends on. A pair reaching zero here would
        measure nothing while looking exactly like the failure under study.
        """
        assert _contacts_at_rest(pair) >= 1

    def test_the_drop_height_reaches_the_document(self) -> None:
        """The height is the one the module declares, not a literal in the XML."""
        assert f'pos="0 0 {start_height("mesh_mesh")}"' in build_pair("mesh_mesh")


class TestConvexPairs:
    """The split the finding is stated over."""

    def test_convex_pairs_are_part_of_the_family(self) -> None:
        """Every convex-dispatched pair is one this module can build."""
        assert set(CONVEX_PAIRS) <= set(COLLISION_PAIRS)

    def test_the_family_covers_both_narrowphases(self) -> None:
        """There are primitive-dispatched pairs too, or the split proves nothing.

        A family that was entirely convex could not distinguish "the convex
        narrowphase fails" from "deterministic mode fails".
        """
        assert set(COLLISION_PAIRS) - set(CONVEX_PAIRS)

    def test_the_convex_pairs_are_the_ones_measured_to_fail(self) -> None:
        """Pinned as a literal so a change to either list is deliberate."""
        assert CONVEX_PAIRS == ("mesh_box", "mesh_mesh", "box_box")
