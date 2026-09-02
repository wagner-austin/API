"""Build the geometry pairs that separate MuJoCo-Warp's two narrowphases.

The scene family in :mod:`navprobe.scenes` is one shape parametrised by size.
This module is the other axis: a fixed-size scene parametrised by *what
collides with what*, because that is the variable MuJoCo-Warp's
``MJ_COLLISION_TABLE`` dispatches on, and it decides which narrowphase runs.

Measured 2026-08-30: every pair the table routes to ``CollisionType.CONVEX``
loses every contact under Warp's deterministic modes while reproducing bit for
bit, and every pair it routes to ``CollisionType.PRIMITIVE`` is untouched. A
verdict taken on a primitive pair therefore says nothing about a convex one,
which is why the pair is a value a result can cite rather than MJCF buried in
whichever script produced it.

Each pair drops a body onto a static geom. The drop height is per-pair rather
than shared, and that is not incidental: the builtin ``sphere`` mesh has a
bounding radius near 1.5 while a primitive sphere here is 0.3, so one shared
height leaves some pairs floating and never touching. A pair that makes no
contact measures nothing, and reads exactly like a pair whose contacts were
dropped -- so the heights below are the ones at which each pair is verified to
start in contact.
"""

from __future__ import annotations

from navprobe import NavProbeError

#: The falling geom, per pair name's first half.
_FALLING = {
    "mesh": '<geom name="falling" type="mesh" mesh="unit_sphere"/>',
    "sphere": '<geom name="falling" type="sphere" size=".3"/>',
    "box": '<geom name="falling" type="box" size=".3 .3 .3"/>',
}

#: The static geom it lands on, per pair name's second half.
_GROUND = {
    "plane": '<geom name="ground" type="plane" size="5 5 .01"/>',
    "box": '<body><geom name="ground" type="box" size=".7 .7 .3"/></body>',
    "mesh": '<body pos="0 0 -2"><geom name="ground" type="mesh" mesh="big_sphere"/></body>',
}

#: Height at which each pair is verified to start in contact. Established by
#: scanning rather than computed: the coarse builtin hulls do not reach as far
#: as the smooth radii their bounding spheres imply, so arithmetic on
#: ``geom_rbound`` overestimates the contact height and silently produces a
#: scene that never touches.
_START_HEIGHT = {
    "mesh_box": 1.0,
    "mesh_plane": 0.28,
    "mesh_mesh": 0.4,
    "sphere_box": 0.55,
    "sphere_plane": 0.28,
    "box_box": 0.58,
}

#: Every pair this module can build, in the order a report lists them:
#: primitive-dispatched pairs first, convex-dispatched pairs after.
COLLISION_PAIRS = (
    "sphere_plane",
    "sphere_box",
    "mesh_plane",
    "mesh_box",
    "mesh_mesh",
    "box_box",
)

#: Pairs MuJoCo-Warp 3.11.0's ``MJ_COLLISION_TABLE`` sends to the CONVEX
#: narrowphase. Declared here so a test can assert the split this module's
#: findings rest on, rather than restating it in prose.
CONVEX_PAIRS = ("mesh_box", "mesh_mesh", "box_box")


class CollisionPairError(NavProbeError):
    """A pair name does not describe a buildable scene.

    Args:
        code: Stable identifier in the ``NP-PAIR-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


def require_pair(pair: str) -> str:
    """Validate a pair name.

    Args:
        pair: The pair to check.

    Returns:
        The pair, unchanged, so a caller can validate inline.

    Raises:
        CollisionPairError: When the name is not one this module builds.
    """
    if pair not in COLLISION_PAIRS:
        raise CollisionPairError(
            "NP-PAIR-001", f"unknown collision pair {pair!r}, expected one of {COLLISION_PAIRS}"
        )
    return pair


def start_height(pair: str) -> float:
    """Report the height at which a pair starts in contact.

    Args:
        pair: The pair name.

    Returns:
        The drop height.

    Raises:
        CollisionPairError: When the name is not one this module builds.
    """
    return _START_HEIGHT[require_pair(pair)]


def build_pair(pair: str) -> str:
    """Turn a pair name into MJCF.

    Args:
        pair: One of :data:`COLLISION_PAIRS`.

    Returns:
        The MJCF document for that pair.

    Raises:
        CollisionPairError: When the name is not one this module builds.
    """
    falling, ground = require_pair(pair).split("_")
    height = _START_HEIGHT[pair]
    return f"""<mujoco>
  <option>
    <flag multiccd="enable"/>
  </option>
  <asset>
    <mesh name="unit_sphere" builtin="sphere" params="0"/>
    <mesh name="big_sphere" builtin="sphere" params="0" scale="2 2 2"/>
  </asset>
  <worldbody>
    <body name="falling_body" pos="0 0 {height}">
      <freejoint/>
      {_FALLING[falling]}
    </body>
    {_GROUND[ground]}
  </worldbody>
</mujoco>
"""


__all__ = [
    "COLLISION_PAIRS",
    "CONVEX_PAIRS",
    "CollisionPairError",
    "build_pair",
    "require_pair",
    "start_height",
]
