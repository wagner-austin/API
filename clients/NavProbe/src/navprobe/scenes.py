"""Build the scenes every determinism sweep is measured over.

A published determinism result is only reproducible if the scene is. Writing
MJCF inline per experiment makes each measurement a one-off: the spacing that
mattered ends up buried in a string literal, and two sweeps that look comparable
can differ in a wall position nobody recorded.

So the scene is a value. :class:`navprobe.records.SceneSpec` carries the five
numbers that define it, this module turns one into MJCF, and a result cites the
spec. Rebuilding the scene a finding was measured on is then reading four
integers off a wiki page.

The family is deliberately narrow — spheres on a plane inside four walls — and
parametrised on the axes that turned out to matter: how many bodies there are,
how wide the lattice is (which decides whether they stack), and whether the
spacing puts them in contact with each other.
"""

from __future__ import annotations

from navprobe import NavProbeError
from navprobe.records import SceneSpec

#: Gap between the outermost lattice site and the wall enclosing it.
WALL_MARGIN = 0.1

#: Half-thickness of each wall slab.
WALL_THICKNESS = 0.02

#: Half-height of each wall slab.
WALL_HEIGHT = 0.3

#: Height the lowest layer is dropped from.
DROP_HEIGHT = 0.12

#: Vertical distance between layers.
LAYER_HEIGHT = 0.08

#: Half-extent of the ground plane.
FLOOR_EXTENT = 5.0


class SceneError(NavProbeError):
    """A scene specification does not describe a buildable scene.

    Args:
        code: Stable identifier in the ``NP-SCENE-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


def require_scene(spec: SceneSpec) -> SceneSpec:
    """Validate a scene specification.

    Every field is checked at the point a scene is built rather than trusted
    from whatever produced the spec, because a decoded spec and a hand-written
    one arrive by different routes and a scene built from an invalid one fails
    somewhere inside MuJoCo's compiler instead of here.

    Args:
        spec: The specification to validate.

    Returns:
        The specification unchanged.

    Raises:
        SceneError: When a field is outside the range that describes a scene.
    """
    if spec["body_count"] < 1:
        raise SceneError(
            "NP-SCENE-001", f"body_count must be one or greater, got {spec['body_count']}"
        )
    if spec["lattice_width"] < 1:
        raise SceneError(
            "NP-SCENE-002", f"lattice_width must be one or greater, got {spec['lattice_width']}"
        )
    if spec["spacing"] <= 0.0:
        raise SceneError("NP-SCENE-003", f"spacing must be positive, got {spec['spacing']}")
    if spec["radius"] <= 0.0:
        raise SceneError("NP-SCENE-004", f"radius must be positive, got {spec['radius']}")
    if spec["timestep"] <= 0.0:
        raise SceneError("NP-SCENE-005", f"timestep must be positive, got {spec['timestep']}")
    return spec


def bodies_touch(spec: SceneSpec) -> bool:
    """Report whether neighbouring lattice sites are within contact range.

    Derived rather than stored. Whether bodies contact *each other* — as opposed
    to only the floor — is the variable that decides GPU reproducibility, and a
    stored flag could disagree with the geometry it claims to describe.

    Args:
        spec: The scene specification.

    Returns:
        ``True`` when neighbouring spheres overlap at their initial positions,
        which is when the spacing is below one diameter.
    """
    return spec["spacing"] < 2.0 * spec["radius"]


def layer_count(spec: SceneSpec) -> int:
    """Report how many layers the lattice occupies.

    Args:
        spec: The scene specification.

    Returns:
        The number of layers, which is one when every body fits in a single
        ``lattice_width`` by ``lattice_width`` grid.
    """
    per_layer = spec["lattice_width"] * spec["lattice_width"]
    return -(-spec["body_count"] // per_layer)


def _body_position(spec: SceneSpec, index: int) -> tuple[float, float, float]:
    """Locate one body's initial position in the lattice.

    Args:
        spec: The scene specification.
        index: Zero-based body number.

    Returns:
        The body's initial ``(x, y, z)``.
    """
    width = spec["lattice_width"]
    layer, within = divmod(index, width * width)
    half = spec["spacing"] * width / 2.0
    return (
        spec["spacing"] * (within % width) - half,
        spec["spacing"] * (within // width) - half,
        DROP_HEIGHT + LAYER_HEIGHT * layer,
    )


def _wall_extent(spec: SceneSpec) -> float:
    """Locate the walls enclosing the lattice.

    Args:
        spec: The scene specification.

    Returns:
        The distance from the origin to each wall.
    """
    return spec["spacing"] * spec["lattice_width"] / 2.0 + WALL_MARGIN


def build_scene(spec: SceneSpec) -> str:
    """Build the MJCF document for a scene specification.

    Args:
        spec: The specification to build. Validated first.

    Returns:
        The MJCF document.

    Raises:
        SceneError: When the specification does not describe a buildable scene.
    """
    require_scene(spec)
    edge = _wall_extent(spec)
    walls = "".join(
        f'<geom type="box" pos="{x:.6f} {y:.6f} {WALL_HEIGHT:.6f}" '
        f'size="{sx:.6f} {sy:.6f} {WALL_HEIGHT:.6f}"/>'
        for x, y, sx, sy in (
            (edge, 0.0, WALL_THICKNESS, edge),
            (-edge, 0.0, WALL_THICKNESS, edge),
            (0.0, edge, edge, WALL_THICKNESS),
            (0.0, -edge, edge, WALL_THICKNESS),
        )
    )
    bodies = ""
    for index in range(spec["body_count"]):
        x, y, z = _body_position(spec, index)
        bodies += (
            f'<body name="b{index}" pos="{x:.6f} {y:.6f} {z:.6f}">'
            f"<freejoint/>"
            f'<geom type="sphere" size="{spec["radius"]:.6f}" density="1000" rgba="1 0 0 1"/>'
            f"</body>"
        )
    return (
        f'<mujoco><option timestep="{spec["timestep"]:.6f}"/>'
        '<visual><global offwidth="64" offheight="64"/></visual>'
        "<worldbody>"
        '<light pos="0 0 3" dir="0 0 -1"/>'
        f'<geom name="floor" type="plane" size="{FLOOR_EXTENT} {FLOOR_EXTENT} 0.1" '
        'rgba="0.8 0.8 0.8 1"/>'
        f"{walls}{bodies}"
        '<camera name="cam0" pos="0 -1.2 0.6" xyaxes="1 0 0 0 0.5 1"/>'
        "</worldbody></mujoco>"
    )


def row_scene(body_count: int, spacing: float, radius: float, timestep: float) -> SceneSpec:
    """Build a single-row scene specification.

    A row is the arrangement that separates body-to-body contact from stacking:
    setting the lattice width to the body count puts every sphere in one line,
    so nothing ever rests on anything.

    Args:
        body_count: Number of spheres in the row.
        spacing: Centre-to-centre distance.
        radius: Sphere radius.
        timestep: Simulation timestep.

    Returns:
        The specification.
    """
    return SceneSpec(
        body_count=body_count,
        lattice_width=body_count,
        spacing=spacing,
        radius=radius,
        timestep=timestep,
    )


__all__ = [
    "DROP_HEIGHT",
    "FLOOR_EXTENT",
    "LAYER_HEIGHT",
    "WALL_HEIGHT",
    "WALL_MARGIN",
    "WALL_THICKNESS",
    "SceneError",
    "bodies_touch",
    "build_scene",
    "layer_count",
    "require_scene",
    "row_scene",
]
