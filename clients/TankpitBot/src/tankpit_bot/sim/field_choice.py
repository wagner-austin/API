"""Playing the sim on any of the shipped fields, not only field01.

The field GIF is the authoritative static terrain, and all 44 ship with
the package (``scripts/download_fields.py``), so the terrain half of
another field is already there. What is field01-specific is where the
scenarios put things: the arena's clearing, the practice layouts'
spawns and roster, the default world's hand-placed containers. On
another field those coordinates can land on rock or water.

:func:`resolve_field` turns a field name into the shipped terrain GIF's
name. :func:`settle_on_field` moves every seed that a scenario placed
on ground this field does not offer to the nearest open tile, so a
scenario keeps its shape (the rival is still near the client, the
roster still spread as the layout spread it) on terrain it was not
written for. The container population needs no settling: it is laid
by a walk over this field's own passable tiles
(:func:`~tankpit_bot.sim.world_seed.seed_field_population`).

The scenarios built around one field01 feature (the ferry lake, the
larder) and the two that replay field01's own records (a ghost capture,
the mined atlas) refuse another field rather than play a different
game under the same name.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.resources import field_gif_path
from tankpit_bot.sim.scenarios import SIM_FIELD
from tankpit_bot.sim.spawn import find_open_tile_near
from tankpit_bot.sim.world import SimWorldDict

SETTLE_RADIUS = 16
"""How far a seed may move to find open ground: one viewport."""


class FieldChoiceError(ValueError):
    """A field that cannot be played (``SIM_FIELD_*`` codes)."""


def resolve_field(image: str) -> str:
    """The shipped terrain GIF's name for a field.

    Args:
        image: The field as the server names it (``field05.gif``) or
            bare (``field05``).

    Returns:
        The terrain GIF's file name (``field05_r.gif``), as a world's
        ``field`` holds it.

    Raises:
        FieldChoiceError: If no shipped minimap answers to the name
            (``SIM_FIELD_UNKNOWN``).
    """
    name = image if image.endswith(".gif") else f"{image}.gif"
    gif = field_gif_path(name)
    if gif is None:
        raise FieldChoiceError(f"SIM_FIELD_UNKNOWN: no shipped minimap for {image!r}")
    return gif.name


def require_field_scenario(
    field: str, *, ferry: bool, larder: bool, atlas: bool, ghost: bool
) -> None:
    """Refuse the field01-bound scenarios on any other field.

    Args:
        field: The resolved terrain GIF name.
        ferry: The ferry scenario was asked for.
        larder: The larder scenario was asked for.
        atlas: A mined-atlas world was asked for.
        ghost: A ghost replay was asked for.

    Raises:
        FieldChoiceError: If one of them was asked for off field01
            (``SIM_FIELD_SCENARIO``).
    """
    if field == SIM_FIELD:
        return
    bound = [
        flag
        for flag, asked in (
            ("--ferry", ferry),
            ("--larder", larder),
            ("--from-atlas", atlas),
            ("--ghost", ghost),
        )
        if asked
    ]
    if bound:
        raise FieldChoiceError(
            f"SIM_FIELD_SCENARIO: {', '.join(bound)} replay or rely on field01"
            f" and cannot play {field}"
        )


def settle_on_field(world: SimWorldDict, terrain: _test_hooks.TerrainMapProtocol) -> int:
    """Move every seed off ground this field does not offer.

    Tanks, fuel containers and equipment standing on impassable tiles
    are each moved to the nearest open tile, ring by ring outward, the
    seeds earlier in the world settled first so later ones avoid them.

    Args:
        world: The seeded world (mutated).
        terrain: This field's terrain.

    Returns:
        How many seeds moved.

    Raises:
        FieldChoiceError: If a seed has no open tile within
            :data:`SETTLE_RADIUS` (``SIM_FIELD_UNSETTLED``).
    """
    moved = 0
    seeds = [
        *(tank for _, tank in sorted(world["tanks"].items())),
        *world["containers"],
        *world["equipment"],
    ]
    for seed in seeds:
        if terrain.is_passable(seed["x"], seed["y"]):
            continue
        spot = find_open_tile_near(
            world,
            terrain,
            seed["x"],
            seed["y"],
            world["tick"],
            min_radius=1,
            max_radius=SETTLE_RADIUS,
        )
        if spot is None:
            raise FieldChoiceError(
                f"SIM_FIELD_UNSETTLED: no open tile within {SETTLE_RADIUS}"
                f" of ({seed['x']},{seed['y']})"
            )
        seed["x"], seed["y"] = spot
        moved += 1
    return moved


__all__ = [
    "SETTLE_RADIUS",
    "FieldChoiceError",
    "require_field_scenario",
    "resolve_field",
    "settle_on_field",
]
