"""The connected tanks of a field played by more than one production bot.

A sim session used to seat exactly one connected tank, the primary client
``SIM_CLIENT_ID``; everything else on the field was driven by a sim
policy. Phase 2 of the multiplayer track (board task b008ab91) seats
several production bots on one field, each over its own connection, and
this module decides who they are: where each rival stands, which side it
plays, and what account its lobby reports.

Rivals MIRROR the primary client — same rank, fuel and equipment — so a
duel measures the bot against itself rather than against a handicap, and
they alternate sides: the first rival plays :data:`RIVAL_TEAM`, the next
the primary's own team, and so on, so every bot has an enemy and, from
three bots up, an ally. Rivals keep the practice-bot name shape
(``red-<id>``), which is what lets the production bot hunt them at all:
a human-shaped name sits behind the consent gate and would never be
engaged first.
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.sim.lobby import SIM_ACCOUNT, SimAccountDict
from tankpit_bot.sim.spawn import find_open_tile_near
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank

RIVAL_TEAM = 1
"""The side the first rival plays: the scripted opponent's team."""

RIVAL_RING_MIN = 8
"""Innermost ring a rival is placed on, in tiles from the primary.

Eight keeps a rival off the primary's doorstep while leaving it inside
the 16-tile join window, so both bots see each other from tick one."""

RIVAL_RING_MAX = 14
"""Outermost ring a rival is placed on."""

MAX_FIELD_CLIENTS = 8
"""Most connected bots one field seats. Each is a full production bot
in this process, so the bound is what one machine plays at a useful
pace, not a protocol limit."""


class FieldSeatError(RuntimeError):
    """Raised when a field cannot seat the bots it was asked for."""


def seed_field_rivals(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    primary_id: int,
    count: int,
) -> tuple[int, ...]:
    """Seat ``count`` rival bots around the primary client.

    Rival ``k`` (1-based) takes id ``primary_id + k`` and the first open
    tile of the ring band around the primary, the band's walk rotated by
    ``k`` so rivals spread round the primary instead of queueing on one
    side.

    Args:
        world: Simulated world (rivals added). The primary is already
            on it.
        terrain: Static terrain for the placement search.
        primary_id: The primary client's tank id.
        count: How many rivals to seat; zero seats none.

    Returns:
        The rivals' ids, in seating order.

    Raises:
        FieldSeatError: If more than :data:`MAX_FIELD_CLIENTS` bots are
            asked for, a rival's id is already taken, or no open tile
            remains in the band.
    """
    if count + 1 > MAX_FIELD_CLIENTS:
        raise FieldSeatError(
            f"SIM_FIELD_SEATS: {count + 1} bots asked for, at most {MAX_FIELD_CLIENTS} per field"
        )
    primary = world["tanks"][primary_id]
    rivals: list[int] = []
    for k in range(1, count + 1):
        tank_id = primary_id + k
        if tank_id in world["tanks"]:
            raise FieldSeatError(f"SIM_FIELD_SEAT_TAKEN: tank id {tank_id} is already on the field")
        spot = find_open_tile_near(
            world,
            terrain,
            primary["x"],
            primary["y"],
            world["tick"] + k,
            min_radius=RIVAL_RING_MIN,
            max_radius=RIVAL_RING_MAX,
        )
        if spot is None:
            raise FieldSeatError(
                f"SIM_FIELD_NO_TILE: no open tile {RIVAL_RING_MIN}-{RIVAL_RING_MAX} tiles "
                f"from the primary at ({primary['x']},{primary['y']})"
            )
        team = RIVAL_TEAM if k % 2 == 1 else primary["team"]
        rival = make_sim_tank(tank_id, team, primary["rank"], spot[0], spot[1], primary["fuel"])
        rival["counts"] = list(primary["counts"])
        rival["enabled"] = list(primary["enabled"])
        world["tanks"][tank_id] = rival
        rivals.append(tank_id)
    return tuple(rivals)


def field_account(world: SimWorldDict, tank_id: int) -> SimAccountDict:
    """The account a rival's lobby join confirms report.

    The same account record as the primary's (:data:`SIM_ACCOUNT`) with
    the rival's own name and rank, so the lobby and the room agree about
    who joined.

    Args:
        world: Simulated world holding the rival.
        tank_id: The rival's tank id.

    Returns:
        The rival's account.
    """
    tank = world["tanks"][tank_id]
    return SimAccountDict(
        game_start=SIM_ACCOUNT["game_start"],
        name=tank["name"],
        rank=tank["rank"],
        active_forces=SIM_ACCOUNT["active_forces"],
    )


__all__ = [
    "MAX_FIELD_CLIENTS",
    "RIVAL_RING_MAX",
    "RIVAL_RING_MIN",
    "RIVAL_TEAM",
    "FieldSeatError",
    "field_account",
    "seed_field_rivals",
]
