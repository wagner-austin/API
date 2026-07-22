"""A deterministic scripted opponent for seam soaks.

NOT a model of real enemy minds (explicitly outside the fidelity
certification, [[physics-module-roadmap]]) — a reproducible aggressor
whose only job is to exercise the client-side paths a passive world
never touches: incoming 0x53 echoes, armor absorption, the fuel
book's enemy-hit feasibility entries, the ammo book's armor channel,
and the bot's damage/retreat behavior.

The policy is a pure function of the world tick, so seam sessions
stay byte-reproducible:

- tick % 4 == 0 -> dodge one tile east/west (alternating) — a moving
  target exercises the homing queue-race on the bot's side and the
  walk-billing laws on the opponent's;
- tick % 4 in (1, 3) -> shoot at the client's current tile (a human
  positional click);
- tick % 4 == 2 -> hold (keeps the damage rate survivable).

The opponent acts only while the client is inside ITS viewport
radius — it cannot see farther than anyone else.
"""

from __future__ import annotations

from tankpit_bot.protocol.commands import CMD_MOVE, CMD_SHOOT
from tankpit_bot.sim.actions import VIEWPORT_RADIUS
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.spawn import find_open_tile_near
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank

DODGE_PERIOD = 4

REVIVE_DELAY_TICKS = 2
"""Harness respawn cadence for the scripted opponent (~4 s).

Real players respawn after deactivation, which is why real rooms
never run out of targets — the exact respawn timing/placement law is
unmeasured, so this is explicit harness policy, tuned fast enough
that the production HUNT owner's ``no_viable_targets`` exit (a fresh
map with no enemies) does not end every session at the first kill.
"""

_REVIVE_FUEL = 500
_REVIVE_COUNTS = (0, 4, 0, 2, 3)


def decide_opponent(world: SimWorldDict, enemy_id: int, client_id: int) -> ClientCommandDict | None:
    """Decide the scripted opponent's command for the coming tick.

    Args:
        world: Simulated world.
        enemy_id: The scripted tank.
        client_id: The connected client's tank.

    Returns:
        The command to queue, or None when the opponent holds (dead
        parties, client out of sight, or the hold beat).
    """
    enemy = world["tanks"][enemy_id]
    client = world["tanks"][client_id]
    if not enemy["alive"] or not client["alive"]:
        return None
    reach = max(abs(enemy["x"] - client["x"]), abs(enemy["y"] - client["y"]))
    if reach > VIEWPORT_RADIUS:
        return None
    beat = world["tick"] % DODGE_PERIOD
    if beat == 0:
        step = 1 if (world["tick"] // DODGE_PERIOD) % 2 == 0 else -1
        return ClientCommandDict(
            kind="move",
            command=CMD_MOVE,
            x=enemy["x"] + step,
            y=enemy["y"],
            target_id=0,
            slot=0,
        )
    if beat == 2:
        return None
    return ClientCommandDict(
        kind="shoot",
        command=CMD_SHOOT,
        x=client["x"],
        y=client["y"],
        target_id=0,
        slot=0,
    )


def maybe_revive_opponent(server: SimServer, enemy_id: int, client_id: int) -> int:
    """Reactivate a dead scripted opponent as a NEW tank near the client.

    Real respawns join with a NEW wire tank id (``persistent_tank_id``
    exists to bridge them), so the killed id stays a corpse and the
    replacement activates fresh: a 0x21 identity broadcast, then the
    per-tick 0x2E cadence, a map blip, and a 0x3D on viewport entry —
    all through the server's existing laws. The placement stays
    within a reachable ring band of the client because a
    corner-of-the-map respawn fails the HUNT owner's affordability
    gates and ends every session at the first kill.

    Args:
        server: The sim server (owns the world and the announcement).
        enemy_id: The scripted tank currently driven by the harness.
        client_id: The connected client (the ring-band anchor).

    Returns:
        The id of the tank the harness should drive from now on —
        unchanged while the opponent lives or revival is not due,
        else the freshly activated id.
    """
    world = server.world
    enemy = world["tanks"][enemy_id]
    if enemy["alive"] or world["tick"] % REVIVE_DELAY_TICKS != 0:
        return enemy_id
    client = world["tanks"][client_id]
    position = find_open_tile_near(
        world,
        server.terrain,
        client["x"],
        client["y"],
        world["tick"],
        min_radius=6,
        max_radius=24,
    )
    if position is None:
        return enemy_id
    new_id = max(world["tanks"]) + 1
    replacement = make_sim_tank(
        new_id, enemy["team"], enemy["rank"], position[0], position[1], _REVIVE_FUEL
    )
    replacement["counts"] = list(_REVIVE_COUNTS)
    world["tanks"][new_id] = replacement
    server.announce_tank(new_id)
    return new_id


__all__ = [
    "DODGE_PERIOD",
    "REVIVE_DELAY_TICKS",
    "decide_opponent",
    "maybe_revive_opponent",
]
