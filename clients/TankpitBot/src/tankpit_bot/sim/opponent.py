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
from tankpit_bot.sim.world import SimWorldDict

DODGE_PERIOD = 4


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


__all__ = [
    "DODGE_PERIOD",
    "decide_opponent",
]
