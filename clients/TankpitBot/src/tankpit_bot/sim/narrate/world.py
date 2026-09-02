"""Narration for the commands that change shared world state.

Pure functions from a resolved outcome to the messages ONE observer
receives; see :mod:`tankpit_bot.sim.narrate.movement` for the shape and
the meaning of ``observer_id``.

Both families here are BROADCAST, which is measured rather than
assumed: a 0x42 naming a tank other than the receiving client appears
in the archive, and 45 sessions received a 0x4A having sent no block
command at all ([[recipient-policy]]).
"""

from __future__ import annotations

from tankpit_bot.protocol.constants import SUPERVISOR_ERROR_CANT_GO
from tankpit_bot.protocol.types import (
    BinaryMessage,
    BuildPickupDict,
    ChatMessageDict,
    SupervisorDict,
    TerrainUpdateDict,
)
from tankpit_bot.sim.blocks import BlockOutcomeDict
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.world import SimWorldDict


def narrate_chat(tank_id: int, command: ClientCommandDict) -> list[BinaryMessage]:
    """Narrate one chat command as the 0x4D broadcast.

    Mirrors the un-muted real server (sniff-20260729-214411): an
    accepted chat comes back as ``M + sender_id + message_id + x + y``
    to everyone, INCLUDING the sender — the echo is the client's
    delivery receipt. It takes no ``observer_id`` because every
    connection receives the identical message.

    The sim does not model the flood mute; bot policy (one greeting
    per human lock) keeps live sends far below the mute threshold.

    Args:
        tank_id: The chatting tank.
        command: The decoded chat command.

    Returns:
        The broadcast message, which every observer receives.
    """
    return [
        ChatMessageDict(
            msg_type=0x4D,
            sender_id=tank_id,
            message_type=command["message_id"],
            x=command["x"],
            y=command["y"],
        )
    ]


def narrate_block_action(
    world: SimWorldDict,
    outcome: BlockOutcomeDict,
    tank_id: int,
    observer_id: int,
) -> list[BinaryMessage]:
    """Narrate one resolved block press to a single observer.

    Success emits the 0x42 BuildPickup event plus the 0x4A tile update
    carrying the tile's post-action value, and BOTH broadcast: the
    archive holds a 0x42 naming a foreign actor, and 45 sessions
    received a 0x4A having sent no block command
    ([[recipient-policy]]). Failures answer only the presser, with the
    measured 0x52 code 1.

    Block operations are FREE — zero fuel delta measured across seven
    pick/drop pairs ([[movable-blocks]]).

    Args:
        world: Simulated world, post-press. Read only.
        outcome: The press's resolved outcome.
        tank_id: The pressing tank.
        observer_id: The connection being narrated for.

    Returns:
        The messages this observer receives, in emission order.
    """
    if outcome["kind"] in ("out_of_reach", "refused"):
        if tank_id != observer_id:
            return []
        return [
            SupervisorDict(
                msg_type=0x52,
                reset_action=1,
                close_map=0,
                error_code=SUPERVISOR_ERROR_CANT_GO,
            )
        ]
    tank = world["tanks"][tank_id]
    return [
        BuildPickupDict(
            msg_type=0x42,
            tank_id=tank_id,
            source_x=tank["x"],
            source_y=tank["y"],
            drop_x=outcome["x"],
            drop_y=outcome["y"],
            direction=outcome["direction"],
            obstacle_type=outcome["tile_value"],
            flag=0,
        ),
        TerrainUpdateDict(
            msg_type=0x4A,
            updates=[(outcome["x"], outcome["y"], outcome["tile_value"])],
        ),
    ]


__all__ = [
    "narrate_block_action",
    "narrate_chat",
]
