"""Narration for the shoot command and the corpse window it opens.

Pure functions from a resolved outcome to the messages ONE observer
receives; see :mod:`tankpit_bot.sim.narrate.movement` for the shape and
the meaning of ``observer_id``.

Combat is the family where the resolve/narrate boundary mattered most.
The old emitter resolved the shot, applied the radar-zero kill reward
to the killer's counts, advanced the deferred-debit and corpse clocks,
and appended the wire messages, all in one call taking the client id as
a parameter. Calling that once per connection would have fired the shot
N times ([[physics-module-roadmap]]). The world effects now belong to
:func:`tankpit_bot.sim.combat.process_shot` and
:class:`tankpit_bot.sim.combat_clock.CombatClock`; what is left here
reads the outcome and decides who hears what.
"""

from __future__ import annotations

from tankpit_bot.container.types import MineDetonationDict
from tankpit_bot.protocol.types import (
    BinaryMessage,
    DeactivationDict,
    EquipmentGainDict,
    ShootEventDict,
    TankRemoveDict,
)
from tankpit_bot.sim.combat import ShotOutcomeDict


def narrate_shot(outcome: ShotOutcomeDict, observer_id: int) -> list[BinaryMessage]:
    """Narrate one resolved shot to a single observer.

    Three of the four families here BROADCAST — the 0x53 echo and the
    0x41 kill announcement are room-wide, and the 0x45 detonation
    outnumbers placements 296 to 23 in the archive, which is only
    possible if observers see other players' blasts
    ([[recipient-policy]]). The kill reward's 0x67 does not: production
    treats any 0x67 as a SELF gain, so only the killer's own bundle
    rides this connection's wire.

    A shot draws NO inventory snapshot. The archive's 11,051 shot
    windows are 92.4% a bare 0x53 echo — the real server never answers
    a shot with 0x49 (response-shape differ 2026-08-01). Counts
    re-sync on the next 0x49-bearing event (radar extra, equipment
    gain), exactly as live.

    Args:
        outcome: The shot's resolved outcome, carrying its own
            shooter team and kill reward so nothing is re-read.
        observer_id: The connection being narrated for.

    Returns:
        The messages this observer receives, in emission order.
    """
    messages: list[BinaryMessage] = [
        ShootEventDict(
            msg_type=0x53,
            team=outcome["shooter_team"],
            shooter_id=outcome["shooter_id"],
            source_x=outcome["source_x"],
            source_y=outcome["source_y"],
            target_x=outcome["impact_x"],
            target_y=outcome["impact_y"],
            aim_x=outcome["aim_x"],
            aim_y=outcome["aim_y"],
            weapon=outcome["weapon"],
        )
    ]
    for packet in outcome["mine_cascade"]:
        messages.append(MineDetonationDict(msg_type=0x45, positions=packet))
    victim_id = outcome["victim_id"]
    if not outcome["victim_deactivated"] or victim_id is None:
        return messages
    messages.append(
        DeactivationDict(
            msg_type=0x41,
            status=1,
            victim_id=victim_id,
            promo_eligible=False,
            killer_id=outcome["shooter_id"],
            is_mine_kill=False,
        )
    )
    mercy = outcome["mercy"]
    if mercy is not None and mercy["killer_id"] == observer_id:
        messages.append(
            EquipmentGainDict(msg_type=0x67, show_message=False, gained=list(mercy["gained"]))
        )
    return messages


def narrate_corpse_removals(tank_ids: list[int]) -> list[BinaryMessage]:
    """Narrate the corpses whose 22-second window closed this tick.

    The corpse 0x58 is the same wire message a living tank's viewport
    exit draws, and the sim has always sent it to the connected client
    unconditionally. It takes no ``observer_id`` for that reason and
    that reason only: whether the real server scopes a corpse removal
    to the connections that could see the corpse is NOT measured —
    the archive question needs each receiving client's window at the
    moment of removal, which no sweep has reconstructed. Narrating it
    room-wide preserves the measured single-client behaviour exactly
    rather than inventing a scoping law ([[recipient-policy]]).

    Args:
        tank_ids: The tanks whose corpse windows closed, ascending.

    Returns:
        One 0x58 per closed window, in the same order.
    """
    return [TankRemoveDict(msg_type=0x58, tank_id=tank_id) for tank_id in tank_ids]


__all__ = [
    "narrate_corpse_removals",
    "narrate_shot",
]
