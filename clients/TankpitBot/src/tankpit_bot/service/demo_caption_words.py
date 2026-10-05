"""What each of the bot's decisions means, in words a stranger can read.

The public demo shows people who may never have played TankPit a bot
playing it (operator, 2026-10-05: "if I don't know the game Idk what
it's doing"). Every decision the bot dispatches already names WHY it
was taken, as a member of the closed
:class:`~tankpit_bot.bot.ai.scoring_types.ReasonKind` vocabulary; this
module is the one translation of that vocabulary into a caption.

Each caption has two parts: ``doing``, a few words for what the tank is
doing now, and ``why``, one plain sentence for the reason. The table
covers EVERY member, and a test holds it to the enum, so a new reason
the bot learns cannot reach the page as a raw code word.

The words describe what the reason's own code does, read from the
module that emits it; where a reason is shared by both modes the words
fit either.
"""

from __future__ import annotations

from typing import Final

from typing_extensions import TypedDict

from tankpit_bot.bot.ai.scoring_types import ReasonKind


class CaptionWordsDict(TypedDict):
    """One caption's words.

    Attributes:
        doing: What the tank is doing, a few words, sentence case.
        why: Why, one plain sentence ending in a full stop.
    """

    doing: str
    why: str


def _words(doing: str, why: str) -> CaptionWordsDict:
    """Build one table entry.

    Args:
        doing: What the tank is doing.
        why: Why it is doing it.

    Returns:
        The entry.
    """
    return CaptionWordsDict(doing=doing, why=why)


DESTROYED_WORDS: Final[CaptionWordsDict] = _words(
    "Destroyed",
    "Its tank was knocked out. It comes back to life shortly and carries on.",
)
"""The caption while the bot's own tank is dead, from its
``self_deactivated`` receipt until its next decision."""

REASON_WORDS: Final[dict[ReasonKind, CaptionWordsDict]] = {
    ReasonKind.SCAN_ON_LANDING: _words(
        "Scanning with radar",
        "It has just teleported, so it sweeps the new area for enemies and supplies.",
    ),
    ReasonKind.EQUIPMENT_LOCKED: _words(
        "Heading to an equipment crate",
        "It has picked out a crate of equipment nearby and is going to collect it.",
    ),
    ReasonKind.FUEL_LOCKED: _words(
        "Heading to fuel",
        "It needs fuel and has picked out a fuel tank to collect.",
    ),
    ReasonKind.EQUIPMENT_RESTOCK: _words(
        "Restocking equipment",
        "It is short of gear it wants, such as radars, so it goes to pick some up.",
    ),
    ReasonKind.EQUIPMENT_HOP: _words(
        "Teleporting toward equipment",
        "The equipment it wants is far away, so it teleports closer to it.",
    ),
    ReasonKind.FORAGE_FRONTIER_HOP: _words(
        "Teleporting to a new area",
        "Nothing useful is left nearby, so it jumps to ground it has not searched yet.",
    ),
    ReasonKind.FUEL_HOP: _words(
        "Teleporting toward fuel",
        "It needs fuel and the fuel it knows about is far away, so it teleports closer.",
    ),
    ReasonKind.FUEL_COLLECT: _words(
        "Picking up fuel",
        "It is right beside a fuel tank and drives onto it to refuel.",
    ),
    ReasonKind.MINE_CLEARANCE_SHOT: _words(
        "Shooting a mine",
        "A mine stands between it and something it wants, so it shoots the mine away.",
    ),
    ReasonKind.MINE_PIN: _words(
        "Laying mines",
        "An enemy is right beside it, so it drops mines around itself to hem them in.",
    ),
    ReasonKind.FORAGE_RADAR: _words(
        "Scanning with radar",
        "It is looking for fuel and equipment on the ground around it.",
    ),
    ReasonKind.FORAGE_SWEEP: _words(
        "Searching the area",
        "It drives to the spot where its next radar sweep will reveal the most new ground.",
    ),
    ReasonKind.FORAGE_FRONTIER_WALK: _words(
        "Driving to new ground",
        "It heads toward ground it has not searched yet, looking for supplies.",
    ),
    ReasonKind.FORAGE_FRONTIER_PAN: _words(
        "Looking further out",
        "It moves its view, which costs nothing, to see ground it has not searched yet.",
    ),
    ReasonKind.QUAD_SWEEP_SHIFT: _words(
        "Searching the map in quarters",
        "Standing still, it turns its view to the next corner of the area to scan it.",
    ),
    ReasonKind.QUAD_SWEEP_RADAR: _words(
        "Scanning with radar",
        "It scans one corner of the area around it, one of four, without moving.",
    ),
    ReasonKind.HARVEST_FRAME_SHIFT: _words(
        "Lining up supplies",
        "It has found supplies just out of view, so it moves its view to bring them in.",
    ),
    ReasonKind.HARVEST_LEG_WALK: _words(
        "Driving to supplies",
        "It is collecting a group of supplies it found and drives to the next one.",
    ),
    ReasonKind.DESYNC_RESCAN: _words(
        "Scanning again",
        "Its picture of the area went out of date, so it scans to see what is really there.",
    ),
    ReasonKind.MINE_HIT_REVEAL_SCAN: _words(
        "Scanning for mines",
        "It just drove over a mine it could not see, so it scans to find the rest.",
    ),
    ReasonKind.SEARCH_COLLECT_LOCAL: _words(
        "Picking up nearby supplies",
        "There are supplies close by, so it collects them before going anywhere else.",
    ),
    ReasonKind.CLAIM_DENIED: _words(
        "Choosing another target",
        "Another of the bots already claimed that supply, so it picks something else.",
    ),
    ReasonKind.WALK_FOR_FUEL: _words(
        "Driving to fuel",
        "Its fuel is too low to teleport, so it drives to the nearest fuel it knows of.",
    ),
    ReasonKind.WALK_FOR_FUEL_PAN: _words(
        "Looking ahead for fuel",
        "Its fuel is too low to teleport, so it moves its view ahead to plan the drive.",
    ),
    ReasonKind.MAP_FOR_DOTS: _words(
        "Checking the map",
        "It opens the full map to see where the other tanks and the fuel are.",
    ),
    ReasonKind.AWAIT_MAP_ANSWER: _words(
        "Reading the map",
        "It has asked the game for the full map and waits for the answer.",
    ),
    ReasonKind.FERRY_SCOPE_SCOUT: _words(
        "Looking for a ferry",
        "What it wants is across water, so it looks along the shore for a ferry to ride.",
    ),
    ReasonKind.GATHERER_HOLD: _words(
        "Waiting",
        "It has collected everything nearby and waits for supplies to reappear.",
    ),
    ReasonKind.FIND_TARGET: _words(
        "Looking for an enemy",
        "It opens the map to find another tank to fight.",
    ),
    ReasonKind.FIND_ENEMIES: _words(
        "Searching for enemies",
        "It has no enemy in reach, so it checks the map for another tank to go after.",
    ),
    ReasonKind.TELEPORT_TARGET: _words(
        "Teleporting to an enemy",
        "It found an enemy tank and jumps close to it to attack.",
    ),
    ReasonKind.GREET_APPROACH: _words(
        "Going to say hello",
        "A human player is in the room, so it teleports to where they can see it.",
    ),
    ReasonKind.WALK_TO_TARGET: _words(
        "Driving toward an enemy",
        "It closes in on an enemy tank to get a clear shot.",
    ),
    ReasonKind.SHOOT_TARGET: _words(
        "Shooting at an enemy",
        "An enemy tank is in range and in line, so it fires.",
    ),
    ReasonKind.COMBAT_FRAME_SHIFT: _words(
        "Keeping the enemy in view",
        "The enemy is near the edge of its view, so it moves its view to keep them in sight.",
    ),
    ReasonKind.OPPORTUNITY_SHOT: _words(
        "Taking a shot",
        "An enemy came into line while it was busy with something else, so it fires.",
    ),
    ReasonKind.DOT_RELAY: _words(
        "Chasing an enemy",
        "The enemy is too far to reach in one jump, so it teleports part of the way, "
        "refuelling as it goes.",
    ),
    ReasonKind.HUNT_REFUEL: _words(
        "Refuelling before a fight",
        "It wants to attack but needs more fuel first, so it collects some.",
    ),
    ReasonKind.CONFIRM_KILL: _words(
        "Checking the kill",
        "It hit an enemy that may be destroyed and checks the map to make sure.",
    ),
    ReasonKind.MANUAL_HOLD: _words(
        "Holding still",
        "Its operator has told it to stay where it is.",
    ),
}
"""Every reason the bot gives, as a caption."""


__all__ = [
    "DESTROYED_WORDS",
    "REASON_WORDS",
    "CaptionWordsDict",
]
