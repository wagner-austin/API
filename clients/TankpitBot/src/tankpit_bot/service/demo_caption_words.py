"""What each of the bot's decisions means, in words a stranger can read.

The public demo shows people who may never have played TankPit a bot
playing it (operator, 2026-10-05: "if I don't know the game Idk what
it's doing"). Every decision the bot dispatches already names WHY it
was taken, as a member of the closed
:class:`~tankpit_bot.bot.ai.scoring_types.ReasonKind` vocabulary; this
module is the one translation of that vocabulary into a caption.

Each decision caption has two parts: ``doing``, a ``Next:`` preview of the
command just sent, and ``why``, one plain sentence explaining the plan. The table
covers EVERY member, and a test holds it to the enum, so a new reason
the bot learns cannot reach the page as a raw code word.

The words describe what the reason's own code does, read from the
module that emits it; where a reason is shared by both modes the words
fit either.
"""

from __future__ import annotations

from typing import Final

from platform_core.json_utils import JSONObject, require_str
from typing_extensions import TypedDict

from tankpit_bot.bot.ai.scoring_types import ReasonKind


class CaptionWordsDict(TypedDict):
    """One caption's words.

    Attributes:
        doing: A Next action preview, or the actual lifecycle state.
        why: Why, one plain sentence ending in a full stop.
    """

    doing: str
    why: str


def _words(doing: str, why: str) -> CaptionWordsDict:
    """Build one table entry.

    Args:
        doing: The planned action or actual lifecycle state.
        why: The plan's reason or an explanation of the state.

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
        "Next: Scan with radar",
        "After teleporting, it plans to check the new area for enemies and supplies.",
    ),
    ReasonKind.EQUIPMENT_LOCKED: _words(
        "Next: Collect equipment",
        "It plans to collect the equipment crate it found nearby.",
    ),
    ReasonKind.FUEL_LOCKED: _words(
        "Next: Collect fuel",
        "It needs fuel and plans to collect the fuel tank it found.",
    ),
    ReasonKind.EQUIPMENT_RESTOCK: _words(
        "Next: Restock equipment",
        "It plans to pick up gear it needs, such as radars.",
    ),
    ReasonKind.EQUIPMENT_HOP: _words(
        "Next: Jump toward equipment",
        "It plans to teleport closer to the equipment it found.",
    ),
    ReasonKind.FORAGE_FRONTIER_HOP: _words(
        "Next: Jump to a new area",
        "Nothing useful is left nearby, so it plans to teleport to unsearched ground.",
    ),
    ReasonKind.FUEL_HOP: _words(
        "Next: Jump toward fuel",
        "It needs fuel and plans to teleport closer to a distant fuel tank.",
    ),
    ReasonKind.FUEL_COLLECT: _words(
        "Next: Pick up fuel",
        "It plans to drive onto the fuel tank beside it to refuel.",
    ),
    ReasonKind.MINE_CLEARANCE_SHOT: _words(
        "Next: Shoot a mine",
        "A mine blocks its route, so it plans to shoot it away.",
    ),
    ReasonKind.MINE_PIN: _words(
        "Next: Lay mines",
        "An enemy is beside it, so it plans to lay mines to hem them in.",
    ),
    ReasonKind.FORAGE_RADAR: _words(
        "Next: Scan with radar",
        "It plans to search nearby ground for fuel and equipment.",
    ),
    ReasonKind.FORAGE_SWEEP: _words(
        "Next: Drive to a scan spot",
        "It plans to drive where its next radar scan can reveal the most new ground.",
    ),
    ReasonKind.FORAGE_FRONTIER_WALK: _words(
        "Next: Drive to new ground",
        "It plans to drive toward unsearched ground to look for supplies.",
    ),
    ReasonKind.FORAGE_FRONTIER_PAN: _words(
        "Next: Look further out",
        "It plans to shift its view to unsearched ground without spending fuel.",
    ),
    ReasonKind.QUAD_SWEEP_SHIFT: _words(
        "Next: Look at another corner",
        "It plans to shift its view to the next corner while keeping the tank still.",
    ),
    ReasonKind.QUAD_SWEEP_RADAR: _words(
        "Next: Scan with radar",
        "It plans to scan one of the four corners around it without moving.",
    ),
    ReasonKind.HARVEST_FRAME_SHIFT: _words(
        "Next: Look toward supplies",
        "It plans to shift its view toward the supplies it found.",
    ),
    ReasonKind.HARVEST_LEG_WALK: _words(
        "Next: Drive to supplies",
        "It plans to drive to the next supply in the group it found.",
    ),
    ReasonKind.DESYNC_RESCAN: _words(
        "Next: Scan again",
        "Its picture of the area is out of date, so it plans to check with radar.",
    ),
    ReasonKind.MINE_HIT_REVEAL_SCAN: _words(
        "Next: Scan for mines",
        "It hit an unseen mine and plans to scan for others.",
    ),
    ReasonKind.SEARCH_COLLECT_LOCAL: _words(
        "Next: Jump toward fuel",
        "It plans to teleport to fuel on the map and search the new area.",
    ),
    ReasonKind.CLAIM_DENIED: _words(
        "Next: Choose another target",
        "Another bot claimed that supply, so it plans to pick a different target.",
    ),
    ReasonKind.WALK_FOR_FUEL: _words(
        "Next: Drive to fuel",
        "Its fuel is too low to teleport, so it plans to drive to the nearest known fuel.",
    ),
    ReasonKind.WALK_FOR_FUEL_PAN: _words(
        "Next: Look ahead for fuel",
        "Its fuel is too low to teleport, so it plans to look ahead for a driving route.",
    ),
    ReasonKind.MAP_FOR_DOTS: _words(
        "Next: Check the map",
        "It plans to open the map to find tanks and fuel.",
    ),
    ReasonKind.AWAIT_MAP_ANSWER: _words(
        "Next: Wait for the map",
        "It has requested the map and plans to wait for the game's answer.",
    ),
    ReasonKind.FERRY_SCOPE_SCOUT: _words(
        "Next: Look for a ferry",
        "Its target is across water, so it plans to search the shore for a ferry.",
    ),
    ReasonKind.GATHERER_HOLD: _words(
        "Next: Wait for supplies",
        "It has collected everything nearby and plans to wait for supplies to reappear.",
    ),
    ReasonKind.FIND_TARGET: _words(
        "Next: Find an enemy",
        "It plans to open the map to find another tank to fight.",
    ),
    ReasonKind.FIND_ENEMIES: _words(
        "Next: Search for enemies",
        "No enemy is in reach, so it plans to check the map for another target.",
    ),
    ReasonKind.TELEPORT_TARGET: _words(
        "Next: Jump toward an enemy",
        "It found an enemy and plans to teleport closer for an attack.",
    ),
    ReasonKind.GREET_APPROACH: _words(
        "Next: Go say hello",
        "A human player is in the room, so it plans to teleport into their view.",
    ),
    ReasonKind.WALK_TO_TARGET: _words(
        "Next: Drive toward an enemy",
        "It plans to move closer to the enemy for a clear shot.",
    ),
    ReasonKind.SHOOT_TARGET: _words(
        "Next: Shoot at an enemy",
        "An enemy is in range and in line, so it plans to fire.",
    ),
    ReasonKind.COMBAT_FRAME_SHIFT: _words(
        "Next: Follow the enemy",
        "The enemy is near the edge, so it plans to shift its view to keep them in sight.",
    ),
    ReasonKind.OPPORTUNITY_SHOT: _words(
        "Next: Take a shot",
        "An enemy came into line during another task, so it plans to take a shot.",
    ),
    ReasonKind.DOT_RELAY: _words(
        "Next: Jump toward an enemy",
        "The enemy is too far for one jump, so it plans a shorter jump via fuel.",
    ),
    ReasonKind.HUNT_REFUEL: _words(
        "Next: Refuel before a fight",
        "It wants to attack but plans to collect more fuel first.",
    ),
    ReasonKind.CONFIRM_KILL: _words(
        "Next: Check the kill",
        "The enemy may be destroyed, so it plans to check the map to confirm.",
    ),
    ReasonKind.MANUAL_HOLD: _words(
        "Next: Hold still",
        "Its operator asked it to stay put, so it plans to hold its position.",
    ),
}
"""Every reason the bot gives, as a caption."""


def caption_words(reason: ReasonKind, resource_kind: str) -> CaptionWordsDict:
    """Describe a decision using its committed resource target.

    Args:
        reason: The decision's reason.
        resource_kind: The held target's kind, or empty when none is held.

    Returns:
        Plain words that distinguish fuel from equipment when a target is known.
    """
    if resource_kind in ("fuel", "equipment"):
        if reason == ReasonKind.HARVEST_FRAME_SHIFT:
            return _words(
                f"Next: Look toward {resource_kind}",
                f"It plans to shift its view toward the {resource_kind} it found.",
            )
        if reason == ReasonKind.HARVEST_LEG_WALK:
            return _words(
                f"Next: Collect {resource_kind}",
                f"It plans to collect the {resource_kind} it found nearby.",
            )
    return REASON_WORDS[reason]


def encode_caption_words(words: CaptionWordsDict) -> JSONObject:
    """Encode the text sent to the game's status panel.

    Args:
        words: The action and its explanation.

    Returns:
        A JSON object containing only text.
    """
    return {"doing": words["doing"], "why": words["why"]}


def decode_caption_words(data: JSONObject) -> CaptionWordsDict:
    """Validate status-panel text at a JSON boundary.

    Args:
        data: Serialized caption words.

    Returns:
        Validated caption text.

    Raises:
        JSONTypeError: Either field is missing or is not text.
    """
    return CaptionWordsDict(doing=require_str(data, "doing"), why=require_str(data, "why"))


__all__ = [
    "DESTROYED_WORDS",
    "REASON_WORDS",
    "CaptionWordsDict",
    "caption_words",
    "decode_caption_words",
    "encode_caption_words",
]
