"""Choosing the raider the reserve should turn on, if any.

The ladder is decided by extractors lost, and the traces say when they are
lost: non-wins hold five to seven until mid-match, then bleed to nought-to-two
while the opponent compounds ([[policy-holding-ground]]). Nothing answers the
raid. Turrets were measured four ways and refuted four ways; the army rallies
at the base while the extractors die out where the pools are; and the released
wave is across the map attacking. The one force that could answer -- the
reserve, gathered and idle -- had no order to.

The engine's own AI answers raids with *mobile* groups flagged Defensive, tied
to zones ([[engine-ai-zones]]). This module is the bot's minimal equivalent:
a hostile standing inside the radius the engine itself gives a resource
outpost, measured from any of our structures, is an intruder, and the deepest
intruder the reserve can actually shoot is the one it turns on.

Pure, like the rest of the policy layer: a sample goes in, a target comes out,
and what to do about it is the wave controller's business.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile, can_engage
from rw_bot.policy.scoreboard import standing_of
from rw_bot.wire.state import Entity, Sample

#: How close to one of our structures a hostile counts as an intruder.
#:
#: The engine's own number, not ours: the AI creates every resource-outpost
#: zone with radius 360, and defends what is inside its zones
#: ([[engine-ai-zones]]). Using the same figure means we call "intrusion"
#: exactly what the opponent calls "territory".
OUTPOST_RADIUS = 360.0


def _depth(hostile: Entity, structures: Sequence[Entity]) -> float:
    """Return the squared distance from a hostile to our nearest structure.

    Args:
        hostile: The hostile entity.
        structures: Our standing buildings.

    Returns:
        Squared world distance to the closest structure.
    """
    return min((hostile["x"] - s["x"]) ** 2 + (hostile["y"] - s["y"]) ** 2 for s in structures)


def deepest_intruder(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    profiles: Mapping[str, CombatProfile],
    reserve: Sequence[Entity],
    targets: Sequence[Entity],
) -> Entity | None:
    """Return the intruder the reserve should turn on, or None.

    Deepest first -- the raider closest to a structure is the one doing damage
    soonest -- and only an intruder at least one reserve unit can engage is
    chosen, because turning the reserve toward a helicopter it cannot shoot is
    the engageable-filter failure all over again
    ([[mechanics-combat-profile]]). Ties break on health then engine id, so
    two runs of one seed choose identically.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for telling a building from a unit.
        profiles: Combat profiles by type name, for what the reserve can shoot.
        reserve: Units gathered at the base and not released to a wave.
        targets: The hostile entities currently visible.

    Returns:
        The intruder to engage, or None when nothing hostile stands inside
        :data:`OUTPOST_RADIUS` of a structure, nothing in the reserve can
        shoot what does, or there is no reserve or no structure at all.
    """
    if not reserve or not targets:
        return None
    structures = standing_of(sample, catalogue)
    if not structures:
        return None
    limit = OUTPOST_RADIUS**2

    def rank(target: Entity) -> tuple[float, float, int]:
        return (_depth(target, structures), target["hp"], target["unit_id"])

    intruders = sorted(
        (target for target in targets if _depth(target, structures) <= limit),
        key=rank,
    )
    for intruder in intruders:
        if any(can_engage(profiles, unit, intruder) for unit in reserve):
            return intruder
    return None


__all__ = ["OUTPOST_RADIUS", "deepest_intruder"]
