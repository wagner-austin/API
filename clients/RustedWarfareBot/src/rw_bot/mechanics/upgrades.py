"""What an owned structure offers to turn itself into.

The build tree says an ``extractorT1`` produces an ``extractorT2``, and the
catalogue prices the tiers at 8, 12, 20 and 30 credits per second. If that edge
is reachable at runtime it is the largest economic lever in the game and the
only one that needs no builder, no travel and no contested ground: the extractor
upgrades **itself**, in place, on ground already held.

Whether it is reachable is a question about the *option stream*, not about the
tree, and the two are not the same. A structure's upgrade may arrive as a placed
action -- which production refuses, because a queue cannot express a position --
or as an ordinary produce action, or not at all. `economy.py` has carried a note
for some time saying no capture had yet shown an owned extractor offering one,
and this is what settles it rather than guessing ([[policy-holding-ground]]).

Pure: a sample goes in and a description of what upgrades are on offer comes
out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.wire.state import Sample

#: Upgrade **paths**, each listing types where every entry converts into the next.
#:
#: Two paths rather than one chain, because the extractor line is not a line.
#: ``.game/assets/units/extractor/extractorT3.ini`` declares *two* conversions
#: on the tier three -- ``action_overclock`` to ``extractorT3_overclocked`` and
#: ``action_reinforce`` to ``extractorT3_reinforced`` -- and neither converts
#: into the other. They are siblings: both carry an ``action_refund`` back to
#: the plain tier three and nothing forward.
#:
#: Listing them as one five-long chain said an overclocked extractor was an
#: upgrade of a reinforced one, which is false in both directions. Two paths
#: sharing a prefix say the true thing instead, and :func:`satisfies` needs no
#: special case for it: it requires both types in the *same* path, and no path
#: holds both siblings.
#:
#: The two are different purchases, not two grades of one. Overclocked pays 30
#: credits a second and drops to 1,100 hit points -- fewer than the tier three
#: it came from; reinforced stays at 20 and goes to 4,700 with an 800 shield
#: and self-regeneration. Income and survivability, priced at 8,000 and 3,000.
#:
#: Only the extractor line is listed, because it is the only upgrade the bot
#: takes. Turrets upgrade too, and those belong here when something starts
#: building them, not before.
TIER_CHAINS: tuple[tuple[str, ...], ...] = (
    ("extractorT1", "extractorT2", "extractorT3", "extractorT3_overclocked"),
    ("extractorT1", "extractorT2", "extractorT3", "extractorT3_reinforced"),
)


def satisfies(held: str, wanted: str) -> bool:
    """Report whether a held type counts as the wanted one.

    **An upgrade must not un-satisfy the thing it improved.** The opening plan
    reads its own progress off the roster by matching type names, which is
    right until a structure can convert itself. The moment the bot could
    upgrade an extractor, the plan stopped seeing the ``extractorT1`` it had
    asked for, ordered another, and did that forever -- so the builder was
    never free, expansion never ran, and a match ended with 41,559 credits
    banked and a plan reading 0 of 8 ([[policy-holding-ground]]).

    A later tier satisfies an earlier one. The reverse does not: asking for a
    tier two is not answered by holding a tier one.

    Args:
        held: The type the player owns.
        wanted: The type the plan asked for.

    Returns:
        True when the held type is the wanted one or an upgrade of it.
    """
    if held == wanted:
        return True
    for chain in TIER_CHAINS:
        if wanted in chain and held in chain:
            return chain.index(held) > chain.index(wanted)
    return False


def next_tier(held: str) -> str | None:
    """Return the single type this one converts into, when there is exactly one.

    **Ambiguity is answered with None rather than with a preference.** The tier
    three offers two conversions and they are not two grades of one thing:
    overclocking pays 30 credits a second and drops the structure to 1,100 hit
    points, reinforcing holds income at 20 and takes it to 4,700 with an 800
    shield. Which is worth more depends on whether the ground is contested,
    which is a question for a measurement rather than for a constant here --
    and on this map, where the opponents finish holding 44 of the 46 pools, it
    is not an obvious one ([[policy-holding-ground]]).

    So the walk goes as far as the paths agree and stops where they fork. A
    tier one names a tier two and a tier two names a tier three, because every
    path says the same thing about those steps.

    Args:
        held: The type the player owns.

    Returns:
        The type it converts into, or None when it converts into nothing or
        into more than one thing.
    """
    successors = {
        chain[chain.index(held) + 1]
        for chain in TIER_CHAINS
        if held in chain and chain.index(held) + 1 < len(chain)
    }
    return successors.pop() if len(successors) == 1 else None


class UpgradeOffer(TypedDict):
    """One owned structure's offer to become something else.

    Attributes:
        unit_id: Engine identity of the structure offering it.
        holder_type: What the structure is now.
        produces: What the option would make.
        placed: Whether the engine wants a position for it. A placed option
            cannot be ordered from a queue, which decides whether production or
            the build policy has to own it.
        available: Whether the engine says it may be used right now. This is
            where tech gating and the unit cap already live, so a false here is
            an answer rather than a gap ([[mechanics-build-actions]]).
    """

    unit_id: int
    holder_type: str
    produces: str
    placed: bool
    available: bool


def upgrade_offers(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
) -> tuple[UpgradeOffer, ...]:
    """Return every option an owned **structure** offers.

    Restricted to structures because that is the question: a factory offering a
    tank is ordinary production and is already understood. A structure offering
    anything at all is either an upgrade of itself or a capability nothing in
    the policy layer has ever reached.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for telling a structure from a unit.

    Returns:
        One entry per option offered by an owned immobile entity, in roster
        order.
    """
    holders = {
        entity["unit_id"]: entity["type_name"]
        for entity in sample["entities"]
        if entity["mine"]
        and entity["complete"]
        and catalogue.get(entity["type_name"], {"speed": 1.0})["speed"] == 0.0
    }
    return tuple(
        UpgradeOffer(
            unit_id=option["unit_id"],
            holder_type=holders[option["unit_id"]],
            produces=option["produces"],
            placed=option["placed"],
            available=option["available"],
        )
        for option in sample["options"]
        if option["unit_id"] in holders
    )


def format_offers(offers: Sequence[UpgradeOffer]) -> tuple[str, ...]:
    """Render the offers as aligned lines.

    Args:
        offers: What the structures offered.

    Returns:
        A header and one line per offer, or a single line saying there were
        none -- which is itself the answer to the question this probe asks.
    """
    if not offers:
        return ("no owned structure offered any option at all",)
    lines = [f"{'holder':<24}{'unit':>8}  {'produces':<28}{'placed':>8}{'available':>11}"]
    lines.extend(
        f"{offer['holder_type']:<24}{offer['unit_id']:>8}  {offer['produces']:<28}"
        f"{offer['placed']!s:>8}{offer['available']!s:>11}"
        for offer in offers
    )
    return tuple(lines)


__all__ = [
    "TIER_CHAINS",
    "UpgradeOffer",
    "format_offers",
    "next_tier",
    "satisfies",
    "upgrade_offers",
]
