"""Keeping what the economy bought: covering a structure that has none.

The third expansion question. Income asks "can I earn more" and throughput asks
"can I spend more"; this one asks **"can I keep what I bought"**, which is the
question the duels turn on -- every seed reaches three extractors and the
verdict follows how many it then loses ([[policy-holding-ground]]).

**It has never yet been shown to help, and the record is kept here rather than
argued away.** Four arms have failed: defence ahead of income, defence from the
surplus, defence aimed at the base, and defence aimed at the extractors the
traces show dying. See :func:`undefended` for why the last of those is recorded
as a refutation rather than retried.

Split out of :mod:`rw_bot.policy.economy` because it answers a different
question from the one that module is named for, and because keeping the failed
arms' evidence beside the code that would be changed by a fifth attempt is the
point of writing it down.

Pure: a world state in, a decision out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.combat_profile import CombatProfile, profile_of
from rw_bot.policy.build_order import PLACEHOLDER_TYPE
from rw_bot.policy.combat import is_mobile
from rw_bot.policy.economy import Expansion, count_extractors, placer, waiting
from rw_bot.policy.siting import clear_site_near, find_anchor
from rw_bot.wire.state import Entity, Sample

#: The defence placed beside a structure that has none.
#:
#: The best damage per credit and the best hit points per credit in the game --
#: 16.40 and 140 against ``c_tank``'s 5.71 and 60 -- and cheaper than the
#: extractor it covers ([[mechanics-unit-value]]). It is also the only defence
#: a Builder can place from the start, which is what makes it reachable at all
#: ([[policy-holding-ground]]).
TURRET_TYPE = "c_turret_t1"

#: The defence placed beside a structure once the opponent has shown aircraft.
#:
#: The other half of a gap the ground turret cannot close: ``c_turret_t1``
#: declares ``canAttackFlyingUnits: false`` and so does every unit the opening
#: plan builds, so before this existed nothing the bot could place touched an
#: aircraft at all ([[policy-holding-ground]]). 600 credits, 250 reach, 800
#: hit points, placeable by the Builder from the start -- same reachability
#: argument as the ground turret's.
AA_TURRET_TYPE = "c_antiAirTurret"


def _range_squared(one: Entity, other: Entity) -> float:
    """Return the squared distance between two entities.

    Left squared because only the ordering and a threshold comparison use it,
    and a square root would cost precision for nothing.

    Args:
        one: The first entity.
        other: The second entity.

    Returns:
        The squared distance in world units.
    """
    return (one["x"] - other["x"]) ** 2 + (one["y"] - other["y"]) ** 2


def undefended(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    profiles: Mapping[str, CombatProfile],
    turret_type: str = TURRET_TYPE,
) -> Entity | None:
    """Return an owned structure with no turret standing near it.

    **Why this exists at all.** Holding a pool, not claiming one, is what
    decides a match. Across the twelve Hard duels every seed reaches a peak of
    three extractors inside the first fifth of the match, and the verdict then
    follows the number it *loses*: nought or one drop won four matches of four,
    two or more lost eight of eight. The bot rebuilds three or four times in
    every run, so it is not failing to expand -- it is paying 700 credits over
    and over for ground it cannot keep ([[policy-holding-ground]]).

    A turret is cheaper than the extractor it protects: 500 against 700, with
    2.3x the hit points per credit of a tank and 2.9x the damage
    ([[mechanics-unit-value]]). So an undefended structure is the site worth
    spending on, and this is what finds one.

    Nearest to the anchor first, so the base is covered before the frontier.

    **Covering the extractors instead was tried, and lost.** The argument for it
    was the strongest this policy has had, and it was not enough. Extractor
    losses decide every duel; the per-loss table puts each death 688-1,766 world
    units from the army's own fighting cloud, so they are raided at sites
    nothing defends; and not one unit died within 900 units of the base across
    two traced runs, which makes nearest-first look like spending in the one
    place never attacked. Restricting cover to extractors, same twelve seeds,
    same rung: **wins 4 -> 0, extractor drops 21 -> 24, and the first two
    defeats in fifty-two duels.** Turrets standing at the end across all twelve
    matches: three.

    So the rule is kept as it was, and the refutation is recorded rather than
    the argument. Three separate arms have now failed -- defence ahead of
    income, defence from surplus, defence aimed at what actually dies -- which
    is enough to say the turret is not the answer to extractor loss, whatever
    the value table says about it. What is *not* established is why so few get
    built at all: at three in twelve matches this has never been a policy that
    ran, only one that was reached ([[policy-holding-ground]]).

    Cover is the turret's own reach, read from the registry rather than chosen
    here: a structure is covered when a turret could actually shoot something
    standing on it ([[mechanics-combat-profile]]).

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for telling a structure from a unit.
        profiles: Combat profiles by type name, for the turret's reach.
        turret_type: The defence that counts as cover.

    Returns:
        The nearest owned structure lacking cover, or None when every one of
        them has some, or when there is no anchor to measure from.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe
            the turret.
    """
    anchor = find_anchor(sample, catalogue)
    if anchor is None:
        return None
    cover = profile_of(profiles, turret_type)["attack_range"] ** 2
    turrets = [
        entity
        for entity in sample["entities"]
        if entity["mine"] and entity["type_name"] == turret_type
    ]

    def from_anchor(entity: Entity) -> float:
        return _range_squared(entity, anchor)

    # The placeholder exclusion is load-bearing here exactly as it is in
    # find_producer: the map editor's unit is an owned immobile entity parked
    # off-map at (-1000, -1000) with 170,000 hit points, and it is eternally
    # bare. It ranked last by distance, so nothing reached it -- until the
    # cover ring made defence WORK, every real structure got covered, and the
    # one bare "structure" left was the placeholder. Defence then poured
    # turrets at an unbuildable off-map point for the rest of the match,
    # walking 46 of 51 builders to their deaths and freezing the workforce
    # the plan was waiting on (log: 2026-07-31).
    bare = [
        entity
        for entity in sample["entities"]
        if entity["mine"]
        and entity["complete"]
        and entity["type_name"] != PLACEHOLDER_TYPE
        and not is_mobile(entity, catalogue)
        and not any(_range_squared(entity, turret) <= cover for turret in turrets)
    ]
    if not bare:
        return None
    return min(bare, key=from_anchor)


def expand_defence(
    sample: Sample,
    catalogue: Mapping[str, UnitStats],
    profiles: Mapping[str, CombatProfile],
    *,
    available: int,
    free: Sequence[Entity],
    turret_type: str = TURRET_TYPE,
) -> Expansion:
    """Decide whether to cover a structure that has none.

    The third expansion question. Income asked "can I earn more" and throughput
    asked "can I spend more"; nothing asked **"can I keep what I bought"**, and
    that is the question the twelve Hard duels turn on: every seed reaches three
    extractors and the verdict follows how many it then loses
    ([[policy-holding-ground]]).

    Asking it has not yet answered it. Three arms have failed, the last of them
    aimed squarely at the extractors the traces show dying, and see
    :func:`undefended` for why that one is recorded as a refutation rather than
    retried. **It reaches the field far too seldom to be judged as a policy** --
    three turrets standing across twelve full matches -- and that, rather than
    the choice of site, is what is unexplained.

    Placed beside the structure it covers rather than on the base ring, because
    a turret at the base does not defend a pool on the far side of the map.

    **The site is the covered structure's own ring now, not a bare offset.**
    For its whole prior life this reached for ``x + RING_SLOT_RADIUS`` without
    checking whether anything stood there -- at an extractor that offset can be
    the pool itself or the next structure over, the engine refuses such an
    order silently, and the scorecards finally priced the damage: 27 paid
    turret orders in one match with about five ever standing. The refusal cost
    a builder walk and a stall window per attempt, invisibly, for four
    measured arms in a row -- so every one of those refutations carries this
    confound, and the first fair reading of defence arrives with this fix
    (log: 2026-07-30). :func:`~rw_bot.policy.siting.clear_site_near` walks a
    cover ring sized inside the turret's own reach with the occupancy
    predicate every other placement already trusts.

    Args:
        sample: One observation of the world.
        catalogue: Unit stats by type name, for prices and for telling a
            structure from a unit.
        profiles: Combat profiles by type name, for the turret's reach.
        available: Credits still unclaimed by higher-priority spenders.
        free: Workers not already carrying out an order.
        turret_type: The defence to place.

    Returns:
        What to do, with the reasoning behind it either way.

    Raises:
        CombatProfileError: ``RW-COMBAT-002`` when the dump does not describe
            the turret.
    """
    stats = catalogue.get(turret_type)
    if stats is None:
        return waiting(f"the catalogue does not price {turret_type}", sample, turret_type)
    # Demand before price, so a priced-out wait is a fact worth saving toward:
    # it now means "a bare structure exists and this balance cannot cover it",
    # never "we are broke and everything is covered anyway". The expander's
    # cross-tick withhold keys on exactly that flag -- the validated champion's
    # own ledger showed defence reached 954 times and acting twice, with the
    # last refusal reading `wanted 500 of 0 available` while the two turrets it
    # DID land were setting the survival and dip records (log 2026-08-01).
    target = undefended(sample, catalogue, profiles, turret_type)
    if target is None:
        return waiting("every structure already has cover", sample, turret_type)
    if available < stats["price"]:
        return waiting(
            f"{turret_type} wanted {stats['price']} of {available} available",
            sample,
            turret_type,
            priced_out=True,
        )
    builder = placer(sample, turret_type, free)
    if builder is None:
        return waiting(f"no free worker can place {turret_type}", sample, turret_type)
    site = clear_site_near(sample, target, catalogue)
    if site is None:
        return waiting(
            f"no clear cover position around {target['type_name']} "
            f"at {target['x']:.0f},{target['y']:.0f}",
            sample,
            turret_type,
        )
    return Expansion(
        build=True,
        priced_out=False,
        reason=f"covering {target['type_name']} at {target['x']:.0f},{target['y']:.0f}",
        type_name=turret_type,
        unit_id=builder["unit_id"],
        x=site[0],
        y=site[1],
        owned=count_extractors(sample),
        occupied=0,
        exposed=0,
        visible=0,
        unreachable=0,
    )


__all__ = ["AA_TURRET_TYPE", "TURRET_TYPE", "expand_defence", "undefended"]
