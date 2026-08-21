"""Landing refusals: the server's stay-at-origin verdict becomes belief.

The mined law ([[teleport-mechanics]] § the refusal law, 137/137
archived receipts, 2026-08-21): beyond ring-1 no ejection exists — a
teleport whose requested tile AND whole ring-1 are blocked is REFUSED,
uncharged, and the tank stays at its origin. The bot perceives that as
"landed at origin"; recording it as ring-blocked evidence is what
stops the identical hop re-certifying against mine-blind beliefs (the
08-05 534-refusal session, the 2026-08-21 marooning loops).
"""

from __future__ import annotations

from tankpit_bot.bot.ai.ferry import FerryAwareTerrain
from tankpit_bot.bot.ai.reachability import find_attainable_landing_tile
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_service_movement import _LANDING_REFUSAL_TTL_MS
from tests.in_memory_terrain_map import InMemoryTerrainMap


def test_refusal_is_recorded_as_ring1_evidence_and_expires() -> None:
    """A refusal blocks exactly the requested tile + ring-1, then ages out.

    One refusal PROVES "requested + ring-1 all blocked" and nothing
    more — the origin distance is where the tank already stood, not a
    hostile radius (the pre-mining model blanketed up to radius 12 and
    was wrong).
    """
    ws = WorldService()

    ws.mark_landing_refused(128, 238, 7, 100_000)

    fresh = ws.hostile_landing_keys(100_000 + 1)
    assert fresh == frozenset(f"{x},{y}" for x in range(127, 130) for y in range(237, 240))

    expired = ws.hostile_landing_keys(100_000 + _LANDING_REFUSAL_TTL_MS)
    assert expired == frozenset()
    # Expiry also prunes the store, so long sessions never accrete.
    assert ws.landing_refusals == {}


def test_routine_ring1_displacement_is_never_a_refusal() -> None:
    """A one-tile shift is the ring-1 displacement law working."""
    ws = WorldService()

    ws.mark_landing_refused(100, 100, 1, 100_000)

    assert ws.landing_refusals == {}
    assert ws.has_fresh_landing_refusal(100_001) is False


def test_has_fresh_landing_refusal_tracks_the_ttl() -> None:
    """The repair-radar gate's predicate follows refusal freshness."""
    ws = WorldService()
    ws.mark_landing_refused(50, 50, 3, 100_000)

    assert ws.has_fresh_landing_refusal(100_000 + 1) is True
    assert ws.has_fresh_landing_refusal(100_000 + _LANDING_REFUSAL_TTL_MS) is False


def test_composed_terrain_refuses_landings_inside_ring_but_not_walks() -> None:
    """Refusal evidence blocks LANDINGS in the ring only; walking is untouched.

    The regression shape of the marooning harvest orbit: with the
    refusal recorded, the selector that certified four identical hops
    at (128,238) answers ``None`` — every service tile (target +
    cardinals) sits inside the proven-blocked ring — and the existing
    unservable/clearance laws take over.
    """
    base = InMemoryTerrainMap()
    ws = WorldService()
    ws.mark_landing_refused(128, 238, 7, 100_000)
    terrain = FerryAwareTerrain(
        base,
        {},
        riding=False,
        hostile_mine_keys=frozenset(),
        occupied_tank_keys=frozenset(),
        refused_landing_keys=ws.hostile_landing_keys(100_001),
    )

    assert terrain.is_landing_attainable(128, 238) is False
    assert terrain.is_landing_attainable(127, 237) is False
    assert terrain.is_passable(128, 238) is True
    assert find_attainable_landing_tile(terrain, 128, 238) is None
    # Outside the proven ring, landings stay attainable.
    assert terrain.is_landing_attainable(130, 238) is True
