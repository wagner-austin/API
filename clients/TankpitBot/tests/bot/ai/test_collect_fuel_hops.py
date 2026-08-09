"""The fuel hops in the collect cascade: the larder harvest and the marooned hop.

Both terrain guards pinned here were mutation survivors (2026-08-08).
Each was exercised against a world that ALSO failed a later gate -- an
empty larder, no affordable candidate -- so the guard and its absence
produced the same ``None`` and the assertions could not tell them apart.
The context below keeps a fuel container that would otherwise qualify on
every other gate, leaving the terrain guard as the only thing that can
decline.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_hops import desperation_fuel_hop, larder_harvest
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.sniffer.world_service import WorldService
from tests._runtime_logging_support import capture_runtime_events, event_fields
from tests.bot.ai._support import (
    make_container,
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _ctx_stocked_larder_unknown_terrain() -> DecideCtx:
    """Build a context whose fuel larder is stocked and whose terrain is unknown.

    Rank 2 caps fuel at 1200 and the tank holds 800, so the larder's
    capacity gate passes. The container sits ten tiles east: teleport
    cost 60 against 800 fuel, so affordability passes too. Volume 500
    against the deficit clears the dreg floor. Every gate but terrain
    is satisfied on purpose.

    Returns:
        A decision context whose ``terrain`` is None.
    """
    ws = WorldService()
    containers = {
        "110,100": make_container(x=110, y=100, volume=500, is_fuel=True),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=800, containers=containers)
    return DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(default_count=15),
        100000,
        None,
        "",
        ws=ws,
    )


def test_larder_harvest_declines_when_terrain_unknown() -> None:
    """The fuel larder declines on unknown terrain, stocked or not.

    Landing resolution requires a non-None terrain, so without the
    guard this raises rather than declining.
    """
    ctx = _ctx_stocked_larder_unknown_terrain()
    assert larder_harvest(ctx, ctx.base) is None


def test_desperation_fuel_hop_declines_when_terrain_unknown() -> None:
    """The marooned hop declines on unknown terrain rather than aiming blind.

    This is the last rung of the marooned ladder, so declining here
    ends the session with ``out_of_fuel`` -- but a blind hop would spend
    the last of the tank on a landing tile nobody has verified.
    """
    ctx = _ctx_stocked_larder_unknown_terrain()
    assert desperation_fuel_hop(ctx, ctx.base) is None


def test_fuel_larder_declines_silently_at_capacity() -> None:
    """A full tank is not a declined hop, so it logs no fuel_larder decline.

    The capacity gate's RETURN VALUE is indistinguishable from the
    profitability gate's: at capacity the deficit is zero, so
    ``min(volume, deficit)`` never clears the hop cost and the selection
    yields no container either way. What the gate actually prevents is a
    spurious ``hop_declined`` beside a live fuel container on every tick
    of a full tank, so this pins the event stream rather than the return
    value (mutation survivor, 2026-08-08).
    """
    ws = WorldService()
    containers = {
        "110,100": make_container(x=110, y=100, volume=500, is_fuel=True),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(default_count=15),
        100000,
        InMemoryTerrainMap(),
        "",
        ws=ws,
    )

    with capture_runtime_events() as records:
        assert larder_harvest(ctx, ctx.base) is None

    declines = [
        (event_fields(record).get("diagnostic_kind"), event_fields(record).get("hop_kind"))
        for record in records
    ]

    # Paired assertion. The equipment decline MUST be captured, or this
    # test proves nothing about the fuel one -- the first draft read a
    # flat ``diagnostic_kind`` off the record instead of the nested
    # ``runtime_fields``, matched nothing, and passed while the mutant
    # lived.
    assert ("hop_declined", "equipment") in declines
    assert ("hop_declined", "fuel_larder") not in declines
