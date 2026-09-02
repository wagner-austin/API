"""The walk-territory responsibility gap: rock-pocketed near stock is served.

Run bot-20260901-210631 ([[flag-triage-20260902]]) livelocked nine
minutes on an in-viewport fuel container in a rock pocket: the walk
lane declined it (no route), the larder ceded it BY GEOMETRY
("too_close"/"own_ground" — walk territory), and nobody served it.
These tests pin the 2026-09-02 law: the walk lane owns only what the
pickup dispatch's own reachability predicate says it can take, and a
walk-blocked near container is teleport fair game.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_hops import hop_toward_equipment
from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.inventory import InventoryState
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import ContainerStateDict, make_container_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def _rock_ring(x: int, y: int) -> dict[tuple[int, int], str]:
    """Return terrain data walling a tile behind its eight neighbors.

    Args:
        x: Pocketed tile X.
        y: Pocketed tile Y.

    Returns:
        Terrain data with every neighbor rock and the tile itself
        ground — teleport-attainable, walk-unreachable.
    """
    ring = {
        (x + dx, y + dy): InMemoryTerrainMap.ROCK
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0)
    }
    ring[(x, y)] = InMemoryTerrainMap.GROUND
    return ring


def _container(x: int, y: int, *, is_fuel: bool, volume: int) -> ContainerStateDict:
    """Build one fresh-sighted container belief.

    Args:
        x: Container X.
        y: Container Y.
        is_fuel: Whether the container is fuel.
        volume: Container volume.

    Returns:
        Container state stamped at the test's decision time.
    """
    return make_container_state(
        x=x,
        y=y,
        is_fuel=is_fuel,
        volume=volume,
        timestamp_ms=100000,
        failed_pickups=0,
    )


def _ctx(
    *,
    fuel: int,
    containers: dict[str, ContainerStateDict],
    terrain: InMemoryTerrainMap,
    inventory: InventoryState,
) -> DecideCtx:
    """Build a decision context around the default (100,100) tank.

    Args:
        fuel: Current fuel.
        containers: Believed containers.
        terrain: Terrain map under test.
        inventory: Inventory state.

    Returns:
        Context with a scanned viewport, ready for the cascade.
    """
    world, self_state = make_world(fuel=fuel, containers=containers)
    return DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        terrain,
        "",
        ws=WorldService(),
    )


def test_a_rock_pocketed_in_viewport_fuel_container_is_served_by_the_larder_hop() -> None:
    """The 2026-09-01 pocket shape resolves in ONE tick via teleport.

    The fuel container sits in-viewport, four tiles east, ringed by
    rock: the walk lane cannot reach it, so pre-fix the larder ceded
    it as walk territory and the cascade fought over the window for
    nine minutes. Now the walk-territory law asks the pickup lane's
    own reachability predicate, the cession fails, and the larder
    prices the hop — landing ON the container, where auto-pick
    finishes the job.
    """
    ctx = _ctx(
        fuel=700,
        containers={"104,100": _container(104, 100, is_fuel=True, volume=700)},
        terrain=InMemoryTerrainMap(_rock_ring(104, 100)),
        inventory=make_inventory(),
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected the larder to serve the pocketed container")
    assert decision["behavior"]["reason_kind"] == "fuel_hop"
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 104
    assert decision["command"]["target_y"] == 100
    updated = decision["updated_ai_state"]
    assert updated["resource_target_kind"] == "fuel"
    assert updated["resource_target_x"] == 104
    assert updated["resource_target_y"] == 100
    assert updated["resource_target_held_ticks"] == 0
    assert updated["suppress_landing_scan"] is True


def test_a_walkable_in_viewport_fuel_container_stays_with_the_walk_pickup_lane() -> None:
    """Same geometry minus the rocks: the deferral still holds.

    The walk lane CAN take this one, so the s9-2/3 movement law is
    unchanged — the tick dispatches the pickup, never a teleport.
    """
    ctx = _ctx(
        fuel=700,
        containers={"104,100": _container(104, 100, is_fuel=True, volume=700)},
        terrain=InMemoryTerrainMap(),
        inventory=make_inventory(),
    )

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected the walk lane to serve the open container")
    assert decision["behavior"]["reason_kind"] == "fuel_collect"
    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["command"]["target_x"] == 104
    assert decision["command"]["target_y"] == 100


def test_a_rock_pocketed_near_equipment_container_is_priced_by_the_hop() -> None:
    """A walk-blocked container two tiles away is teleport fair game.

    The equipment hop's ``own_ground`` deferral (2026-08-13) refused
    any landing within the walk-dominant range by DISTANCE alone,
    re-opening the 2026-07-30 flag-4 gap one gate lower: near stock
    the walk lane had already disproven still got no teleport. The
    deferral now asks the same reachability predicate the pickup
    dispatch uses.
    """
    ctx = _ctx(
        fuel=1200,
        containers={"102,100": _container(102, 100, is_fuel=False, volume=0)},
        terrain=InMemoryTerrainMap(_rock_ring(102, 100)),
        inventory=make_inventory(default_count=15),
    )

    decision = hop_toward_equipment(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected the hop to price the pocketed container")
    assert decision["behavior"]["reason_kind"] == "equipment_hop"
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 102
    assert decision["command"]["target_y"] == 100
    updated = decision["updated_ai_state"]
    assert updated["resource_target_kind"] == "equipment"
    assert updated["resource_target_x"] == 102
    assert updated["resource_target_y"] == 100


def test_a_walkable_near_equipment_container_stays_own_ground() -> None:
    """Same two-tile container, no rocks: the hop still defers.

    Ground within walking reach that the walk lane can actually take
    belongs to the pickup steps (s8-2, HUD flag 1) — the hop declines
    and no teleport is spent.
    """
    ctx = _ctx(
        fuel=1200,
        containers={"102,100": _container(102, 100, is_fuel=False, volume=0)},
        terrain=InMemoryTerrainMap(),
        inventory=make_inventory(default_count=15),
    )

    assert hop_toward_equipment(ctx, ctx.base) is None
