"""Tests for the larder fuel scorer."""

from __future__ import annotations

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.larder import select_fuel_larder_hop
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.state.types import ContainerStateDict, TerrainTileDict
from tests.bot.ai._support import (
    make_container,
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

_RANK = 2


def _never_blacklisted(x: int, y: int) -> bool:
    del x, y
    return False


def _ctx(
    *,
    fuel: int,
    containers: dict[str, ContainerStateDict],
    terrain: TerrainMapProtocol | None = None,
) -> DecideCtx:
    world, self_state = make_world(self_x=100, self_y=100, fuel=fuel, containers=containers)
    return DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain if terrain is not None else InMemoryTerrainMap(),
        "",
    )


def _fuel_at_deficit(deficit: int) -> int:
    return fuel_capacity(_RANK) - deficit


def test_big_deficit_prefers_the_rich_far_container() -> None:
    """Uncapped by the deficit, volume/cost picks the 600 at 20 tiles."""
    containers = {
        "120,100": make_container(120, 100, 600, is_fuel=True),
        "110,100": make_container(110, 100, 250, is_fuel=True),
    }
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    winner = selection["container"]
    assert winner == containers["120,100"]
    assert (selection["landing_x"], selection["landing_y"]) == (120, 100)
    assert selection["cost"] == 120
    assert selection["candidates"] == 2


def test_small_deficit_prefers_the_near_container() -> None:
    """The deficit clamp flips the argmax to the closer 250."""
    containers = {
        "120,100": make_container(120, 100, 600, is_fuel=True),
        "110,100": make_container(110, 100, 250, is_fuel=True),
    }
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(150), containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] == containers["110,100"]
    assert selection["cost"] == 60


def test_skips_equipment_empty_failed_blacklisted_and_adjacent() -> None:
    """Only live believed fuel beyond auto-pick reach enters scoring."""
    failed = make_container(130, 100, 400, is_fuel=True)
    failed["failed_pickups"] = 2
    containers = {
        "115,100": make_container(115, 100, 0, is_fuel=False),
        "116,100": make_container(116, 100, 0, is_fuel=True),
        "130,100": failed,
        "125,100": make_container(125, 100, 400, is_fuel=True),
        "101,101": make_container(101, 101, 400, is_fuel=True),
        "120,100": make_container(120, 100, 500, is_fuel=True),
    }

    def _blacklist_125(x: int, y: int) -> bool:
        return (x, y) == (125, 100)

    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers=containers),
        is_blacklisted=_blacklist_125,
    )
    assert selection["container"] == containers["120,100"]
    assert selection["candidates"] == 2
    assert selection["too_close"] == 1


def test_water_locked_container_counts_no_landing() -> None:
    """A container with no passable tile in reach is skipped, tallied."""
    terrain = InMemoryTerrainMap(
        {
            (120, 100): "W",
            (121, 100): "W",
            (119, 100): "W",
            (120, 99): "W",
            (120, 101): "W",
        }
    )
    containers = {"120,100": make_container(120, 100, 500, is_fuel=True)}
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers=containers, terrain=terrain),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] is None
    assert selection["no_landing"] == 1


def test_shore_container_lands_on_the_cardinal_neighbor() -> None:
    """A water-sitting container is harvested from its shore tile."""
    terrain = InMemoryTerrainMap({(120, 100): "W"})
    containers = {"120,100": make_container(120, 100, 500, is_fuel=True)}
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers=containers, terrain=terrain),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] == containers["120,100"]
    assert (selection["landing_x"], selection["landing_y"]) == (121, 100)


def test_reserve_gates_the_transaction_not_the_transit() -> None:
    """A hop whose pickup lands the tank back above reserve is allowed.

    The F16 death regression ([[flag-triage-20260729]]): at fuel 216
    the old post-teleport-only gate blocked every refuel hop (the
    200-250 dead zone) and the bot died standing. Net of the pickup,
    a 355-volume container 8 tiles out leaves 523 - well clear.
    """
    containers = {"108,100": make_container(108, 100, 355, is_fuel=True)}
    selection = select_fuel_larder_hop(
        _ctx(fuel=216, containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] == containers["108,100"]
    assert selection["reserve_blocked"] == 0


def test_reserve_blocked_hop_is_declined() -> None:
    """A hop still below reserve after its own pickup never wins."""
    containers = {"120,100": make_container(120, 100, 105, is_fuel=True)}
    selection = select_fuel_larder_hop(
        _ctx(fuel=210, containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] is None
    assert selection["reserve_blocked"] == 1


def test_gain_below_cost_is_unprofitable() -> None:
    """A sliver container far away hands the tick to discovery."""
    containers = {"120,100": make_container(120, 100, 50, is_fuel=True)}
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] is None
    assert selection["unprofitable"] == 1


def test_empty_registry_returns_no_candidates() -> None:
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers={}),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] is None
    assert selection["candidates"] == 0


def test_dreg_below_floor_is_skipped_for_the_real_container() -> None:
    """The flag s3-9 shape: a close sub-floor dreg loses to a far 355.

    ``gain/cost`` favored the dreg before the floor
    ([[flag-triage-20260729]] F13); the dreg now tallies and the
    355-volume container wins outright. (Both sit outside the
    viewport -- in-viewport stock is the walk step's business.)
    """
    containers = {
        "110,100": make_container(110, 100, 90, is_fuel=True),
        "131,100": make_container(131, 100, 355, is_fuel=True),
    }
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] == containers["131,100"]
    assert selection["dreg"] == 1


def test_dreg_gain_completing_the_deficit_is_taken() -> None:
    """A clamped gain that fills the tank to capacity beats the floor.

    The F1 top-off microscope: refusing the last points below the
    floor would force a wasteful dot hop to finish hunt readiness.
    """
    containers = {
        "110,100": make_container(110, 100, 900, is_fuel=True),
    }
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(80), containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    winner = selection["container"]
    assert winner == containers["110,100"]
    assert selection["dreg"] == 0


def test_dreg_waiver_needs_hunt_ready_inventory() -> None:
    """Below combat-ready the deficit-completing waiver does not apply.

    Flag s8-1 (2026-07-30): at fuel 1076/1100 with zero radars the
    larder spent a map open + teleport on a 24-fuel top-off that
    unlocked nothing -- hunting was blocked by equipment regardless,
    and the walk step had just refused the same area's fuel as not
    worth a 5-tile walk. Topping the last points only matters when
    fuel is the FINAL hunt requirement.
    """
    containers = {
        "110,100": make_container(110, 100, 900, is_fuel=True),
    }
    world, self_state = make_world(
        self_x=100, self_y=100, fuel=_fuel_at_deficit(80), containers=containers
    )
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(default_count=0),
        100000,
        InMemoryTerrainMap(),
        "",
    )

    selection = select_fuel_larder_hop(ctx, is_blacklisted=_never_blacklisted)

    assert selection["container"] is None
    assert selection["dreg"] == 1


def test_desperation_fuel_is_reserve_blocked_not_dreg_gated() -> None:
    """At or below fuel_low_threshold the reserve gate owns the decline.

    The dreg floor never sees desperation fuel: every hop from
    fuel <= reserve is reserve_blocked first, and the dedicated
    desperation path (collect_mode._desperation_fuel_hop) handles that
    regime — so the floor needs no desperation exemption.
    """
    containers = {
        "110,100": make_container(110, 100, 90, is_fuel=True),
    }
    selection = select_fuel_larder_hop(
        _ctx(fuel=150, containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["container"] is None
    assert selection["reserve_blocked"] == 1
    assert selection["dreg"] == 0


def test_in_viewport_containers_belong_to_the_walk_step() -> None:
    """Same-viewport stock is never a larder hop (movement law).

    Flag s9-2/3 (2026-07-30): a 3-tile in-viewport larder teleport
    paid a map open, got displaced, and spent a landing radar for
    ground a walk served. The larder is cross-viewport machinery.
    """
    containers = {
        "102,100": make_container(102, 100, 500, is_fuel=True),
        "106,103": make_container(106, 103, 400, is_fuel=True),
    }
    selection = select_fuel_larder_hop(
        _ctx(fuel=_fuel_at_deficit(600), containers=containers),
        is_blacklisted=_never_blacklisted,
    )
    assert selection["too_close"] == 2
    assert selection["container"] is None


def test_walk_dominant_range_excludes_close_offscreen_containers() -> None:
    """Manhattan <= 2 stays the walk step's business even off-viewport.

    A tank that walked to its viewport edge can stand 2 tiles from a
    container just outside the frame -- still a walk, never a hop.
    """
    containers = {
        "109,100": make_container(109, 100, 500, is_fuel=True),
        "131,100": make_container(131, 100, 400, is_fuel=True),
    }
    ctx = _ctx(fuel=_fuel_at_deficit(600), containers=containers)
    ctx.world["viewport"]["left"] = 92
    ctx.self_state["x"] = 107

    selection = select_fuel_larder_hop(ctx, is_blacklisted=_never_blacklisted)

    assert selection["too_close"] == 1
    assert selection["container"] == containers["131,100"]


def _ferry_tile(x: int, y: int, observed_ms: int) -> TerrainTileDict:
    """Return a wire-terrain ferry belief at the given tile."""
    from tankpit_bot.state.types import make_terrain_tile
    from tankpit_bot.state.types.constants import TERRAIN_FERRY

    return make_terrain_tile(x, y, TERRAIN_FERRY, observed_ms=observed_ms)


def _water_everywhere_terrain() -> InMemoryTerrainMap:
    """Terrain where a lake spans the container's whole neighborhood."""
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(115, 135):
        for y in range(90, 110):
            terrain_data[(x, y)] = "W"
    return InMemoryTerrainMap(terrain_data=terrain_data)


def test_water_locked_fuel_is_ferry_served() -> None:
    """A fresh believed ferry near a lake container becomes the landing.

    [[flag-triage-20260729]] F5: the larder previously tallied every
    water-locked container ``no_landing`` (15/15 in
    bot-20260730-000038); with a ferry sighted at (122,101) the hop
    boards it and the held lock rides to the pickup.
    """
    containers = {
        "125,100": make_container(125, 100, 800, is_fuel=True),
    }
    ctx = _ctx(
        fuel=_fuel_at_deficit(600),
        containers=containers,
        terrain=_water_everywhere_terrain(),
    )
    ctx.world["terrain"]["122,101"] = _ferry_tile(122, 101, observed_ms=100000)
    # A farther second ferry and a plain-ground belief must not win.
    ctx.world["terrain"]["130,105"] = _ferry_tile(130, 105, observed_ms=100000)
    from tankpit_bot.state.types import make_terrain_tile

    ctx.world["terrain"]["124,100"] = make_terrain_tile(124, 100, 0, observed_ms=100000)

    selection = select_fuel_larder_hop(ctx, is_blacklisted=_never_blacklisted)

    assert selection["container"] == containers["125,100"]
    assert (selection["landing_x"], selection["landing_y"]) == (122, 101)
    assert selection["ferry_served"] == 1
    assert selection["no_landing"] == 0


def test_ferry_on_a_disjoint_pond_stays_no_landing() -> None:
    """A ferry on a separate water body cannot serve the container.

    Live deadlocks 2026-08-04/05 (runs bot-20260804-234008 and
    bot-20260805-070006): a ferry docked on a pool one land ridge
    away from the container's pond was served as the boarding tile
    purely on distance; the ride could never reach the pickup, and
    the hop + lock + one-step-disembark contract cycled for minutes.
    A land column at x=127 splits the lake: ferry on the far side,
    container on the near side — the ride does not exist, so the
    candidate is ``no_landing``.
    """
    containers = {
        "125,100": make_container(125, 100, 800, is_fuel=True),
    }
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(115, 135):
        for y in range(90, 110):
            terrain_data[(x, y)] = "W"
    for y in range(90, 110):
        terrain_data[(127, y)] = InMemoryTerrainMap.GROUND
    ctx = _ctx(
        fuel=_fuel_at_deficit(600),
        containers=containers,
        terrain=InMemoryTerrainMap(terrain_data=terrain_data),
    )
    ctx.world["terrain"]["129,101"] = _ferry_tile(129, 101, observed_ms=100000)

    selection = select_fuel_larder_hop(ctx, is_blacklisted=_never_blacklisted)

    assert selection["container"] is None
    assert selection["no_landing"] == 1
    assert selection["ferry_served"] == 0


def test_old_ferry_belief_still_boards() -> None:
    """Ferry memory is positional, not clocked (user ruling 2026-08-05).

    A sighting 61 s old was the OLD TTL's rot boundary; under the
    no-drift law ([[ferry-mechanics]]: 136/148 movements
    rider-attributed, zero spontaneous) an uncontradicted belief keeps
    serving — the 0x4A move pairs, re-observation patches, and
    displacement disproof are what retire it, never a clock.
    """
    containers = {
        "125,100": make_container(125, 100, 800, is_fuel=True),
    }
    ctx = _ctx(
        fuel=_fuel_at_deficit(600),
        containers=containers,
        terrain=_water_everywhere_terrain(),
    )
    ctx.world["terrain"]["122,101"] = _ferry_tile(122, 101, observed_ms=100000 - 61000)

    selection = select_fuel_larder_hop(ctx, is_blacklisted=_never_blacklisted)

    container = selection["container"]
    if container is None:
        raise AssertionError("expected the old ferry belief to board the water pickup")
    assert (container["x"], container["y"]) == (125, 100)
    assert selection["ferry_served"] == 1
    assert (selection["landing_x"], selection["landing_y"]) == (122, 101)


def test_far_ferry_is_not_a_boarding_target() -> None:
    """A ferry outside the search radius does not serve the container."""
    containers = {
        "125,100": make_container(125, 100, 800, is_fuel=True),
    }
    ctx = _ctx(
        fuel=_fuel_at_deficit(600),
        containers=containers,
        terrain=_water_everywhere_terrain(),
    )
    ctx.world["terrain"]["150,130"] = _ferry_tile(150, 130, observed_ms=100000)

    selection = select_fuel_larder_hop(ctx, is_blacklisted=_never_blacklisted)

    assert selection["container"] is None
    assert selection["no_landing"] == 1
    assert selection["ferry_served"] == 0


def test_in_viewport_water_container_is_still_larder_business() -> None:
    """A floating container in view is never ceded to the walk step.

    F5 completion (2026-08-01): the walk-territory gate handed every
    in-viewport container to the walk economics, but a water container
    is walk-unreachable from land at ANY distance — ceding it stranded
    in-view water fuel with nobody serving it. On water the larder
    keeps the candidate and its landing resolves to the ferry
    boarding tile.
    """
    containers = {
        "105,100": make_container(105, 100, 800, is_fuel=True),
    }
    terrain_data = {(x, y): "W" for x in range(103, 112) for y in range(96, 105)}
    ctx = _ctx(
        fuel=_fuel_at_deficit(600),
        containers=containers,
        terrain=InMemoryTerrainMap(terrain_data=terrain_data),
    )
    ctx.world["terrain"]["108,100"] = _ferry_tile(108, 100, observed_ms=100000)

    selection = select_fuel_larder_hop(ctx, is_blacklisted=_never_blacklisted)

    assert selection["too_close"] == 0
    assert selection["container"] == containers["105,100"]
    assert (selection["landing_x"], selection["landing_y"]) == (108, 100)
    assert selection["ferry_served"] == 1
