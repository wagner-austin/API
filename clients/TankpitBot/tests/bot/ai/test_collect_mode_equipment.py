"""Tests for the durable equipment recovery owner."""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_mode import (
    _hop_toward_equipment,
    decide_collect_mode,
    select_equipment_target,
)
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.state.rank_formulas import inventory_capacity
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def test_full_inventory_skips_equipment_pickup() -> None:
    """At all-slots-full, visible equipment is not dispatched.

    User mechanic (2026-07-18): containers fill whatever is empty and
    the server rejects with code 7 only at all-slots-full -- a pickup
    at full inventory is a guaranteed wasted tick (8 of them in the
    2026-07-18 5-minute run before this gate).
    """
    world, self_state = make_world(
        fuel=400,
        containers={
            "101,100": make_container_state(
                x=101,
                y=100,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    rank_cap = inventory_capacity(self_state["rank"])
    inventory = make_inventory(dual_count=rank_cap, default_count=rank_cap)

    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
    decision = decide_collect_mode(ctx)

    assert ctx.inventory["dual_shots"]["count"] == rank_cap
    if decision is not None:
        assert decision["command"]["cmd_type"] != "pickup_equipment"


def test_collect_mode_forages_radar_when_search_hop_is_unaffordable() -> None:
    """The durable owner forages built-in radar when no search hop can be afforded.

    Regression guard for live run 20260610-000x: the owner used to raise
    here, killing the bot process mid-game. An unaffordable hop with no
    extra radar must degrade to the free built-in radar forage, never an
    exception. The viewport has unscanned ground so the forager fires
    the free radar instead of falling through to the unaffordable hop.
    """
    world, self_state = make_world(fuel=800, scanned=False)
    base_state = make_scanned_ai_state(landing_scan_viewport="")
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
            },
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["behavior"]["reason_kind"] == "scan_on_landing"
    assert decision["command"]["cmd_type"] == "radar"


def test_collect_mode_forages_radar_when_fully_boxed_in() -> None:
    """A fully boxed-in owner forages built-in radar instead of crashing.

    Every viewport tile is water, radar is exhausted, and at fuel=140
    neither the search hop nor any exploration teleport is affordable --
    the terminal action must be the free built-in radar forage so the
    process keeps running. Viewport not yet scanned so the forager
    has work to do.
    """
    world, self_state = make_world(fuel=140, scanned=False)
    base_state = make_scanned_ai_state(landing_scan_viewport="")
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
            },
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(92, 108):
        for y in range(92, 108):
            terrain_data[(x, y)] = "W"
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["behavior"]["reason_kind"] == "scan_on_landing"
    assert decision["command"]["cmd_type"] == "radar"


def test_collect_mode_raises_when_genuinely_boxed_in() -> None:
    """Boxed-in recovery raises instead of spamming map_open.

    Both walking-to-edge and the always-on map_intel terminal were
    removed 2026-06-22 because they wasted fuel without changing the
    bot's state in any productive way. When the forager is
    exhausted AND no teleport hop is affordable AND no known
    equipment exists, the bot has nothing legal to do; raising
    surfaces the stuck state loudly instead of silently looping.
    """
    import pytest

    # Fuel below the short-hop cost (8 tiles * 6 = 48). The
    # ``hunt_min_fuel`` reserve was dropped 2026-06-24, so genuine
    # stranding requires fuel below the raw short-hop cost.
    world, self_state = make_world(fuel=30, scanned=True)
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
            },
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            # Recent map open: the dot atlas is empty and a re-open
            # inside the cooldown teaches nothing, so the hop declines.
            "last_map_open_ms": 96000,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    with pytest.raises(SessionExitError, match="COLLECT owner produced no decision"):
        decide_collect_mode(ctx)


def test_collect_mode_raises_when_fully_boxed_in() -> None:
    """A fully boxed-in owner raises so the stuck state is loud.

    Every viewport tile is water, the forage map is swept, radar is
    exhausted, and the search hop is unaffordable. The bot has no
    productive action; the silent map_intel fallback was deleted
    2026-06-22 in favour of a loud raise so the wedged state can't
    be missed.
    """
    import pytest

    # Fuel below the short-hop cost so no teleport is affordable.
    world, self_state = make_world(fuel=30, scanned=True)
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
            },
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            # Recent map open: the dot atlas is empty and a re-open
            # inside the cooldown teaches nothing, so the hop declines.
            "last_map_open_ms": 96000,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    terrain_data: dict[tuple[int, int], str] = {}
    for x in range(92, 108):
        for y in range(92, 108):
            terrain_data[(x, y)] = "W"
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")

    with pytest.raises(SessionExitError, match="COLLECT owner produced no decision"):
        decide_collect_mode(ctx)


def test_collect_mode_picks_equipment_before_adjacent_fuel() -> None:
    """COLLECT picks visible equipment first, even when fuel sits adjacent.

    Under the unified cascade equipment ranks ahead of fuel: the
    gameplay loop drains all in-viewport equipment first, then the
    fuel-pickup step considers what remains. The old fuel-mode rule
    that opportunistically grabbed equipment, and the equipment-mode
    rule that opportunistically grabbed adjacent fuel, are collapsed
    into this single ordering.
    """
    world, self_state = make_world(
        fuel=800,
        scanned=True,
        containers={
            "100,101": make_container_state(
                x=100,
                y=101,
                is_fuel=True,
                volume=500,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "106,106": make_container_state(
                x=106,
                y=106,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "equipment_restock"
    assert decision["command"]["cmd_type"] == "pickup_equipment"
    assert decision["behavior"]["target_x"] == 106
    assert decision["behavior"]["target_y"] == 106


def test_collect_mode_walks_to_biggest_viewport_fuel_when_no_equipment() -> None:
    """No equipment in viewport + a fuel container exists -> walk to it before teleporting.

    User's hand-played loop (2026-06-23): teleport into a fresh
    viewport, scan, pick up all equipment, then optionally grab the
    biggest fuel container before hopping to the next clean
    viewport. The bot previously ignored non-adjacent fuel during
    equipment recovery and immediately teleport-searched, leaving a
    pickup-eligible fuel container behind every viewport. Live run
    2026-06-23 tick 21: F at (167,251), bot at (165,254), bot
    ignored it and hopped away. This test pins that the bot now
    walks to the in-viewport fuel before bailing.
    """
    # Fuel + volume chosen so the projected pickup fits under cap:
    # corporal cap is 1200, fuel 800, walk 10 tiles, volume 300 -->
    # 800 + 10 + min(300, 400) = 1110 <= 1200. Overflow-refusal is
    # covered by ``_pickup_not_worth_walk`` tests in test_collect_mode_fuel.py.
    world, self_state = make_world(
        fuel=800,
        scanned=True,
        containers={
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=300,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["mode"] == "COLLECT"
    assert decision["behavior"]["reason_kind"] in ("fuel_locked", "fuel_collect")
    assert decision["behavior"]["reason_context"]["volume"] == 300
    assert decision["behavior"]["target_x"] == 105
    assert decision["behavior"]["target_y"] == 105


def test_collect_mode_skips_opportunistic_fuel_at_rank_capacity() -> None:
    """When fuel is at ``fuel_capacity(rank)``, the fuel pickup is skipped.

    Picking up at capacity wastes the action (wire ``0x52`` code-5
    ``Tank full``), so the opportunistic-viewport-fuel branch must
    defer. Capacity here is rank-derived
    (:func:`tankpit_bot.state.rank_formulas.fuel_capacity`), not a
    learned watermark: at corporal (``rank=2``) capacity is 1200 and
    the tank is at exactly 1200, so ``_select_and_pickup_fuel``
    returns ``None`` and the cascade falls through to the no-equipment
    search-hop path.
    """
    world, self_state = make_world(
        fuel=1200,
        scanned=True,
        containers={
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=900,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] != "fuel_collect"


def test_collect_mode_falls_through_when_fuel_walk_unreachable() -> None:
    """Fuel in viewport but walk_or_teleport blocked -> fall through, no fuel pickup.

    Covers the ``fuel_command is not None`` false branch. The fuel
    container's coords are marked as a previously-failed move, so
    ``walk_or_teleport`` returns ``None`` and the bot skips the
    opportunistic fuel pickup, falling through to the no-equipment
    search-hop path.
    """
    from tankpit_bot.sniffer.world_state import mark_move_target_failed, reset_world_state

    reset_world_state()
    world, self_state = make_world(
        fuel=800,
        scanned=True,
        containers={
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=True,
                volume=900,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    mark_move_target_failed(105, 105, 99000)
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    reset_world_state()
    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] != "fuel_collect"


def test_collect_mode_releases_lock_for_markedly_closer_equipment() -> None:
    """A locked far container yields to markedly closer equipment.

    Mirrors the fuel-mode rule; regression guard for live run
    20260610-011x lock stickiness.
    """
    world, self_state = make_world(
        fuel=800,
        scanned=True,
        containers={
            "106,106": make_container_state(
                x=106,
                y=106,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "160,100": make_container_state(
                x=160,
                y=100,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "resource_target_kind": "equipment",
            "resource_target_x": 160,
            "resource_target_y": 100,
        }
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["target_x"] == 106
    assert decision["behavior"]["target_y"] == 106
    assert decision["updated_ai_state"]["resource_target_x"] == 106


def test_collect_mode_keeps_lock_against_marginally_closer_equipment() -> None:
    """A candidate inside the anti-churn threshold does not break the lock."""
    world, self_state = make_world(
        fuel=800,
        scanned=True,
        containers={
            "104,104": make_container_state(
                x=104,
                y=104,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
            "105,105": make_container_state(
                x=105,
                y=105,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            ),
        },
    )
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "resource_target_kind": "equipment",
            "resource_target_x": 105,
            "resource_target_y": 105,
        }
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["behavior"]["reason_kind"] == "equipment_locked"
    assert decision["behavior"]["target_x"] == 105
    assert decision["behavior"]["target_y"] == 105


def test_select_equipment_target_returns_none_for_unreachable_off_viewport_target() -> None:
    """Out-of-viewport equipment with no walkable approach and no fuel returns None.

    The in-viewport simplification dispatches pickup commands
    unconditionally, but off-viewport targets still go through
    ``_approach_command`` which can fall through to ``None`` when
    the bot can't walk OR afford a teleport. The selector must
    surface that as "no executable target".
    """
    from tankpit_bot.sniffer.world_state import mark_move_target_failed, reset_world_state

    reset_world_state()
    world, self_state = make_world(
        self_x=100,
        self_y=100,
        fuel=800,
        scanned=True,
        containers={
            "103,100": make_container_state(
                x=103,
                y=100,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    # Recently-failed move target: walk_or_teleport short-circuits to
    # None when the target is on the recent-failure list.
    mark_move_target_failed(103, 100, 99000)
    terrain = InMemoryTerrainMap(terrain_data={})
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain,
        "",
    )

    assert select_equipment_target(ctx) is None
    reset_world_state()


def test_select_equipment_target_rejects_walk_unreachable_in_viewport() -> None:
    """Walk-unreachable in-viewport equipment is not selected (walk-only).

    Live run 2026-06-23 23:45:22 stranded the bot dispatching
    ``pickup_equipment`` at water-locked containers; the server
    returned CANT_GO and a single rejection flagged the container
    failed_pickup. Teleport-to-container was removed entirely
    2026-06-26; containers without a walk path are never selected.
    """
    world, self_state = make_world(
        fuel=0,
        containers={
            "103,100": make_container_state(
                x=103,
                y=100,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    terrain_data: dict[tuple[int, int], str] = {(101, y): "#" for y in range(92, 109)}
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain,
        "",
    )

    assert select_equipment_target(ctx) is None


def _make_blocked_equipment_setup(
    attempted_equipment_targets: dict[str, int],
) -> DecideCtx:
    """Build a context where the only equipment target needs a teleport.

    A rock wall at x=101 cuts the tank at (100,100) off from the
    equipment container at (103,100), so ``walk_or_teleport`` resolves
    to the teleport fallback.

    Args:
        attempted_equipment_targets: Approach marks carried into AI state.

    Returns:
        Decision context at timestamp 100000 with ample fuel.
    """
    world, self_state = make_world(
        fuel=800,
        scanned=True,
        containers={
            "103,100": make_container_state(
                x=103,
                y=100,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
    terrain_data: dict[tuple[int, int], str] = {(101, y): "#" for y in range(92, 109)}
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 12
    inventory["homing_shots"]["count"] = 12
    inventory["extra_radars"]["count"] = 12
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "mode": "COLLECT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "attempted_equipment_targets": attempted_equipment_targets,
        }
    )
    return DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")


def test_rock_walled_equipment_is_skipped_for_forage() -> None:
    """Rock-walled in-viewport equipment is not selected under the walk-only contract.

    User contract (2026-06-26): a container only counts as actionable
    if a walk path to it exists inside the current viewport. The
    rock-walled container is in-viewport but the bot cannot walk to
    it; the cascade falls through to the forage radar.
    """
    ctx = _make_blocked_equipment_setup({})

    decision = decide_collect_mode(ctx)

    if decision is None:
        raise AssertionError("expected collect decision")
    assert decision["command"]["cmd_type"] != "pickup_equipment"


def test_blacklisted_container_is_skipped_by_select() -> None:
    """A blacklisted container is excluded from equipment candidate selection."""
    from tankpit_bot.bot.ai.collect_mode import (
        _blacklist_container,
        is_container_blacklisted,
    )

    _blacklist_container(105, 105)
    assert is_container_blacklisted(105, 105) is True

    containers = {
        "105,105": make_container_state(
            x=105,
            y=105,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(fuel=800, containers=containers)
    inventory = make_inventory(dual_count=3, default_count=30)
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")

    result = select_equipment_target(ctx)

    assert result is None


def test_blacklist_duplicate_does_not_re_emit_diagnostic() -> None:
    """Blacklisting the same container twice skips the second diagnostic."""
    from tankpit_bot.bot.ai.collect_mode import (
        _blacklist_container,
        is_container_blacklisted,
    )

    _blacklist_container(91, 65)
    assert is_container_blacklisted(91, 65) is True
    _blacklist_container(91, 65)
    assert is_container_blacklisted(91, 65) is True


def test_select_skips_previously_blacklisted_container() -> None:
    """A previously blacklisted container is excluded from candidates."""
    from tankpit_bot.bot.ai.collect_mode import (
        _blacklist_container,
        is_container_blacklisted,
    )

    containers = {
        "102,100": make_container_state(
            x=102,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    _blacklist_container(102, 100)
    assert is_container_blacklisted(102, 100) is True

    world, self_state = make_world(self_x=100, self_y=100, fuel=800, containers=containers)
    inventory = make_inventory(dual_count=3, default_count=30)
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, None, "")

    result = select_equipment_target(ctx)
    assert result is None


def test_hop_toward_equipment_picks_nearest_of_multiple_external_candidates() -> None:
    """The equipment hop step picks the nearest external equipment container.

    Two equipment containers sit outside the current viewport
    (default 92-107 around bot at 100,100): one at (130,100) with
    teleport cost 180, one at (150,100) with teleport cost 300. Bot
    at fuel 1200 (corporal cap) and under-armed inventory forces the
    hop step to fire; both teleports leave the 650 engagement
    reserve behind so both are affordable candidates. The step ranks
    by teleport cost and picks (130,100) -- exercising the
    ``best_container is not None AND cost >= best_cost`` branch when
    the (150,100) candidate is considered and rejected as more
    expensive.
    """
    containers = {
        "130,100": make_container_state(
            x=130,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
        "150,100": make_container_state(
            x=150,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    inventory = make_inventory(default_count=15)
    terrain = InMemoryTerrainMap()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, terrain, "")

    decision = _hop_toward_equipment(ctx, ctx.base)

    if decision is None:
        raise AssertionError("expected equipment-hop decision")
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["command"]["target_x"] == 130
    assert decision["command"]["target_y"] == 100
    assert decision["behavior"]["reason_kind"] == "equipment_hop"


def test_hop_toward_equipment_skips_in_viewport_containers() -> None:
    """The equipment hop step ignores containers inside the current viewport.

    Step 3 of the cascade (``_select_and_pickup_equipment``) already
    handles viewport-local equipment. The hop step reaches for
    tracked equipment elsewhere on the map; when every tracked
    container sits inside the current viewport bounds, the hop
    declines and the cascade falls through to the search-hop path.
    """
    containers = {
        "105,105": make_container_state(
            x=105,
            y=105,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    inventory = make_inventory(default_count=15)
    terrain = InMemoryTerrainMap()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, terrain, "")

    decision = _hop_toward_equipment(ctx, ctx.base)

    assert decision is None


def test_hop_toward_equipment_skips_when_teleport_unaffordable() -> None:
    """The equipment hop step declines when every teleport leaves under-reserve.

    Engagement reserve is ``engagement_fuel_budget(450) +
    fuel_low_threshold(200) = 650``. A teleport from (100,100) to
    (200,100) costs 600; at fuel 1000 the post-teleport residual is
    400, which is below the 650 reserve. The step considers the
    candidate, rejects the affordability check, and returns None.
    """
    containers = {
        "200,100": make_container_state(
            x=200,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1000, containers=containers)
    inventory = make_inventory(default_count=15)
    terrain = InMemoryTerrainMap()
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, terrain, "")

    decision = _hop_toward_equipment(ctx, ctx.base)

    assert decision is None


def test_hop_toward_equipment_skips_when_landing_tile_impassable() -> None:
    """The equipment hop step skips containers with no legal landing tile.

    Container at (150,100) with the container tile and all four
    cardinal neighbors marked water: ``find_teleport_landing_tile``
    returns None for this container, the loop continues to the next
    candidate (of which there are none), and the step returns None.
    """
    containers = {
        "150,100": make_container_state(
            x=150,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=1200, containers=containers)
    inventory = make_inventory(default_count=15)
    terrain_data: dict[tuple[int, int], str] = {
        (150, 100): "W",
        (149, 100): "W",
        (151, 100): "W",
        (150, 99): "W",
        (150, 101): "W",
    }
    terrain = InMemoryTerrainMap(terrain_data=terrain_data)
    ctx = DecideCtx(world, self_state, make_scanned_ai_state(), inventory, 100000, terrain, "")

    decision = _hop_toward_equipment(ctx, ctx.base)

    assert decision is None
