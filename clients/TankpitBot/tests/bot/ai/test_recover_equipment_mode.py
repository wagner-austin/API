"""Tests for the durable equipment recovery owner."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.recover_equipment_mode import (
    decide_recover_equipment_mode,
    select_equipment_target,
    try_search_critical_equipment,
)
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap

# The tile-aware forager calls is_viewport_fully_covered on the
# default 16x16 viewport centered on the tank. Covering every tile
# in that viewport makes plan_forage_search return None so the test
# can exercise the recovery fallback beneath it.
_VIEWPORT_HALF_EXTENT = 8


def _exhausted_viewport_tiles(self_x: int, self_y: int, now_ms: int) -> dict[str, int]:
    """Return a ``local_scan_tiles`` dict covering the entire viewport.

    When every tile in the current viewport carries a fresh scan mark
    ``plan_forage_search`` returns ``None`` and the caller falls
    through to the search-hop / edge-walk / map-intel recovery
    fallback.

    Args:
        self_x: Tank X coordinate.
        self_y: Tank Y coordinate.
        now_ms: Timestamp to stamp each tile with.

    Returns:
        Coverage dict keyed by ``"x,y"`` covering every tile in the
        16x16 viewport centered on the tank.
    """
    left = self_x - _VIEWPORT_HALF_EXTENT
    top = self_y - _VIEWPORT_HALF_EXTENT
    right = self_x + _VIEWPORT_HALF_EXTENT - 1
    bottom = self_y + _VIEWPORT_HALF_EXTENT - 1
    return {f"{x},{y}": now_ms for y in range(top, bottom + 1) for x in range(left, right + 1)}


def test_recover_equipment_mode_forages_radar_when_search_hop_is_unaffordable() -> None:
    """The durable owner forages built-in radar when no search hop can be afforded.

    Regression guard for live run 20260610-000x: the owner used to raise
    here, killing the bot process mid-game. An unaffordable hop with no
    extra radar must degrade to the free built-in radar forage, never an
    exception.
    """
    world, self_state = make_world(fuel=800, scanned=True)
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
                "equip_search_hop_distance": 150,
            },
            "mode": "RECOVER_EQUIPMENT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_equipment_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
    assert decision["behavior"]["reason"] == "forage_radar"
    assert decision["command"]["cmd_type"] == "radar"


def test_recover_equipment_mode_forages_radar_when_fully_boxed_in() -> None:
    """A fully boxed-in owner forages built-in radar instead of crashing.

    Every viewport tile is water, radar is exhausted, and at fuel=140
    neither the search hop nor any exploration teleport is affordable --
    the terminal action must be the free built-in radar forage so the
    process keeps running.
    """
    world, self_state = make_world(fuel=140, scanned=True)
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
                "equip_search_hop_distance": 150,
            },
            "mode": "RECOVER_EQUIPMENT",
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

    decision = decide_recover_equipment_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
    assert decision["behavior"]["reason"] == "forage_radar"
    assert decision["command"]["cmd_type"] == "radar"


def test_try_search_critical_equipment_forages_radar_when_hop_is_unaffordable() -> None:
    """The emergency search helper forages built-in radar, never raises.

    This validates the helper-level contract separately from the durable
    owner: emergency equipment search owns the tick but cannot afford the
    hop and has no extra radars, so it must degrade to the free built-in
    radar forage.
    """
    world, self_state = make_world(fuel=800, scanned=True)
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
                "equip_search_hop_distance": 150,
            },
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 3
    inventory["homing_shots"]["count"] = 3
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = try_search_critical_equipment(ctx)

    if decision is None:
        raise AssertionError("expected forage-radar fallback decision")
    assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
    assert decision["behavior"]["reason"] == "forage_radar"
    assert decision["command"]["cmd_type"] == "radar"


def test_recover_equipment_mode_short_hops_when_standard_hop_unaffordable() -> None:
    """The durable owner does a short hop when the standard hop is unaffordable.

    With the forage grid fully swept and the standard 150-tile hop
    unaffordable at fuel=800, the bot tries a short 8-tile hop instead
    of falling back to a useless edge walk.
    """
    world, self_state = make_world(fuel=800, scanned=True)
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
                "equip_search_hop_distance": 150,
            },
            "mode": "RECOVER_EQUIPMENT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "local_scan_tiles": _exhausted_viewport_tiles(100, 100, 100000),
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_equipment_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
    assert decision["behavior"]["reason"] == "search_equipment_local"
    assert decision["command"]["cmd_type"] == "teleport"


def test_recover_equipment_mode_raises_when_genuinely_boxed_in() -> None:
    """Boxed-in recovery raises instead of spamming map_open.

    Both walking-to-edge and the always-on map_intel terminal were
    removed 2026-06-22 because they wasted fuel without changing the
    bot's state in any productive way. When the forager is
    exhausted AND no teleport hop is affordable AND no known
    equipment exists, the bot has nothing legal to do; raising
    surfaces the stuck state loudly instead of silently looping.
    """
    import pytest

    world, self_state = make_world(fuel=120, scanned=True)
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
                "equip_search_hop_distance": 150,
            },
            "mode": "RECOVER_EQUIPMENT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "local_scan_tiles": _exhausted_viewport_tiles(100, 100, 100000),
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    with pytest.raises(ValueError, match="RECOVER_EQUIPMENT owner produced no decision"):
        decide_recover_equipment_mode(ctx)


def test_recover_equipment_mode_raises_when_fully_boxed_in() -> None:
    """A fully boxed-in owner raises so the stuck state is loud.

    Every viewport tile is water, the forage map is swept, radar is
    exhausted, and the search hop is unaffordable. The bot has no
    productive action; the silent map_intel fallback was deleted
    2026-06-22 in favour of a loud raise so the wedged state can't
    be missed.
    """
    import pytest

    world, self_state = make_world(fuel=140, scanned=True)
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
                "equip_search_hop_distance": 150,
            },
            "mode": "RECOVER_EQUIPMENT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "local_scan_tiles": _exhausted_viewport_tiles(100, 100, 100000),
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

    with pytest.raises(ValueError, match="RECOVER_EQUIPMENT owner produced no decision"):
        decide_recover_equipment_mode(ctx)


def test_try_search_critical_equipment_short_hops_when_standard_unaffordable() -> None:
    """The emergency search helper does a short hop when the standard is unaffordable.

    With the forage grid fully swept and the 150-tile hop unaffordable,
    the emergency helper uses a cheap short hop to keep searching.
    """
    world, self_state = make_world(fuel=800, scanned=True)
    base_state = make_scanned_ai_state()
    ai_state = AIStateDict(
        **{
            **base_state,
            "config": {
                **base_state["config"],
                "equip_search_hop_distance": 150,
            },
            "local_scan_tiles": _exhausted_viewport_tiles(100, 100, 100000),
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 3
    inventory["homing_shots"]["count"] = 3
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = try_search_critical_equipment(ctx)

    if decision is None:
        raise AssertionError("expected short-hop decision")
    assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
    assert decision["behavior"]["reason"] == "search_equipment_local"
    assert decision["command"]["cmd_type"] == "teleport"


def test_recover_equipment_mode_grabs_adjacent_fuel_opportunistically() -> None:
    """Equipment recovery picks up fuel it is standing next to.

    Mirrors the fuel-mode rule from live run 20260610-011x: resource
    modes must not walk past the other resource kind at arm's reach.
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
            "mode": "RECOVER_EQUIPMENT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
        }
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_equipment_mode(ctx)

    assert decision["behavior"]["reason"] == "opportunistic_fuel"
    assert decision["command"]["cmd_type"] == "pickup_fuel"
    assert decision["behavior"]["target_x"] == 100
    assert decision["behavior"]["target_y"] == 101


def test_recover_equipment_mode_releases_lock_for_markedly_closer_equipment() -> None:
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
            "mode": "RECOVER_EQUIPMENT",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "resource_target_kind": "equipment",
            "resource_target_x": 160,
            "resource_target_y": 100,
        }
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_equipment_mode(ctx)

    assert decision["behavior"]["target_x"] == 106
    assert decision["behavior"]["target_y"] == 106
    assert decision["updated_ai_state"]["resource_target_x"] == 106


def test_recover_equipment_mode_keeps_lock_against_marginally_closer_equipment() -> None:
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
            "mode": "RECOVER_EQUIPMENT",
            "mode_state": "APPROACH",
            "mode_started_ms": 90000,
            "resource_target_kind": "equipment",
            "resource_target_x": 105,
            "resource_target_y": 105,
        }
    )
    inventory = make_inventory(default_count=15)
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_equipment_mode(ctx)

    assert decision["behavior"]["reason"] == "equipment_locked"
    assert decision["behavior"]["target_x"] == 105
    assert decision["behavior"]["target_y"] == 105


def test_try_search_critical_equipment_returns_none_when_not_in_emergency() -> None:
    """Emergency search helper is a no-op when reserves are not broken."""
    world, self_state = make_world(fuel=800, scanned=True)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        None,
        "",
    )

    assert try_search_critical_equipment(ctx) is None


def test_try_search_critical_equipment_returns_radar_when_scan_is_needed() -> None:
    """Emergency search helper senses the current viewport before teleport search.

    With extras > 0 the forager dispatches an extra radar (whole viewport
    revealed) -- the same wire command as the free 5x5 scan, but the
    server consumes one extra and the coverage map fills with all 256
    viewport tiles.
    """
    world, self_state = make_world(fuel=800, scanned=False)
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 3
    inventory["homing_shots"]["count"] = 3
    inventory["extra_radars"]["count"] = 5
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        None,
        "",
    )

    decision = try_search_critical_equipment(ctx)

    if decision is None:
        raise AssertionError("expected radar search decision")
    assert decision["command"]["cmd_type"] == "radar"
    assert decision["behavior"]["reason"] == "forage_radar"
    # Extras > 0 -- the recorded coverage matches the whole 16x16 viewport.
    assert len(decision["updated_ai_state"]["local_scan_tiles"]) == 16 * 16


def test_try_search_critical_equipment_does_not_spam_radar_in_covered_viewport() -> None:
    """Forager respects the viewport-level coverage map.

    Live capture 2026-06-21 19:46:33+: bot fired the radar every 2 s
    for 80+ s after a failed pickup, because the old gate checked a
    server-side viewport-scan flag that a 5x5 scan never closes. The
    tile-aware forager replaces that gate with
    ``is_viewport_fully_covered(local_scan_tiles, ...)``. When every
    tile in the current viewport has a fresh scan mark the forager
    returns ``None`` and the caller hops to a fresh sector instead of
    re-firing.
    """
    world, self_state = make_world(self_x=131, self_y=126, fuel=800, scanned=False)
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 3
    inventory["homing_shots"]["count"] = 3
    inventory["extra_radars"]["count"] = 5
    ai_state = AIStateDict(
        **{
            **make_scanned_ai_state(),
            "local_scan_tiles": _exhausted_viewport_tiles(131, 126, 99500),
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = try_search_critical_equipment(ctx)

    if decision is None:
        raise AssertionError("expected a fallback search decision, got None")
    assert decision["command"]["cmd_type"] != "radar", (
        "bot fired radar on a fully-covered viewport; the radar-spam regression "
        "from 2026-06-21 19:46:33 has come back"
    )


def test_try_search_critical_equipment_uses_regular_radar_when_extra_is_empty() -> None:
    """At zero extras, emergency search scans the free built-in radar.

    The forager is the only scan path, so the scan is the free
    built-in 5x5 (reason ``forage_radar``). Same radar command, with
    the coverage map recording only the intersection of (tank±2) with
    the viewport bounds.
    """
    world, self_state = make_world(fuel=800, scanned=False)
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 3
    inventory["homing_shots"]["count"] = 3
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        inventory,
        100000,
        None,
        "",
    )

    decision = try_search_critical_equipment(ctx)

    if decision is None:
        raise AssertionError("expected built-in-radar forage decision")
    assert decision["command"]["cmd_type"] == "radar"
    assert decision["behavior"]["reason"] == "forage_radar"


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

    assert select_equipment_target(ctx, allow_unreachable=True) is None
    reset_world_state()


def test_select_equipment_target_dispatches_pickup_for_in_viewport_target() -> None:
    """In-viewport equipment dispatches pickup_equipment regardless of walkability.

    Pre-2026-06-21 the bot tried to walk-or-teleport to the
    container; rock walls + no fuel for teleport meant it gave up
    (returned ``None``). The new path is simpler: ``pickup_equipment``
    is one command and the server handles the routing, so the
    decision is "is the container in the viewport" -- if yes,
    dispatch and let the server walk.
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

    selected = select_equipment_target(ctx, allow_unreachable=True)
    if selected is None:
        raise AssertionError("expected select_equipment_target to dispatch a pickup")
    _container, command = selected
    assert command["cmd_type"] == "pickup_equipment"
    assert command["target_x"] == 103
    assert command["target_y"] == 100


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
            "mode": "RECOVER_EQUIPMENT",
            "mode_state": "SEARCH",
            "mode_started_ms": 90000,
            "attempted_equipment_targets": attempted_equipment_targets,
        }
    )
    return DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")


def test_equipment_approach_dispatches_pickup_without_attempt_mark() -> None:
    """Pickup commands rely on server reject signals, not the attempt mark.

    The attempt-mark mechanism was added to prevent teleport orbits
    around blocked containers. ``pickup_equipment`` is server-routed
    -- if the container can't be reached or is empty, the server
    surfaces that via the ``failed_pickups`` counter on the
    container, which the planner already gates on. The attempt
    mark stays teleport-only to avoid double-bookkeeping.
    """
    ctx = _make_blocked_equipment_setup({})

    decision = decide_recover_equipment_mode(ctx)

    assert decision["command"]["cmd_type"] == "pickup_equipment"
    assert decision["updated_ai_state"]["attempted_equipment_targets"] == {}


def test_select_equipment_target_skips_recently_attempted_container() -> None:
    """A live approach mark excludes the container from re-selection."""
    ctx = _make_blocked_equipment_setup({"103,100": 99000})

    assert select_equipment_target(ctx, allow_unreachable=True) is None


def test_select_equipment_target_allows_expired_attempt_mark() -> None:
    """An expired approach mark no longer vetoes the container."""
    expired_mark = {"103,100": 100000 - 120001}
    ctx = _make_blocked_equipment_setup(expired_mark)

    decision = decide_recover_equipment_mode(ctx)

    assert decision["behavior"]["target_x"] == 103
    assert decision["behavior"]["target_y"] == 100
    assert decision["command"]["cmd_type"] == "pickup_equipment"
    # Pickup dispatch leaves the (now-expired) mark unchanged; the
    # attempt mark is teleport-only bookkeeping.
    assert decision["updated_ai_state"]["attempted_equipment_targets"] == expired_mark


def test_blacklisted_container_is_skipped_by_select() -> None:
    """A blacklisted container is excluded from equipment candidate selection."""
    from tankpit_bot.bot.ai.recover_equipment_mode import (
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

    result = select_equipment_target(ctx, allow_unreachable=True)

    assert result is None


def test_blacklist_duplicate_does_not_re_emit_diagnostic() -> None:
    """Blacklisting the same container twice skips the second diagnostic."""
    from tankpit_bot.bot.ai.recover_equipment_mode import (
        _blacklist_container,
        is_container_blacklisted,
    )

    _blacklist_container(91, 65)
    assert is_container_blacklisted(91, 65) is True
    _blacklist_container(91, 65)
    assert is_container_blacklisted(91, 65) is True


def test_select_skips_previously_blacklisted_container() -> None:
    """A previously blacklisted container is excluded from candidates."""
    from tankpit_bot.bot.ai.recover_equipment_mode import (
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

    result = select_equipment_target(ctx, allow_unreachable=True)
    assert result is None
