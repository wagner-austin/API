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

# The forage grid sweep checks chebyshev rings 0..12 from the tank's
# cell. Covering all cells within that ring makes plan_forage_search
# return None so the test can exercise the recovery fallback beneath it.
_FORAGE_RING_LIMIT = 12


def _exhausted_forage_cells(self_x: int, self_y: int, now_ms: int) -> dict[str, int]:
    """Return a local_scan_cells dict that covers the entire forage ring.

    When every cell within the forage search ring is marked as recently
    scanned, ``plan_forage_search`` returns ``None`` and the caller
    falls through to the edge-walk / map-intel recovery fallback.

    Args:
        self_x: Tank X coordinate.
        self_y: Tank Y coordinate.
        now_ms: Timestamp to stamp each cell with.

    Returns:
        Coverage dict keyed by ``"cx,cy"`` covering every cell in the
        forage search ring plus the tank's own cell.
    """
    center_cx = self_x // 5
    center_cy = self_y // 5
    cells: dict[str, int] = {}
    for cx in range(center_cx - _FORAGE_RING_LIMIT, center_cx + _FORAGE_RING_LIMIT + 1):
        for cy in range(center_cy - _FORAGE_RING_LIMIT, center_cy + _FORAGE_RING_LIMIT + 1):
            cells[f"{cx},{cy}"] = now_ms
    return cells


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
    inventory["dual_shots"]["count"] = 5
    inventory["homing_shots"]["count"] = 5
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = try_search_critical_equipment(ctx)

    if decision is None:
        raise AssertionError("expected forage-radar fallback decision")
    assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
    assert decision["behavior"]["reason"] == "forage_radar"
    assert decision["command"]["cmd_type"] == "radar"


def test_recover_equipment_mode_edge_walks_when_search_hop_is_unaffordable() -> None:
    """The durable owner edge-walks when forage is exhausted and no hop is affordable.

    With the forage grid fully swept and the search hop unaffordable,
    the recovery fallback produces a cheap edge walk so the bot
    keeps making progress instead of stalling.
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
            "local_scan_cells": _exhausted_forage_cells(100, 100, 100000),
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 15
    inventory["homing_shots"]["count"] = 15
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_equipment_mode(ctx)

    assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
    assert decision["behavior"]["reason"] == "edge_for_equipment"
    assert decision["command"]["cmd_type"] == "move"


def test_recover_equipment_mode_opens_map_when_fully_boxed_in() -> None:
    """A fully boxed-in owner opens the map for intel when forage is exhausted.

    Every viewport tile is water, the forage grid is swept, radar is
    exhausted, and the search hop is unaffordable -- the terminal action
    must be the free map-intel decision so the process keeps running.
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
            "local_scan_cells": _exhausted_forage_cells(100, 100, 100000),
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
    assert decision["behavior"]["reason"] == "map_intel_for_equipment"
    assert decision["command"]["cmd_type"] == "map_open"


def test_try_search_critical_equipment_edge_walks_when_hop_is_unaffordable() -> None:
    """The emergency search helper edge-walks when forage is exhausted.

    With the forage grid fully swept and the hop unaffordable, the
    emergency helper degrades to a cheap edge walk rather than raising.
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
            "local_scan_cells": _exhausted_forage_cells(100, 100, 100000),
        }
    )
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 5
    inventory["homing_shots"]["count"] = 5
    inventory["extra_radars"]["count"] = 0
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = try_search_critical_equipment(ctx)

    if decision is None:
        raise AssertionError("expected edge-walk fallback decision")
    assert decision["behavior"]["mode"] == "COLLECT_EQUIPMENT"
    assert decision["behavior"]["reason"] == "edge_for_equipment"
    assert decision["command"]["cmd_type"] == "move"


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
    """Emergency search helper senses the current viewport before teleport search."""
    world, self_state = make_world(fuel=800, scanned=False)
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 5
    inventory["homing_shots"]["count"] = 5
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
    assert decision["behavior"]["reason"] == "radar_for_equipment"


def test_try_search_critical_equipment_uses_regular_radar_when_extra_is_empty() -> None:
    """At zero extras, emergency search scans the free built-in radar.

    The forager owns the 0-extra search leg, so the scan is the
    free built-in 5x5 (reason ``forage_radar``) rather than a viewport
    sweep -- the same radar command, directed by the grid sweep that
    breaks the zero-extra-radar spiral.
    """
    world, self_state = make_world(fuel=800, scanned=False)
    inventory = make_inventory(default_count=30)
    inventory["dual_shots"]["count"] = 5
    inventory["homing_shots"]["count"] = 5
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


def test_select_equipment_target_returns_none_when_teleport_is_unaffordable() -> None:
    """Blocked equipment is rejected when teleport fallback exceeds current fuel."""
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

    assert select_equipment_target(ctx, allow_unreachable=True) is None


def test_recover_equipment_mode_approaches_known_off_viewport_equipment_before_search() -> None:
    """Known tracked equipment is pursued before generic search hops."""
    world, self_state = make_world(
        self_x=100,
        self_y=100,
        fuel=800,
        scanned=True,
        containers={
            "120,100": make_container_state(
                x=120,
                y=100,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
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
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_equipment_mode(ctx)

    assert decision["behavior"]["reason"] == "known_equipment"
    assert decision["command"]["cmd_type"] == "move"
    assert decision["command"]["target_x"] == 107
    assert decision["command"]["target_y"] == 100
    assert decision["updated_ai_state"]["attempted_equipment_targets"] == {}


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


def test_equipment_teleport_approach_records_attempt_mark() -> None:
    """A teleport approach at an equipment target writes its attempt mark.

    Regression guard for live run 20260612-071918: teleports land
    scattered and never ON a blocked container, so without the mark the
    same unreachable target was re-approached 7 times in one session.
    """
    ctx = _make_blocked_equipment_setup({})

    decision = decide_recover_equipment_mode(ctx)

    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["updated_ai_state"]["attempted_equipment_targets"] == {"103,100": 100000}


def test_select_equipment_target_skips_recently_attempted_container() -> None:
    """A live approach mark excludes the container from re-selection."""
    ctx = _make_blocked_equipment_setup({"103,100": 99000})

    assert select_equipment_target(ctx, allow_unreachable=True) is None


def test_select_equipment_target_allows_expired_attempt_mark() -> None:
    """An expired approach mark no longer vetoes the container."""
    ctx = _make_blocked_equipment_setup({"103,100": 100000 - 120001})

    decision = decide_recover_equipment_mode(ctx)

    assert decision["behavior"]["target_x"] == 103
    assert decision["behavior"]["target_y"] == 100
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["updated_ai_state"]["attempted_equipment_targets"] == {"103,100": 100000}


def test_known_equipment_skips_recently_attempted_target() -> None:
    """A marked known target falls through to generic search."""
    world, self_state = make_world(
        self_x=100,
        self_y=100,
        fuel=800,
        scanned=True,
        containers={
            "120,100": make_container_state(
                x=120,
                y=100,
                is_fuel=False,
                volume=0,
                timestamp_ms=100000,
                failed_pickups=0,
            )
        },
    )
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
            "attempted_equipment_targets": {"120,100": 99000},
        }
    )
    ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")

    decision = decide_recover_equipment_mode(ctx)

    assert decision["behavior"]["reason"] != "known_equipment"
