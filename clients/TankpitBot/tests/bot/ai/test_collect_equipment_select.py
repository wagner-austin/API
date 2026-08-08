"""Equipment target selection and lock-steal executability."""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
from tankpit_bot.bot.ai.collect_pickups import select_equipment_target
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_container_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


def test_select_equipment_target_returns_none_for_unreachable_off_viewport_target() -> None:
    """Out-of-viewport equipment with no walkable approach and no fuel returns None.

    The in-viewport simplification dispatches pickup commands
    unconditionally, but off-viewport targets still go through
    ``_approach_command`` which can fall through to ``None`` when
    the bot can't walk OR afford a teleport. The selector must
    surface that as "no executable target".
    """

    ws = WorldService()
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
    ws.mark_move_target_failed(103, 100, 99000)
    terrain = InMemoryTerrainMap(terrain_data={})
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        terrain,
        "",
        ws=ws,
    )

    assert select_equipment_target(ctx) is None


def test_select_equipment_target_rejects_walk_unreachable_in_viewport() -> None:
    """Walk-unreachable in-viewport equipment is not selected (walk-only).

    Live run 2026-06-23 23:45:22 stranded the bot dispatching
    ``pickup_equipment`` at water-locked containers; the server
    returned CANT_GO and a single rejection flagged the container
    failed_pickup. Teleport-to-container was removed entirely
    2026-06-26; containers without a walk path are never selected.
    """
    ws = WorldService()
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
        ws=ws,
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
    ws = WorldService()
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
    return DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "", ws=ws)


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


def test_lock_steal_requires_an_executable_candidate() -> None:
    """Closer-but-unexecutable equipment never steals a viable lock.

    Session-12 ferry livelock (2026-07-30, ~70 laps): the steal test's
    reachability said the closer container was walkable, its stolen
    lock stalled "not executable" after one disembark leg, and the
    cascade hopped back toward the original target forever -- every
    action succeeding, so no disproof could fire. Superiority now
    demands a command THIS TICK, the same bar execution applies.
    """
    from tankpit_bot.bot.ai.collect_locks import _superior_equipment_candidate

    ws = WorldService()
    containers = {
        "130,100": make_container_state(
            x=130,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
        "103,100": make_container_state(
            x=103,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=900, containers=containers)
    ws.mark_move_target_failed(103, 100, 99000)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        None,
        "",
        ws=ws,
    )

    result = _superior_equipment_candidate(ctx, containers["130,100"])

    assert result is None


def test_lock_steal_allows_an_executable_closer_candidate() -> None:
    """A genuinely executable markedly-closer candidate still steals."""
    from tankpit_bot.bot.ai.collect_locks import _superior_equipment_candidate

    ws = WorldService()
    containers = {
        "130,100": make_container_state(
            x=130,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
        "103,100": make_container_state(
            x=103,
            y=100,
            is_fuel=False,
            volume=0,
            timestamp_ms=100000,
            failed_pickups=0,
        ),
    }
    world, self_state = make_world(self_x=100, self_y=100, fuel=900, containers=containers)
    ctx = DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        None,
        "",
        ws=ws,
    )

    result = _superior_equipment_candidate(ctx, containers["130,100"])

    if result is None:
        raise AssertionError("an executable closer candidate must steal")
    assert (result["x"], result["y"]) == (103, 100)
