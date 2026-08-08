"""Integration tests for COLLECT-mode equipment search recovery."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.movement_exploration import (
    select_exploration_command,
    viewport_exploration_candidates,
)
from tankpit_bot.bot.ai.types import (
    AIConfigDict,
    AIStateDict,
    make_default_ai_config,
)
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    ContainerStateDict,
    TankStateDict,
    make_viewport_state,
)
from tests.bot.ai._collect_integration_fixtures import _enemy
from tests.bot.ai._support import (
    make_container,
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestRecoverEquipmentSearch:
    """Tests for equipment search and related recovery transitions."""

    def test_critical_equipment_search_uses_radar_when_ready(self) -> None:
        """Critical equipment depletion scans before relocating when radar is ready."""
        ws = WorldService()
        world, self_state = make_world(fuel=800, scanned=False)
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=30)

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(landing_scan_viewport=""),
            inventory,
            100000,
            None,
            ws=ws,
        )

        assert decision["behavior"]["reason_kind"] == "scan_on_landing"
        assert decision["command"]["cmd_type"] == "radar"

    def test_critical_equipment_new_unscanned_viewport_ignores_scan_cooldown(self) -> None:
        """A new unscanned viewport bypasses the global radar cooldown."""
        ws = WorldService()
        world, self_state = make_world(fuel=800, scanned=False)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(landing_scan_viewport=""),
                "last_scan_ms": 99999,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=30)

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["reason_kind"] == "scan_on_landing"
        assert decision["command"]["cmd_type"] == "radar"

    def test_critical_equipment_search_relocates_when_viewport_fully_swept(self) -> None:
        """Critical equipment depletion relocates when the viewport is fully scanned.

        The OLD gate was the global scan cooldown; the new gate is the
        tile-coverage map. A fully-covered viewport routes the bot to
        the search hop instead of re-firing the radar.
        """
        ws = WorldService()
        world, self_state = make_world(fuel=800)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 94000,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=30)

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            None,
            map_fuel_dots=((140, 100),),
            ws=ws,
        )

        assert decision["behavior"]["reason_kind"] == "search_collect_local"
        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 140
        assert decision["command"]["target_y"] == 100

    def test_equipment_search_bails_out_after_max_failures(self) -> None:
        """Critical equipment search stays in recovery after hitting the failure cap.

        With the current viewport already swept by the bot's tile-coverage
        map, the forager yields and the search-hop path runs -- exercising
        the failure-counter reset branch in ``_plan_equipment_search``.
        """
        ws = WorldService()
        world, self_state = make_world(fuel=800)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 94000,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=30)

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"

    def test_equipment_search_falls_back_to_forage_radar(self) -> None:
        """Critical equipment search forages free radar before considering teleport hops.

        Regression guard for live run 20260610-000x: this path used to
        raise and kill the bot process mid-game. The unified COLLECT
        cascade tries ``plan_forage_search`` BEFORE the teleport hop,
        so an unscanned viewport with affordable radar always produces
        a free-radar decision instead of raising.
        """
        ws = WorldService()
        world, self_state = make_world(fuel=550, scanned=False)
        config = AIConfigDict(
            **{
                **make_default_ai_config(),
            }
        )
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(landing_scan_viewport=""),
                "config": config,
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=0)

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "scan_on_landing"
        assert decision["command"]["cmd_type"] == "radar"

    def test_equipment_search_skips_when_fuel_too_low(self) -> None:
        """Equipment search defers to fuel recovery when fuel is already low."""
        ws = WorldService()
        world, self_state = make_world(fuel=150)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=0)
        inventory["extra_radars"]["count"] = 1

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            None,
            map_fuel_dots=((140, 100),),
            ws=ws,
        )

        assert decision["behavior"]["mode"] == "COLLECT"

    def test_reachable_container_behind_wall_uses_final_pickup_target(self) -> None:
        """In-viewport terrain detours preserve the final pickup target."""
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "103,100": make_container(103, 100, 700, True),
        }
        world, self_state = make_world(fuel=150, containers=containers)
        terrain = InMemoryTerrainMap({(102, 100): "#"})

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            terrain,
            ws=ws,
        )

        assert decision["command"]["cmd_type"] == "pickup_fuel"
        assert decision["command"]["target_x"] == 103
        assert decision["command"]["target_y"] == 100

    def test_low_fuel_without_targets_uses_radar(self) -> None:
        """Low fuel scans when no actionable fuel target exists."""
        ws = WorldService()
        world, self_state = make_world(fuel=150, scanned=False)

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(landing_scan_viewport=""),
            make_inventory(),
            100000,
            None,
            ws=ws,
        )

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "scan_on_landing"
        assert decision["command"]["cmd_type"] == "radar"

    def test_low_fuel_unscanned_viewport_scans_before_collecting(self) -> None:
        """An unscanned viewport scans first even when a visible fuel target exists.

        Mirrors HUNT's scan_on_landing: the COLLECT cascade fires one
        radar on a fresh-landing viewport so the planner has the full
        picture (0x5A entries plus radar reveals) before committing to
        a pickup. Without this gate the bot would commit to the first
        0x5A-visible container and miss any extra containers radar
        would have surfaced.
        """
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "104,100": make_container(104, 100, 700, True),
        }
        world, self_state = make_world(fuel=150, containers=containers)
        world["scanned_tiles"] = {}

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(landing_scan_viewport=""),
            make_inventory(),
            100000,
            None,
            ws=ws,
        )

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "scan_on_landing"
        assert decision["command"]["cmd_type"] == "radar"

    def test_low_fuel_new_unscanned_viewport_ignores_global_scan_cooldown(self) -> None:
        """A newly entered unconfirmed viewport radars immediately."""
        ws = WorldService()
        world, self_state = make_world(fuel=150, scanned=False)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(landing_scan_viewport=""),
                "last_scan_ms": 99999,
            }
        )

        decision = decide(world, self_state, ai_state, make_inventory(), 100000, None, ws=ws)

        assert decision["behavior"]["reason_kind"] == "scan_on_landing"
        assert decision["command"]["cmd_type"] == "radar"

    def test_low_fuel_fully_covered_viewport_walks_instead_of_repeating_radar(self) -> None:
        """Tile-level coverage suppresses immediate radar retry.

        The tile-aware forager treats a fully scanned viewport as
        exhausted: it returns ``None`` to the fuel-recovery owner,
        which then teleports to a fresh sector. This is the
        regression guard that replaces the old server-side
        ``mark_scan_viewport_failed`` gate -- the new gate is the
        per-tile ``world.scanned_tiles`` map populated by the
        wire-side radar handler.
        """
        ws = WorldService()
        world, self_state = make_world(fuel=150, scanned=False)
        viewport_left = world["viewport"]["left"]
        viewport_top = world["viewport"]["top"]
        viewport_right = viewport_left + world["viewport"]["width"] - 1
        viewport_bottom = viewport_top + world["viewport"]["height"] - 1
        world["scanned_tiles"] = {
            f"{x},{y}": 100000
            for y in range(viewport_top, viewport_bottom + 1)
            for x in range(viewport_left, viewport_right + 1)
        }
        ai_state = make_scanned_ai_state()

        decision = decide(
            world,
            self_state,
            ai_state,
            make_inventory(),
            100000,
            None,
            # In-block dot stays: at fuel 150 the sweep is gated off
            # (fuel-low), and a farther dot would be unaffordable.
            map_fuel_dots=((116, 100),),
            ws=ws,
        )

        assert decision["behavior"]["reason_kind"] == "search_collect_local"
        assert decision["command"]["cmd_type"] == "teleport"

    def test_low_fuel_blocked_search_with_visible_threats_falls_back_to_map(self) -> None:
        """Blocked low-fuel exploration does not break recovery ownership."""
        ws = WorldService()
        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=150,
            tanks={"50": _enemy(x=120, y=100)},
        )
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_scan_ms": 99999})
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "", ws=ws)
        terrain_data: dict[tuple[int, int], str] = dict.fromkeys(
            viewport_exploration_candidates(ctx),
            "W",
        )
        for candidate_x, candidate_y in viewport_exploration_candidates(ctx):
            terrain_data[(candidate_x - 1, candidate_y)] = "#"
            terrain_data[(candidate_x + 1, candidate_y)] = "#"
            terrain_data[(candidate_x, candidate_y - 1)] = "#"
            terrain_data[(candidate_x, candidate_y + 1)] = "#"
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"

    def test_exploration_candidates_omit_self_and_duplicates(self) -> None:
        """Exploration candidates omit the current tile and duplicate entries."""
        ws = WorldService()
        world, self_state = make_world(self_x=107, self_y=100, fuel=800)
        world["viewport"] = make_viewport_state(left=92, top=92, width=16, height=16)
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

        candidates = viewport_exploration_candidates(ctx)

        assert (107, 100) not in candidates
        assert len(candidates) == len(set(candidates))

    def test_exploration_skips_blocked_target_and_uses_next_candidate(self) -> None:
        """Exploration skips blocked edges and falls through to the next candidate."""
        ws = WorldService()
        world, self_state = make_world(self_x=100, self_y=100, fuel=550)
        terrain = InMemoryTerrainMap(
            terrain_data={
                (107, 107): "W",
                (106, 107): "W",
                (107, 106): "W",
                (108, 107): "#",
                (107, 108): "#",
            }
        )
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

        exploration = select_exploration_command(ctx)

        if exploration is None:
            raise AssertionError("expected exploration command")
        candidate_x, candidate_y, command = exploration
        assert (candidate_x, candidate_y) != (107, 107)
        assert command["cmd_type"] in ("move", "teleport")

    def test_locked_combat_with_zero_dual_releases_to_equipment(self) -> None:
        """Combat lock releases once dual shots are critically depleted."""
        ws = WorldService()
        world, self_state = make_world(fuel=800, tanks={"50": _enemy()})
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 103,
                "combat_target_y": 103,
            }
        )
        inventory = make_inventory(dual_count=0, dual_enabled=False, default_count=30)
        inventory["extra_radars"]["count"] = 30

        decision = decide(
            world,
            self_state,
            ai_state,
            inventory,
            100000,
            None,
            map_fuel_dots=((140, 100),),
            ws=ws,
        )

        assert decision["behavior"]["mode"] == "COLLECT"

    def test_killed_target_releases_combat_lock_for_recovery(self) -> None:
        """Killed locked targets release combat so recovery can proceed."""
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "101,100": make_container(101, 100, 700, True),
        }
        tanks: dict[str, TankStateDict] = {
            "50": _enemy(timestamp_ms=100000),
            "60": _enemy(tank_id=60, x=105, y=105, name="red-23", timestamp_ms=100000),
        }
        world, self_state = make_world(fuel=150, tanks=tanks, containers=containers)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_scan_ms": 99500,
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 103,
                "combat_target_y": 103,
                "killed_tank_ids": {"50": 99000},
            }
        )

        decision = decide(world, self_state, ai_state, make_inventory(), 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"

    def test_new_target_selection_skips_recently_killed_enemy(self) -> None:
        """Threat acquisition skips enemies still on the kill cooldown.

        With both enemies wire-fresh, HUNT/ACQUIRE locks the
        non-cooldowned ``LiveEnemy`` and teleports directly toward it.
        The semantic invariant is that the cooldowned ``DeadEnemy`` is
        not picked; the specific verb (teleport vs map-open-then-teleport)
        depends on per-target freshness, not on this test's intent.
        """
        ws = WorldService()
        tanks: dict[str, TankStateDict] = {
            "50": _enemy(name="red-24", timestamp_ms=100000),
            "60": _enemy(tank_id=60, x=104, y=103, name="red-25", timestamp_ms=100000),
        }
        world, self_state = make_world(fuel=1200, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "killed_tank_ids": {"50": 99900},
            }
        )

        decision = decide(world, self_state, ai_state, make_inventory(), 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "HUNT"
        # LiveEnemy sits 7 tiles away inside the viewport, so the
        # acquire engages from the current tile (in-view shot
        # short-circuit) rather than teleporting.
        assert decision["behavior"]["reason_kind"] == "shoot_target"
        assert decision["behavior"]["reason_context"]["target_name"] == "red-25"
        assert "red-24" not in decision["behavior"]["reason_kind"]
