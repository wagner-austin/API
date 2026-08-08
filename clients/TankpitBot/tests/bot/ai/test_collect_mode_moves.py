"""Tests for COLLECT-mode move and pickup selection."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import (
    DecideCtx,
)
from tankpit_bot.bot.ai.movement import walk_or_teleport
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import ContainerStateDict, make_mine_state
from tests.bot.ai._collect_helper_fixtures import _enemy
from tests.bot.ai._support import (
    make_container,
    make_inventory,
    make_scanned_ai_state,
    make_world,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestCollectModeMoves:
    """Tests for COLLECT-mode move and pickup selection."""

    def test_decide_picks_up_visible_edge_equipment(self) -> None:
        """Visible edge equipment is actionable without an approach step."""
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "66,63": make_container(66, 63, 0, False),
        }
        world, self_state = make_world(self_x=64, self_y=64, fuel=800, containers=containers)
        inventory = make_inventory(default_count=30)
        inventory["dual_shots"]["count"] = 0
        inventory["dual_shots"]["enabled"] = False

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(landing_scan_viewport="56,56"),
            inventory,
            100000,
            InMemoryTerrainMap(),
            ws=ws,
        )

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["command"]["cmd_type"] == "pickup_equipment"
        assert decision["command"]["target_x"] == 66
        assert decision["command"]["target_y"] == 63

    def test_low_equipment_preempts_hunt_to_restock_first(self) -> None:
        """Any weapon/radar below the resume threshold preempts HUNT.

        Per the 2026-06-22 user-defined gameplay loop ("restock -> hunt
        -> kill -> restock"), the bot enters COLLECT
        whenever ANY counter is below its resume threshold (duals < 25,
        homings < 25, radars < 20). At duals=20 and radars=20, both
        are below their resume thresholds, so HUNT yields to
        equipment recovery.
        """
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "106,106": make_container(106, 106, 30, False),
        }
        world, self_state = make_world(fuel=800, containers=containers)
        inventory = make_inventory(default_count=30)
        inventory["missile_shots"]["count"] = 5
        inventory["dual_shots"]["count"] = 20
        inventory["extra_radars"]["count"] = 20

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            inventory,
            100000,
            None,
            ws=ws,
        )

        assert decision["behavior"]["mode"] == "COLLECT"

    def test_non_emergency_equipment_low_does_not_enter_recovery_search(self) -> None:
        """Non-emergency equipment depletion leaves HUNT in charge of the tick."""
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "107,107": make_container(107, 107, 0, False),
        }
        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=1200,
            containers=containers,
            tanks={"50": _enemy(x=120, y=100)},
        )
        inventory = make_inventory(default_count=30)
        inventory["missile_shots"]["count"] = 5
        terrain = InMemoryTerrainMap(
            terrain_data={
                (107, 107): "W",
                (108, 107): "W",
                (106, 107): "W",
                (107, 108): "#",
                (107, 106): "#",
            }
        )

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            inventory,
            100000,
            terrain,
            ws=ws,
        )

        assert decision["behavior"]["mode"] == "HUNT"

    def test_non_emergency_equipment_low_keeps_hunt_even_with_visible_equipment(self) -> None:
        """Visible equipment does not override HUNT when reserves are not at break levels.

        The semantic invariant is the behavior mode label: HUNT rather
        than COLLECT. The specific HUNT command (teleport when
        the wire-sourced target position is fresh, map_open when stale)
        is incidental to the assertion.
        """
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "103,100": make_container(103, 100, 0, False),
            "106,100": make_container(106, 100, 0, False),
        }
        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=1200,
            containers=containers,
            tanks={"50": _enemy(x=103, y=100, timestamp_ms=100000)},
        )
        inventory = make_inventory(default_count=30)
        inventory["missile_shots"]["count"] = 5

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            inventory,
            100000,
            InMemoryTerrainMap(),
            ws=ws,
        )

        assert decision["behavior"]["mode"] == "HUNT"

    def test_non_emergency_equipment_low_does_not_force_outer_ring_search(self) -> None:
        """Blocked outer-ring equipment does not start COLLECT outside break thresholds.

        The semantic invariant is that the durable AI owner is HUNT --
        the bot does not abandon HUNT to chase a water-locked container
        while reserves are above break levels. Whether HUNT internally
        defers to ``decide_collect_mode`` for a refuel sub-pursuit is
        incidental: the resulting decision still carries the HUNT
        durable owner via ``apply_mode_to_decision``.
        """
        ws = WorldService()
        containers: dict[str, ContainerStateDict] = {
            "129,184": make_container(129, 184, 0, False),
        }
        world, self_state = make_world(
            self_x=138,
            self_y=192,
            fuel=1200,
            containers=containers,
            tanks={"50": _enemy(x=120, y=100)},
        )
        inventory = make_inventory(default_count=30)
        inventory["missile_shots"]["count"] = 5
        terrain = InMemoryTerrainMap(
            terrain_data={
                (130, 184): "W",
                (131, 184): "W",
                (129, 184): "W",
                (130, 185): "#",
                (130, 183): "#",
            }
        )

        decision = decide(
            world,
            self_state,
            make_scanned_ai_state(),
            inventory,
            100000,
            terrain,
            ws=ws,
        )

        assert decision["updated_ai_state"]["mode"] == "HUNT"

    def test_waypoint_clamped_to_viewport_bounds(self) -> None:
        """A* waypoints never produce moves outside the visible viewport."""
        ws = WorldService()
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        terrain = InMemoryTerrainMap(
            terrain_data={(row, col): "#" for row in range(92, 100) for col in range(92, 100)}
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

        result = walk_or_teleport(ctx, 91, 91, pickup_kind=None)

        if result is not None and result["cmd_type"] == "move":
            viewport = ctx.world["viewport"]
            left = viewport["left"]
            top = viewport["top"]
            right = left + viewport["width"] - 1
            bottom = top + viewport["height"] - 1
            assert left <= result["target_x"] <= right
            assert top <= result["target_y"] <= bottom

    def test_walk_or_teleport_rejects_failed_move_target(self) -> None:
        """Recently failed move targets are skipped."""
        ws = WorldService()
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            InMemoryTerrainMap(),
            "",
            ws=ws,
        )
        ws.mark_move_target_failed(107, 100, 90000)

        assert walk_or_teleport(ctx, 107, 100, pickup_kind=None) is None

    def test_walk_or_teleport_rejects_enemy_occupied_direct_move(self) -> None:
        """Direct moves to occupied enemy tiles are rejected."""
        ws = WorldService()
        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=150,
            tanks={"50": _enemy(x=107, y=100, timestamp_ms=100000)},
        )
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            InMemoryTerrainMap(),
            "",
            ws=ws,
        )

        assert walk_or_teleport(ctx, 107, 100, pickup_kind=None) is None

    def test_walk_or_teleport_rejects_occupied_move_without_terrain(self) -> None:
        """Enemy occupancy still blocks direct moves without terrain."""
        ws = WorldService()
        world, self_state = make_world(
            self_x=100,
            self_y=100,
            fuel=150,
            tanks={"50": _enemy(x=107, y=100, timestamp_ms=100000)},
        )
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

        assert walk_or_teleport(ctx, 107, 100, pickup_kind=None) is None

    def test_walk_or_teleport_returns_none_for_out_of_bounds_target(self) -> None:
        """Out-of-bounds target returns None via teleport fallback (landing=None)."""
        ws = WorldService()
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        terrain = InMemoryTerrainMap(
            terrain_data={
                (100, 100): InMemoryTerrainMap.GROUND,
                (101, 100): InMemoryTerrainMap.ROCK,
                (100, 101): InMemoryTerrainMap.ROCK,
                (99, 100): InMemoryTerrainMap.ROCK,
                (100, 99): InMemoryTerrainMap.ROCK,
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

        assert walk_or_teleport(ctx, 300, 300, pickup_kind="fuel") is None

    def test_walk_or_teleport_rejects_mined_move_without_terrain(self) -> None:
        """Mine occupancy blocks direct moves without terrain."""
        ws = WorldService()
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        world["mines"] = {"107,100": make_mine_state(x=107, y=100, mine_type=0, tank_id=-1, team=0)}
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

        assert walk_or_teleport(ctx, 107, 100, pickup_kind=None) is None
