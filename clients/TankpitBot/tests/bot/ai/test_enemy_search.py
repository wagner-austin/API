"""Enemy-search and exploration fallback integration tests for decide()."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.movement_exploration import viewport_exploration_candidates
from tankpit_bot.bot.ai.types import (
    AIStateDict,
    make_default_ai_config,
)
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.bot.session_exit import SessionExitError
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state.types import SelfStateDict, TankStateDict, WorldStateDict, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestDecideMapOpen:
    """Tests for top-level enemy search routing."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_map_open_when_no_enemies(self) -> None:
        """decide() triggers map open when no live enemies are visible."""
        world, self_state = make_world(fuel=1200)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason_kind"] == "find_enemies"

    def test_no_map_open_when_enemy_visible(self) -> None:
        """decide() skips generic map-open fallback when a live enemy is visible.

        The enemy's wire-sourced position is fresh
        (``last_position_update_ms`` at the current tick), so HUNT can
        teleport directly instead of dropping to the find_enemies
        search.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="red-1",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_map_open_ms": 99000})
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["reason_kind"] != "find_enemies"

    def test_fallback_exits_when_fresh_map_shows_no_viable_target(self) -> None:
        """A fresh map snapshot with nothing viable ends the session.

        Replaces the pre-2026-07-02 behavior of dispatching another
        ``map_open`` every targetless tick: with the snapshot already
        fresh, another refresh cannot change the answer, so looping on
        it is exactly the churn the user rejected. The session exits
        with ``no_viable_targets`` instead. A stale snapshot still
        dispatches a refresh (see
        ``test_hunt_search_dispatches_map_open_not_radar_during_acquire``).
        """
        world, self_state = make_world(fuel=1200, scanned=False)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "config": make_default_ai_config(),
                "last_map_open_ms": 99000,
            }
        )
        inventory = make_inventory()

        with pytest.raises(SessionExitError, match="no_viable_targets"):
            decide(world, self_state, ai_state, inventory, 100000, None)


class TestDecideBlockedEdgeSearch:
    """Tests for blocked viewport-edge scouting paths."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def _blocked_exploration_terrain(
        self,
        world: WorldStateDict,
        self_state: SelfStateDict,
    ) -> InMemoryTerrainMap:
        """Build terrain that blocks every exploration candidate and landing tile.

        Args:
            world: World state under test.
            self_state: Player state under test.

        Returns:
            InMemoryTerrainMap with all exploration targets and their adjacent
            teleport landing tiles blocked.
        """
        ctx = DecideCtx(
            world,
            self_state,
            make_scanned_ai_state(),
            make_inventory(),
            100000,
            None,
            "",
        )
        terrain_data: dict[tuple[int, int], str] = {}
        for candidate_x, candidate_y in viewport_exploration_candidates(ctx):
            terrain_data[(candidate_x, candidate_y)] = "W"
            terrain_data[(candidate_x - 1, candidate_y)] = "#"
            terrain_data[(candidate_x + 1, candidate_y)] = "#"
            terrain_data[(candidate_x, candidate_y - 1)] = "#"
            terrain_data[(candidate_x, candidate_y + 1)] = "#"
        return InMemoryTerrainMap(terrain_data=terrain_data)

    def test_low_fuel_blocked_search_yields_to_fuel_recovery(self) -> None:
        """Blocked terrain with low fuel routes the tick to fuel recovery.

        The exploration helper still exists for resource recovery edge
        walks (``edge_for_fuel`` / ``edge_for_equipment``); only HUNT's
        enemy-search edge walk was removed on 2026-06-22. This test
        guards that low-fuel HUNT still delegates to fuel recovery
        rather than wandering or crashing when the viewport's
        exploration candidates are all blocked.
        """
        world, self_state = make_world(self_x=100, self_y=100, fuel=150)
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_scan_ms": 99999})
        inventory = make_inventory()
        terrain = self._blocked_exploration_terrain(world, self_state)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "COLLECT"
