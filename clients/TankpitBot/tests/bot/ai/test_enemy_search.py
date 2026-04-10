"""Enemy-search and exploration fallback integration tests for decide()."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.movement import viewport_exploration_candidates
from tankpit_bot.bot.ai.types import AIStateDict, make_default_ai_config
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state.types import SelfStateDict, TankStateDict, WorldStateDict, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.fakes import FakeTerrainMap


class TestDecideMapOpen:
    """Tests for top-level enemy search routing."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_map_open_when_no_enemies(self) -> None:
        """decide() triggers map open when no live enemies are visible."""
        world, self_state = make_world(fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "find_enemies"

    def test_no_map_open_when_enemy_visible(self) -> None:
        """decide() skips generic map-open fallback when a live enemy is visible."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=105,
                y=105,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_map_open_ms": 99000})
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["reason"] != "find_enemies"

    def test_fallback_uses_radar_when_map_on_cooldown(self) -> None:
        """Fallback uses radar instead of map_open when the map is on cooldown."""
        world, self_state = make_world(fuel=800, scanned=False)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "config": make_default_ai_config(),
                "last_map_open_ms": 99000,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "radar"
        assert decision["behavior"]["reason"] == "radar_for_enemies"

    def test_fallback_walks_when_map_and_radar_on_cooldown(self) -> None:
        """Fallback walks to the viewport edge when both map and radar are cooling down."""
        world, self_state = make_world(fuel=800)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "config": make_default_ai_config(),
                "last_scan_ms": 99000,
                "last_map_open_ms": 99000,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "move"
        assert decision["behavior"]["reason"] == "edge_for_enemies"

    def test_fallback_does_not_repeat_radar_in_already_scanned_viewport(self) -> None:
        """Fallback walks instead of rescanning an already confirmed viewport."""
        world, self_state = make_world(fuel=800)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "config": make_default_ai_config(),
                "last_map_open_ms": 99000,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "move"
        assert decision["behavior"]["reason"] == "edge_for_enemies"

    def test_fallback_opens_map_when_edge_walk_blocked(self) -> None:
        """Fallback reopens the map when exploration edges are fully blocked."""
        world, self_state = make_world(fuel=800)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "config": make_default_ai_config(),
                "last_scan_ms": 99000,
                "last_map_open_ms": 99000,
            }
        )
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        terrain_data: dict[tuple[int, int], str] = {}
        for candidate_x, candidate_y in viewport_exploration_candidates(ctx):
            terrain_data[(candidate_x, candidate_y)] = "W"
            terrain_data[(candidate_x - 1, candidate_y)] = "#"
            terrain_data[(candidate_x + 1, candidate_y)] = "#"
            terrain_data[(candidate_x, candidate_y - 1)] = "#"
            terrain_data[(candidate_x, candidate_y + 1)] = "#"
        terrain = FakeTerrainMap(terrain_data=terrain_data)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "find_enemies"


class TestDecideBlockedEdgeSearch:
    """Tests for blocked viewport-edge scouting paths."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def _blocked_exploration_terrain(
        self,
        world: WorldStateDict,
        self_state: SelfStateDict,
    ) -> FakeTerrainMap:
        """Build terrain that blocks every exploration candidate and landing tile.

        Args:
            world: World state under test.
            self_state: Player state under test.

        Returns:
            FakeTerrainMap with all exploration targets and their adjacent
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
        return FakeTerrainMap(terrain_data=terrain_data)

    def test_fallback_uses_alternate_edge_when_preferred_candidate_blocked(self) -> None:
        """Fallback rotates to another edge candidate before reopening the map."""
        world, self_state = make_world(fuel=800)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "config": make_default_ai_config(),
                "last_scan_ms": 99000,
                "last_map_open_ms": 99000,
            }
        )
        inventory = make_inventory()
        terrain = FakeTerrainMap(
            terrain_data={
                (107, 107): "#",
                (106, 107): "#",
                (108, 107): "#",
                (107, 106): "#",
                (107, 108): "#",
            }
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "move"
        assert decision["behavior"]["reason"] == "edge_for_enemies"
        assert (decision["behavior"]["target_x"], decision["behavior"]["target_y"]) != (107, 107)

    def test_low_fuel_blocked_edge_search_falls_through(self) -> None:
        """Blocked edge scouting with low fuel yields to fuel recovery."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=300)
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_scan_ms": 99999})
        inventory = make_inventory()
        terrain = self._blocked_exploration_terrain(world, self_state)

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["behavior"]["mode"] == "COLLECT_FUEL"
