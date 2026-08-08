"""Tests for the remaining strategy-cascade branches.

Radar-for-equipment, the low-fuel exploration guard, and the
hunt-only-when-full contract. ``test_strategy_coverage.py`` was 1,089
lines; locked targets and search fallbacks are now siblings.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.types import (
    AIStateDict,
)
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import (
    TankStateDict,
)
from tests.bot.ai._strategy_fixtures import (
    _c,
    _make_inventory,
    _make_world,
    _scanned_ai_state,
)


class TestRadarForEquipment:
    """Tests for radar-for-equipment in unscanned viewport."""

    def setup_method(self) -> None:
        """Reset world state."""
        self.ws = WorldService()
        self.ws.update_world_state_from_position(100, 100)

    def test_forage_radar_in_unscanned_viewport(self) -> None:
        """Equipment recovery forages with a radar in an unscanned viewport.

        The forager owns the scan path regardless of extras count: it
        fires the radar when any viewport tile is unscanned and the
        radar fuel cost is payable.
        """
        ws = self.ws
        world, self_state = _make_world(fuel=800, scanned=False)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
                "last_landing_scan_viewport": "",
            }
        )
        # default_count=15: below low (20) but above break (12) → _try_collect_equipment
        # radar_count=15: above break (12) so _try_search_critical doesn't fire
        inventory = _make_inventory(default_count=15, radar_count=15)

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["behavior"]["reason_kind"] == "scan_on_landing"
        assert decision["command"]["cmd_type"] == "radar"


class TestExplorationSkipsTeleportLowFuel:
    """Tests for exploration rejecting teleport when fuel too low."""

    def setup_method(self) -> None:
        """Reset world state."""
        self.ws = WorldService()
        self.ws.update_world_state_from_position(100, 100)

    def test_exploration_skips_teleport_when_cant_afford(self) -> None:
        """Exploration skips teleport candidates when fuel reserve too low."""
        from tankpit_bot.bot.ai.context import DecideCtx
        from tankpit_bot.bot.ai.movement_exploration import select_exploration_command
        from tests.in_memory_terrain_map import InMemoryTerrainMap

        # All edge tiles are water — only teleport is possible, but fuel is too low
        ws = self.ws
        terrain_data: dict[tuple[int, int], str] = {}
        for x in range(92, 108):
            for y in range(92, 108):
                terrain_data[(x, y)] = "W"
        terrain = InMemoryTerrainMap(terrain_data=terrain_data)

        # Fuel below the short-hop cost so even cheap exploration teleports
        # are unaffordable (reserve gate dropped 2026-06-24; the new floor
        # is raw teleport cost).
        world, self_state = _make_world(fuel=30)
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
            }
        )
        inventory = _make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "", ws=ws)

        result = select_exploration_command(ctx)

        # All candidates blocked or teleport-unaffordable → None
        assert result is None


class TestHuntOnlyWhenFull:
    """Tests for the hunt-only-when-full mode-selector contract.

    User contract (2026-07-25): "it should never hunt when its low on
    fuel or equipment. it should never hunt if it is not full on
    everything except -5 max radar." The 2026-07-13 cardinal-adjacent
    override that outranked every reserve check is deleted -- it
    produced the practice-room fight-to-death (bot traded down from
    384 fuel to 0 against a gang-up because an enemy was always one
    tile away). Ignoring an adjacent enemy while collecting is safe:
    bots never initiate, they only return fire.
    """

    def setup_method(self) -> None:
        """Reset world state."""
        self.ws = WorldService()
        self.ws.update_world_state_from_position(100, 100)

    def _make_cardinal_enemy(self) -> dict[str, TankStateDict]:
        """Return a viewport-fresh live enemy at (101,100), Manhattan 1 from (100,100)."""
        from tankpit_bot.state.types import make_tank_state

        return {
            "50": make_tank_state(
                tank_id=50,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="CardinalEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }

    def _make_distant_enemy(self) -> dict[str, TankStateDict]:
        """Return a viewport-fresh live enemy at (104,100), Manhattan 4 from (100,100)."""
        from tankpit_bot.state.types import make_tank_state

        return {
            "50": make_tank_state(
                tank_id=50,
                x=104,
                y=100,
                team=2,
                rank=1,
                name="DistantEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_wire_seen_ms=100000,
                last_position_update_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }

    def test_cardinal_enemy_cannot_divert_an_understocked_collect_tick(self) -> None:
        """An adjacent enemy never flips an under-stocked COLLECT tick.

        Bot durably in COLLECT with a fuel target lock on (105,105),
        fuel 800 (below the 1100 full threshold), and an enemy one
        tile away at (101,100). Under the deleted 2026-07-13 override
        this tick became HUNT and the bot opened a fight it could not
        fund; under the 2026-07-25 contract COLLECT keeps ownership
        and dispatches the fuel pickup. The adjacent bot is no danger:
        bots never initiate, they only return fire.
        """
        ws = self.ws
        containers = {"105,105": _c(105, 105, 400, True)}
        world, self_state = _make_world(
            fuel=800, containers=containers, tanks=self._make_cardinal_enemy()
        )
        ai_state = AIStateDict(
            **{
                **_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "APPROACH",
                "mode_started_ms": 90000,
                "resource_target_kind": "fuel",
                "resource_target_x": 105,
                "resource_target_y": 105,
            },
        )
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"
        assert decision["command"]["cmd_type"] == "pickup_fuel"

    def test_full_stock_with_adjacent_enemy_hunts(self) -> None:
        """A fully stocked bot (fuel full, weapons at cap) hunts the
        adjacent enemy through the ordinary selector path -- no
        override needed once readiness is genuine."""
        ws = self.ws
        world, self_state = _make_world(fuel=1100, tanks=self._make_cardinal_enemy())
        ai_state = _scanned_ai_state()
        inventory = _make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "HUNT"

    def test_understocked_weapons_never_hunt_even_at_full_fuel(self) -> None:
        """Full fuel with duals below cap stays COLLECT -- "never hunt
        if it is not full on everything except -5 max radar"."""
        ws = self.ws
        containers = {"105,105": _c(105, 105, 0, False)}
        world, self_state = _make_world(
            fuel=1100, containers=containers, tanks=self._make_cardinal_enemy()
        )
        ai_state = _scanned_ai_state()
        inventory = _make_inventory(dual_count=3)

        decision = decide(world, self_state, ai_state, inventory, 100000, None, ws=ws)

        assert decision["behavior"]["mode"] == "COLLECT"
