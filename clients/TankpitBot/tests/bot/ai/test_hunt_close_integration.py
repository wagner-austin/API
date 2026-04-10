"""Close-phase integration tests for HUNT routing through decide()."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import combat_landing_tile as _combat_landing_tile
from tankpit_bot.bot.ai.context import DecideCtx, filter_killed_tanks
from tankpit_bot.bot.ai.types import AIStateDict, EnemyThreatDict
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
from tests.fakes import FakeTerrainMap


class TestDecideTeleportToFarTarget:
    """Tests for HUNT close/teleport integration through decide()."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_far_target_starts_with_map_open(self) -> None:
        """A new far target starts by opening the map and locking combat state."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_map_open_ms": 94000})
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_recent_map_intel_teleports_without_reopening_map(self) -> None:
        """Fresh map intel allows direct teleport into the close phase."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="MappedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_map_open_ms": 99000})
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["behavior"]["target_x"] == 119
        assert decision["behavior"]["target_y"] == 100
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_locked_phase_one_target_teleports_to_existing_enemy(self) -> None:
        """Locked targets keep their target identity during close teleports."""
        tanks: dict[str, TankStateDict] = {
            "60": make_tank_state(
                tank_id=60,
                x=101,
                y=100,
                team=2,
                rank=1,
                name="CloserEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="LockedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 120,
                "combat_target_y": 100,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["behavior"]["target_x"] == 119
        assert decision["behavior"]["target_y"] == 100
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_locked_phase_one_target_uses_passable_adjacent_combat_landing(self) -> None:
        """Close teleport selects the passable adjacent landing tile near the target."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=197,
                y=86,
                team=2,
                rank=1,
                name="LockedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = make_world(self_x=180, self_y=80, fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 197,
                "combat_target_y": 86,
            }
        )
        inventory = make_inventory()
        terrain = FakeTerrainMap(
            terrain_data={
                (197, 86): "W",
                (198, 86): "W",
                (197, 87): "W",
            }
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "teleport"
        assert decision["command"]["target_x"] == 196
        assert decision["command"]["target_y"] == 86

    def test_locked_phase_one_target_without_landing_tile_resets_target(self) -> None:
        """When no adjacent landing exists, the target is cleared and reacquisition starts."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=197,
                y=86,
                team=2,
                rank=1,
                name="LockedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = make_world(self_x=180, self_y=80, fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "last_map_open_ms": 99500,
                "combat_target_id": 50,
                "combat_target_x": 197,
                "combat_target_y": 86,
            }
        )
        inventory = make_inventory()
        terrain = FakeTerrainMap(
            terrain_data={
                (198, 86): "W",
                (196, 86): "W",
                (197, 87): "#",
                (197, 85): "#",
            }
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["updated_ai_state"]["combat_target_id"] == -1

    def test_combat_landing_tile_without_terrain_skips_out_of_bounds_candidates(self) -> None:
        """Landing selection without terrain still respects world bounds."""
        world, self_state = make_world(self_x=10, self_y=10, fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        target = EnemyThreatDict(
            tank_id=50,
            x=0,
            y=0,
            distance=20,
            damage_state=0,
            rank=1,
            team=2,
            name="EdgeEnemy",
            is_bot=False,
            timestamp_ms=0,
        )

        landing_x, landing_y = _combat_landing_tile(ctx, target)

        assert (landing_x, landing_y) == (1, 0)

    def test_missing_locked_target_confirms_before_reacquiring_new_enemy(self) -> None:
        """Missing locked targets clear stale combat state before reacquiring."""
        tanks: dict[str, TankStateDict] = {
            "60": make_tank_state(
                tank_id=60,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="NewEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "HUNT",
                "mode_state": "CLOSE",
                "mode_started_ms": 90000,
                "last_map_open_ms": 94000,
                "combat_target_id": 50,
                "combat_target_x": 110,
                "combat_target_y": 100,
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason"] == "confirm_kill"
        assert decision["updated_ai_state"]["combat_target_id"] == -1

    def test_stale_killed_target_is_not_reacquired_from_old_sighting(self) -> None:
        """Older sightings stay suppressed while kill cooldown is active."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="KilledEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=90000,
            ),
        }
        world, _ = make_world(fuel=800, tanks=tanks)

        filtered = filter_killed_tanks(world, {"50": 95000})

        assert "50" not in filtered["tanks"]

    def test_killed_target_can_return_after_newer_sighting(self) -> None:
        """Newer post-kill sightings are allowed back into the threat set."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="RespawnedEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=96000,
            ),
        }
        world, _ = make_world(fuel=800, tanks=tanks)

        filtered = filter_killed_tanks(world, {"50": 95000})

        assert "50" in filtered["tanks"]

    def test_no_teleport_when_fuel_too_low(self) -> None:
        """Teleport close is skipped when fuel cannot satisfy the guard."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="FarEnemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=0,
            ),
        }
        world, self_state = make_world(fuel=50, tanks=tanks)
        ai_state = AIStateDict(**{**make_scanned_ai_state(), "last_map_open_ms": 99500})
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] != "teleport"
