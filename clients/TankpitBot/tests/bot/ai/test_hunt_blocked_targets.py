"""Blocked-target HUNT integration tests through decide()."""

from __future__ import annotations

from tankpit_bot.bot.ai.combat_strategy import combat_landing_tile as _combat_landing_tile
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict, make_default_ai_config, make_enemy_threat
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.sniffer.world_state import mark_move_target_failed, reset_world_state
from tankpit_bot.state.types import ContainerStateDict, TankStateDict, make_tank_state
from tests.bot.ai._support import make_container, make_inventory, make_scanned_ai_state, make_world
from tests.in_memory_terrain_map import InMemoryTerrainMap


class TestDecideBlockedCombatTargets:
    """Tests for blocked combat target memory."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def test_blocked_target_is_skipped_on_reacquire(self) -> None:
        """Blocked targets are not reacquired as new HUNT threats."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=1200, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "blocked_combat_targets": {"50": 99000},
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["behavior"]["reason_kind"] == "find_enemies"

    def test_boxed_target_is_shot_from_position(self) -> None:
        """A water-boxed target inside shot range is SHOT from the current tile.

        User ruling 2026-07-29: no landing is needed for an in-view
        target -- the dual fires from here. The old behavior teleported
        at the boxed tile and relied on server displacement.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=105,
                y=100,
                team=2,
                rank=1,
                name="Boxed",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_viewport_observation_ms=100000,
            ),
            "60": make_tank_state(
                tank_id=60,
                x=103,
                y=100,
                team=2,
                rank=1,
                name="Reachable",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "combat_target_id": 50,
                "combat_target_x": 105,
                "combat_target_y": 100,
                "last_map_open_ms": 99500,
            }
        )
        inventory = make_inventory()
        terrain = InMemoryTerrainMap(
            terrain_data={
                (106, 100): "W",
                (104, 100): "W",
                (105, 101): "W",
                (105, 99): "W",
            }
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_x"] == 105
        assert decision["command"]["target_y"] == 100
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_failed_combat_landing_blocks_target_and_switches(self) -> None:
        """Failed combat landing at the target's coords blocks and switches to viable one."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="FailedLanding",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_viewport_observation_ms=100000,
            ),
            "60": make_tank_state(
                tank_id=60,
                x=103,
                y=100,
                team=2,
                rank=1,
                name="Reachable",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "combat_target_id": 50,
                "combat_target_x": 120,
                "combat_target_y": 100,
            }
        )
        inventory = make_inventory()
        mark_move_target_failed(120, 100, 99000)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason_kind"] == "find_target"
        assert decision["behavior"]["reason_context"]["target_name"] == "Reachable"
        assert "50" in decision["updated_ai_state"]["blocked_combat_targets"]

    def test_failed_combat_landing_blocks_target_and_reopens_map_when_no_viable_threat(
        self,
    ) -> None:
        """Failed landing with no other enemies falls through to find_enemies map open.

        Covers ``block_combat_target_and_replan``'s no-viable-threats
        fallthrough: the locked target's landing is marked failed and
        no other enemy is in the threat list, so the bot blocks the
        target, clears the lock, and reopens the map to find new
        threats rather than firing or close-engaging.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=120,
                y=100,
                team=2,
                rank=1,
                name="OnlyTarget",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "combat_target_id": 50,
                "combat_target_x": 120,
                "combat_target_y": 100,
            }
        )
        inventory = make_inventory()
        mark_move_target_failed(120, 100, 99000)

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["command"]["cmd_type"] == "map_open"
        assert decision["behavior"]["reason_kind"] == "find_enemies"
        assert "50" in decision["updated_ai_state"]["blocked_combat_targets"]
        assert decision["updated_ai_state"]["combat_target_id"] == -1

    def test_walk_close_ringed_target_is_shot_from_position(self) -> None:
        """A walk-range water-ringed target is SHOT from the current tile.

        User ruling 2026-07-29: an in-view, in-range target needs no
        landing at all -- the ring is irrelevant to the shot.
        """
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=104,
                y=100,
                team=2,
                rank=1,
                name="RingedNear",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "combat_target_id": 50,
                "combat_target_x": 104,
                "combat_target_y": 100,
                "last_map_open_ms": 99500,
            }
        )
        inventory = make_inventory()
        terrain = InMemoryTerrainMap(
            terrain_data={
                (105, 100): "W",
                (103, 100): "W",
                (104, 101): "W",
                (104, 99): "W",
            }
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_x"] == 104
        assert decision["command"]["target_y"] == 100
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_boxed_solo_target_is_shot_from_position(self) -> None:
        """A boxed solo target inside shot range is SHOT from the current tile."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=105,
                y=100,
                team=2,
                rank=1,
                name="Boxed",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=800, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "combat_target_id": 50,
                "combat_target_x": 105,
                "combat_target_y": 100,
                "last_map_open_ms": 99500,
            }
        )
        inventory = make_inventory()
        terrain = InMemoryTerrainMap(
            terrain_data={
                (106, 100): "W",
                (104, 100): "W",
                (105, 101): "W",
                (105, 99): "W",
            }
        )

        decision = decide(world, self_state, ai_state, inventory, 100000, terrain)

        assert decision["command"]["cmd_type"] == "shoot"
        assert decision["command"]["target_x"] == 105
        assert decision["command"]["target_y"] == 100
        assert decision["updated_ai_state"]["combat_target_id"] == 50

    def test_combat_landing_skips_dynamic_occupiers(self) -> None:
        """Combat landing avoids adjacent tiles occupied by containers."""
        containers: dict[str, ContainerStateDict] = {
            "104,100": make_container(104, 100, 0, False),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=800, containers=containers)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")
        target = make_enemy_threat(
            tank_id=50,
            x=105,
            y=100,
            distance=5,
            damage_state=0,
            rank=1,
            team=2,
            name="Enemy",
            is_bot=False,
            timestamp_ms=100000,
        )

        landing = _combat_landing_tile(ctx, target)

        assert landing != (104, 100)

    def test_combat_landing_skips_adjacent_enemy_occupier(self) -> None:
        """Combat landing avoids adjacent tiles occupied by tanks."""
        tanks: dict[str, TankStateDict] = {
            "60": make_tank_state(
                tank_id=60,
                x=104,
                y=100,
                team=2,
                rank=1,
                name="Blocker",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=800, tanks=tanks)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")
        target = make_enemy_threat(
            tank_id=50,
            x=105,
            y=100,
            distance=5,
            damage_state=0,
            rank=1,
            team=2,
            name="Enemy",
            is_bot=False,
            timestamp_ms=100000,
        )

        landing = _combat_landing_tile(ctx, target)

        assert landing != (104, 100)

    def test_combat_landing_returns_target_coords_when_all_adjacent_impassable(self) -> None:
        """Combat landing returns target coords; server handles displacement."""
        world, self_state = make_world(self_x=100, self_y=100, fuel=800)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        terrain = InMemoryTerrainMap(
            terrain_data={
                (106, 100): "W",
                (104, 100): "W",
                (105, 101): "W",
                (105, 99): "W",
            }
        )
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, terrain, "")
        target = make_enemy_threat(
            tank_id=50,
            x=105,
            y=100,
            distance=5,
            damage_state=0,
            rank=1,
            team=2,
            name="Enemy",
            is_bot=False,
            timestamp_ms=100000,
        )

        landing = _combat_landing_tile(ctx, target)

        assert landing == (105, 100)

    def test_combat_landing_returns_target_coords_when_all_adjacent_occupied(self) -> None:
        """Combat landing returns target coords even when adjacent tiles occupied."""
        containers: dict[str, ContainerStateDict] = {
            "106,100": make_container(106, 100, 0, False),
            "104,100": make_container(104, 100, 0, False),
            "105,101": make_container(105, 101, 0, False),
            "105,99": make_container(105, 99, 0, False),
        }
        world, self_state = make_world(self_x=100, self_y=100, fuel=800, containers=containers)
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()
        ctx = DecideCtx(world, self_state, ai_state, inventory, 100000, None, "")
        target = make_enemy_threat(
            tank_id=50,
            x=105,
            y=100,
            distance=5,
            damage_state=0,
            rank=1,
            team=2,
            name="Enemy",
            is_bot=False,
            timestamp_ms=100000,
        )

        landing = _combat_landing_tile(ctx, target)

        assert landing == (105, 100)

    def test_blocked_target_expires_after_ttl(self) -> None:
        """Blocked combat targets expire after the cooldown window."""
        tanks: dict[str, TankStateDict] = {
            "50": make_tank_state(
                tank_id=50,
                x=103,
                y=100,
                team=2,
                rank=1,
                name="Enemy",
                is_self=False,
                is_bot=False,
                damage_state=0,
                timestamp_ms=100000,
                last_viewport_observation_ms=100000,
            ),
        }
        world, self_state = make_world(fuel=1200, tanks=tanks)
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "config": make_default_ai_config(),
                "blocked_combat_targets": {"50": 50000},
            }
        )
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "HUNT"
        # The re-acquired enemy sits 3 tiles away inside the viewport,
        # so the acquire engages from the current tile (flag 1 of run
        # bot-20260730-000030) instead of paying a close teleport.
        assert decision["behavior"]["reason_kind"] == "shoot_target"
        assert decision["behavior"]["reason_context"]["target_name"] == "Enemy"
