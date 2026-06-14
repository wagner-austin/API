"""Tests for sniffer world state dispatch handling of container/tile messages."""

from __future__ import annotations

from tankpit_bot.container import RadarContainerDict
from tankpit_bot.sniffer import (
    dispatch_world_state_update,
    reset_world_state,
    update_world_state_from_radar,
    world_state,
)
from tankpit_bot.sniffer.world_state_combat import check_and_clear_combat_hit


class TestDispatchTilePatchUpdates:
    """Tests for absolute tile patch dispatch in world state."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_overlay_update_preserves_existing_cache_and_terrain(self) -> None:
        """Top-level 0x40 updates only the overlay layer for existing tiles."""
        from tankpit_bot.protocol import OverlayUpdateDict, TerrainUpdateDict

        dispatch_world_state_update(TerrainUpdateDict(msg_type=0x4A, updates=[(70, 80, 6)]))
        dispatch_world_state_update(OverlayUpdateDict(msg_type=0x40, updates=[(70, 80, 9)]))

        tile = world_state._world_state["terrain"]["70,80"]
        assert tile["terrain_type"] == 6
        assert tile["cache_value"] == 0
        assert tile["overlay_value"] == 9

    def test_dispatch_cache_update_updates_terrain_only(self) -> None:
        """Top-level 0x43 updates tile cache without creating targets."""
        from tankpit_bot.protocol import CacheUpdateDict

        dispatch_world_state_update(CacheUpdateDict(msg_type=0x43, updates=[(33, 44, 600)]))

        tile = world_state._world_state["terrain"]["33,44"]
        assert tile["terrain_type"] == 0
        assert tile["cache_value"] == 600
        assert tile["overlay_value"] == 255
        assert "33,44" not in world_state._world_state["containers"]

        dispatch_world_state_update(CacheUpdateDict(msg_type=0x43, updates=[(33, 44, 0)]))

        cleared_tile = world_state._world_state["terrain"]["33,44"]
        assert cleared_tile["cache_value"] == 0
        assert "33,44" not in world_state._world_state["containers"]

    def test_dispatch_cache_clear_does_not_override_radar_container(self) -> None:
        """A 0x43 cache clear does not erase radar-confirmed container truth."""
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.protocol import CacheUpdateDict, MovementResponseDict
        from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar

        dispatch_world_state_update(
            MovementResponseDict(
                msg_type=0x3D,
                team=1,
                tank_id=1300,
                x=33,
                y=44,
                direction=0,
                rank=1,
                leaderboard_position=5,
            )
        )
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        self_state["fuel"] = 250

        update_world_state_from_radar([RadarContainerDict(x=33, y=44, volume=600)], [])
        dispatch_world_state_update(CacheUpdateDict(msg_type=0x43, updates=[(33, 44, 0)]))

        assert world_state._world_state["terrain"]["33,44"]["cache_value"] == 0
        assert "33,44" in world_state._world_state["containers"]
        assert world_state._world_state["containers"]["33,44"]["volume"] == 600
        self_state = world_state._world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["fuel"] == 250

    def test_dispatch_combined_tile_update_applies_cache_and_overlay_sections(self) -> None:
        """Top-level 0x4F applies both cache and overlay sections visually only."""
        from tankpit_bot.protocol import CombinedTileUpdateDict, TerrainUpdateDict

        dispatch_world_state_update(TerrainUpdateDict(msg_type=0x4A, updates=[(90, 91, 4)]))
        dispatch_world_state_update(
            CombinedTileUpdateDict(
                msg_type=0x4F,
                cache_updates=[(90, 91, -1)],
                overlay_updates=[(90, 91, 12)],
            )
        )

        tile = world_state._world_state["terrain"]["90,91"]
        assert tile["terrain_type"] == 4
        assert tile["cache_value"] == -1
        assert tile["overlay_value"] == 12
        assert "90,91" not in world_state._world_state["containers"]


class TestDispatchContainerCombatEvents:
    """Tests for container combat events: combat_hit, deactivation_kill, deactivation_death."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_dispatch_combat_hit_marks_hit_for_self(self) -> None:
        """Dispatch combat_hit calls mark_combat_hit when attacker is self."""
        from tankpit_bot.container import CombatHitDict
        from tankpit_bot.protocol import MovementResponseDict

        # Set up self with tank_id=10
        first = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=10,
            x=100,
            y=100,
            direction=0,
            rank=2,
            leaderboard_position=5,
        )
        dispatch_world_state_update(first)

        # combat_hit where attacker_id matches self
        msg = CombatHitDict(
            msg_type="combat_hit",
            direction=0x09,
            attacker_id=10,
            combat_data=b"\x00\x01\x02\x03\x04\x05",
            is_outgoing=True,
        )
        dispatch_world_state_update(msg)

        # Verify mark_combat_hit was called
        assert check_and_clear_combat_hit() is True

    def test_dispatch_combat_hit_ignores_other_attacker(self) -> None:
        """Dispatch combat_hit does not mark hit when attacker is not self."""
        from tankpit_bot.container import CombatHitDict
        from tankpit_bot.protocol import MovementResponseDict

        # Set up self with tank_id=10
        first = MovementResponseDict(
            msg_type=0x3D,
            team=1,
            tank_id=10,
            x=100,
            y=100,
            direction=0,
            rank=2,
            leaderboard_position=5,
        )
        dispatch_world_state_update(first)

        # combat_hit from a different tank
        msg = CombatHitDict(
            msg_type="combat_hit",
            direction=0x03,
            attacker_id=99,
            combat_data=b"\x00\x01\x02\x03\x04\x05",
            is_outgoing=False,
        )
        dispatch_world_state_update(msg)

        # No hit recorded for self
        assert check_and_clear_combat_hit() is False

    def test_dispatch_deactivation_kill_invalidates_victim(self) -> None:
        """Dispatch deactivation_kill invalidates victim tank position."""
        from tankpit_bot.container import DeactivationKillDict
        from tankpit_bot.protocol import TankEntryDict

        entry = TankEntryDict(msg_type=0x28, tank_id=900, x=100, y=100, name="Killed")
        dispatch_world_state_update(entry)

        msg = DeactivationKillDict(
            msg_type="deactivation_kill",
            victim_id=900,
            killer_id=1,
        )
        dispatch_world_state_update(msg)

        assert world_state._world_state["tanks"]["900"]["x"] == 0
        assert world_state._world_state["tanks"]["900"]["y"] == 0

    def test_dispatch_deactivation_death_is_handled(self) -> None:
        """Dispatch deactivation_death is handled without error."""
        from tankpit_bot.container import DeactivationDeathDict

        msg = DeactivationDeathDict(
            msg_type="deactivation_death",
            flags=0,
            killer_id=42,
            extra_data=b"\x00\x01\x02",
        )
        dispatch_world_state_update(msg)  # should not raise


class TestIncrementContainerFailedPickups:
    """Tests for increment_container_failed_pickups."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_increments_failed_pickups(self) -> None:
        """Incrementing raises the failed_pickups counter by 1."""
        from tankpit_bot.sniffer.world_state_containers import increment_container_failed_pickups

        update_world_state_from_radar(
            [RadarContainerDict(x=50, y=60, volume=100)],
            [],
        )
        assert world_state._world_state["containers"]["50,60"]["failed_pickups"] == 0
        increment_container_failed_pickups(50, 60)
        assert world_state._world_state["containers"]["50,60"]["failed_pickups"] == 1
        increment_container_failed_pickups(50, 60)
        assert world_state._world_state["containers"]["50,60"]["failed_pickups"] == 2

    def test_noop_for_missing_container(self) -> None:
        """Incrementing a missing container is a no-op."""
        from tankpit_bot.sniffer.world_state_containers import increment_container_failed_pickups

        increment_container_failed_pickups(99, 99)
        assert len(world_state._world_state["containers"]) == 0


class TestRemoveContainerAt:
    """Tests for remove_container_at world state mutation."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state after each test."""
        reset_world_state()

    def test_removes_existing_container(self) -> None:
        """remove_container_at removes a container at the given coordinates."""
        from tankpit_bot.sniffer.world_state_containers import remove_container_at

        update_world_state_from_radar(
            [RadarContainerDict(x=50, y=60, volume=100)],
            [],
        )
        assert "50,60" in world_state._world_state["containers"]
        remove_container_at(50, 60)
        assert "50,60" not in world_state._world_state["containers"]

    def test_noop_for_missing_container(self) -> None:
        """remove_container_at is a no-op when container doesn't exist."""
        from tankpit_bot.sniffer.world_state_containers import remove_container_at

        remove_container_at(99, 99)
        assert len(world_state._world_state["containers"]) == 0
