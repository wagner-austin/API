"""Tests for sniffer world state dispatch handling of container/tile messages."""

from __future__ import annotations

import logging

import pytest

from tankpit_bot.protocol import RadarContainerDict
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar


class TestDuplicatePickupBroadcast:
    """The server sends every pickup twice; only one may reach the book."""

    def test_the_second_broadcast_of_one_pickup_is_not_booked_again(self) -> None:
        """A repeated ContainerPickup body books one fuel credit, not two.

        The server broadcasts every pickup twice -- once to the picker,
        once to the world view -- both inside ~200 ms, measured at a
        43.9% duplicate rate across 13 sniff sessions. Every call to
        ``update_world_state_from_container_pickup`` records a ``pickup``
        entry in the fuel book, so without the dedup check each real
        pickup is credited to the bot twice.

        This is not hypothetical. It is the pickup double-count
        corruption that makes every run before 2026-06-24 ineligible for
        the teleport-cost validator, which is why
        ``POST_FUEL_FIX_DATE`` exists at all.
        """
        from tankpit_bot.container.types import ContainerPickupDict, ContainerPickupRecordDict

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)
        message = ContainerPickupDict(
            msg_type="container_pickup",
            pickups=(ContainerPickupRecordDict(x=50, y=60, remaining_volume=0),),
        )

        dispatch_world_state_update(ws, message)
        dispatch_world_state_update(ws, message)

        assert [entry["kind"] for entry in ws.fuel_book["entries"]] == ["pickup"]

    def test_control_two_distinct_pickups_are_both_booked(self) -> None:
        """Control: different tiles are different signatures, so both count."""
        from tankpit_bot.container.types import ContainerPickupDict, ContainerPickupRecordDict

        ws = WorldService()
        ws.update_world_state_from_position(100, 100)

        for x, y in ((50, 60), (51, 60)):
            dispatch_world_state_update(
                ws,
                ContainerPickupDict(
                    msg_type="container_pickup",
                    pickups=(ContainerPickupRecordDict(x=x, y=y, remaining_volume=0),),
                ),
            )

        assert [entry["kind"] for entry in ws.fuel_book["entries"]] == ["pickup", "pickup"]


class TestDispatchTilePatchUpdates:
    """Tests for absolute tile patch dispatch in world state."""

    def test_dispatch_overlay_update_creates_mine_and_keeps_terrain(self) -> None:
        """Top-level 0x40 lifts overlay bytes into world.mines and leaves terrain alone."""
        from tankpit_bot.protocol import OverlayUpdateDict, TerrainUpdateDict

        ws = WorldService()
        dispatch_world_state_update(ws, TerrainUpdateDict(msg_type=0x4A, updates=[(70, 80, 6)]))
        dispatch_world_state_update(ws, OverlayUpdateDict(msg_type=0x40, updates=[(70, 80, 1)]))

        tile = ws.world_state["terrain"]["70,80"]
        assert tile["terrain_type"] == 6
        # Overlay byte 1 = mine present, team = 1 (low 2 bits).
        mine = ws.world_state["mines"]["70,80"]
        assert mine["team"] == 1
        assert mine["source"] == "viewport"

    def test_dispatch_overlay_clear_removes_existing_mine(self) -> None:
        """0x40 with overlay byte >= 8 explicitly empties the mine layer."""
        from tankpit_bot.protocol import OverlayUpdateDict

        ws = WorldService()
        dispatch_world_state_update(ws, OverlayUpdateDict(msg_type=0x40, updates=[(70, 80, 2)]))
        assert "70,80" in ws.world_state["mines"]

        dispatch_world_state_update(ws, OverlayUpdateDict(msg_type=0x40, updates=[(70, 80, 255)]))
        assert "70,80" not in ws.world_state["mines"]

    def test_dispatch_cache_update_creates_container(self) -> None:
        """Top-level 0x43 lifts cache bytes directly into world.containers."""
        from tankpit_bot.protocol import CacheUpdateDict

        ws = WorldService()
        dispatch_world_state_update(ws, CacheUpdateDict(msg_type=0x43, updates=[(33, 44, 600)]))

        container = ws.world_state["containers"]["33,44"]
        assert container["is_fuel"] is True
        assert container["volume"] == 600
        assert container["source"] == "viewport"
        assert container["refresh_kind"] == "viewport_patch"

        dispatch_world_state_update(ws, CacheUpdateDict(msg_type=0x43, updates=[(33, 44, 0)]))

        assert "33,44" not in ws.world_state["containers"]

    def test_dispatch_cache_clear_removes_previously_radared_container(self) -> None:
        """A 0x43 cache_value=0 supersedes a radar entry: the tile is now empty.

        Per-tile wire updates are authoritative for the tiles they
        enumerate. The radar may have seen a container before pickup;
        the 0x43 CacheUpdate is the wire's later "this tile is empty"
        signal and removes it.
        """
        from tankpit_bot.protocol import CacheUpdateDict, MovementResponseDict, RadarContainerDict
        from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar

        ws = WorldService()
        dispatch_world_state_update(
            ws,
            MovementResponseDict(
                msg_type=0x3D,
                team=1,
                tank_id=1300,
                x=33,
                y=44,
                direction=0,
                damage_state=0,
                rank=1,
                lb_score=5,
                carrying=0,
            ),
        )
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        self_state["fuel"] = 250

        update_world_state_from_radar(ws, [RadarContainerDict(x=33, y=44, volume=600)], [], [])
        assert "33,44" in ws.world_state["containers"]

        dispatch_world_state_update(ws, CacheUpdateDict(msg_type=0x43, updates=[(33, 44, 0)]))

        assert "33,44" not in ws.world_state["containers"]
        self_state = ws.world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should not be None")
        assert self_state["fuel"] == 250

    def test_dispatch_radar_scan_result_applies_container_and_mine_sections(self) -> None:
        """A 0x4F radar scan result lands both entry kinds in their registries."""
        from tankpit_bot.protocol import (
            RadarContainerDict,
            RadarMineDict,
            RadarScanResultDict,
            TerrainUpdateDict,
        )

        ws = WorldService()
        dispatch_world_state_update(ws, TerrainUpdateDict(msg_type=0x4A, updates=[(90, 91, 4)]))
        dispatch_world_state_update(
            ws,
            RadarScanResultDict(
                msg_type=0x4F,
                containers=[RadarContainerDict(x=90, y=91, volume=-1)],
                mines=[RadarMineDict(x=90, y=91, team=3)],
                mine_clears=[],
            ),
        )

        tile = ws.world_state["terrain"]["90,91"]
        assert tile["terrain_type"] == 4
        container = ws.world_state["containers"]["90,91"]
        assert container["is_fuel"] is False
        assert container["source"] == "radar"
        mine = ws.world_state["mines"]["90,91"]
        assert mine["team"] == 3
        assert mine["source"] == "radar"


class TestDispatchShootEvent:
    """Tests for 0x53 ShootEvent dispatch (protocol path)."""

    def test_own_shot_with_tank_at_target_marks_hit(self) -> None:
        """Own shot landing on a tracked tank marks confirmed hit."""
        from tankpit_bot.protocol import MovementResponseDict, ShootEventDict
        from tankpit_bot.sniffer.world_state_combat import check_and_clear_combat_hit
        from tankpit_bot.state.types import make_tank_state

        ws = WorldService()
        dispatch_world_state_update(
            ws,
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=155,
                y=154,
                direction=0,
                damage_state=0,
                rank=1,
                lb_score=5,
                carrying=0,
            ),
        )
        ws.world_state["tanks"]["534"] = make_tank_state(
            tank_id=534,
            x=155,
            y=155,
            team=3,
            rank=1,
            name="orange-8",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=1000,
        )

        msg = ShootEventDict(
            msg_type=0x53,
            team=2,
            shooter_id=1301,
            source_x=155,
            source_y=154,
            target_x=155,
            target_y=155,
            aim_x=155,
            aim_y=155,
            weapon=1,
        )
        dispatch_world_state_update(ws, msg)
        assert check_and_clear_combat_hit(ws) is True
        assert ws.last_shot_victim_id == 534

    def test_own_shot_on_empty_tile_is_miss(self) -> None:
        """A free single (weapon=0) records a miss and no victim.

        Consumption = hit (user contract 2026-07-02): the server picks
        ``weapon=0`` exactly when the shot resolves against empty
        ground and spends nothing.
        """
        from tankpit_bot.protocol import MovementResponseDict, ShootEventDict
        from tankpit_bot.sniffer.world_state_combat import (
            check_and_clear_combat_hit,
            check_and_clear_our_shot_response,
        )

        ws = WorldService()
        dispatch_world_state_update(
            ws,
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=155,
                y=154,
                direction=0,
                damage_state=0,
                rank=1,
                lb_score=5,
                carrying=0,
            ),
        )

        msg = ShootEventDict(
            msg_type=0x53,
            team=2,
            shooter_id=1301,
            source_x=155,
            source_y=154,
            target_x=170,
            target_y=174,
            aim_x=170,
            aim_y=174,
            weapon=0,
        )
        dispatch_world_state_update(ws, msg)
        assert check_and_clear_combat_hit(ws) is False
        assert check_and_clear_our_shot_response(ws) is True

    def test_shooter_id_zero_is_ignored(self) -> None:
        """shooter_id=0 (no real shooter) updates nothing."""
        from tankpit_bot.protocol import MovementResponseDict, ShootEventDict
        from tankpit_bot.sniffer.world_state_combat import (
            check_and_clear_combat_hit,
            check_and_clear_our_shot_response,
        )

        ws = WorldService()
        dispatch_world_state_update(
            ws,
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=155,
                y=154,
                direction=0,
                damage_state=0,
                rank=1,
                lb_score=5,
                carrying=0,
            ),
        )

        msg = ShootEventDict(
            msg_type=0x53,
            team=0,
            shooter_id=0,
            source_x=10,
            source_y=10,
            target_x=10,
            target_y=10,
            aim_x=10,
            aim_y=10,
            weapon=0,
        )
        dispatch_world_state_update(ws, msg)
        # shooter_id 0 falls through neither branch
        assert check_and_clear_combat_hit(ws) is False
        assert check_and_clear_our_shot_response(ws) is False

    def test_enemy_shot_updates_enemy_position(self) -> None:
        """Enemy shot updates that enemy's tracked position from src tile."""
        from tankpit_bot.protocol import MovementResponseDict, ShootEventDict
        from tankpit_bot.state.types import make_tank_state

        ws = WorldService()
        dispatch_world_state_update(
            ws,
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=155,
                y=154,
                direction=0,
                damage_state=0,
                rank=1,
                lb_score=5,
                carrying=0,
            ),
        )
        ws.world_state["tanks"]["534"] = make_tank_state(
            tank_id=534,
            x=100,
            y=100,  # stale belief
            team=3,
            rank=1,
            name="orange-8",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=1000,
        )

        msg = ShootEventDict(
            msg_type=0x53,
            team=3,
            shooter_id=534,
            source_x=155,
            source_y=155,
            target_x=155,
            target_y=154,
            aim_x=155,
            aim_y=154,
            weapon=0,
        )
        dispatch_world_state_update(ws, msg)
        # Position updated to where the enemy fired from
        assert ws.world_state["tanks"]["534"]["x"] == 155
        assert ws.world_state["tanks"]["534"]["y"] == 155

    def test_own_homing_does_not_overwrite_locked_target_position(self) -> None:
        """Own homing/missile shot leaves the locked target's registry untouched.

        User contract (2026-06-26): the bot stays put and fires homing
        at off-viewport targets repeatedly until the kill. The server's
        homing seeker resolves to wherever the target actually is, but
        that tile is often off-viewport. Overwriting the registry with
        that off-viewport coord poisons the next shoot dispatch -- the
        planner aims at the off-viewport tile and the server rejects
        with ``command_error`` because shoot commands must target a
        tile inside the 18x18 viewport (see [[shot-range]]). The
        registry keeps the last on-viewport coord, the bot keeps
        aiming there, and the server auto-tracks every homing.
        """
        from tankpit_bot.protocol import MovementResponseDict, ShootEventDict
        from tankpit_bot.state.types import make_tank_state

        ws = WorldService()
        dispatch_world_state_update(
            ws,
            MovementResponseDict(
                msg_type=0x3D,
                team=2,
                tank_id=1301,
                x=155,
                y=154,
                direction=0,
                damage_state=0,
                rank=1,
                lb_score=5,
                carrying=0,
            ),
        )
        ws.world_state["tanks"]["517"] = make_tank_state(
            tank_id=517,
            x=180,
            y=147,
            team=1,
            rank=1,
            name="purple-9",
            is_self=False,
            is_bot=True,
            damage_state=0,
            timestamp_ms=1000,
        )
        ws.last_shot_combat_target_id = 517
        msg = ShootEventDict(
            msg_type=0x53,
            team=2,
            shooter_id=1301,
            source_x=155,
            source_y=154,
            target_x=198,
            target_y=152,
            aim_x=198,
            aim_y=152,
            weapon=3,
        )

        dispatch_world_state_update(ws, msg)

        assert ws.world_state["tanks"]["517"]["x"] == 180
        assert ws.world_state["tanks"]["517"]["y"] == 147


class TestDispatchProtocolDeactivation:
    """Tests for protocol-path 0x41 Deactivation dispatch.

    0x41 moved out of container into the protocol layer 2026-06-19;
    dispatch_world_state_update routes the integer msg_type 0x41.
    """

    def test_dispatch_deactivation_marks_liveness_deactivated(self) -> None:
        """Dispatch 0x41 marks the victim ``liveness="deactivated"`` and
        preserves the death tile.

        Replaces the prior ``position-set-to-(0,0)`` sentinel with the
        explicit liveness state machine introduced 2026-06-20.
        """
        from tankpit_bot.protocol import DeactivationDict, TankEntryDict

        ws = WorldService()
        entry = TankEntryDict(
            msg_type=0x28, team=0, tank_id=900, rank=0, damage_state=0, score=0, x=100, y=100
        )
        dispatch_world_state_update(ws, entry)

        msg = DeactivationDict(
            msg_type=0x41,
            status=0,
            victim_id=900,
            promo_eligible=True,
            killer_id=1,
            is_mine_kill=False,
        )
        dispatch_world_state_update(ws, msg)

        tank = ws.world_state["tanks"]["900"]
        assert tank["liveness"] == "deactivated"
        assert tank["x"] == 100
        assert tank["y"] == 100

    # Container deactivation_death dispatch test deleted 2026-06-20 after
    # the container DeactivationDeath decoder was removed. Tank
    # deactivation flows through 0x41 Deactivation on the protocol path.


class TestIncrementContainerFailedPickups:
    """Tests for increment_container_failed_pickups."""

    def test_increments_failed_pickups(self) -> None:
        """Incrementing raises the failed_pickups counter by 1."""
        from tankpit_bot.sniffer.world_state_containers import increment_container_failed_pickups

        ws = WorldService()
        update_world_state_from_radar(ws, [RadarContainerDict(x=50, y=60, volume=100)], [], [])
        assert ws.world_state["containers"]["50,60"]["failed_pickups"] == 0
        increment_container_failed_pickups(ws, 50, 60)
        assert ws.world_state["containers"]["50,60"]["failed_pickups"] == 1
        increment_container_failed_pickups(ws, 50, 60)
        assert ws.world_state["containers"]["50,60"]["failed_pickups"] == 2

    def test_noop_for_missing_container(self) -> None:
        """Incrementing a missing container is a no-op."""
        from tankpit_bot.sniffer.world_state_containers import increment_container_failed_pickups

        ws = WorldService()
        increment_container_failed_pickups(ws, 99, 99)
        assert len(ws.world_state["containers"]) == 0


class TestRemoveContainerAt:
    """Tests for remove_container_at world state mutation."""

    def test_removes_existing_container(self) -> None:
        """remove_container_at removes a container at the given coordinates."""
        from tankpit_bot.sniffer.world_state_containers import remove_container_at

        ws = WorldService()
        update_world_state_from_radar(ws, [RadarContainerDict(x=50, y=60, volume=100)], [], [])
        assert "50,60" in ws.world_state["containers"]
        remove_container_at(ws, 50, 60)
        assert "50,60" not in ws.world_state["containers"]

    def test_noop_for_missing_container(self) -> None:
        """remove_container_at is a no-op when container doesn't exist."""
        from tankpit_bot.sniffer.world_state_containers import remove_container_at

        ws = WorldService()
        remove_container_at(ws, 99, 99)
        assert len(ws.world_state["containers"]) == 0

    def test_missing_container_does_not_log_a_removal(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Nothing was removed, so nothing claims a removal.

        The state assertion above cannot see this: ``remove_container``
        returns the SAME world-state object when the key is absent, so
        the store is untouched either way and only the log line differs.
        A "Removed unreachable container" for a container that was never
        there is a false entry in the record a session post-mortem reads
        to explain where a belief went.
        """
        from tankpit_bot.sniffer.world_state_containers import remove_container_at

        ws = WorldService()

        with caplog.at_level(logging.INFO):
            remove_container_at(ws, 99, 99)

        assert not any("Removed unreachable container" in r.message for r in caplog.records)

    def test_control_removing_a_real_container_does_log(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Control: a genuine removal is announced, so silence above is the guard."""
        from tankpit_bot.sniffer.world_state_containers import remove_container_at

        ws = WorldService()
        update_world_state_from_radar(ws, [RadarContainerDict(x=50, y=60, volume=100)], [], [])

        with caplog.at_level(logging.INFO):
            remove_container_at(ws, 50, 60)

        assert any("Removed unreachable container" in r.message for r in caplog.records)
