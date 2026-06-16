"""Tests for sniffer world state dispatch handling of misc messages."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.sniffer.world_state import (
    get_world_service,
    reset_world_state,
    update_world_state_from_position,
)
from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar


class TestDispatchOther:
    """Tests for dispatch_world_state_update with other message types."""

    def setup_method(self) -> None:
        """Reset world state before each test."""
        reset_world_state()

    def teardown_method(self) -> None:
        """Reset world state and hooks after each test."""
        reset_world_state()
        _test_hooks.path_exists = _test_hooks._real_path_exists
        _test_hooks.load_terrain_map = _test_hooks._real_load_terrain_map

    def test_dispatch_ignores_other_messages(self) -> None:
        """Test dispatch ignores non-radar/movement messages."""
        from tankpit_bot.protocol import SyncDict

        initial_self = get_world_service().world_state["self_state"]

        msg = SyncDict(msg_type=0x3F)

        dispatch_world_state_update(get_world_service(), msg)

        assert get_world_service().world_state["self_state"] == initial_self

    def test_dispatch_tank_registry_container(self) -> None:
        """Tank-registry container messages do not populate resource truth."""
        from tankpit_bot.container import TankRegistryDict
        from tankpit_bot.sniffer import viewport

        viewport.update_viewport_origin(200, 0)

        msg = TankRegistryDict(
            msg_type="tank_registry",
            flags=0,
            tank_id=1000,
            info_bytes=b"",
            team="red",
            tank_name="",
            military_rank=0,
            badge_count=0,
            is_bot=False,
            is_container=True,
            container_x=None,
            container_y=75,
            container_viewport_x=3,
            tank_y=None,
            tank_viewport_x=None,
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert "203,75" not in state["containers"]

        viewport.reset_viewport_tracking()

    def test_dispatch_fuel_gain_message(self) -> None:
        """Test dispatch handles FuelGain (0x44) message."""
        from tankpit_bot.protocol import FuelGainDict

        # First set up a position to create self_state
        update_world_state_from_position(100, 100)

        msg = FuelGainDict(msg_type=0x44, fuel_total=25, is_free=False)
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        # fuel_total is an absolute value, not a delta
        assert state["self_state"]["fuel"] == 25

    def test_dispatch_fuel_deposit_message(self) -> None:
        """Test dispatch handles FuelDeposit (0x64) message."""
        from tankpit_bot.protocol import FuelDepositDict

        # First set up a position to create self_state
        update_world_state_from_position(100, 100)

        msg = FuelDepositDict(msg_type=0x64, fuel_total=30)
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        # fuel_total is an absolute value, not a delta
        assert state["self_state"]["fuel"] == 30

    def test_dispatch_tank_status_sync_with_fuel(self) -> None:
        """Test dispatch handles TankStatusSync (0x2E) with fuel field."""
        from tankpit_bot.protocol import TankStatusSyncDict

        # First set up a position to create self_state
        update_world_state_from_position(100, 100)

        msg = TankStatusSyncDict(
            msg_type=0x2E,
            subtype=3,
            tank_id=1227,
            damage_state=2,
            rank=4,
            flags=b"\x00\x22\x84",
            leaderboard_position=8,
            fuel=1400,
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        assert state["self_state"]["fuel"] == 1400

    def test_dispatch_tank_status_sync_without_fuel(self) -> None:
        """Test dispatch ignores TankStatusSync (0x2E) without fuel (short format)."""
        from tankpit_bot.protocol import TankStatusSyncDict

        update_world_state_from_position(100, 100)
        # Set fuel to a known value first
        update_world_state_from_fuel_total(get_world_service(), 500)

        msg = TankStatusSyncDict(
            msg_type=0x2E,
            subtype=1,
            tank_id=1227,
            damage_state=2,
            rank=4,
            flags=b"\x00\x22\x84",
            leaderboard_position=0,
            fuel=None,
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        # Fuel unchanged — short-format TankStatusSync has no fuel
        assert state["self_state"]["fuel"] == 500

    def test_dispatch_container_pickup_message(self) -> None:
        """Test dispatch handles container_pickup message."""
        from tankpit_bot.container import (
            ContainerPickupDict,
            RadarContainerDict,
            RadarMineDict,
        )

        # First set up a position to create self_state
        update_world_state_from_position(100, 100)

        # Add a container via radar
        containers: list[RadarContainerDict] = [RadarContainerDict(x=80, y=90, volume=50)]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(get_world_service(), containers, mines)

        state = get_world_service().world_state
        assert "80,90" in state["containers"]

        # Dispatch container pickup
        msg = ContainerPickupDict(msg_type="container_pickup", x=80, y=90, volume=50, is_fuel=True)
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        # Container should be removed
        assert "80,90" not in state["containers"]
