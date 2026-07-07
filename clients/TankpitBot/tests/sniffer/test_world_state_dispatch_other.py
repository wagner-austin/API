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

    # Container tank_registry dispatch test deleted 2026-06-20 after the
    # container TankRegistry decoder was removed. Container resource
    # truth is now sourced exclusively from 0x4F RadarResponse on the
    # protocol path.

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
            lb_score=8,
            promo_state=0,
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
            lb_score=0,
            promo_state=None,
            fuel=None,
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        if state["self_state"] is None:
            raise AssertionError("self_state should not be None")
        # Fuel unchanged — short-format TankStatusSync has no fuel
        assert state["self_state"]["fuel"] == 500

    def test_dispatch_container_pickup_full_removes_container(self) -> None:
        """Full pickup (remaining_volume=0) removes the container.

        ``remaining_volume == 0`` is the wire's "container emptied"
        signal -- the picker took everything (or it was equipment with
        no fuel). The world-state mutator drops the tile so the planner
        stops targeting it.
        """
        from tankpit_bot.container import ContainerPickupDict, ContainerPickupRecordDict
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict

        update_world_state_from_position(100, 100)

        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=80, y=90, volume=500),
        ]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(get_world_service(), containers, mines, [])
        assert "80,90" in get_world_service().world_state["containers"]

        msg = ContainerPickupDict(
            msg_type="container_pickup",
            pickups=(ContainerPickupRecordDict(x=80, y=90, remaining_volume=0),),
        )
        dispatch_world_state_update(get_world_service(), msg)
        assert "80,90" not in get_world_service().world_state["containers"]

    def test_dispatch_container_pickup_partial_updates_volume(self) -> None:
        """Partial pickup (remaining_volume>0) keeps the container at the new volume.

        Empirical 2026-06-20: when the picker hits the 1100 fuel cap
        before draining the container, the server reports the leftover.
        Pre-fix the dispatcher unconditionally removed the container
        and the bot lost track of the residual fuel until the next
        radar; the new behaviour keeps the tile in state with its
        volume reduced to ``remaining_volume`` so the planner can come
        back for the rest.

        The 0x43 ContainerPickup record DOES NOT update ``self_state["fuel"]``.
        That is the wire's job via 0x44 FuelGain / 0x2E TankStatusSync /
        0x64 FuelDeposit (the absolute-fuel messages). Computing the
        delta here on top of the wire's absolute value double-counted
        the pickup (live observation 2026-06-23: 438-volume container
        added +438 ghost beyond the wire's correct 633). The fuel-delta
        update was removed from ``pickup_container`` 2026-06-23.
        """
        from tankpit_bot.container import ContainerPickupDict, ContainerPickupRecordDict
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict

        update_world_state_from_position(100, 100)

        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=80, y=90, volume=500),
        ]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(get_world_service(), containers, mines, [])
        assert "80,90" in get_world_service().world_state["containers"]
        initial_self_state = get_world_service().world_state["self_state"]
        if initial_self_state is None:
            raise AssertionError("self_state should be populated")
        initial_fuel = initial_self_state["fuel"]

        msg = ContainerPickupDict(
            msg_type="container_pickup",
            pickups=(ContainerPickupRecordDict(x=80, y=90, remaining_volume=300),),
        )
        dispatch_world_state_update(get_world_service(), msg)

        state = get_world_service().world_state
        assert "80,90" in state["containers"]
        assert state["containers"]["80,90"]["volume"] == 300
        self_state = state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should be populated")
        assert self_state["fuel"] == initial_fuel
