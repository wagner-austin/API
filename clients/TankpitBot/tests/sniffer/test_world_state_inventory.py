"""Tests for inventory tracking detail."""

from __future__ import annotations

from tankpit_bot.protocol.types import (
    EquipmentGainDict,
    EquipmentToggleDict,
    InventoryDict,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_dispatch import dispatch_world_state_update
from tankpit_bot.sniffer.world_state_inventory import (
    get_inventory_state,
    update_inventory_from_protocol,
    update_inventory_from_toggle,
)


class TestInventoryTrackingDetail:
    """Tests for inventory tracking detail."""

    def test_update_from_toggle_changes_enabled(self) -> None:
        """Test 0x74 message updates enabled flags."""
        ws = WorldService()
        update_inventory_from_protocol(
            ws,
            counts=[10, 5, 3, 2, 1],
            enabled=[True, True, True, True, True],
        )
        update_inventory_from_toggle(
            ws,
            enabled=[False, True, False, True, False],
        )

        inv = get_inventory_state(ws)
        assert inv["armor_shields"]["enabled"] is False
        assert inv["dual_shots"]["enabled"] is True
        assert inv["missile_shots"]["enabled"] is False
        assert inv["homing_shots"]["enabled"] is True
        assert inv["extra_radars"]["enabled"] is False

    def test_update_from_toggle_preserves_counts(self) -> None:
        """Test 0x74 message does not change counts."""
        ws = WorldService()
        update_inventory_from_protocol(
            ws,
            counts=[10, 5, 3, 2, 1],
            enabled=[True, True, True, True, True],
        )
        update_inventory_from_toggle(
            ws,
            enabled=[False, False, False, False, False],
        )

        inv = get_inventory_state(ws)
        assert inv["armor_shields"]["count"] == 10
        assert inv["dual_shots"]["count"] == 5
        assert inv["missile_shots"]["count"] == 3

    def test_update_from_toggle_returns_changes(self) -> None:
        """Test 0x74 message returns changes for toggled items."""
        ws = WorldService()
        update_inventory_from_protocol(
            ws,
            counts=[10, 5, 3, 2, 1],
            enabled=[True, True, True, True, True],
        )
        changes = update_inventory_from_toggle(
            ws,
            enabled=[False, True, True, True, True],
        )
        assert len(changes) == 1
        assert changes[0]["item"] == "armor_shields"
        assert changes[0]["enabled_changed"] is True
        assert changes[0]["now_enabled"] is False

    def test_update_from_protocol_logs_used_on_decrease(self) -> None:
        """Test 0x49 message with decreased counts returns negative delta."""
        ws = WorldService()
        update_inventory_from_protocol(
            ws,
            counts=[10, 5, 3, 2, 1],
            enabled=[True, True, True, True, True],
        )
        changes = update_inventory_from_protocol(
            ws,
            counts=[9, 5, 3, 2, 1],
            enabled=[True, True, True, True, True],
        )
        assert len(changes) == 1
        assert changes[0]["item"] == "armor_shields"
        assert changes[0]["delta"] == -1
        assert changes[0]["old_count"] == 10
        assert changes[0]["new_count"] == 9

    def test_a_fresh_service_carries_an_empty_inventory(self) -> None:
        """Stocked inventory belongs to the service that was stocked."""
        stocked = WorldService()
        update_inventory_from_protocol(
            stocked,
            counts=[40, 30, 20, 10, 5],
            enabled=[False, False, False, False, False],
        )
        assert get_inventory_state(stocked)["armor_shields"]["count"] == 40

        inv = get_inventory_state(WorldService())
        assert inv["armor_shields"]["count"] == 0
        assert inv["armor_shields"]["enabled"] is False

    def test_dispatch_inventory_message(self) -> None:
        """Test dispatch_world_state_update handles 0x49 message."""
        ws = WorldService()
        msg = InventoryDict(
            msg_type=0x49,
            show=False,
            alternate=True,
            counts=[40, 30, 20, 10, 5],
            enabled=[True, True, True, True, True],
        )
        dispatch_world_state_update(ws, msg)

        inv = get_inventory_state(ws)
        assert inv["armor_shields"]["count"] == 40
        assert inv["extra_radars"]["count"] == 5

    def test_dispatch_equipment_gain_message(self) -> None:
        """Test dispatch_world_state_update handles 0x67 message."""
        ws = WorldService()
        update_inventory_from_protocol(
            ws,
            counts=[10, 10, 10, 10, 10],
            enabled=[True, True, True, True, True],
        )
        msg = EquipmentGainDict(
            msg_type=0x67,
            show_message=True,
            gained=[5, 3, 0, 0, 2],
        )
        dispatch_world_state_update(ws, msg)

        inv = get_inventory_state(ws)
        assert inv["armor_shields"]["count"] == 15
        assert inv["dual_shots"]["count"] == 13
        assert inv["missile_shots"]["count"] == 10
        assert inv["extra_radars"]["count"] == 12

    def test_dispatch_equipment_toggle_message(self) -> None:
        """Test dispatch_world_state_update handles 0x74 message."""
        ws = WorldService()
        update_inventory_from_protocol(
            ws,
            counts=[10, 10, 10, 10, 10],
            enabled=[True, True, True, True, True],
        )
        msg = EquipmentToggleDict(
            msg_type=0x74,
            enabled=[False, True, False, True, False],
        )
        dispatch_world_state_update(ws, msg)

        inv = get_inventory_state(ws)
        assert inv["armor_shields"]["enabled"] is False
        assert inv["dual_shots"]["enabled"] is True
        assert inv["missile_shots"]["enabled"] is False
