"""Coverage tests for executor.py: _dispatch_tracked_target_command failure paths.

Exercises the ValueError for unsupported command types and the dispatched=False
branches for move, pickup_fuel, and pickup_equipment.
"""

from __future__ import annotations

import pytest

from tankpit_bot.bot.executor import _dispatch_tracked_target_command
from tankpit_bot.bot.types import (
    make_move_command,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_shoot_command,
)
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.state import WorldStateDict, make_empty_world_state


class _DispatchFailBot:
    """Bot double where move/pickup methods always return False."""

    def __init__(self) -> None:
        """Initialize with world state."""
        self._world = make_empty_world_state()
        self._cdp = None
        self._cdp_message_buffer: list[str] = []

    def get_world_state(self) -> WorldStateDict:
        """Return the injected world-state snapshot."""
        return self._world

    def move_to(self, x: int, y: int) -> bool:
        """Always fails to dispatch."""
        return False

    def pickup_fuel_to(self, x: int, y: int) -> bool:
        """Always fails to dispatch."""
        return False

    def pickup_equipment_to(self, x: int, y: int) -> bool:
        """Always fails to dispatch."""
        return False

    def teleport_to(self, x: int, y: int) -> bool:
        """Unused stub."""
        return False

    def shoot_at(self, x: int, y: int, target_id: int) -> bool:
        """Unused stub."""
        return False

    def use_radar(self) -> bool:
        """Unused stub."""
        return False

    def send_chat(self, message_id: int, x: int, y: int) -> bool:
        """Unused stub."""
        _ = (message_id, x, y)
        return False

    def open_map(self) -> bool:
        """Unused stub."""
        return False

    def close_map(self) -> bool:
        """Unused stub."""
        return False

    def captured_message_count(self) -> int:
        """Unused stub."""
        return 0

    def enable_equipment(self, slot: int) -> bool:
        """Unused stub."""
        return False

    def disable_equipment(self, slot: int) -> bool:
        """Unused stub."""
        return False

    def _has_equipment_stock(self, slot: int) -> bool:
        """Unused stub."""
        return False


class TestDispatchTrackedTargetCommand:
    """Tests for _dispatch_tracked_target_command failure paths."""

    def test_unsupported_command_type_raises_value_error(self) -> None:
        """A command type outside move/pickup_fuel/pickup_equipment raises ValueError."""
        bot = _DispatchFailBot()
        with pytest.raises(ValueError, match="Not a tracked-target command"):
            _dispatch_tracked_target_command(bot, make_shoot_command(100, 100))

    def test_move_dispatch_failure_returns_false(self) -> None:
        """When move_to returns False, the dispatch reports failure."""
        reset_world_state()
        bot = _DispatchFailBot()
        command = make_move_command(150, 160)

        result = _dispatch_tracked_target_command(bot, command)

        assert result is False

    def test_pickup_fuel_dispatch_failure_returns_false(self) -> None:
        """When pickup_fuel_to returns False, the dispatch reports failure."""
        reset_world_state()
        bot = _DispatchFailBot()
        command = make_pickup_fuel_command(80, 90)

        result = _dispatch_tracked_target_command(bot, command)

        assert result is False

    def test_pickup_equipment_dispatch_failure_returns_false(self) -> None:
        """When pickup_equipment_to returns False, the dispatch reports failure."""
        reset_world_state()
        bot = _DispatchFailBot()
        command = make_pickup_equipment_command(120, 130)

        result = _dispatch_tracked_target_command(bot, command)

        assert result is False
