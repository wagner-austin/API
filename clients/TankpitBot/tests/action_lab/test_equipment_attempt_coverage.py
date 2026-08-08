"""Tests for equipment-probe attempt coverage: the happy paths.

``test_equipment_attempt_coverage.py`` was 684 lines; the terminal
outcomes are now a sibling.
"""

from __future__ import annotations

from tests.action_lab._equipment_attempt_harness import (
    _world,
)

from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_protocol


class TestEquipmentProbeAttemptCoverage:
    """TestEquipmentProbeAttemptCoverage tests."""

    def setup_method(self) -> None:
        self.ws = WorldService()
        self.ws.world_state = _world()
        update_inventory_from_protocol(
            self.ws,
            [0, 0, 0, 0, 0],
            [False] * 5,
        )
