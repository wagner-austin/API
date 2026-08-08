"""Tests for equipment slot management.

Per-mode enable/disable decisions, slot queries, and stock checks.
"""

from __future__ import annotations

from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_inventory import (
    update_inventory_from_protocol,
)
from tests.conftest import FakeEnv


class TestBotEquipmentSlots:
    """Equipment enable/disable decisions and slot queries."""

    def test_apply_equipment_hunt_critical_enemy(self, fake_env: FakeEnv) -> None:
        """HUNT with critically damaged enemy enables radar, dual, homing."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(ws, counts, [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [2, 4, 5])
        # Enable: 5 (radar), 2 (dual), 4 (homing) = 3 toggles
        assert len(fake_cdp._sent_methods) == 3

    def test_apply_equipment_hunt_healthy_no_homing(self, fake_env: FakeEnv) -> None:
        """HUNT with healthy enemy enables radar and dual only."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(ws, counts, [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [2, 5])
        # Enable: 5 (radar), 2 (dual) = 2 toggles
        assert len(fake_cdp._sent_methods) == 2

    def test_apply_equipment_defend_enables_armor(self, fake_env: FakeEnv) -> None:
        """DEFEND mode enables radar (5) and armor (1), disables dual+homing."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(ws, counts, [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [1, 5])
        # Enable: 5 (radar), 1 (armor) = 2 toggles
        assert len(fake_cdp._sent_methods) == 2

    def test_apply_equipment_collect_fuel_critical_shields(self, fake_env: FakeEnv) -> None:
        """COLLECT with critical fuel enables radar and shields."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(ws, counts, [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [1, 5])
        # Enable: 5 (radar), 1 (shields) = 2 toggles
        assert len(fake_cdp._sent_methods) == 2

    def test_apply_equipment_collect_fuel_low_no_shields(self, fake_env: FakeEnv) -> None:
        """COLLECT with low (not critical) fuel: radar only."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(ws, counts, [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [5])
        # Enable: 5 (radar) only = 1 toggle
        assert len(fake_cdp._sent_methods) == 1

    def test_apply_equipment_patrol_only_radar(self, fake_env: FakeEnv) -> None:
        """PATROL mode only enables extra radar (5)."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(ws, counts, [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [5])
        # Enable: 5 (radar) = 1 toggle
        assert len(fake_cdp._sent_methods) == 1

    def test_apply_equipment_disables_unneeded(self, fake_env: FakeEnv) -> None:
        """Disables combat equipment when switching from HUNT to PATROL."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tests.fakes import FakeCDPSession

        # Simulate: homing+dual+radar enabled, shields disabled
        ws = WorldService()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(ws, counts, [False, True, False, True, True])
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [5])
        # Radar already on (skip), disable dual (2), disable homing (4) = 2
        assert len(fake_cdp._sent_methods) == 2

    def test_apply_equipment_no_stock_skips_enable(self, fake_env: FakeEnv) -> None:
        """Does not enable equipment when stock is depleted."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tests.fakes import FakeCDPSession

        # All counts zero, all disabled
        ws = WorldService()
        update_inventory_from_protocol(ws, [0, 0, 0, 0, 0], [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [2, 4, 5])
        # Nothing enabled — no stock available
        assert len(fake_cdp._sent_methods) == 0

    def test_is_equipment_enabled_all_slots(self, fake_env: FakeEnv) -> None:
        """is_equipment_enabled returns correct state for all 5 slots."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_toggle

        ws = WorldService()
        update_inventory_from_toggle(ws, [True, False, True, False, True])
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        assert bot.is_equipment_enabled(1) is True
        assert bot.is_equipment_enabled(2) is False
        assert bot.is_equipment_enabled(3) is True
        assert bot.is_equipment_enabled(4) is False
        assert bot.is_equipment_enabled(5) is True

    def test_disable_equipment_invalid_slot(self, fake_env: FakeEnv) -> None:
        """disable_equipment returns False for out-of-range slot."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot.disable_equipment(0) is False
        assert bot.disable_equipment(6) is False

    def test_has_equipment_stock_missile_slot(self, fake_env: FakeEnv) -> None:
        """_has_equipment_stock returns True for slot 3 (missile) with stock."""
        from tankpit_bot.bot.base import Bot

        ws = WorldService()
        update_inventory_from_protocol(ws, [0, 0, 10, 0, 0], [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        assert bot._has_equipment_stock(3) is True

    def test_has_equipment_stock_invalid_slot(self, fake_env: FakeEnv) -> None:
        """_has_equipment_stock returns False for invalid slot number."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot._has_equipment_stock(99) is False
