"""Tests for executor module: apply_equipment and dispatch_command."""

from __future__ import annotations

from tankpit_bot.bot.ai.types import make_behavior_score, make_initial_ai_state
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.executor import apply_equipment, dispatch_command, execute
from tankpit_bot.bot.tick_loop_types import make_tick_decision
from tankpit_bot.bot.types import (
    make_map_open_command,
    make_move_command,
    make_pickup_move_command,
    make_radar_command,
    make_shoot_command,
    make_teleport_command,
)
from tankpit_bot.sniffer.world_state import (
    reset_world_state,
    update_inventory_from_toggle,
    update_world_state_from_fuel_total,
    update_world_state_from_position,
)
from tests.conftest import FakeEnv
from tests.fakes import FakeCDPSession


def _make_bot(fake_env: FakeEnv) -> tuple[Bot, FakeCDPSession]:
    """Create a Bot with FakeCDPSession in IDLE state."""
    reset_world_state()
    update_world_state_from_position(100, 100)
    update_world_state_from_fuel_total(800)
    bot = Bot("https://test.tankpit.com/", headless=True)
    fake_cdp = FakeCDPSession()
    bot._cdp = fake_cdp
    bot._state_data = bot._state_data.copy()
    bot._state_data["state"] = "IDLE"
    return bot, fake_cdp


class TestApplyEquipment:
    """Tests for apply_equipment."""

    def test_enables_desired_slots(self, fake_env: FakeEnv) -> None:
        """Enables desired combat slots that have stock."""
        from tankpit_bot.sniffer.world_state import update_inventory_from_gain

        bot, fake_cdp = _make_bot(fake_env)
        # Give slot 2 (dual) stock so _has_equipment_stock returns True
        update_inventory_from_gain([0, 5, 0, 0, 0])
        # Disable slot 2 via toggle so enable triggers a toggle command
        update_inventory_from_toggle([True, False, True, True, True])
        apply_equipment(bot, [2, 5])
        # slot 1: not desired, enabled → disable → toggle (1 CDP)
        # slot 2: desired, disabled, has stock → enable → toggle (1 CDP)
        # slot 4: not desired, enabled → disable → toggle (1 CDP)
        assert fake_cdp._sent_methods.count("Runtime.evaluate") == 3

    def test_disables_undesired_slots(self, fake_env: FakeEnv) -> None:
        """Disables combat slots not in desired list."""
        bot, fake_cdp = _make_bot(fake_env)
        # Set all slots to enabled so we can test disabling
        update_inventory_from_toggle([True, True, True, True, True])
        apply_equipment(bot, [5])
        # Should disable slots 1, 2, 4 (3 CDP calls)
        assert fake_cdp._sent_methods.count("Runtime.evaluate") == 3

    def test_skips_already_correct_state(self, fake_env: FakeEnv) -> None:
        """No toggles when equipment already matches desired state."""
        bot, fake_cdp = _make_bot(fake_env)
        # Set inventory to match: 1=off, 2=on, 4=off, 5=on
        update_inventory_from_toggle([False, True, True, False, True])
        apply_equipment(bot, [2, 5])
        # slot 1: not in desired, already disabled → no toggle
        # slot 2: in desired, already enabled → no toggle
        # slot 4: not in desired, already disabled → no toggle
        assert len(fake_cdp._sent_methods) == 0


class TestDispatchCommand:
    """Tests for dispatch_command."""

    def test_dispatch_move(self, fake_env: FakeEnv) -> None:
        """Dispatches move command via bot.move_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_move_command(150, 160))
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_pickup_move(self, fake_env: FakeEnv) -> None:
        """Dispatches pickup_move command via bot.pickup_move_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_pickup_move_command(80, 90))
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_shoot(self, fake_env: FakeEnv) -> None:
        """Dispatches shoot command via bot.shoot_at."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_shoot_command(105, 103))
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_radar(self, fake_env: FakeEnv) -> None:
        """Dispatches radar command via bot.use_radar."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_radar_command())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_map_open(self, fake_env: FakeEnv) -> None:
        """Dispatches map_open command via bot.open_map."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_map_open_command())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_teleport(self, fake_env: FakeEnv) -> None:
        """Dispatches teleport command via bot.teleport_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_teleport_command(200, 200))
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods


class TestExecute:
    """Tests for execute (apply_equipment + dispatch_command)."""

    def test_execute_applies_equipment_then_dispatches(self, fake_env: FakeEnv) -> None:
        """Execute applies equipment changes before dispatching command."""
        bot, fake_cdp = _make_bot(fake_env)
        # Set all slots to enabled so execute needs to disable 1, 2, 4
        update_inventory_from_toggle([True, True, True, True, True])
        behavior = make_behavior_score("HUNT", 50, 100, 200, "patrol_waypoint")
        decision = make_tick_decision(
            command=make_move_command(100, 200),
            behavior=behavior,
            updated_ai_state=make_initial_ai_state(),
            desired_equipment=[5],
        )
        execute(bot, decision)
        # Equipment toggles (disable 1, 2, 4) + move command
        assert len(fake_cdp._sent_methods) == 4
