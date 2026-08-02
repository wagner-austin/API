"""Tests for executor dispatch_command (split from test_executor, 2026-08-01)."""

from __future__ import annotations

from tankpit_bot.bot.executor import (
    dispatch_command,
)
from tankpit_bot.bot.types import (
    make_hold_command,
    make_map_open_command,
    make_move_command,
    make_pickup_equipment_command,
    make_pickup_fuel_command,
    make_radar_command,
    make_shoot_command,
    make_teleport_command,
)
from tests.bot._executor_support import (
    _make_bot,
    _make_snapshot,
    _make_world,
    _WorldOnlyBot,
)
from tests.conftest import FakeEnv


class TestDispatchCommand:
    """Tests for dispatch_command."""

    def test_dispatch_move(self, fake_env: FakeEnv) -> None:
        """Dispatches move command via bot.move_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_move_command(150, 160), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_pickup_fuel(self, fake_env: FakeEnv) -> None:
        """Dispatches pickup_fuel command via bot.pickup_fuel_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_pickup_fuel_command(80, 90), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_pickup_equipment(self, fake_env: FakeEnv) -> None:
        """Dispatches pickup_equipment command via bot.pickup_equipment_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_pickup_equipment_command(80, 90), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_shoot(self, fake_env: FakeEnv) -> None:
        """Dispatches shoot command via bot.shoot_at."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_shoot_command(105, 103), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_radar(self, fake_env: FakeEnv) -> None:
        """Dispatches radar command via bot.use_radar."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_radar_command(), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_map_open(self, fake_env: FakeEnv) -> None:
        """Dispatches map_open command via bot.open_map."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_map_open_command(), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_chat(self, fake_env: FakeEnv) -> None:
        """Dispatches chat command via bot.send_chat."""
        from tankpit_bot.bot.types import make_chat_command

        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_chat_command(41, 100, 100), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_chat_without_cdp_fails(self, fake_env: FakeEnv) -> None:
        """Chat dispatch reports False when no CDP session is attached."""
        from tankpit_bot.bot.types import make_chat_command

        bot, _fake_cdp = _make_bot(fake_env)
        bot._cdp = None
        result = dispatch_command(bot, make_chat_command(41, 100, 100), _make_snapshot())
        assert result is False

    def test_dispatch_scope_shift(self, fake_env: FakeEnv) -> None:
        """Dispatches scope_shift via bot.scope_shift onto the wire."""
        from tankpit_bot.bot.types import make_scope_shift_command
        from tankpit_bot.protocol.commands import SCOPE_EAST

        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_scope_shift_command(SCOPE_EAST), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_scope_shift_without_cdp_fails(self, fake_env: FakeEnv) -> None:
        """Scope dispatch reports False when no CDP session is attached."""
        from tankpit_bot.bot.types import make_scope_shift_command
        from tankpit_bot.protocol.commands import SCOPE_WEST

        bot, _fake_cdp = _make_bot(fake_env)
        bot._cdp = None
        result = dispatch_command(bot, make_scope_shift_command(SCOPE_WEST), _make_snapshot())
        assert result is False

    def test_dispatch_hold_sends_nothing(self, fake_env: FakeEnv) -> None:
        """Hold command returns True and does not touch the wire.

        The SPA-driven idle tick must not dispatch any CDP command;
        the fake CDP session confirms no ``Runtime.evaluate`` reached
        it while ``dispatch_command`` still reports success (the
        desired effect — do nothing — was achieved).
        """
        bot, fake_cdp = _make_bot(fake_env)
        assert "Runtime.evaluate" not in fake_cdp._sent_methods
        result = dispatch_command(bot, make_hold_command(), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" not in fake_cdp._sent_methods

    def test_dispatch_map_open_sends_wire_only_when_already_visible(
        self, fake_env: FakeEnv
    ) -> None:
        """A visible map dispatches CMD_MAP_OPEN with no client-side keypress.

        Regression guard for capture 20260620-183916: CMD_MAP_OPEN is
        idempotent on the server -- every wire dispatch produces a fresh
        MAP_DATA payload regardless of overlay visibility. The previous
        close-then-reopen hack added a synthetic 'm' keypress before
        every redundant intel refresh; the wire dispatch alone is what
        the server actually needs.
        """
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(
            bot,
            make_map_open_command(),
            _make_snapshot(map_visible=True),
        )
        assert result is True
        # No synthetic 'm' keypress dispatched -- only the wire command.
        key_events = [m for m in fake_cdp._sent_methods if m == "Input.dispatchKeyEvent"]
        assert key_events == []
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_teleport(self, fake_env: FakeEnv) -> None:
        """Dispatches teleport command via bot.teleport_to."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_teleport_command(200, 200), _make_snapshot())
        assert result is True
        assert "Runtime.evaluate" in fake_cdp._sent_methods

    def test_dispatch_teleport_records_no_attempt_when_send_fails(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """A failed teleport send leaves no pending attempt to mislabel later."""
        from tankpit_bot.ledger.outcome.teleport import emit_teleport_landed

        world = _make_world()
        result = dispatch_command(
            _WorldOnlyBot(world),
            make_teleport_command(200, 200),
            _make_snapshot(map_visible=True),
        )

        assert result is False
        landed = emit_teleport_landed(
            duration_ms=0, target_x=200, target_y=200, landed_x=200, landed_y=200, messages=[]
        )
        assert landed["detail"]["sent_window"] == "(none)"

    def test_dispatch_teleport_skips_open_map_when_map_already_visible(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Teleport skips the precondition map_open when the map is already open."""
        bot, fake_cdp = _make_bot(fake_env)
        result = dispatch_command(
            bot,
            make_teleport_command(200, 200),
            _make_snapshot(map_visible=True),
        )
        assert result is True
        sent_methods = fake_cdp._sent_methods
        runtime_calls = [m for m in sent_methods if m == "Runtime.evaluate"]
        assert len(runtime_calls) == 1
