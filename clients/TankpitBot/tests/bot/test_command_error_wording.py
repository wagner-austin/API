"""Tests for the wording of the 0x52 sync line.

Split from ``test_tick_loop_command_error.py`` (618 lines, over the
600-line ceiling): that file owns which 0x52 codes apply to which
in-flight action, this one owns how an applicable code is described --
a collect's close is a receipt, not a rejection.
"""

from __future__ import annotations

from tests._runtime_logging_support import capture_runtime_events
from tests.conftest import FakeEnv


class TestCommandErrorWording:
    """A collect's 0x52 close is a receipt; only one wording may be logged."""

    def test_a_collect_receipt_is_not_also_announced_as_a_rejection(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """The receipt wording is logged, and the rejection wording is not.

        Code 5 is the clamp SUCCESS close -- the transfer landed in the
        same batch -- so calling it a rejection is false. Logging them
        all as rejections is what hid the 2026-08-03 nope-fight ground
        truth through three read passes: 32 "rejections" that were
        successful drinks.

        Both branches only log, so the wording is the entire observable
        difference, and without the early return BOTH go out for the same
        receipt: a line saying the server closed the collect and a line
        saying it rejected it and the bot is replanning. That reinstates
        exactly the confusion the split was written to end.
        """
        from tankpit_bot.bot.tick_loop_command_errors import _emit_command_error_sync
        from tankpit_bot.protocol.constants import SUPERVISOR_ERROR_TANK_FULL

        with capture_runtime_events() as records:
            _emit_command_error_sync("collect", 40, 50, SUPERVISOR_ERROR_TANK_FULL)

        messages = [record.getMessage() for record in records]
        assert [m for m in messages if "closed by server receipt" in m] != []
        assert [m for m in messages if "rejected by server" in m] == []

    def test_control_a_genuine_rejection_keeps_the_word(self, fake_env: FakeEnv) -> None:
        """Control: a code outside the receipt set IS a rejection.

        Same kind, same tile, different code -- so the wording above is
        the receipt set being honoured rather than the rejection line
        being unreachable.
        """
        from tankpit_bot.bot.tick_loop_command_errors import _emit_command_error_sync
        from tankpit_bot.protocol.constants import SUPERVISOR_ERROR_CANT_GO

        with capture_runtime_events() as records:
            _emit_command_error_sync("collect", 40, 50, SUPERVISOR_ERROR_CANT_GO)

        messages = [record.getMessage() for record in records]
        assert [m for m in messages if "rejected by server" in m] != []
        assert [m for m in messages if "closed by server receipt" in m] == []
