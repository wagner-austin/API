"""Tests for ground-aimed shot resolution: the 0x53 echo receipt.

A clearance shot targets a tile, not a tank, so the id-keyed combat
classifier never resolves it — before the resolver existed every
clearance decision rotted into ``superseded`` and the liveness counter
misread the silence as a livelock (soak bot-20260821-013519: 13 wire
dispatches, 12/12 superseded, 0 completions). These tests pin the
executor's pending-ground-shot mark, the tick resolver's ``fired`` /
``command_rejected`` receipts, and the flag hygiene that keeps a
ground echo from leaking into the next combat shot's classification.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.scoring_types import make_behavior_score
from tankpit_bot.bot.ai.types import make_initial_ai_state
from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.executor import dispatch_command, execute
from tankpit_bot.bot.tick_combat_feedback import _resolve_pending_ground_shot
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import make_shoot_command
from tankpit_bot.ledger.ring import outcome_counts, recent_outcomes
from tankpit_bot.protocol.constants import SUPERVISOR_ERROR_CANT_DO
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_combat import (
    mark_combat_hit,
    mark_pending_ground_shot,
)
from tests.bot._executor_support import _make_bot, _make_snapshot
from tests.conftest import FakeEnv


def _make_feedback_bot() -> Bot:
    """Return a real Bot over a fresh world for resolver tests."""
    return Bot("https://test.tankpit.com/", headless=True, world=WorldService())


class TestExecutorGroundShotMark:
    """The dispatch side: which shots get the pending-ground mark."""

    def test_ground_aimed_shot_marks_pending(self, fake_env: FakeEnv) -> None:
        """A ``target_id == 0`` shot records aim tile and dispatch time."""
        bot, _fake_cdp = _make_bot(fake_env)
        result = dispatch_command(bot, make_shoot_command(105, 103), _make_snapshot())
        assert result is True
        assert bot.world.pending_ground_shot_aim_x == 105
        assert bot.world.pending_ground_shot_aim_y == 103
        assert bot.world.pending_ground_shot_dispatch_ms > 0

    def test_tank_targeted_shot_clears_a_stale_ground_mark(self, fake_env: FakeEnv) -> None:
        """A tracked shot owns the next echo — the stale mark must go."""
        bot, _fake_cdp = _make_bot(fake_env)
        mark_pending_ground_shot(bot.world, 50, 51, 1000)
        result = dispatch_command(bot, make_shoot_command(105, 103, 534), _make_snapshot())
        assert result is True
        assert bot.world.pending_ground_shot_dispatch_ms == 0
        assert bot.world.last_shot_combat_target_id == 534

    def test_failed_dispatch_marks_nothing(self, fake_env: FakeEnv) -> None:
        """A shot that never left (no CDP) records no pending mark."""
        bot, _fake_cdp = _make_bot(fake_env)
        bot._cdp = None
        result = dispatch_command(bot, make_shoot_command(105, 103), _make_snapshot())
        assert result is False
        assert bot.world.pending_ground_shot_dispatch_ms == 0


class TestGroundShotResolver:
    """The tick side: receipts resolve the pending ground shot."""

    def test_echo_resolves_fired_and_consumes_the_flags(self, fake_env: FakeEnv) -> None:
        """The own 0x53 echo yields ``shoot:fired`` and clean flags.

        The echo's side flags (response, victim lookup, snapshot) must
        be consumed with it: left latched, they would hand the NEXT
        tank-targeted shot an instant stale classification.
        """
        bot = _make_feedback_bot()
        ws = bot.world
        mark_pending_ground_shot(ws, 227, 171, 1000)
        ws.pending_shot_inventory_snapshot = ws.inventory_state
        mark_combat_hit(ws, 0, -1)

        _resolve_pending_ground_shot(bot)

        fired = recent_outcomes(ws.ledger, "shoot", 1)[0]
        assert fired["outcome"] == "fired"
        assert fired["detail"] == {"aim_x": 227, "aim_y": 171}
        assert ws.pending_ground_shot_dispatch_ms == 0
        assert ws.got_our_shot_response is False
        assert ws.got_confirmed_hit is False
        assert ws.last_shot_victim_id == -1
        assert ws.pending_shot_inventory_snapshot is None

    def test_echo_with_weapon_debit_still_resolves_fired(self, fake_env: FakeEnv) -> None:
        """An enemy wandering onto the aim tile does not corrupt the receipt.

        ``mark_combat_hit`` latches ``got_confirmed_hit`` on a weapon
        debit; the resolver consumes it so it cannot leak, and the
        shot still resolves ``fired`` — ground shots carry no hit/miss
        semantics.
        """
        bot = _make_feedback_bot()
        ws = bot.world
        mark_pending_ground_shot(ws, 227, 171, 1000)
        mark_combat_hit(ws, 1, 534)

        _resolve_pending_ground_shot(bot)

        assert recent_outcomes(ws.ledger, "shoot", 1)[0]["outcome"] == "fired"
        assert ws.got_confirmed_hit is False
        assert ws.last_shot_victim_id == -1

    def test_shot_rejecting_error_resolves_command_rejected(self, fake_env: FakeEnv) -> None:
        """A 0x52 refusal means no echo will ever come — resolve now.

        [[shot-range]]: five out-of-range dispatches drew code-0
        rejections, zero echoes, zero ammo deltas.
        """
        bot = _make_feedback_bot()
        ws = bot.world
        mark_pending_ground_shot(ws, 227, 171, 1000)
        ws.last_command_error = SUPERVISOR_ERROR_CANT_DO

        _resolve_pending_ground_shot(bot)

        rejected = recent_outcomes(ws.ledger, "shoot", 1)[0]
        assert rejected["outcome"] == "command_rejected"
        assert rejected["detail"]["error_code"] == SUPERVISOR_ERROR_CANT_DO
        assert ws.pending_ground_shot_dispatch_ms == 0
        assert ws.last_command_error == -1

    def test_no_receipt_keeps_waiting(self, fake_env: FakeEnv) -> None:
        """Neither echo nor rejection: the mark stays, nothing is emitted."""
        bot = _make_feedback_bot()
        ws = bot.world
        mark_pending_ground_shot(ws, 227, 171, 1000)

        _resolve_pending_ground_shot(bot)

        assert ws.pending_ground_shot_dispatch_ms == 1000
        assert outcome_counts(ws.ledger, "shoot") == {}

    def test_without_a_pending_mark_the_resolver_is_inert(self, fake_env: FakeEnv) -> None:
        """No pending ground shot: combat receipts are left untouched."""
        bot = _make_feedback_bot()
        ws = bot.world
        mark_combat_hit(ws, 0, -1)

        _resolve_pending_ground_shot(bot)

        assert ws.got_our_shot_response is True
        assert outcome_counts(ws.ledger, "shoot") == {}


def _clearance_decision() -> TickDecisionDict:
    """Return a clearance-shot tick decision (ground aim, id 0)."""
    return make_tick_decision(
        command=make_shoot_command(105, 103),
        behavior=make_behavior_score("COLLECT", 925, 105, 103, "mine_clearance_shot"),
        updated_ai_state=make_initial_ai_state(),
        desired_equipment=[],
    )


class TestExecuteDispatchMark:
    """``execute`` marks the recorded decision once its command dispatches."""

    def test_dispatched_decision_is_marked(self, fake_env: FakeEnv) -> None:
        """A successful dispatch adds the decision id to the mark set."""
        bot, _fake_cdp = _make_bot(fake_env)

        result = execute(bot, _clearance_decision(), _make_snapshot())

        assert result is True
        assert len(bot.world.ledger.dispatched_decision_ids) == 1

    def test_failed_dispatch_stays_unmarked(self, fake_env: FakeEnv) -> None:
        """A dispatch that never left the process earns no mark."""
        bot, _fake_cdp = _make_bot(fake_env)
        bot._cdp = None

        result = execute(bot, _clearance_decision(), _make_snapshot())

        assert result is False
        assert bot.world.ledger.dispatched_decision_ids == set()
