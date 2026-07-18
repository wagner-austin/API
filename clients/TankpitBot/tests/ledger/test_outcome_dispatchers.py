"""Coverage for the kind-routing dispatchers and discard branches."""

from __future__ import annotations

import pytest

from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.executor import _is_valid_move_destination
from tankpit_bot.bot.tick_loop_actions import (
    _emit_command_rejected_outcome,
    _emit_stall_outcome,
)
from tankpit_bot.bot.types import make_move_command, make_teleport_command
from tankpit_bot.contracts.base import LedgerInvariantError
from tankpit_bot.ledger.outcome.teleport import TeleportDispatchContract
from tankpit_bot.ledger.ring import outcome_counts, recent_outcomes
from tankpit_bot.state.types import make_mine_state
from tankpit_bot.state.types.world import make_empty_world_state
from tests.conftest import FakeEnv


def test_move_to_hostile_mine_records_move_discard() -> None:
    """A move onto a hostile mine records the move discard outcome."""
    world = make_empty_world_state()
    world["mines"]["10,20"] = make_mine_state(10, 20, 0, -1, 1, source="radar")
    assert _is_valid_move_destination(world, make_move_command(10, 20)) is False
    records = recent_outcomes("move", 1)
    assert records[0]["outcome"] == "discarded_hostile_mine"
    assert records[0]["detail"] == {"target_x": 10, "target_y": 20}


def test_teleport_to_hostile_mine_records_teleport_discard() -> None:
    """A teleport onto a hostile mine records the teleport discard."""
    world = make_empty_world_state()
    world["mines"]["10,20"] = make_mine_state(10, 20, 0, -1, 1, source="radar")
    assert _is_valid_move_destination(world, make_teleport_command(10, 20)) is False
    records = recent_outcomes("teleport", 1)
    assert records[0]["outcome"] == "discarded_hostile_mine"


def test_command_rejected_dispatcher_routes_every_kind(fake_env: FakeEnv) -> None:
    """Each action kind's 0x52 rejection routes to its typed emitter."""
    bot = Bot("https://test.tankpit.com/", headless=True)
    _emit_command_rejected_outcome(bot, "move", 1, 2, 100, 0)
    _emit_command_rejected_outcome(bot, "collect", 1, 2, 100, 5)
    _emit_command_rejected_outcome(bot, "teleport", 1, 2, 100, 8)
    _emit_command_rejected_outcome(bot, "scan", 1, 2, 100, 0)
    _emit_command_rejected_outcome(bot, "map_open", 1, 2, 100, 0)
    for kind in ("move", "collect", "teleport", "scan", "map_open"):
        assert outcome_counts(kind) == {"command_rejected": 1}


def test_stall_dispatcher_routes_every_kind(fake_env: FakeEnv) -> None:
    """Each action kind's stall timeout routes to its typed emitter."""
    bot = Bot("https://test.tankpit.com/", headless=True)
    _emit_stall_outcome(bot, "move", 1, 2, 10000, 10000)
    _emit_stall_outcome(bot, "collect", 1, 2, 10000, 10000)
    _emit_stall_outcome(bot, "teleport", 1, 2, 10000, 10000)
    _emit_stall_outcome(bot, "scan", 1, 2, 10000, 10000)
    _emit_stall_outcome(bot, "map_open", 1, 2, 10000, 10000)
    for kind in ("move", "collect", "teleport", "scan", "map_open"):
        assert outcome_counts(kind) == {"stall_timeout": 1}


def test_dispatchers_ignore_shoot_kind(fake_env: FakeEnv) -> None:
    """Shoot resolutions ride the combat-feedback classifier, not these.

    Documents the contract: the in-flight dispatchers route only the
    five HFSM-tracked kinds; a ``shoot`` (fire-and-forget) or ``none``
    kind falls through without emitting.
    """
    bot = Bot("https://test.tankpit.com/", headless=True)
    _emit_command_rejected_outcome(bot, "shoot", 1, 2, 100, 0)
    _emit_stall_outcome(bot, "none", 1, 2, 100, 100)
    assert outcome_counts("shoot") == {}


def test_teleport_dispatch_contract_names_itself_and_rejects_bad_input() -> None:
    """The dispatch contract exposes its name and raises on violations."""
    contract = TeleportDispatchContract()
    assert contract.name == "teleport_dispatch"
    with pytest.raises(LedgerInvariantError) as exc:
        contract.check(target_x=300, target_y=20, message_index=0, sent_window="w")
    assert exc.value.details == {"target_x": "300", "target_y": "20"}
    with pytest.raises(LedgerInvariantError) as exc:
        contract.check(target_x=10, target_y=20, message_index=-1, sent_window="w")
    assert exc.value.details == {"message_index": "-1", "sent_window": "w"}
