"""Tests for the under-fire escape rungs of the collect cascade."""

from __future__ import annotations

from tankpit_bot.bot.ai.collect_mode import _hop_escapes_attacker
from tankpit_bot.bot.ai.context import make_decision
from tankpit_bot.bot.ai.types import AIStateDict, make_initial_ai_state
from tankpit_bot.bot.tick_loop_types import TickDecisionDict
from tankpit_bot.bot.types import make_pickup_fuel_command, make_teleport_command


def _locked_state(*, attacker_x: int = 100, attacker_y: int = 100) -> AIStateDict:
    """AI state holding the escape's combat lock on the attacker.

    Args:
        attacker_x: Attacker X coordinate.
        attacker_y: Attacker Y coordinate.

    Returns:
        AI state with the lock the escape carries while fleeing.
    """
    return AIStateDict(
        **{
            **make_initial_ai_state(),
            "combat_target_id": 50,
            "combat_target_x": attacker_x,
            "combat_target_y": attacker_y,
        }
    )


def _teleport_decision(state: AIStateDict, tx: int, ty: int) -> TickDecisionDict:
    """Build a minimal hop decision landing at the given tile.

    Args:
        state: AI state for the decision.
        tx: Landing X.
        ty: Landing Y.

    Returns:
        Teleport decision shaped like the larder/search hop output.
    """
    return make_decision(
        make_teleport_command(tx, ty),
        "COLLECT",
        925,
        tx,
        ty,
        "fuel_hop",
        state,
        [],
    )


def test_far_landing_clears_the_attacker_envelope() -> None:
    """A hop landing a full viewport away counts as a real escape."""
    state = _locked_state()

    decision = _teleport_decision(state, 120, 100)

    assert _hop_escapes_attacker(state, decision) is True


def test_near_landing_stays_in_the_kill_zone() -> None:
    """A hop landing beside the attacker is not an escape.

    Flag 1 of run bot-20260730-025x: the escape teleported ONE tile,
    then three — both map-open ticks paid, both landings still under
    red-6's guns — because the larder score structurally favors the
    nearest fuel.
    """
    state = _locked_state()

    decision = _teleport_decision(state, 103, 100)

    assert _hop_escapes_attacker(state, decision) is False


def test_no_known_attacker_accepts_any_landing() -> None:
    """Without a combat lock there is no envelope to clear."""
    state = AIStateDict(**{**make_initial_ai_state(), "combat_target_id": -1})

    decision = _teleport_decision(state, 101, 100)

    assert _hop_escapes_attacker(state, decision) is True


def test_non_teleport_decisions_pass_through() -> None:
    """Only teleport landings are judged; other commands pass.

    User movement law (flag 4, 2026-07-30): a walk is one tick and at
    most one hit, so an in-viewport fuel WALK is always an acceptable
    under-fire action and never filtered by the envelope rule.
    """
    state = _locked_state()
    decision = make_decision(
        make_pickup_fuel_command(101, 100),
        "COLLECT",
        925,
        101,
        100,
        "fuel_locked",
        state,
        [],
    )

    assert _hop_escapes_attacker(state, decision) is True
