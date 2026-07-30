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


def test_trapped_escape_takes_the_near_hop_over_standing_still() -> None:
    """When no hop clears the attacker, the near larder hop still goes.

    The envelope rule prefers real exits, but a trapped tank (every
    known landing inside the attacker's reach) must still move -- any
    hop beats standing in the firing line drinking dregs.
    """
    from tankpit_bot.bot.ai.collect_mode import decide_collect_mode
    from tankpit_bot.bot.ai.context import DecideCtx
    from tankpit_bot.ledger.damage_book import confirm_incoming_damage, record_incoming_shot
    from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
    from tankpit_bot.state.types import make_container_state
    from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world
    from tests.in_memory_terrain_map import InMemoryTerrainMap

    reset_world_state()
    try:
        book = get_world_service().damage_book
        for i in range(4):
            ts = 95000 + i * 1000
            record_incoming_shot(book, 60, "Yuppler", 1, ts)
            confirm_incoming_damage(book, -90, ts + 100)
        world, self_state = make_world(
            fuel=800,
            containers={
                "110,100": make_container_state(
                    x=110,
                    y=100,
                    is_fuel=True,
                    volume=400,
                    timestamp_ms=100000,
                    failed_pickups=0,
                )
            },
        )
        ai_state = AIStateDict(
            **{
                **make_scanned_ai_state(),
                "mode": "COLLECT",
                "mode_state": "SEARCH",
                "mode_started_ms": 90000,
                "last_map_open_ms": 99000,
                "combat_target_id": 50,
                "combat_target_x": 112,
                "combat_target_y": 100,
            }
        )
        ctx = DecideCtx(
            world,
            self_state,
            ai_state,
            make_inventory(),
            100000,
            InMemoryTerrainMap(),
            "",
        )
        decision = decide_collect_mode(ctx)
    finally:
        reset_world_state()

    if decision is None:
        raise AssertionError("expected trapped-escape hop decision")
    assert decision["command"]["cmd_type"] == "teleport"
    assert decision["behavior"]["reason_kind"] == "fuel_hop"
