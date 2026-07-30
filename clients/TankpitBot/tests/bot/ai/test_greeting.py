"""Tests for the one-shot HELLO greeting on human target acquisition."""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.greeting import attach_human_greeting
from tankpit_bot.bot.ai.types import AIStateDict, make_behavior_score, make_initial_ai_state
from tankpit_bot.bot.tick_loop_types import TickDecisionDict, make_tick_decision
from tankpit_bot.bot.types import (
    make_chat_command,
    make_radar_command,
    make_teleport_command,
)
from tankpit_bot.protocol.chat import CHAT_HELLO
from tankpit_bot.state.types import TankStateDict, make_tank_state
from tests.bot.ai._support import make_inventory, make_scanned_ai_state, make_world


def _tank(tank_id: int, name: str) -> TankStateDict:
    """A registry tank the greeting can classify by name."""
    return make_tank_state(
        tank_id=tank_id,
        x=120,
        y=100,
        team=2,
        rank=4,
        name=name,
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=100000,
    )


def _ctx(tanks: dict[str, TankStateDict]) -> DecideCtx:
    """A decision context whose registry holds the given tanks."""
    world, self_state = make_world(tanks=tanks)
    return DecideCtx(
        world,
        self_state,
        make_scanned_ai_state(),
        make_inventory(),
        100000,
        None,
        "",
    )


def _decision(
    *,
    combat_target_id: int,
    greeted_target_id: int = -1,
    with_secondary: bool = False,
) -> TickDecisionDict:
    """A HUNT-owner decision with the given lock and greet latch."""
    ai_state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "combat_target_id": combat_target_id,
            "greeted_target_id": greeted_target_id,
        }
    )
    secondary = make_radar_command() if with_secondary else None
    return make_tick_decision(
        command=make_teleport_command(119, 100),
        behavior=make_behavior_score("HUNT", 800, 119, 100, "teleport_target"),
        updated_ai_state=ai_state,
        desired_equipment=[2, 5],
        secondary_command=secondary,
    )


def test_greets_new_human_lock_with_hello_secondary() -> None:
    """A fresh human lock attaches HELLO from the bot's current tile."""
    ctx = _ctx({"50": _tank(50, "Yuppler")})
    decision = _decision(combat_target_id=50)

    result = attach_human_greeting(ctx, decision)

    assert result["secondary_command"] == make_chat_command(
        CHAT_HELLO, ctx.self_state["x"], ctx.self_state["y"]
    )
    assert result["updated_ai_state"]["greeted_target_id"] == 50
    assert result["command"] == decision["command"]
    assert result["behavior"] == decision["behavior"]
    assert result["desired_equipment"] == decision["desired_equipment"]


def test_no_greeting_without_a_lock() -> None:
    """No combat lock means nothing to greet."""
    ctx = _ctx({"50": _tank(50, "Yuppler")})
    decision = _decision(combat_target_id=-1)

    result = attach_human_greeting(ctx, decision)

    assert result is decision


def test_no_regreeting_of_the_same_target() -> None:
    """The latch stops a re-greet while the same lock is re-derived.

    Server flood-mute discipline ([[chat-messages]]): the lock is
    re-asserted every hunt tick, and each duplicate HELLO would count
    toward the mute that silences the bot for the whole session.
    """
    ctx = _ctx({"50": _tank(50, "Yuppler")})
    decision = _decision(combat_target_id=50, greeted_target_id=50)

    result = attach_human_greeting(ctx, decision)

    assert result is decision


def test_practice_bot_lock_is_never_greeted() -> None:
    """Farming a practice bot stays silent."""
    ctx = _ctx({"50": _tank(50, "red-6")})
    decision = _decision(combat_target_id=50)

    result = attach_human_greeting(ctx, decision)

    assert result is decision
    assert result["updated_ai_state"]["greeted_target_id"] == -1


def test_unknown_target_id_is_not_greeted() -> None:
    """A lock on a tank missing from the registry cannot be classified."""
    ctx = _ctx({})
    decision = _decision(combat_target_id=50)

    result = attach_human_greeting(ctx, decision)

    assert result is decision


def test_existing_secondary_command_is_never_displaced() -> None:
    """A planned secondary keeps its slot; the un-latched greet retries."""
    ctx = _ctx({"50": _tank(50, "Yuppler")})
    decision = _decision(combat_target_id=50, with_secondary=True)

    result = attach_human_greeting(ctx, decision)

    assert result is decision
    assert result["updated_ai_state"]["greeted_target_id"] == -1


class TestGreetingThroughDecide:
    """End-to-end: the arbitrator's HUNT path carries the greeting."""

    def setup_method(self) -> None:
        """Reset module-level world state before each test."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

    def teardown_method(self) -> None:
        """Reset module-level world state after each test."""
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

    def test_decide_attaches_hello_when_locking_a_visible_human(self) -> None:
        """A full-stock tick that acquires a human lock says HELLO.

        The user contract this pins (2026-07-29): the bot finishes
        collecting (HUNT is a privilege of a full tank), targets the
        human, and greets them in the same tick window as the
        acquisition teleport.
        """
        from tankpit_bot.bot.ai_strategy import decide

        human = make_tank_state(
            tank_id=50,
            x=105,
            y=100,
            team=2,
            rank=4,
            name="Yuppler",
            is_self=False,
            is_bot=False,
            damage_state=0,
            timestamp_ms=100000,
            last_wire_seen_ms=100000,
            last_position_update_ms=100000,
            last_viewport_observation_ms=100000,
        )
        world, self_state = make_world(fuel=2000, tanks={"50": human})
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["updated_ai_state"]["combat_target_id"] == 50
        assert decision["secondary_command"] == make_chat_command(
            CHAT_HELLO, self_state["x"], self_state["y"]
        )
        assert decision["updated_ai_state"]["greeted_target_id"] == 50

    def test_decide_stays_silent_when_locking_a_practice_bot(self) -> None:
        """The same acquisition against a practice bot attaches nothing."""
        from tankpit_bot.bot.ai_strategy import decide

        bot_tank = make_tank_state(
            tank_id=51,
            x=105,
            y=100,
            team=2,
            rank=4,
            name="red-6",
            is_self=False,
            is_bot=True,
            damage_state=0,
            timestamp_ms=100000,
            last_wire_seen_ms=100000,
            last_position_update_ms=100000,
            last_viewport_observation_ms=100000,
        )
        world, self_state = make_world(fuel=2000, tanks={"51": bot_tank})
        ai_state = make_scanned_ai_state()
        inventory = make_inventory()

        decision = decide(world, self_state, ai_state, inventory, 100000, None)

        assert decision["behavior"]["mode"] == "HUNT"
        assert decision["updated_ai_state"]["combat_target_id"] == 51
        assert decision["secondary_command"] is None
        assert decision["updated_ai_state"]["greeted_target_id"] == -1
