"""Tests for the one-shot HELLO greeting on human viewport encounter."""

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


def _tank(
    tank_id: int,
    name: str,
    *,
    x: int = 105,
    y: int = 100,
    timestamp_ms: int = 100000,
) -> TankStateDict:
    """A registry tank the greeting can classify by name.

    ``timestamp_ms`` is the map-freshness stamp the hello-anytime
    rule reads (user ruling 2026-07-31): fresh means "on the map
    logged in"; position and viewport presence never gate the HELLO.
    """
    return make_tank_state(
        tank_id=tank_id,
        x=x,
        y=y,
        team=2,
        rank=4,
        name=name,
        is_self=False,
        is_bot=False,
        damage_state=0,
        timestamp_ms=timestamp_ms,
        last_viewport_observation_ms=timestamp_ms,
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
    greeted_tank_ids: dict[str, int] | None = None,
    with_secondary: bool = False,
) -> TickDecisionDict:
    """A HUNT-owner decision with the given lock and greet latch."""
    ai_state = AIStateDict(
        **{
            **make_initial_ai_state(),
            "combat_target_id": combat_target_id,
            "greeted_tank_ids": greeted_tank_ids if greeted_tank_ids is not None else {},
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
    """A viewport human attaches HELLO from the bot's current tile."""
    ctx = _ctx({"50": _tank(50, "Yuppler")})
    decision = _decision(combat_target_id=50)

    result = attach_human_greeting(ctx, decision)

    assert result["secondary_command"] == make_chat_command(
        CHAT_HELLO, ctx.self_state["x"], ctx.self_state["y"]
    )
    assert result["updated_ai_state"]["greeted_tank_ids"] == {"50": 100000}
    assert result["command"] == decision["command"]
    assert result["behavior"] == decision["behavior"]
    assert result["desired_equipment"] == decision["desired_equipment"]


def test_greets_viewport_human_without_a_lock() -> None:
    """Encounter semantics (2026-07-30): no lock is needed to greet.

    The consent contract forbids locking an unresponsive human, so
    the HELLO fires on viewport presence alone -- the greeting
    approach lands a few tiles off and this hook says hello.
    """
    ctx = _ctx({"50": _tank(50, "Yuppler")})
    decision = _decision(combat_target_id=-1)

    result = attach_human_greeting(ctx, decision)

    assert result["secondary_command"] == make_chat_command(
        CHAT_HELLO, ctx.self_state["x"], ctx.self_state["y"]
    )
    assert result["updated_ai_state"]["greeted_tank_ids"] == {"50": 100000}


def test_map_stale_human_is_not_greeted() -> None:
    """A human whose registry stamp went map-stale gets no HELLO.

    Map freshness is the "on the map logged in" proxy (user ruling
    2026-07-31): 0x4C map opens and the global 0x2E sync refresh the
    stamp for everyone actually in the game, so a stale stamp means
    nobody has vouched for them recently — wait for the next map.
    """
    ctx = _ctx({"50": _tank(50, "Yuppler", timestamp_ms=10)})
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
    decision = _decision(combat_target_id=50, greeted_tank_ids={"50": 90000})

    result = attach_human_greeting(ctx, decision)

    assert result is decision


def test_practice_bot_lock_is_never_greeted() -> None:
    """Farming a practice bot stays silent."""
    ctx = _ctx({"50": _tank(50, "red-6")})
    decision = _decision(combat_target_id=50)

    result = attach_human_greeting(ctx, decision)

    assert result is decision
    assert result["updated_ai_state"]["greeted_tank_ids"] == {}


def test_unknown_target_id_is_not_greeted() -> None:
    """An empty registry offers nobody to greet."""
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
    assert result["updated_ai_state"]["greeted_tank_ids"] == {}


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

    def test_decide_greets_but_never_locks_an_unconsented_human(self) -> None:
        """The consent contract end-to-end: HELLO yes, lock no.

        User ruling 2026-07-30 ("the human must respond hello or
        engage the bot first"): a full-stock tick with an unresponsive
        human in view attaches the greeting but acquires NO combat
        lock -- the fight waits for their answer.
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

        assert decision["updated_ai_state"]["combat_target_id"] == -1
        assert decision["secondary_command"] == make_chat_command(
            CHAT_HELLO, self_state["x"], self_state["y"]
        )
        assert decision["updated_ai_state"]["greeted_tank_ids"] == {"50": 100000}

    def test_decide_locks_a_consented_human(self) -> None:
        """A chat response consents the human into a normal lock."""
        from tankpit_bot.bot.ai_strategy import decide
        from tankpit_bot.sniffer.world_state import get_world_service

        get_world_service().chat_seen_tank_ids.add(50)
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
        assert decision["updated_ai_state"]["greeted_tank_ids"] == {"50": 100000}

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
        assert decision["updated_ai_state"]["greeted_tank_ids"] == {}


def test_greets_the_nearest_of_two_viewport_humans() -> None:
    """With two ungreeted humans in view the nearest gets the HELLO."""
    ctx = _ctx({"50": _tank(50, "Yuppler"), "60": _tank(60, "guest", x=107)})
    decision = _decision(combat_target_id=-1)

    result = attach_human_greeting(ctx, decision)

    assert result["updated_ai_state"]["greeted_tank_ids"] == {"50": 100000}


def test_far_off_viewport_human_is_greeted_anyway() -> None:
    """Chat is global: a map-fresh human across the field gets the HELLO.

    User ruling 2026-07-31, verbatim: "hello can run anytime... as
    long as the other player is on the map logged in. you dont have
    to be near them." The stand-off VISIT keeps its own latch and
    proximity machinery; the hello never waits for it.
    """
    ctx = _ctx({"50": _tank(50, "Yuppler", x=240, y=30)})
    decision = _decision(combat_target_id=-1)

    result = attach_human_greeting(ctx, decision)

    assert result["updated_ai_state"]["greeted_tank_ids"] == {"50": 100000}
    assert result["secondary_command"] == make_chat_command(
        CHAT_HELLO, ctx.self_state["x"], ctx.self_state["y"]
    )


def test_position_unsynced_human_is_greeted_anyway() -> None:
    """A logged-in human whose position has not synced still gets the HELLO.

    The (0,0) sentinel gates targeting and the stand-off visit (both
    need real coordinates), never the chat — they are in the game the
    moment their identity broadcast lands.
    """
    ctx = _ctx({"50": _tank(50, "Yuppler", x=0, y=0)})
    decision = _decision(combat_target_id=-1)

    result = attach_human_greeting(ctx, decision)

    assert result["updated_ai_state"]["greeted_tank_ids"] == {"50": 100000}
