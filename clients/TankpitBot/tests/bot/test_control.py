"""Tests for the control verbs a running bot takes from its ``CONTROL`` file.

The file is real: every read, write and removal goes to ``tmp_path``
through the production hooks, so what the fleet manager writes is exactly
what the tick loop reads. The AI state is a real one, and each verb is
checked against the fields the AI already reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import InvalidJsonError, JSONTypeError

from tankpit_bot.bot.ai.combat_opportunity import collect_return_fire
from tankpit_bot.bot.ai.combat_target import clear_combat_target
from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.bot.ai.types import AIStateDict
from tankpit_bot.bot.ai_strategy import decide
from tankpit_bot.bot.combat_feedback import CombatFeedback
from tankpit_bot.bot.control import (
    CONTROL_FILE_NAME,
    HOLDABLE_MODES,
    ControlVerb,
    apply_control,
    apply_pending_control,
    control_file_path,
    control_pending,
    decode_control_command,
    encode_control_command,
    make_control_command,
    take_pending_control,
    write_control,
)
from tankpit_bot.bot.session_exit import SessionExitError, SessionExitReason
from tankpit_bot.fleetshare.types import EngagementDoctrine
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state.types import make_tank_state
from tankpit_bot.types.modes import AIMode, AIModeState
from tests.bot.ai._support import (
    make_inventory,
    make_scanned_ai_state,
    make_world,
    seed_confirmed_incoming,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

_NOW = 100000
_ATTACKER_ID = 60
"""The shooter ``seed_confirmed_incoming`` records confirmed hits from."""


def _engaged() -> AIStateDict:
    """A pinned-to-HUNT state holding a combat lock, as in a long fight."""
    return AIStateDict(
        **{**make_scanned_ai_state(), "combat_target_id": 42, "manual_mode": AIMode.HUNT}
    )


def _under_fire(ai_state: AIStateDict) -> DecideCtx:
    """A stocked COLLECT tick with the attacker in view and its hits confirmed.

    Args:
        ai_state: The AI state the tick decides from.

    Returns:
        The decision context.
    """
    ws = WorldService()
    seed_confirmed_incoming(ws, 3)
    attacker = make_tank_state(
        tank_id=_ATTACKER_ID,
        x=103,
        y=100,
        team=2,
        rank=1,
        name="ganker",
        is_self=False,
        is_bot=False,
        damage_state=3,
        timestamp_ms=_NOW,
        last_wire_seen_ms=_NOW,
        last_position_update_ms=_NOW,
        last_viewport_observation_ms=_NOW,
    )
    world, self_state = make_world(fuel=800, tanks={str(_ATTACKER_ID): attacker})
    state = AIStateDict(**{**ai_state, "mode_state": AIModeState.SEARCH, "mode_started_ms": 90000})
    return DecideCtx(
        world,
        self_state,
        state,
        make_inventory(),
        _NOW,
        InMemoryTerrainMap(),
        CombatFeedback.NONE,
        ws=ws,
    )


class TestCommands:
    """Building, encoding and decoding a command."""

    @pytest.mark.parametrize(
        ("verb", "argument"),
        [
            (ControlVerb.HOLD, "HUNT"),
            (ControlVerb.HOLD, "COLLECT"),
            (ControlVerb.DISENGAGE, ""),
            (ControlVerb.WIND_DOWN, ""),
            (ControlVerb.RELEASE, ""),
            (ControlVerb.DOCTRINE, "passive"),
        ],
    )
    def test_every_valid_command_round_trips(self, verb: ControlVerb, argument: str) -> None:
        """A valid command encodes to plain words and decodes back to itself."""
        command = make_control_command(verb, argument)
        encoded = encode_control_command(command)
        assert encoded == {"verb": verb.value, "argument": argument}
        assert decode_control_command(encoded) == command

    def test_only_hunt_and_collect_can_be_held(self) -> None:
        """``UNSET`` is not a mode to hold; release clears a pin instead."""
        assert HOLDABLE_MODES == (AIMode.HUNT, AIMode.COLLECT)

    @pytest.mark.parametrize(
        ("verb", "argument", "message"),
        [
            (ControlVerb.HOLD, "UNSET", "hold needs a mode to hold (HUNT, COLLECT), got 'UNSET'"),
            (ControlVerb.HOLD, "", "hold needs a mode to hold (HUNT, COLLECT), got ''"),
            (
                ControlVerb.DOCTRINE,
                "berserk",
                "doctrine needs an engagement doctrine (skirmish, swarm, duelist, passive), "
                "got 'berserk'",
            ),
            (ControlVerb.DISENGAGE, "now", "disengage takes no argument, got 'now'"),
            (ControlVerb.RELEASE, "HUNT", "release takes no argument, got 'HUNT'"),
        ],
    )
    def test_an_argument_that_does_not_fit_the_verb_is_refused(
        self, verb: ControlVerb, argument: str, message: str
    ) -> None:
        """Each verb takes exactly its own argument, or none."""
        with pytest.raises(JSONTypeError) as raised:
            make_control_command(verb, argument)
        assert str(raised.value) == message

    def test_an_unknown_verb_is_a_hard_error(self) -> None:
        """The file comes from the manager's own validation; anything else is corruption."""
        with pytest.raises(JSONTypeError, match="verb"):
            decode_control_command({"verb": "self_destruct", "argument": ""})

    def test_a_missing_argument_field_is_refused(self) -> None:
        """Both fields are required, even for a verb whose argument is empty."""
        with pytest.raises(JSONTypeError, match="argument"):
            decode_control_command({"verb": "release"})


class TestFile:
    """The ``CONTROL`` file beside ``STOP``."""

    def test_the_file_sits_beside_stop(self, tmp_path: Path) -> None:
        """One run directory, two sentinels."""
        assert CONTROL_FILE_NAME == "CONTROL"
        assert control_file_path(tmp_path) == tmp_path / "CONTROL"

    def test_a_written_verb_is_pending_until_taken_and_taken_once(self, tmp_path: Path) -> None:
        """Written by the manager, consumed on ingest, gone after."""
        assert control_pending(tmp_path) is False
        assert take_pending_control(tmp_path) is None
        command = make_control_command(ControlVerb.HOLD, "COLLECT")
        write_control(tmp_path, command)
        assert control_pending(tmp_path) is True
        assert (tmp_path / "CONTROL").read_text(
            encoding="utf-8"
        ) == '{"verb":"hold","argument":"COLLECT"}'
        assert take_pending_control(tmp_path) == command
        assert control_pending(tmp_path) is False
        assert take_pending_control(tmp_path) is None

    def test_a_file_that_is_not_json_raises_and_is_consumed(self, tmp_path: Path) -> None:
        """Corruption is loud, and does not repeat on every later tick."""
        (tmp_path / "CONTROL").write_text("hold HUNT", encoding="utf-8")
        with pytest.raises(InvalidJsonError):
            take_pending_control(tmp_path)
        assert control_pending(tmp_path) is False


class TestApply:
    """Each verb writes a field the AI already honours."""

    def test_hold_pins_the_mode(self) -> None:
        """``hold`` sets ``manual_mode``, the pin the mode controller reads."""
        after = apply_control(
            make_scanned_ai_state(), make_control_command(ControlVerb.HOLD, "HUNT"), _NOW
        )
        assert after["manual_mode"] is AIMode.HUNT

    def test_disengage_drops_blocks_and_pins(self) -> None:
        """The lock drops, its target is blocked for the cooldown, COLLECT is pinned."""
        before = AIStateDict(**{**_engaged(), "blocked_combat_targets": {"7": 90000}})
        after = apply_control(before, make_control_command(ControlVerb.DISENGAGE, ""), _NOW)
        assert after["combat_target_id"] == -1
        assert after["blocked_combat_targets"] == {"7": 90000, "42": _NOW}
        assert after["manual_mode"] is AIMode.COLLECT
        assert after["wind_down"] is False

    def test_disengage_with_no_lock_blocks_nobody(self) -> None:
        """Nothing to walk away from: the pin alone."""
        after = apply_control(
            make_scanned_ai_state(), make_control_command(ControlVerb.DISENGAGE, ""), _NOW
        )
        assert after["blocked_combat_targets"] == {}
        assert after["manual_mode"] is AIMode.COLLECT

    def test_disengage_silences_return_fire_at_the_target_it_left(self) -> None:
        """Through the real collect return-fire rung, on a real attacker.

        Clearing the lock alone leaves the lock-free COLLECT tick
        returning fire at the enemy just left (live on sedona
        2026-09-29: six opportunity shots at it); the block is what
        makes it a disengage.
        """
        engaged = AIStateDict(
            **{**_engaged(), "combat_target_id": _ATTACKER_ID, "mode": AIMode.COLLECT}
        )
        lock_cleared = AIStateDict(
            **{**clear_combat_target(engaged), "manual_mode": AIMode.COLLECT}
        )
        cleared_ctx = _under_fire(lock_cleared)
        refought = collect_return_fire(cleared_ctx, cleared_ctx.base)
        if refought is None:
            raise AssertionError("expected the lock-free tick to return fire")
        assert refought["command"]["cmd_type"] == "shoot"
        assert refought["updated_ai_state"]["last_shot_target_id"] == _ATTACKER_ID

        disengaged = apply_control(engaged, make_control_command(ControlVerb.DISENGAGE, ""), _NOW)
        ctx = _under_fire(disengaged)
        assert collect_return_fire(ctx, ctx.base) is None

    def test_wind_down_raises_the_session_flag(self) -> None:
        """The same flag the session clock and the kill target raise."""
        after = apply_control(_engaged(), make_control_command(ControlVerb.WIND_DOWN, ""), _NOW)
        assert after["wind_down"] is True
        # A pin skips the arbitration that holds the stocked exit.
        assert after["manual_mode"] is None
        # The lock stays: a live fight finishes first, as for the clock.
        assert after["combat_target_id"] == 42

    def test_wind_down_on_a_pinned_stocked_bot_ends_the_session(self) -> None:
        """Through the real ``decide``: the pin cleared, the stocked exit fires.

        A stocked bot pinned to HUNT never reaches the arbitration that
        holds the wind-down exit, so the same flag without the cleared
        pin decides a hunt tick instead of ending.
        """
        world, self_state = make_world(fuel=1200)
        pinned = AIStateDict(**{**make_scanned_ai_state(), "manual_mode": AIMode.HUNT})

        flag_only = AIStateDict(**{**pinned, "wind_down": True})
        kept = decide(
            world, self_state, flag_only, make_inventory(), 100000, None, ws=WorldService()
        )
        assert kept["behavior"]["mode"] == "HUNT"

        wound = apply_control(pinned, make_control_command(ControlVerb.WIND_DOWN, ""), _NOW)
        with pytest.raises(SessionExitError) as raised:
            decide(world, self_state, wound, make_inventory(), 100000, None, ws=WorldService())
        assert raised.value.reason is SessionExitReason.SESSION_COMPLETE

    def test_release_clears_the_pin(self) -> None:
        """Auto-arbitration again."""
        after = apply_control(_engaged(), make_control_command(ControlVerb.RELEASE, ""), _NOW)
        assert after["manual_mode"] is None
        assert after["combat_target_id"] == 42

    def test_doctrine_rewrites_the_config_in_place(self) -> None:
        """A bot reads its environment config once, so the verb is the only way to change it."""
        before = make_scanned_ai_state()
        after = apply_control(before, make_control_command(ControlVerb.DOCTRINE, "duelist"), _NOW)
        assert after["config"]["doctrine"] is EngagementDoctrine.DUELIST
        assert before["config"]["doctrine"] is EngagementDoctrine.SKIRMISH
        assert {**after["config"], "doctrine": EngagementDoctrine.SKIRMISH} == before["config"]

    def test_no_pending_verb_leaves_the_state_as_it_was(self, tmp_path: Path) -> None:
        """The common tick: nothing written, nothing changed."""
        state = _engaged()
        assert apply_pending_control(state, tmp_path, _NOW) is state

    def test_a_pending_verb_is_taken_and_applied(self, tmp_path: Path) -> None:
        """The whole path the tick loop runs."""
        write_control(tmp_path, make_control_command(ControlVerb.DISENGAGE, ""))
        after = apply_pending_control(_engaged(), tmp_path, _NOW)
        assert after["combat_target_id"] == -1
        assert after["blocked_combat_targets"] == {"42": _NOW}
        assert after["manual_mode"] is AIMode.COLLECT
        assert control_pending(tmp_path) is False
