"""The kill ledger carries victim rank; the DOM verdict is witnessed.

The points-floor survey (operator flags 5/6/8/12/13, 2026-09-01):
World kills render a verdict — "You earned extra points" or "Enemy's
rank was too low" — but no ledger row ever paired a verdict with the
victim rank it judged, and "kill registered" itself never named the
victim. These tests pin the two witness channels that make the floor
fittable from run artifacts alone.
"""

from __future__ import annotations

import logging as stdlib_logging

from tankpit_bot.bot.base import Bot
from tankpit_bot.bot.tick_combat_feedback import _merge_protocol_kills
from tankpit_bot.browser.dom_scraper import GameLogEntry
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_combat import mark_tank_killed
from tankpit_bot.state.types import WorldStateDict, make_self_state, make_tank_state
from tests._runtime_logging_support import capture_runtime_events, event_fields
from tests.conftest import FakeEnv

_BANNER_RULE = "********************************************"


def _world_with_self(rank: int = 4) -> WorldService:
    """A world whose own tank is established at the given rank."""
    ws = WorldService()
    ws.update_world_state_from_position(100, 100)
    ws.world_state = WorldStateDict(
        **{
            **ws.world_state,
            "self_state": make_self_state(
                tank_id=2731,
                x=100,
                y=100,
                team=0,
                rank=rank,
                fuel=900,
                leaderboard_position=0,
            ),
        }
    )
    return ws


def _register_enemy(ws: WorldService, tank_id: int, name: str, rank: int) -> None:
    """Plant an enemy in the registry the way the wire would."""
    ws.world_state["tanks"][str(tank_id)] = make_tank_state(
        tank_id=tank_id,
        x=104,
        y=100,
        team=1,
        rank=rank,
        name=name,
        is_self=False,
        is_bot=False,
        damage_state=2,
        timestamp_ms=100000,
        last_wire_seen_ms=100000,
        last_position_update_ms=100000,
    )


def _diagnostics(
    records: list[stdlib_logging.LogRecord], kind: str
) -> list[dict[str, str | int | float | bool]]:
    """Extract the fields of every captured diagnostic of one kind."""
    return [
        event_fields(record)
        for record in records
        if event_fields(record).get("diagnostic_kind") == kind
    ]


def test_own_kill_ledgers_the_victims_name_and_rank(fake_env: FakeEnv) -> None:
    """A registry-known victim's rank rides the kill_registered row."""
    ws = _world_with_self()
    _register_enemy(ws, 517, "purple-9", 1)
    bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
    mark_tank_killed(ws, 517, 2731)

    with capture_runtime_events() as records:
        new_state = _merge_protocol_kills(bot.world, bot._ai_state)

    assert new_state["session_kill_count"] == 1
    assert _diagnostics(records, "kill_registered") == [
        {
            "diagnostic_kind": "kill_registered",
            "victim_id": 517,
            "victim_name": "purple-9",
            "victim_rank": 1,
        }
    ]


def test_own_kill_of_an_unregistered_victim_reports_rank_unknown(fake_env: FakeEnv) -> None:
    """A victim the 0x58 cleanup already removed ledgers rank -1, never a guess."""
    ws = _world_with_self()
    bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
    mark_tank_killed(ws, 599, 2731)

    with capture_runtime_events() as records:
        _merge_protocol_kills(bot.world, bot._ai_state)

    assert _diagnostics(records, "kill_registered") == [
        {
            "diagnostic_kind": "kill_registered",
            "victim_id": 599,
            "victim_name": "",
            "victim_rank": -1,
        }
    ]


def _feed(bot: Bot, lines: list[str]) -> list[stdlib_logging.LogRecord]:
    """Run one witness poll over the given DOM lines, capturing events."""
    entries = [GameLogEntry(text=line, category="other") for line in lines]
    with capture_runtime_events() as records:
        bot._record_game_log_witness(entries)
    return records


def test_rank_too_low_verdict_is_paired_with_the_banner_victim(fake_env: FakeEnv) -> None:
    """The five-line banner plus verdict emits one paired diagnostic."""
    ws = _world_with_self(rank=4)
    _register_enemy(ws, 517, "purple-9", 1)
    bot = Bot("https://test.tankpit.com/", headless=True, world=ws)

    records = _feed(
        bot,
        [
            _BANNER_RULE,
            "purple-9",
            "has been deactivated by you",
            _BANNER_RULE,
            "Enemy's rank was too low.",
            "No extra points are given",
        ],
    )

    assert _diagnostics(records, "kill_points_outcome") == [
        {
            "diagnostic_kind": "kill_points_outcome",
            "victim_name": "purple-9",
            "victim_rank": 1,
            "self_rank": 4,
            "outcome": "rank_too_low",
        }
    ]


def test_extra_points_verdict_survives_a_poll_boundary(fake_env: FakeEnv) -> None:
    """A banner split across two polls still pairs; unknown names rank -1."""
    ws = _world_with_self(rank=4)
    # A registry holding OTHER tanks must not lend the unknown victim
    # a rank — the name lookup has to walk past non-matches to its -1.
    _register_enemy(ws, 523, "blue-6", 2)
    bot = Bot("https://test.tankpit.com/", headless=True, world=ws)

    first = _feed(bot, [_BANNER_RULE, "orange-5", "has been deactivated by you"])
    second = _feed(bot, [_BANNER_RULE, "You earned extra points"])

    assert _diagnostics(first, "kill_points_outcome") == []
    assert _diagnostics(second, "kill_points_outcome") == [
        {
            "diagnostic_kind": "kill_points_outcome",
            "victim_name": "orange-5",
            "victim_rank": -1,
            "self_rank": 4,
            "outcome": "extra_points",
        }
    ]


def test_practice_banner_without_a_verdict_emits_nothing(fake_env: FakeEnv) -> None:
    """No verdict line (practice rooms) means no diagnostic — witness only."""
    ws = _world_with_self()
    bot = Bot("https://test.tankpit.com/", headless=True, world=ws)

    records = _feed(
        bot,
        [
            _BANNER_RULE,
            "blue-1",
            "has been deactivated by you",
            _BANNER_RULE,
            "5 dual shots gained",
        ],
    )

    assert _diagnostics(records, "kill_points_outcome") == []


def test_verdict_with_no_pending_banner_emits_nothing(fake_env: FakeEnv) -> None:
    """A stray verdict line pairs with nothing rather than fabricating a victim."""
    ws = _world_with_self()
    bot = Bot("https://test.tankpit.com/", headless=True, world=ws)

    records = _feed(bot, ["You earned extra points"])

    assert _diagnostics(records, "kill_points_outcome") == []
