"""Tests for shared probe-session envelope helpers."""

from __future__ import annotations

from collections.abc import Generator

from tests.action_lab._replay_page import ReplayClock

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.probe_runtime import ProbeCommandReadyContextDict
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.state import SelfStateDict, make_self_state


class _ProbeHarness:
    def __init__(self) -> None:
        self._start_timestamp_ms = 1000
        self._target_url = "https://tankpit.com/play"

    @property
    def session_id(self) -> str:
        return "probe-session"


def _spawn() -> SelfStateDict:
    return make_self_state(
        tank_id=1,
        x=120,
        y=121,
        team=2,
        rank=1,
        fuel=900,
        leaderboard_position=5,
    )


def _context(spawn: SelfStateDict) -> ProbeCommandReadyContextDict:
    return ProbeCommandReadyContextDict(
        game_ready_timestamp_ms=1100,
        intel_ready_timestamp_ms=1200,
        initial_sync_started_ms=1300,
        initial_world_timestamp_ms=1500,
        spawn=spawn,
        command_ready_timestamp_ms=1600,
    )


def _restore_clock() -> Generator[None, None, None]:
    original_get_time = action_hooks.get_current_time_ms
    yield
    action_hooks.get_current_time_ms = original_get_time


def test_build_probe_session_envelope_returns_shared_session_fields() -> None:
    restore = _restore_clock()
    next(restore)
    action_hooks.get_current_time_ms = ReplayClock(1900)
    probe = _ProbeHarness()
    spawn = _spawn()

    try:
        envelope = build_probe_session_envelope(
            probe,
            context=_context(spawn),
            first_attempt_started_ms=1700,
        )
    finally:
        next(restore, None)

    assert envelope.session_id == "probe-session"
    assert envelope.start_timestamp_ms == 1000
    assert envelope.end_timestamp_ms == 1900
    assert envelope.base_url == "https://tankpit.com/play"
    assert envelope.spawn_x == 120
    assert envelope.spawn_y == 121
    assert envelope.startup_timing["command_ready_to_first_attempt_ms"] == 100


def test_build_probe_session_envelope_preserves_missing_first_attempt() -> None:
    restore = _restore_clock()
    next(restore)
    action_hooks.get_current_time_ms = ReplayClock(1900)
    probe = _ProbeHarness()
    spawn = _spawn()

    try:
        envelope = build_probe_session_envelope(
            probe,
            context=_context(spawn),
            first_attempt_started_ms=None,
        )
    finally:
        next(restore, None)

    assert envelope.startup_timing["first_attempt_started_ms"] is None
    assert envelope.startup_timing["command_ready_to_first_attempt_ms"] is None
