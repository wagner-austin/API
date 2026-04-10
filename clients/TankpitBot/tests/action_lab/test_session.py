"""Tests for shared action-lab session helpers."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.session import (
    ActionLabSessionError,
    wait_for_initial_self_state,
    wait_for_world_sync,
)
from tankpit_bot.state import (
    ViewportStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)


class _Clock:
    def __init__(self, start_ms: int) -> None:
        self._now_ms = start_ms

    def __call__(self) -> int:
        return self._now_ms

    def advance(self, delta_ms: int) -> None:
        self._now_ms += delta_ms


class _SequencedProvider:
    def __init__(self, worlds: list[WorldStateDict]) -> None:
        self._worlds = worlds
        self._index = 0
        self._cdp_message_buffer: list[str] = []

    def get_world_state(self) -> WorldStateDict:
        return self._worlds[self._index]

    def advance(self) -> None:
        if self._index + 1 < len(self._worlds):
            self._index += 1


class _FakePage:
    def __init__(self, clock: _Clock, provider: _SequencedProvider) -> None:
        self._clock = clock
        self._provider = provider
        self.waits: list[float] = []

    def wait_for_timeout(self, timeout: float) -> None:
        self.waits.append(timeout)
        self._clock.advance(int(timeout))
        self._provider.advance()


def _make_world(
    timestamp_ms: int,
    x: int,
    y: int,
    fuel: int,
    *,
    self_state_available: bool,
) -> WorldStateDict:
    world = make_empty_world_state()
    return WorldStateDict(
        self_state=(
            make_self_state(
                tank_id=1,
                x=x,
                y=y,
                team=2,
                rank=1,
                fuel=fuel,
                leaderboard_position=1,
            )
            if self_state_available
            else None
        ),
        tanks=world["tanks"],
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=ViewportStateDict(left=0, top=0, width=16, height=16),
        scanned_viewports=world["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


@pytest.fixture(autouse=True)
def _restore_action_hooks() -> Generator[None, None, None]:
    original_get_time = action_hooks.get_current_time_ms
    original_drain = action_hooks.drain_buffered_messages
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.drain_buffered_messages = original_drain


def test_wait_for_world_sync_returns_newer_timestamp() -> None:
    clock = _Clock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(1200, 100, 100, 900, self_state_available=True),
        ]
    )
    page = _FakePage(clock, provider)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    assert wait_for_world_sync(page, provider, 1000, 500) == 1200


def test_wait_for_world_sync_times_out() -> None:
    clock = _Clock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(900, 100, 100, 900, self_state_available=True),
        ]
    )
    page = _FakePage(clock, provider)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    assert wait_for_world_sync(page, provider, 1000, 250) is None


def test_wait_for_initial_self_state_returns_fresh_self_state() -> None:
    clock = _Clock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=False),
            _make_world(1200, 101, 102, 875, self_state_available=True),
        ]
    )
    page = _FakePage(clock, provider)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    timestamp_ms, self_state = wait_for_initial_self_state(page, provider, 1000, 500)

    assert timestamp_ms == 1200
    assert self_state["x"] == 101
    assert self_state["y"] == 102
    assert self_state["fuel"] == 875


def test_wait_for_initial_self_state_raises_on_timeout() -> None:
    clock = _Clock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=False),
            _make_world(1200, 100, 100, 900, self_state_available=False),
            _make_world(1300, 100, 100, 900, self_state_available=False),
        ]
    )
    page = _FakePage(clock, provider)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    with pytest.raises(
        ActionLabSessionError,
        match="initial self state is unavailable after initial sync wait",
    ):
        wait_for_initial_self_state(page, provider, 1000, 250)


class _StartupBot:
    def __init__(self, states: list[str]) -> None:
        self._states = states
        self._index = 0

    def get_state(self) -> str:
        return self._states[self._index]

    def _update_state_from_world(self) -> None:
        if self._index + 1 < len(self._states):
            self._index += 1


def test_advance_startup_state_reaches_idle() -> None:
    bot = _StartupBot(["INITIALIZING", "WAITING_FOR_POSITION", "IDLE"])

    action_session.advance_startup_state(bot)

    assert bot.get_state() == "IDLE"


def test_advance_startup_state_raises_when_state_does_not_progress() -> None:
    bot = _StartupBot(["INITIALIZING"])

    with pytest.raises(ActionLabSessionError, match="startup state did not advance"):
        action_session.advance_startup_state(bot)


def test_wait_for_initial_self_state_drains_buffered_messages_before_reading() -> None:
    clock = _Clock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=False),
            _make_world(1200, 105, 106, 880, self_state_available=True),
        ]
    )
    page = _FakePage(clock, provider)
    action_hooks.get_current_time_ms = clock

    def _drain(source: BufferedMessageSourceProtocol, /) -> int:
        _ = source
        provider.advance()
        return 1

    action_hooks.drain_buffered_messages = _drain

    timestamp_ms, self_state = wait_for_initial_self_state(page, provider, 1000, 500)

    assert timestamp_ms == 1200
    assert self_state["x"] == 105
    assert self_state["y"] == 106
    assert page.waits == []
