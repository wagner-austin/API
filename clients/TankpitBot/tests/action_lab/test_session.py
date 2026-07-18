"""Tests for shared action-lab session helpers."""

from __future__ import annotations

from collections.abc import Callable, Generator

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError
from tests.action_lab._replay_core import ClockAdvancingPage, ReplayClock, StubSnapshotCDPSession

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol, CDPSessionProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.session import (
    ActionLabSessionError,
    capture_teleport_page_snapshot,
    wait_for_initial_self_state,
    wait_for_radar_sync,
    wait_for_world_sync,
)
from tankpit_bot.state import (
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.state.types import make_viewport_state
from tankpit_bot.types import CapturedMessage


class _SequencedProvider:
    def __init__(self, worlds: list[WorldStateDict]) -> None:
        self._worlds = worlds
        self._index = 0
        self._cdp_message_buffer: list[str] = []
        self._messages: list[CapturedMessage] = []
        self._magic: str | None = None

    def get_world_state(self) -> WorldStateDict:
        return self._worlds[self._index]

    def advance(self) -> None:
        if self._index + 1 < len(self._worlds):
            self._index += 1

    @property
    def messages(self) -> list[CapturedMessage]:
        return self._messages

    @property
    def magic(self) -> str | None:
        return self._magic


class _InspectingCDPSession:
    def __init__(self, snapshot: JSONObject) -> None:
        self._snapshot = snapshot
        self.expression = ""

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        assert method == "Runtime.evaluate"
        assert params is not None
        self.expression = str(params.get("expression", ""))
        return {"result": {"value": self._snapshot}}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        _ = (event, handler)

    def detach(self) -> None:
        return None


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
        viewport=make_viewport_state(left=0, top=0, width=16, height=16),
        scanned_tiles=world["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


@pytest.fixture(autouse=True)
def _restore_action_hooks() -> Generator[None, None, None]:
    original_get_time = action_hooks.get_current_time_ms
    original_drain = action_hooks.drain_buffered_messages
    original_check_radar = action_hooks.check_and_clear_radar_scan_complete
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.drain_buffered_messages = original_drain
    action_hooks.check_and_clear_radar_scan_complete = original_check_radar


def test_wait_for_world_sync_returns_newer_timestamp() -> None:
    clock = ReplayClock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(1200, 100, 100, 900, self_state_available=True),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    assert wait_for_world_sync(page, provider, 1000, 500) == 1200


def test_wait_for_world_sync_times_out() -> None:
    clock = ReplayClock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(900, 100, 100, 900, self_state_available=True),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    assert wait_for_world_sync(page, provider, 1000, 250) is None


def test_wait_for_radar_sync_returns_completion_timestamp() -> None:
    clock = ReplayClock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(1200, 100, 100, 900, self_state_available=True),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0
    radar_results = [False, False, True]

    def _check_radar_complete() -> bool:
        return radar_results.pop(0)

    action_hooks.check_and_clear_radar_scan_complete = _check_radar_complete

    assert wait_for_radar_sync(page, provider, 1000, 500) == 1200


def test_wait_for_radar_sync_times_out() -> None:
    clock = ReplayClock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(900, 100, 100, 900, self_state_available=True),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0
    action_hooks.check_and_clear_radar_scan_complete = lambda: False

    assert wait_for_radar_sync(page, provider, 1000, 250) is None


def test_wait_for_radar_sync_ignores_stale_completion_without_new_activity() -> None:
    clock = ReplayClock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=True),
            _make_world(900, 100, 100, 900, self_state_available=True),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
    action_hooks.get_current_time_ms = clock
    drain_results = [0, 1]

    def _drain(source: BufferedMessageSourceProtocol, /) -> int:
        _ = source
        if len(drain_results) == 0:
            return 0
        return drain_results.pop(0)

    action_hooks.drain_buffered_messages = _drain
    radar_results = [True, True]

    def _check_radar_complete() -> bool:
        if len(radar_results) == 0:
            return False
        return radar_results.pop(0)

    action_hooks.check_and_clear_radar_scan_complete = _check_radar_complete

    assert wait_for_radar_sync(page, provider, 1000, 500) == 1100


def test_wait_for_initial_self_state_returns_fresh_self_state() -> None:
    clock = ReplayClock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=False),
            _make_world(1200, 101, 102, 875, self_state_available=True),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0

    timestamp_ms, self_state = wait_for_initial_self_state(page, provider, 1000, 500)

    assert timestamp_ms == 1200
    assert self_state["x"] == 101
    assert self_state["y"] == 102
    assert self_state["fuel"] == 875


def test_wait_for_initial_self_state_raises_on_timeout() -> None:
    clock = ReplayClock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=False),
            _make_world(1200, 100, 100, 900, self_state_available=False),
            _make_world(1300, 100, 100, 900, self_state_available=False),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
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
    clock = ReplayClock(1000)
    provider = _SequencedProvider(
        [
            _make_world(900, 100, 100, 900, self_state_available=False),
            _make_world(1200, 105, 106, 880, self_state_available=True),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
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


def test_capture_teleport_page_snapshot_returns_validated_state() -> None:
    """Teleport page snapshots decode strict CDP evaluation results."""
    cdp: CDPSessionProtocol = StubSnapshotCDPSession(
        {
            "timestamp_ms": 1234,
            "client_present": True,
            "map_visible": True,
            "client_state": 13,
            "client_busy": False,
            "pending_actions": 0,
            "heartbeat_age_ms": 50,
            "last_page_client_send_age_ms": 75,
            "last_bot_send_age_ms": 5,
            "ws_ready_state": 1,
            "current_send_label": None,
            "sent_frame_meta_queue_length": 0,
            "self_fields": {},
            "world_fields": {},
            "map_fields": {},
            "world_collections": {},
        }
    )

    snapshot = capture_teleport_page_snapshot(cdp, "after_map_data")

    assert snapshot["phase"] == "after_map_data"
    assert snapshot["client_state"] == 13
    assert snapshot["last_page_client_send_age_ms"] == 75


def test_capture_teleport_page_snapshot_rejects_invalid_payload() -> None:
    """Teleport page snapshots reject malformed CDP values."""
    cdp: CDPSessionProtocol = StubSnapshotCDPSession({"timestamp_ms": "bad"})

    with pytest.raises(JSONTypeError, match=r"timestamp_ms|client_present"):
        capture_teleport_page_snapshot(cdp, "timeout")


def test_capture_teleport_page_snapshot_rejects_missing_value_field() -> None:
    """Teleport page snapshots reject CDP responses without a value field."""

    class _MissingValueCDPSession:
        def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
            _ = (method, params)
            return {"result": {}}

        def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
            _ = (event, handler)

        def detach(self) -> None:
            return None

    cdp: CDPSessionProtocol = _MissingValueCDPSession()

    with pytest.raises(ValueError, match="missing value"):
        capture_teleport_page_snapshot(cdp, "before_map_open")


def test_capture_teleport_page_snapshot_reads_injected_active_game_handle() -> None:
    """Teleport snapshot JS reads the injected game handle instead of window.W."""
    cdp = _InspectingCDPSession(
        {
            "timestamp_ms": 1234,
            "client_present": False,
            "map_visible": None,
            "client_state": None,
            "client_busy": None,
            "pending_actions": None,
            "heartbeat_age_ms": None,
            "last_page_client_send_age_ms": None,
            "last_bot_send_age_ms": None,
            "ws_ready_state": 1,
            "current_send_label": None,
            "sent_frame_meta_queue_length": 0,
            "self_fields": {},
            "world_fields": {},
            "map_fields": {},
            "world_collections": {},
        }
    )

    capture_teleport_page_snapshot(cdp, "timeout")

    assert "window.__tankpitActiveGame" in cdp.expression
    assert "typeof W !== 'undefined'" not in cdp.expression
