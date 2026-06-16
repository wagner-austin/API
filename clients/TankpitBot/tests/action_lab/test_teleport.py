"""Tests for live teleport probe helpers."""

from __future__ import annotations

import base64
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Literal

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import (
    ClockAdvancingPage,
    ReplayClock,
    StubbedBootstrapMixin,
    StubSnapshotCDPSession,
    WorldStateOverrideMixin,
)
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    CDPSessionProtocol,
    PageProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.teleport import (
    TeleportProbe,
    run_teleport_probe,
)
from tankpit_bot.action_lab.teleport_attempt import TrackedTeleportAttempt
from tankpit_bot.action_lab.teleport_helpers import (
    TeleportProbeError,
    _find_map_data_message_index,
    _format_attempt_window_entries,
    _format_page_snapshots,
    _limit_targets,
    _start_teleport_page_snapshots,
    _wait_for_teleport_outcome,
    build_box_targets,
    format_teleport_probe_summary,
    parse_targets_arg,
)
from tankpit_bot.action_lab.teleport_phase import TeleportOutcomeWaiterProtocol
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportProbeSessionDict,
    TeleportTargetDict,
)
from tankpit_bot.action_lab.types_codecs import decode_teleport_probe_session
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.state import (
    SelfStateDict,
    ViewportStateDict,
    WorldStateDict,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.types import CapturedMessage, decode_capture_session

_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


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


class _AckSequence:
    def __init__(self, values: list[bool]) -> None:
        self._values = values
        self._index = 0

    def __call__(self) -> bool:
        if self._index >= len(self._values):
            return False
        value = self._values[self._index]
        self._index += 1
        return value


def _make_world(timestamp_ms: int, x: int, y: int, fuel: int) -> WorldStateDict:
    world = make_empty_world_state()
    return WorldStateDict(
        self_state=make_self_state(
            tank_id=1,
            x=x,
            y=y,
            team=2,
            rank=1,
            fuel=fuel,
            leaderboard_position=5,
        ),
        tanks=world["tanks"],
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=ViewportStateDict(left=0, top=0, width=16, height=16),
        scanned_viewports=world["scanned_viewports"],
        map_fuel_dots={},
        timestamp_ms=timestamp_ms,
    )


def _make_attempt(
    status: Literal["landed_exact", "landed_offset", "map_sync_timeout", "teleport_timeout"],
) -> TeleportAttemptResultDict:
    return TeleportAttemptResultDict(
        target=TeleportTargetDict(label=status, x=150, y=171),
        teleport_cycle_id=1,
        status=status,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200 if status != "map_sync_timeout" else None,
        teleport_started_ms=1300 if status != "map_sync_timeout" else None,
        completion_timestamp_ms=1500,
        map_sync_elapsed_ms=200 if status != "map_sync_timeout" else None,
        teleport_elapsed_ms=200 if status in ("landed_exact", "landed_offset") else None,
        fuel_before=900,
        fuel_after=840,
        world_timestamp_before=950,
        world_timestamp_after=1450,
        landed_signal_received=status in ("landed_exact", "landed_offset"),
        landed_x=150,
        landed_y=171,
        message_start_index=10,
        message_end_index=14,
        page_snapshots=[],
    )


def _make_page_snapshot(
    phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
) -> TeleportPageSnapshotDict:
    """Build a sample teleport page snapshot."""
    return TeleportPageSnapshotDict(
        phase=phase,
        timestamp_ms=1000,
        client_present=True,
        map_visible=True,
        client_state=13,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=12,
        last_page_client_send_age_ms=250,
        last_bot_send_age_ms=10,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        map_fields={},
        world_collections={},
    )


@pytest.fixture(autouse=True)
def _restore_action_hooks() -> Generator[None, None, None]:
    original_get_time = action_hooks.get_current_time_ms
    original_check_landed = action_hooks.check_and_clear_teleport_landed
    yield
    action_hooks.get_current_time_ms = original_get_time
    action_hooks.check_and_clear_teleport_landed = original_check_landed


def test_build_box_targets_creates_ten_targets() -> None:
    targets = build_box_targets(100, 100, 8, 6)
    assert len(targets) == 10
    assert targets[0]["label"] == "box_r0_c0"
    assert targets[-1]["label"] == "box_r1_c4"
    assert targets[0]["x"] == 84
    assert targets[0]["y"] == 94
    assert targets[-1]["x"] == 116
    assert targets[-1]["y"] == 106


def test_build_box_targets_clamps_edges() -> None:
    targets = build_box_targets(2, 2, 8, 8)
    assert targets[0]["x"] == 0
    assert targets[0]["y"] == 0


def test_build_box_targets_clamps_upper_edges() -> None:
    targets = build_box_targets(254, 254, 8, 8)
    assert targets[-1]["x"] == 255
    assert targets[-1]["y"] == 255


def test_build_box_targets_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="step_x"):
        build_box_targets(100, 100, 0, 8)
    with pytest.raises(ValueError, match="step_y"):
        build_box_targets(100, 100, 8, 0)


def test_format_attempt_window_entries_filters_direction_and_reports_more() -> None:
    provider = _SequencedProvider([_make_world(900, 100, 100, 900)])
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload="$1|0",
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1001,
            direction="sent",
            payload="%AUTH",
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1002,
            direction="sent",
            payload="%MOVE",
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1003,
            direction="sent",
            payload="%RADAR",
            ws_url="wss://tankpit.com/ws/",
        ),
    ]

    summary = _format_attempt_window_entries(
        provider,
        message_start_index=0,
        direction="sent",
        limit=2,
    )

    assert "1:" in summary
    assert "2:" in summary
    assert "...+1 more" in summary


def test_format_attempt_window_entries_returns_exact_window_without_more_suffix() -> None:
    provider = _SequencedProvider([_make_world(900, 100, 100, 900)])
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1001,
            direction="sent",
            payload="%AUTH",
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1002,
            direction="sent",
            payload="%MOVE",
            ws_url="wss://tankpit.com/ws/",
        ),
    ]

    summary = _format_attempt_window_entries(
        provider,
        message_start_index=0,
        direction="sent",
        limit=2,
    )

    assert "0:" in summary
    assert "1:" in summary
    assert "...+" not in summary


def test_format_attempt_window_entries_for_received_messages_omits_sent_metadata() -> None:
    provider = _SequencedProvider([_make_world(900, 100, 100, 900)])
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload="bad",
            ws_url="wss://tankpit.com/ws/",
        )
    ]

    summary = _format_attempt_window_entries(
        provider,
        message_start_index=0,
        direction="received",
        limit=6,
    )

    assert "origin=" not in summary


def test_format_attempt_window_entries_includes_sent_origin_metadata() -> None:
    provider = _SequencedProvider([_make_world(900, 100, 100, 900)])
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1001,
            direction="sent",
            payload="%AUTH",
            ws_url="wss://tankpit.com/ws/",
            sent_origin="bot_injected",
            sent_label="teleport(129,106)",
        ),
        CapturedMessage(
            timestamp_ms=1002,
            direction="sent",
            payload="%MOVE",
            ws_url="wss://tankpit.com/ws/",
            sent_origin="page_client",
        ),
    ]

    summary = _format_attempt_window_entries(
        provider,
        message_start_index=0,
        direction="sent",
        limit=6,
    )

    assert "origin=bot_injected label=teleport(129,106)" in summary
    assert "origin=page_client" in summary


def test_format_page_snapshots_returns_none_for_empty_list() -> None:
    assert _format_page_snapshots([]) == "none"


def test_find_map_data_message_index_skips_earlier_and_sent_messages() -> None:
    provider = _SequencedProvider([_make_world(900, 100, 100, 900)])
    map_data_payload = base64.b64encode(bytes([0, 0, 0x2E]) + bytes(600)).decode("ascii")
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload=map_data_payload,
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1001,
            direction="sent",
            payload=map_data_payload,
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1002,
            direction="received",
            payload=map_data_payload,
            ws_url="wss://tankpit.com/ws/",
        ),
    ]

    result = _find_map_data_message_index(
        provider,
        message_start_index=1,
        scan_start_index=0,
    )

    assert result == 2


def test_find_map_data_message_index_skips_non_map_data_received_messages() -> None:
    provider = _SequencedProvider([_make_world(900, 100, 100, 900)])
    map_data_payload = base64.b64encode(bytes([0, 0, 0x2E]) + bytes(600)).decode("ascii")
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload="bad",
            ws_url="wss://tankpit.com/ws/",
        ),
        CapturedMessage(
            timestamp_ms=1001,
            direction="received",
            payload=map_data_payload,
            ws_url="wss://tankpit.com/ws/",
        ),
    ]

    result = _find_map_data_message_index(
        provider,
        message_start_index=0,
        scan_start_index=0,
    )

    assert result == 1


def test_start_teleport_page_snapshots_rejects_missing_cdp() -> None:
    with pytest.raises(TeleportProbeError, match="cdp session is unavailable"):
        _start_teleport_page_snapshots(
            cdp=None,
            capture_before_map_open=True,
            unavailable_error=TeleportProbeError,
            unavailable_message="cdp session is unavailable",
        )


def test_start_teleport_page_snapshots_can_skip_initial_capture() -> None:
    snapshots, capture_page_snapshot = _start_teleport_page_snapshots(
        cdp=StubSnapshotCDPSession(),
        capture_before_map_open=False,
        unavailable_error=TeleportProbeError,
        unavailable_message="cdp session is unavailable",
    )

    assert snapshots == []
    snapshot = capture_page_snapshot("timeout")
    assert snapshot["phase"] == "timeout"


def test_limit_targets_rejects_non_positive_max_targets() -> None:
    with pytest.raises(ValueError, match="max_targets must be positive"):
        _limit_targets([TeleportTargetDict(label="target_0", x=1, y=2)], 0)


def test_parse_targets_arg_parses_targets() -> None:
    targets = parse_targets_arg("156:170,147:166")
    assert targets == [
        TeleportTargetDict(label="target_0", x=156, y=170),
        TeleportTargetDict(label="target_1", x=147, y=166),
    ]


def test_parse_targets_arg_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        parse_targets_arg("   ")
    with pytest.raises(ValueError, match="expected x:y"):
        parse_targets_arg("156-170")
    with pytest.raises(ValueError, match=r"outside 0\.\.255"):
        parse_targets_arg("999:10")


def test_wait_for_teleport_outcome_records_exact_landing() -> None:
    clock = ReplayClock(1200)
    provider = _SequencedProvider(
        [
            _make_world(1200, 100, 100, 900),
            _make_world(1300, 100, 100, 900),
            _make_world(1500, 156, 170, 720),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.check_and_clear_teleport_landed = _AckSequence([False, False, True])
    result = _wait_for_teleport_outcome(
        page,
        provider,
        TeleportTargetDict(label="target_0", x=156, y=170),
        teleport_cycle_id=1,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        fuel_before=900,
        world_timestamp_before=950,
        timeout_ms=1000,
        page_snapshots=[],
        capture_page_snapshot=_make_page_snapshot,
    )
    assert result["status"] == "landed_exact"
    assert result["landed_signal_received"] is True
    assert result["landed_x"] == 156
    assert result["fuel_after"] == 720


def test_wait_for_teleport_outcome_captures_after_map_data_snapshot() -> None:
    clock = ReplayClock(1200)
    provider = _SequencedProvider(
        [
            _make_world(1200, 100, 100, 900),
            _make_world(1400, 156, 170, 720),
        ]
    )
    map_data_payload = base64.b64encode(bytes([0, 0, 0x2E]) + bytes(600)).decode("ascii")
    provider._messages = [
        CapturedMessage(
            timestamp_ms=1200,
            direction="received",
            payload=map_data_payload,
            ws_url="wss://tankpit.com/ws/",
        )
    ]
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.check_and_clear_teleport_landed = _AckSequence([False, True])
    page_snapshots = [_make_page_snapshot("before_map_open")]
    captured_phases: list[str] = []

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        captured_phases.append(phase)
        return _make_page_snapshot(phase)

    result = _wait_for_teleport_outcome(
        page,
        provider,
        TeleportTargetDict(label="target_0", x=156, y=170),
        teleport_cycle_id=1,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        fuel_before=900,
        world_timestamp_before=950,
        timeout_ms=1000,
        page_snapshots=page_snapshots,
        capture_page_snapshot=_capture_page_snapshot,
    )

    assert captured_phases == ["after_map_data", "landed"]
    assert result["status"] == "landed_exact"


def test_wait_for_teleport_outcome_records_offset_landing() -> None:
    clock = ReplayClock(1200)
    provider = _SequencedProvider(
        [
            _make_world(1200, 100, 100, 900),
            _make_world(1350, 100, 100, 900),
            _make_world(1600, 159, 170, 860),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.check_and_clear_teleport_landed = _AckSequence([False, False, True])
    result = _wait_for_teleport_outcome(
        page,
        provider,
        TeleportTargetDict(label="target_0", x=156, y=170),
        teleport_cycle_id=1,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        fuel_before=900,
        world_timestamp_before=950,
        timeout_ms=1000,
        page_snapshots=[],
        capture_page_snapshot=_make_page_snapshot,
    )
    assert result["status"] == "landed_offset"
    assert result["landed_x"] == 159


def test_wait_for_teleport_outcome_raises_when_self_state_missing_after_landing() -> None:
    clock = ReplayClock(1200)
    world = _make_world(1200, 100, 100, 900)
    missing_self = WorldStateDict(
        self_state=None,
        tanks=world["tanks"],
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=world["viewport"],
        scanned_viewports=world["scanned_viewports"],
        map_fuel_dots={},
        timestamp_ms=1500,
    )
    provider = _SequencedProvider([world, missing_self])
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.check_and_clear_teleport_landed = _AckSequence([False, True])
    with pytest.raises(TeleportProbeError, match="self state disappeared after teleport landed"):
        _wait_for_teleport_outcome(
            page,
            provider,
            TeleportTargetDict(label="target_0", x=156, y=170),
            teleport_cycle_id=1,
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            fuel_before=900,
            world_timestamp_before=950,
            timeout_ms=1000,
            page_snapshots=[],
            capture_page_snapshot=_make_page_snapshot,
        )


def test_wait_for_teleport_outcome_times_out() -> None:
    clock = ReplayClock(1200)
    provider = _SequencedProvider(
        [
            _make_world(1200, 100, 100, 900),
            _make_world(1300, 100, 100, 900),
            _make_world(1400, 100, 100, 900),
        ]
    )
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.check_and_clear_teleport_landed = _AckSequence([False, False, False])
    result = _wait_for_teleport_outcome(
        page,
        provider,
        TeleportTargetDict(label="target_0", x=156, y=170),
        teleport_cycle_id=1,
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1200,
        teleport_started_ms=1300,
        fuel_before=900,
        world_timestamp_before=950,
        timeout_ms=250,
        page_snapshots=[],
        capture_page_snapshot=_make_page_snapshot,
    )
    assert result["status"] == "teleport_timeout"
    assert result["landed_signal_received"] is False


def test_wait_for_teleport_outcome_raises_when_self_state_missing_on_timeout() -> None:
    clock = ReplayClock(1200)
    world = _make_world(1200, 100, 100, 900)
    missing_self = WorldStateDict(
        self_state=None,
        tanks=world["tanks"],
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=world["viewport"],
        scanned_viewports=world["scanned_viewports"],
        map_fuel_dots={},
        timestamp_ms=1500,
    )
    provider = _SequencedProvider([world, missing_self, missing_self])
    page = ClockAdvancingPage(clock, on_wait=provider.advance)
    action_hooks.get_current_time_ms = clock
    action_hooks.check_and_clear_teleport_landed = _AckSequence([False, False, False])
    with pytest.raises(
        TeleportProbeError,
        match="self state disappeared while waiting for teleport timeout",
    ):
        _wait_for_teleport_outcome(
            page,
            provider,
            TeleportTargetDict(label="target_0", x=156, y=170),
            teleport_cycle_id=1,
            map_open_started_ms=1000,
            map_sync_timestamp_ms=1200,
            teleport_started_ms=1300,
            fuel_before=900,
            world_timestamp_before=950,
            timeout_ms=250,
            page_snapshots=[],
            capture_page_snapshot=_make_page_snapshot,
        )


def test_format_teleport_probe_summary_counts_statuses() -> None:
    session = TeleportProbeSessionDict(
        session_id="summary",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        teleport_strategy="sync_before_teleport",
        max_targets=4,
        capture_session_path="teleport_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 100,
            "intel_ready_timestamp_ms": 150,
            "initial_sync_started_ms": 200,
            "initial_world_timestamp_ms": 250,
            "command_ready_timestamp_ms": 300,
            "first_attempt_started_ms": 325,
            "game_ready_to_intel_ready_ms": 50,
            "intel_ready_to_initial_world_ms": 100,
            "initial_world_to_command_ready_ms": 50,
            "command_ready_to_first_attempt_ms": 25,
        },
        map_sync_timeout_ms=3000,
        teleport_timeout_ms=10000,
        settle_delay_ms=500,
        targets=[],
        attempts=[
            _make_attempt("landed_exact"),
            _make_attempt("landed_offset"),
            _make_attempt("map_sync_timeout"),
            _make_attempt("teleport_timeout"),
        ],
    )
    assert format_teleport_probe_summary(session) == (
        "Teleport probe complete: strategy=sync_before_teleport attempts=4 exact=1 "
        "offset=1 map_sync_timeout=1 teleport_timeout=1 "
        "session_to_initial_sync_ms=199 initial_sync_to_command_ready_ms=100"
    )


class _ProbeMethodHarness(TeleportProbe):
    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=False)
        self._self_state: SelfStateDict | None = make_self_state(
            tank_id=1,
            x=158,
            y=132,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )
        self._world_state = _make_world(1000, 158, 132, 900)
        self._fake_page = ClockAdvancingPage(
            ReplayClock(1000),
            on_wait=_SequencedProvider([self._world_state]).advance,
        )
        self._cdp = StubSnapshotCDPSession()
        self.map_open_result = True
        self.teleport_result = True
        self.teleport_calls: list[tuple[int, int]] = []

    def _require_page(self) -> PageProtocol:
        return self._fake_page

    def get_world_state(self) -> WorldStateDict:
        return self._world_state

    def get_self_state(self) -> SelfStateDict | None:
        return self._self_state

    def open_map(self) -> bool:
        return self.map_open_result

    def teleport_to(self, x: int, y: int) -> bool:
        self.teleport_calls.append((x, y))
        return self.teleport_result


class _ProbeMissingPageHarness(_ProbeMethodHarness):
    def _require_page(self) -> PageProtocol:
        raise TeleportProbeError("page is unavailable")


def test_probe_helpers_cover_guards_and_clear_action() -> None:
    probe = _ProbeMethodHarness()
    assert probe._require_self_state()["x"] == 158
    assert probe._require_page() is probe._fake_page

    probe._clear_in_flight_action()
    probe._reset_probe_state_to_idle()
    assert probe.get_state() == "IDLE"

    probe._self_state = None
    from tankpit_bot.action_lab.probe_base import ProbeError

    with pytest.raises(ProbeError, match="self state is unavailable"):
        probe._require_self_state()

    probe = _ProbeMissingPageHarness()
    with pytest.raises(ProbeError, match="page is unavailable"):
        probe._require_page()


def test_base_require_page_returns_page_and_raises_when_missing() -> None:
    from tankpit_bot.action_lab.probe_base import ProbeError

    probe = TeleportProbe("https://tankpit.com/play", headless=True, prefer_account=False)
    with pytest.raises(ProbeError, match="page is unavailable"):
        probe._require_page()
    fake_page = ClockAdvancingPage(
        ReplayClock(1000),
        on_wait=_SequencedProvider([_make_world(900, 158, 132, 900)]).advance,
    )
    probe._page = fake_page
    assert probe._require_page() is fake_page


def test_probe_single_target_returns_map_sync_timeout_and_settles() -> None:
    probe = _ProbeMethodHarness()
    page = probe._fake_page
    original_wait = action_hooks.wait_for_world_sync

    def _wait_sync_timeout(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page_arg, provider, started_ms, timeout_ms)
        return None

    wait_sync_name = "wait_for_world_sync"
    setattr(action_hooks, wait_sync_name, _wait_sync_timeout)
    try:
        result = probe._probe_single_target(
            TeleportTargetDict(label="target_0", x=150, y=171),
            teleport_strategy="sync_before_teleport",
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=250,
        )
    finally:
        setattr(action_hooks, wait_sync_name, original_wait)
    assert result["status"] == "map_sync_timeout"
    assert probe.teleport_calls == []
    assert result["message_start_index"] == 0
    assert result["message_end_index"] == 0
    assert page.waits[-1] == 250.0


def test_probe_single_target_returns_map_sync_timeout_without_settle() -> None:
    probe = _ProbeMethodHarness()
    page = probe._fake_page
    original_wait = action_hooks.wait_for_world_sync

    def _wait_sync_timeout(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page_arg, provider, started_ms, timeout_ms)
        return None

    wait_sync_name = "wait_for_world_sync"
    setattr(action_hooks, wait_sync_name, _wait_sync_timeout)
    try:
        result = probe._probe_single_target(
            TeleportTargetDict(label="target_0", x=150, y=171),
            teleport_strategy="sync_before_teleport",
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=0,
        )
    finally:
        setattr(action_hooks, wait_sync_name, original_wait)
    assert result["status"] == "map_sync_timeout"
    assert page.waits == []


def test_probe_single_target_returns_wait_result_without_settle() -> None:
    from tankpit_bot.action_lab import teleport as teleport_module

    probe = _ProbeMethodHarness()
    page = probe._fake_page
    expected = _make_attempt("landed_exact")
    original_wait_sync = action_hooks.wait_for_world_sync
    original_wait_outcome = teleport_module._wait_for_teleport_outcome

    def _wait_sync_success(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page_arg, provider, started_ms, timeout_ms)
        return 1200

    def _wait_outcome(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        _ = (
            page_arg,
            provider,
            target,
            teleport_cycle_id,
            message_start_index,
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        return expected

    wait_sync_name = "wait_for_world_sync"
    wait_outcome_name = "_wait_for_teleport_outcome"
    setattr(action_hooks, wait_sync_name, _wait_sync_success)
    setattr(teleport_module, wait_outcome_name, _wait_outcome)
    try:
        result = probe._probe_single_target(
            TeleportTargetDict(label="target_0", x=150, y=171),
            teleport_strategy="sync_before_teleport",
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=0,
        )
    finally:
        setattr(action_hooks, wait_sync_name, original_wait_sync)
        setattr(teleport_module, wait_outcome_name, original_wait_outcome)
    assert result == expected
    assert probe.teleport_calls == [(150, 171)]
    assert result["message_start_index"] == 10
    assert result["message_end_index"] == 14
    assert page.waits == []


def test_probe_single_target_rejects_missing_teleport_result_after_acquisition() -> None:
    from tankpit_bot.action_lab import teleport as teleport_module

    probe = _ProbeMethodHarness()
    original_attempt_runner = teleport_module.run_tracked_teleport_attempt

    def _run_attempt(
        page_arg: action_session.WaitPageProtocol,
        probe_arg: TeleportProbe,
        target_arg: TeleportTargetDict,
        *,
        cdp: CDPSessionProtocol | None,
        attempt_label: str,
        fuel_before: int,
        world_timestamp_before: int,
        send_acquisition_command: Callable[[], bool],
        acquisition_command_name: str,
        capture_before_map_open: bool,
        wait_for_acquisition_sync: bool,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        acquisition_dispatch_failure_message: str,
        teleport_dispatch_failure_message: str,
        unavailable_error: type[Exception],
        unavailable_message: str,
        unexpected_result_error: type[Exception],
        unexpected_result_message: str,
        reset_to_idle_before_start: bool = True,
    ) -> TrackedTeleportAttempt:
        _ = (
            page_arg,
            probe_arg,
            target_arg,
            cdp,
            attempt_label,
            fuel_before,
            world_timestamp_before,
            send_acquisition_command,
            acquisition_command_name,
            capture_before_map_open,
            wait_for_acquisition_sync,
            acquisition_timeout_ms,
            teleport_timeout_ms,
            wait_for_outcome,
            dispatch_failure_error,
            acquisition_dispatch_failure_message,
            teleport_dispatch_failure_message,
            unavailable_error,
            unavailable_message,
            unexpected_result_error,
            unexpected_result_message,
            reset_to_idle_before_start,
        )
        return TrackedTeleportAttempt(
            message_start_index=0,
            teleport_cycle=ActionPhaseCycleDict(phase="teleport", cycle_id=1, started_ms=1000),
            acquisition_started_ms=1000,
            acquisition_sync_timestamp_ms=1200,
            page_snapshots=[],
            capture_page_snapshot=lambda phase: _make_page_snapshot(phase),
            teleport_result=None,
            teleport_started_ms=None,
        )

    attempt_runner_name = "run_tracked_teleport_attempt"
    setattr(teleport_module, attempt_runner_name, _run_attempt)
    try:
        with pytest.raises(
            TeleportProbeError,
            match="teleport attempt ended before teleport dispatch",
        ):
            probe._probe_single_target(
                TeleportTargetDict(label="target_0", x=150, y=171),
                teleport_strategy="sync_before_teleport",
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                settle_delay_ms=0,
            )
    finally:
        setattr(teleport_module, attempt_runner_name, original_attempt_runner)


def test_probe_single_target_returns_wait_result_with_settle() -> None:
    from tankpit_bot.action_lab import teleport as teleport_module

    probe = _ProbeMethodHarness()
    page = probe._fake_page
    expected = _make_attempt("landed_exact")
    original_wait_sync = action_hooks.wait_for_world_sync
    original_wait_outcome = teleport_module._wait_for_teleport_outcome

    def _wait_sync_success(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page_arg, provider, started_ms, timeout_ms)
        return 1200

    def _wait_outcome(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        _ = (
            page_arg,
            provider,
            target,
            teleport_cycle_id,
            message_start_index,
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        return expected

    wait_sync_name = "wait_for_world_sync"
    wait_outcome_name = "_wait_for_teleport_outcome"
    setattr(action_hooks, wait_sync_name, _wait_sync_success)
    setattr(teleport_module, wait_outcome_name, _wait_outcome)
    try:
        result = probe._probe_single_target(
            TeleportTargetDict(label="target_0", x=150, y=171),
            teleport_strategy="sync_before_teleport",
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=250,
        )
    finally:
        setattr(action_hooks, wait_sync_name, original_wait_sync)
        setattr(teleport_module, wait_outcome_name, original_wait_outcome)
    assert result == expected
    assert result["message_start_index"] == 10
    assert result["message_end_index"] == 14
    assert page.waits[-1] == 250.0


def test_probe_single_target_immediate_strategy_skips_map_sync_wait() -> None:
    from tankpit_bot.action_lab import teleport as teleport_module

    probe = _ProbeMethodHarness()
    expected = _make_attempt("landed_exact")
    original_wait_sync = action_hooks.wait_for_world_sync
    original_wait_outcome = teleport_module._wait_for_teleport_outcome
    wait_sync_calls: list[int] = []

    def _wait_sync_unexpected(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page_arg, provider, started_ms, timeout_ms)
        wait_sync_calls.append(1)
        return 1200

    def _wait_outcome(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        _ = (
            page_arg,
            provider,
            target,
            teleport_cycle_id,
            message_start_index,
            map_open_started_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        assert map_sync_timestamp_ms is None
        return expected

    wait_sync_name = "wait_for_world_sync"
    wait_outcome_name = "_wait_for_teleport_outcome"
    setattr(action_hooks, wait_sync_name, _wait_sync_unexpected)
    setattr(teleport_module, wait_outcome_name, _wait_outcome)
    try:
        result = probe._probe_single_target(
            TeleportTargetDict(label="target_0", x=150, y=171),
            teleport_strategy="immediate_after_map_open",
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=0,
        )
    finally:
        setattr(action_hooks, wait_sync_name, original_wait_sync)
        setattr(teleport_module, wait_outcome_name, original_wait_outcome)
    assert result == expected
    assert wait_sync_calls == []
    assert probe.teleport_calls == [(150, 171)]


class _ExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, TeleportProbe):
    def __init__(self) -> None:
        TeleportProbe.__init__(
            self, "https://tankpit.com/play", headless=False, prefer_account=True
        )
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 158, 132, 900)
        self.probed_targets: list[TeleportTargetDict] = []
        self.result_attempts: list[TeleportAttemptResultDict] = []

    def _probe_single_target(
        self,
        target: TeleportTargetDict,
        *,
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
    ) -> TeleportAttemptResultDict:
        assert teleport_strategy == "sync_before_teleport"
        assert map_sync_timeout_ms == 3000
        assert teleport_timeout_ms == 10000
        assert settle_delay_ms == 500
        self.probed_targets.append(target)
        return self.result_attempts[len(self.probed_targets) - 1]


def test_execute_raises_when_playwright_is_missing() -> None:
    from tankpit_bot import _test_hooks as core_hooks

    probe = _ProbeMethodHarness()
    original_playwright = core_hooks.sync_playwright
    core_hooks.sync_playwright = None
    try:
        with pytest.raises(PlaywrightNotInstalledError):
            probe.execute(
                explicit_targets=[],
                box_step_x=8,
                box_step_y=8,
                max_targets=None,
                teleport_strategy="sync_before_teleport",
                initial_sync_timeout_ms=10000,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                settle_delay_ms=500,
            )
    finally:
        core_hooks.sync_playwright = original_playwright


def test_execute_rejects_empty_explicit_targets_and_cleans_up() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    original_sync = core_hooks.sync_playwright
    original_wait_initial = action_hooks.wait_for_initial_self_state
    core_hooks.sync_playwright = recorded.sync_playwright_factory

    def _wait_initial(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        _ = (page_arg, provider, started_ms, timeout_ms)
        return 1200, make_self_state(
            tank_id=1,
            x=158,
            y=132,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )

    wait_initial_name = "wait_for_initial_self_state"
    setattr(action_hooks, wait_initial_name, _wait_initial)
    try:
        with pytest.raises(TeleportProbeError, match="requires at least one target"):
            probe.execute(
                explicit_targets=[],
                box_step_x=8,
                box_step_y=8,
                max_targets=None,
                teleport_strategy="sync_before_teleport",
                initial_sync_timeout_ms=10000,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                settle_delay_ms=500,
            )
    finally:
        core_hooks.sync_playwright = original_sync
        setattr(action_hooks, wait_initial_name, original_wait_initial)
    assert recorded.browser_type.launches == [False]


def test_execute_builds_default_targets_and_collects_attempts() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    probe.result_attempts = [_make_attempt("landed_exact") for _ in range(10)]
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    original_sync = core_hooks.sync_playwright
    original_wait_initial = action_hooks.wait_for_initial_self_state
    core_hooks.sync_playwright = recorded.sync_playwright_factory

    def _wait_initial(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        _ = (page_arg, provider, started_ms, timeout_ms)
        return 1200, make_self_state(
            tank_id=1,
            x=158,
            y=132,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )

    wait_initial_name = "wait_for_initial_self_state"
    setattr(action_hooks, wait_initial_name, _wait_initial)
    try:
        session = probe.execute(
            explicit_targets=None,
            box_step_x=8,
            box_step_y=8,
            max_targets=3,
            teleport_strategy="sync_before_teleport",
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=500,
        )
    finally:
        core_hooks.sync_playwright = original_sync
        setattr(action_hooks, wait_initial_name, original_wait_initial)
    assert len(session["targets"]) == 3
    assert len(session["attempts"]) == 3
    assert len(probe.probed_targets) == 3
    assert recorded.browser_type.launches == [False]
    assert session["teleport_strategy"] == "sync_before_teleport"
    assert session["max_targets"] == 3
    assert session["initial_sync_timeout_ms"] == 10000
    assert session["startup_timing"]["initial_world_timestamp_ms"] == 1200
    assert session["startup_timing"]["first_attempt_started_ms"] == 1000


class _FakeTeleportProbe:
    def __init__(
        self,
        target_url: str,
        *,
        headless: bool,
        prefer_account: bool,
        cdp_service: CDPService | None = None,
        command_service: CommandService | None = None,
    ) -> None:
        _ = (cdp_service, command_service)
        self.target_url = target_url
        self.headless = headless
        self.prefer_account = prefer_account

    def execute(
        self,
        *,
        explicit_targets: list[TeleportTargetDict] | None,
        box_step_x: int,
        box_step_y: int,
        max_targets: int | None,
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
        initial_sync_timeout_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
    ) -> TeleportProbeSessionDict:
        targets = (
            explicit_targets
            if explicit_targets is not None
            else build_box_targets(100, 100, box_step_x, box_step_y)
        )
        limited_targets = targets if max_targets is None else targets[:max_targets]
        return TeleportProbeSessionDict(
            session_id="fake-session",
            start_timestamp_ms=10,
            end_timestamp_ms=20,
            base_url=self.target_url,
            spawn_x=100,
            spawn_y=100,
            teleport_strategy=teleport_strategy,
            max_targets=max_targets,
            capture_session_path="",
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            startup_timing={
                "game_ready_timestamp_ms": 100,
                "intel_ready_timestamp_ms": 150,
                "initial_sync_started_ms": 200,
                "initial_world_timestamp_ms": 400,
                "command_ready_timestamp_ms": 450,
                "first_attempt_started_ms": 500,
                "game_ready_to_intel_ready_ms": 50,
                "intel_ready_to_initial_world_ms": 250,
                "initial_world_to_command_ready_ms": 50,
                "command_ready_to_first_attempt_ms": 50,
            },
            map_sync_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            settle_delay_ms=settle_delay_ms,
            targets=limited_targets,
            attempts=[],
        )

    @property
    def messages(self) -> list[CapturedMessage]:
        return []

    @property
    def magic(self) -> str | None:
        return None

    @property
    def session_id(self) -> str:
        return "fake-session"


def test_run_teleport_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    from tankpit_bot.action_lab import teleport as teleport_module

    original_probe_class = teleport_module.TeleportProbe
    probe_class_name = "TeleportProbe"
    setattr(teleport_module, probe_class_name, _FakeTeleportProbe)
    try:
        session = run_teleport_probe(
            "https://tankpit.com/play",
            "teleport_probe.json",
            explicit_targets=[TeleportTargetDict(label="target_0", x=150, y=171)],
        )
    finally:
        setattr(teleport_module, probe_class_name, original_probe_class)

    written = fake_fs.read_text(Path("teleport_probe.json"))
    decoded = decode_teleport_probe_session(narrow_json_to_dict(load_json_str(written)))
    capture_written = fake_fs.read_text(Path("teleport_probe.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))
    assert session == decoded
    assert session["capture_session_path"] == "teleport_probe.capture_session.json"
    assert session["targets"] == [TeleportTargetDict(label="target_0", x=150, y=171)]
    assert capture_decoded["session_id"] == "fake-session"


# =========================================================================
# TeleportProbe composition tests (Phase D coverage)
# =========================================================================


def test_send_bytes_delegates_to_command_service() -> None:
    """_send_bytes syncs CDP and delegates to CommandService."""

    probe = TeleportProbe("https://tankpit.com/play", headless=True)
    sent: list[str] = []

    def _fake_send(cdp: CDPSessionProtocol, data: bytes, label: str) -> str:
        sent.append(label)
        return ""

    probe._commands._send_ws_bytes = _fake_send
    probe._cdp = StubSnapshotCDPSession()
    probe._commands.xor_table = b"\x00" * 256
    assert probe._send_bytes(b"\x04\x00!test", "test_cmd") is True
    assert sent == ["test_cmd"]


def test_send_bytes_returns_false_when_cdp_none() -> None:
    """_send_bytes returns False when no CDP session is attached."""
    probe = TeleportProbe("https://tankpit.com/play", headless=True)
    assert probe._send_bytes(b"\x04\x00!test", "test_cmd") is False


def test_teleport_to_returns_false_when_cdp_none() -> None:
    """teleport_to returns False early when no CDP session."""
    probe = TeleportProbe("https://tankpit.com/play", headless=True)
    assert probe.teleport_to(100, 200) is False


def test_on_message_captured_buffers_received() -> None:
    """_on_message_captured appends received payloads to buffer."""
    from tankpit_bot.types import CapturedMessage

    probe = TeleportProbe("https://tankpit.com/play", headless=True)
    msg = CapturedMessage(
        direction="received",
        payload="dGVzdA==",
        timestamp_ms=1000,
        ws_url="wss://tankpit.com/ws/",
    )
    probe._on_message_captured(msg)
    assert probe._cdp_message_buffer == ["dGVzdA=="]


def test_on_message_captured_ignores_sent() -> None:
    """_on_message_captured does not buffer sent messages."""
    from tankpit_bot.types import CapturedMessage

    probe = TeleportProbe("https://tankpit.com/play", headless=True)
    msg = CapturedMessage(
        direction="sent",
        payload="dGVzdA==",
        timestamp_ms=1000,
        ws_url="wss://tankpit.com/ws/",
    )
    probe._on_message_captured(msg)
    assert probe._cdp_message_buffer == []


def test_on_magic_captured_builds_xor_table(fake_fs: FakeFileSystem) -> None:
    """_on_magic_captured builds XOR table on CommandService."""
    probe = TeleportProbe("https://tankpit.com/play", headless=True)
    assert probe._commands.xor_table is None
    probe._on_magic_captured("TESTMAGIC" * 5)
    xor_table = probe._commands.xor_table
    if xor_table is None:
        raise AssertionError("xor_table should be set after magic capture")
    assert len(xor_table) == 1000
