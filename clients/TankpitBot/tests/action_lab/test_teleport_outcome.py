"""Tests for ``wait_for_teleport_outcome``.

Exact and offset landings, the post-landing snapshot, and both
missing-self-state failure modes.
"""

from __future__ import annotations

import base64
from typing import Literal

import pytest
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)
from tests.action_lab._teleport_harness import (
    _AckSequence,
    _make_page_snapshot,
    _make_world,
    _SequencedProvider,
)

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.teleport_helpers import (
    TeleportProbeError,
    _wait_for_teleport_outcome,
)
from tankpit_bot.action_lab.types import (
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.state import (
    WorldStateDict,
)
from tankpit_bot.types import (
    CapturedMessage,
)


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
        scanned_tiles=world["scanned_tiles"],
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
        scanned_tiles=world["scanned_tiles"],
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
