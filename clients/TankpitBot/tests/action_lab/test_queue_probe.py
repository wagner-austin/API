"""Tests for queue-probe execution, validation, and the summary.

``test_queue_probe.py`` was 896 lines; waits and experiments are now
siblings.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.action_lab._queue_probe_harness import (
    _CAPTURE_PATH,
    _advance_startup_state_stub,
    _ExecuteHarness,
    _FakeQueueProbeForRunner,
    _make_experiment_result,
    _SteppingClock,
    _wait_for_initial_self_state_spawn,
)
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.conftest import FakeFileSystem

import tankpit_bot.action_lab.queue_probe as queue_probe_module
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import queue_experiments as queue_experiments_module
from tankpit_bot.action_lab.queue_experiments import QueueExperimentProbeProtocol
from tankpit_bot.action_lab.queue_probe import (
    QueueProbe,
    format_queue_probe_summary,
    run_queue_probe,
)
from tankpit_bot.action_lab.queue_probe_types import (
    QueueCommandTimingDict,
    QueueExperimentKind,
    QueueExperimentResultDict,
    QueueProbeSessionDict,
)
from tankpit_bot.action_lab.types import TeleportStartupTimingDict


class TestFormatQueueProbeSummary:
    def test_formats_session(self) -> None:
        timing = TeleportStartupTimingDict(
            game_ready_timestamp_ms=1000,
            intel_ready_timestamp_ms=2000,
            initial_sync_started_ms=3000,
            initial_world_timestamp_ms=4000,
            command_ready_timestamp_ms=5000,
            first_attempt_started_ms=6000,
            game_ready_to_intel_ready_ms=1000,
            intel_ready_to_initial_world_ms=2000,
            initial_world_to_command_ready_ms=1000,
            command_ready_to_first_attempt_ms=1000,
        )
        session = QueueProbeSessionDict(
            session_id="test-001",
            start_timestamp_ms=1000,
            end_timestamp_ms=5000,
            base_url="https://tankpit.com/play",
            spawn_x=128,
            spawn_y=128,
            capture_session_path="",
            initial_sync_timeout_ms=10000,
            experiment_timeout_ms=5000,
            startup_timing=timing,
            experiments=[
                QueueExperimentResultDict(
                    kind="shoot_then_pickup",
                    status="both_processed",
                    primary=QueueCommandTimingDict(
                        label="shoot", sent_ms=100, ack_ms=200, elapsed_ms=100
                    ),
                    secondary=QueueCommandTimingDict(
                        label="pickup_fuel", sent_ms=105, ack_ms=210, elapsed_ms=105
                    ),
                    inter_send_delay_ms=5,
                    total_elapsed_ms=115,
                    message_start_index=0,
                    message_end_index=10,
                ),
            ],
        )
        summary = format_queue_probe_summary(session)
        assert "test-001" in summary
        assert "shoot_then_pickup" in summary
        assert "both_processed" in summary
        assert "(128, 128)" in summary

    def test_formats_empty_experiments(self) -> None:
        timing = TeleportStartupTimingDict(
            game_ready_timestamp_ms=1000,
            intel_ready_timestamp_ms=2000,
            initial_sync_started_ms=3000,
            initial_world_timestamp_ms=4000,
            command_ready_timestamp_ms=5000,
            first_attempt_started_ms=None,
            game_ready_to_intel_ready_ms=1000,
            intel_ready_to_initial_world_ms=2000,
            initial_world_to_command_ready_ms=1000,
            command_ready_to_first_attempt_ms=None,
        )
        session = QueueProbeSessionDict(
            session_id="test-002",
            start_timestamp_ms=1000,
            end_timestamp_ms=2000,
            base_url="https://tankpit.com/play",
            spawn_x=64,
            spawn_y=64,
            capture_session_path="",
            initial_sync_timeout_ms=10000,
            experiment_timeout_ms=5000,
            startup_timing=timing,
            experiments=[],
        )
        summary = format_queue_probe_summary(session)
        assert "Experiments: 0" in summary


class TestExecuteProbeIntegration:
    def test_execute_probe_runs_experiments(self) -> None:
        harness = _ExecuteHarness()
        clock = _SteppingClock(1000, 100)
        action_hooks.get_current_time_ms = clock
        recorded = RecordedChromiumSession.from_capture_path(harness, _CAPTURE_PATH)
        core_hooks.sync_playwright = recorded.sync_playwright_factory
        action_hooks.wait_for_initial_self_state = _wait_for_initial_self_state_spawn
        action_hooks.advance_startup_state = _advance_startup_state_stub

        experiment_results = [_make_experiment_result("shoot_then_pickup")]

        def _fake_run_single(
            probe: QueueExperimentProbeProtocol,
            kind: QueueExperimentKind,
            *,
            timeout_ms: int,
        ) -> QueueExperimentResultDict:
            _ = (probe, timeout_ms)
            return experiment_results.pop(0)

        queue_experiments_module.run_single_experiment = _fake_run_single

        session = harness.execute_probe(
            initial_sync_timeout_ms=5000,
            experiment_timeout_ms=3000,
            experiment_kinds=["shoot_then_pickup"],
        )
        assert session["spawn_x"] == 101
        assert session["spawn_y"] == 102
        assert len(session["experiments"]) == 1
        assert session["experiments"][0]["kind"] == "shoot_then_pickup"
        assert session["experiment_timeout_ms"] == 3000
        assert harness._page is None
        assert harness._cdp is None

    def test_execute_probe_raises_when_playwright_missing(self) -> None:
        from tankpit_bot.browser.types import PlaywrightNotInstalledError

        harness = _ExecuteHarness()
        core_hooks.sync_playwright = None
        with pytest.raises(PlaywrightNotInstalledError):
            harness.execute_probe(
                initial_sync_timeout_ms=5000,
                experiment_timeout_ms=3000,
                experiment_kinds=["shoot_then_pickup"],
            )


class TestQueueProbeValidation:
    def test_negative_timeout_raises(self) -> None:
        probe = QueueProbe.__new__(QueueProbe)
        with pytest.raises(ValueError, match="experiment_timeout_ms must be positive"):
            probe.execute_probe(
                initial_sync_timeout_ms=10000,
                experiment_timeout_ms=0,
                experiment_kinds=["shoot_then_pickup"],
            )

    def test_empty_kinds_raises(self) -> None:
        probe = QueueProbe.__new__(QueueProbe)
        with pytest.raises(ValueError, match="experiment_kinds must not be empty"):
            probe.execute_probe(
                initial_sync_timeout_ms=10000,
                experiment_timeout_ms=5000,
                experiment_kinds=[],
            )


def test_create_queue_probe_factory_returns_queue_probe() -> None:
    """The factory returns a real QueueProbe instance."""
    probe = queue_probe_module._create_queue_probe(
        "https://tankpit.com/play",
        headless=True,
        prefer_account=False,
    )
    assert probe._target_url == "https://tankpit.com/play"
    assert probe._headless is True


def test_run_queue_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    """run_queue_probe persists the session and capture JSON."""

    from platform_core.json_utils import load_json_str, narrow_json_to_dict

    from tankpit_bot.action_lab.queue_probe_types import decode_queue_probe_session

    original_factory = queue_probe_module._create_queue_probe
    queue_probe_module._create_queue_probe = (
        lambda target_url, *, headless, prefer_account: _FakeQueueProbeForRunner(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
        )
    )
    try:
        session = run_queue_probe(
            "https://tankpit.com/play",
            "queue_probe.json",
        )
    finally:
        queue_probe_module._create_queue_probe = original_factory

    written = fake_fs.read_text(Path("queue_probe.json"))
    decoded = decode_queue_probe_session(narrow_json_to_dict(load_json_str(written)))
    assert session == decoded
    assert session["session_id"] == "fake-queue-session"
