"""Tests for ``execute_probe`` and the movement-probe entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
)
from tests.action_lab._movement_probe_harness import (
    _FUEL_CAPTURE_PATH,
    _advance_startup_state_stub,
    _ExecuteSuccessHarness,
    _FakeMovementProbe,
    _make_attempt,
    _SteppingClock,
    _TerrainHarness,
    _TerrainMapStub,
    _wait_for_initial_self_state_101_102,
    _wait_for_initial_self_state_103_104,
)
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    TerrainMapProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import movement_probe as movement_probe_module
from tankpit_bot.action_lab.movement_probe import (
    MovementProbe,
    MovementProbeError,
    run_movement_probe,
)
from tankpit_bot.action_lab.movement_probe_types import (
    decode_movement_probe_session,
)
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.state import (
    make_self_state,
)
from tankpit_bot.types import (
    decode_capture_session,
)


def test_execute_probe_rejects_non_positive_max_targets() -> None:
    probe = MovementProbe("https://tankpit.com/play", headless=False, prefer_account=True)
    with pytest.raises(ValueError, match="max_targets must be positive"):
        probe.execute_probe(
            explicit_targets=None,
            max_targets=0,
            initial_sync_timeout_ms=10000,
            move_timeout_ms=5000,
            queue_map_open_during_move=False,
            map_open_delay_ms=0,
            settle_delay_ms=500,
        )


def test_execute_probe_raises_when_playwright_is_missing() -> None:
    probe = MovementProbe("https://tankpit.com/play", headless=False, prefer_account=True)
    original_playwright = core_hooks.sync_playwright
    core_hooks.sync_playwright = None
    try:
        with pytest.raises(PlaywrightNotInstalledError):
            probe.execute_probe(
                explicit_targets=[TeleportTargetDict(label="move_1", x=120, y=121)],
                max_targets=1,
                initial_sync_timeout_ms=10000,
                move_timeout_ms=5000,
                queue_map_open_during_move=False,
                map_open_delay_ms=0,
                settle_delay_ms=500,
            )
    finally:
        core_hooks.sync_playwright = original_playwright


def test_build_default_targets_raises_when_terrain_is_missing() -> None:
    probe = _TerrainHarness(make_self_state(1, 100, 104, 2, 1, 900, 5))
    probe.world.terrain_map = None
    with pytest.raises(MovementProbeError, match="terrain map is unavailable"):
        probe._build_default_targets(max_targets=1)


def test_build_default_targets_uses_spawn_position_and_limits() -> None:
    probe = _TerrainHarness(make_self_state(1, 100, 104, 2, 1, 900, 5))
    expected = [TeleportTargetDict(label="move_1", x=104, y=108)]
    captured: dict[str, int] = {}
    probe.world.terrain_map = _TerrainMapStub()

    def _fake_build_targets(
        origin_x: int,
        origin_y: int,
        terrain: TerrainMapProtocol,
        *,
        max_targets: int,
    ) -> list[TeleportTargetDict]:
        _ = terrain
        captured["x"] = origin_x
        captured["y"] = origin_y
        captured["max_targets"] = max_targets
        return expected

    movement_probe_module._build_probe_targets = _fake_build_targets
    assert probe._build_default_targets(max_targets=2) == expected
    assert captured == {
        "x": 100,
        "y": 104,
        "max_targets": 2,
    }


def test_execute_probe_rejects_negative_map_open_delay() -> None:
    probe = MovementProbe("https://tankpit.com/play", headless=False, prefer_account=True)
    with pytest.raises(ValueError, match="map_open_delay_ms must be non-negative"):
        probe.execute_probe(
            explicit_targets=None,
            max_targets=1,
            initial_sync_timeout_ms=10000,
            move_timeout_ms=5000,
            queue_map_open_during_move=False,
            map_open_delay_ms=-1,
            settle_delay_ms=500,
        )


def test_execute_probe_rejects_negative_settle_delay() -> None:
    probe = MovementProbe("https://tankpit.com/play", headless=False, prefer_account=True)
    with pytest.raises(ValueError, match="settle_delay_ms must be non-negative"):
        probe.execute_probe(
            explicit_targets=None,
            max_targets=1,
            initial_sync_timeout_ms=10000,
            move_timeout_ms=5000,
            queue_map_open_during_move=False,
            map_open_delay_ms=0,
            settle_delay_ms=-1,
        )


def test_execute_probe_runs_successfully_with_explicit_targets() -> None:
    attempts = [_make_attempt("arrived_exact")]
    harness = _ExecuteSuccessHarness(
        attempts=attempts,
        default_targets=[],
    )
    clock = _SteppingClock(1000, 100)
    action_hooks.get_current_time_ms = clock
    recorded = RecordedChromiumSession.from_capture_path(harness, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    action_hooks.wait_for_initial_self_state = _wait_for_initial_self_state_101_102
    action_hooks.advance_startup_state = _advance_startup_state_stub

    explicit_targets = [TeleportTargetDict(label="move_1", x=120, y=121)]
    session = harness.execute_probe(
        explicit_targets=explicit_targets,
        max_targets=1,
        initial_sync_timeout_ms=5000,
        move_timeout_ms=3000,
        queue_map_open_during_move=True,
        map_open_delay_ms=150,
        settle_delay_ms=250,
    )
    assert session["targets"] == explicit_targets
    assert session["attempts"] == attempts
    assert session["spawn_x"] == 101
    assert session["spawn_y"] == 102
    assert harness.probed_targets == explicit_targets
    assert harness._page is None
    assert harness._cdp is None


def test_execute_probe_uses_default_targets_when_explicit_targets_are_absent() -> None:
    default_targets = [TeleportTargetDict(label="move_1", x=120, y=121)]
    attempts = [_make_attempt("move_timeout")]
    harness = _ExecuteSuccessHarness(
        attempts=attempts,
        default_targets=default_targets,
    )
    clock = _SteppingClock(1000, 100)
    action_hooks.get_current_time_ms = clock
    recorded = RecordedChromiumSession.from_capture_path(harness, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    action_hooks.wait_for_initial_self_state = _wait_for_initial_self_state_103_104
    action_hooks.advance_startup_state = _advance_startup_state_stub

    session = harness.execute_probe(
        explicit_targets=None,
        max_targets=1,
        initial_sync_timeout_ms=5000,
        move_timeout_ms=3000,
        queue_map_open_during_move=False,
        map_open_delay_ms=0,
        settle_delay_ms=0,
    )
    assert session["targets"] == default_targets
    assert session["attempts"] == attempts
    assert harness.probed_targets == default_targets


def test_execute_probe_raises_when_target_builder_returns_empty_list() -> None:
    harness = _ExecuteSuccessHarness(
        attempts=[],
        default_targets=[],
    )
    clock = _SteppingClock(1000, 100)
    action_hooks.get_current_time_ms = clock
    recorded = RecordedChromiumSession.from_capture_path(harness, _FUEL_CAPTURE_PATH)
    core_hooks.sync_playwright = recorded.sync_playwright_factory
    action_hooks.wait_for_initial_self_state = _wait_for_initial_self_state_103_104
    action_hooks.advance_startup_state = _advance_startup_state_stub

    with pytest.raises(MovementProbeError, match="requires at least one target"):
        harness.execute_probe(
            explicit_targets=None,
            max_targets=1,
            initial_sync_timeout_ms=5000,
            move_timeout_ms=3000,
            queue_map_open_during_move=False,
            map_open_delay_ms=0,
            settle_delay_ms=0,
        )


def test_run_movement_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    original_factory = movement_probe_module._create_movement_probe
    movement_probe_module._create_movement_probe = lambda target_url, *, headless, prefer_account: (
        _FakeMovementProbe(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
        )
    )
    try:
        session = run_movement_probe(
            "https://tankpit.com/play",
            "movement_probe.json",
            explicit_targets=[TeleportTargetDict(label="move_1", x=120, y=121)],
            queue_map_open_during_move=True,
            map_open_delay_ms=150,
        )
    finally:
        movement_probe_module._create_movement_probe = original_factory

    written = fake_fs.read_text(Path("movement_probe.json"))
    decoded = decode_movement_probe_session(narrow_json_to_dict(load_json_str(written)))
    capture_written = fake_fs.read_text(Path("movement_probe.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))
    assert session == decoded
    assert session["capture_session_path"] == "movement_probe.capture_session.json"
    assert session["targets"] == [TeleportTargetDict(label="move_1", x=120, y=121)]
    assert capture_decoded["session_id"] == "fake-session"
