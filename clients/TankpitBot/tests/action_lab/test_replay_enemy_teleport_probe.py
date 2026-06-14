"""End-to-end replay test: drive the real EnemyTeleportProbe through a capture.

The harness exercises the production
:class:`tankpit_bot.action_lab.enemy_teleport.EnemyTeleportProbe`
end-to-end. Real decoders ingest the recorded bytes; the real
``_probe_single_enemy_attempt`` body sequences through enemy
acquisition (map_open or nearest_enemy) and teleport phases; real
``capture_page_client_snapshot`` calls hit the
:class:`WorldStateDerivedCDP` substitute; ``_send_bytes`` is captured.

No fakes of probe behavior. The only substitutes are at the OS-level
boundary -- the same boundary the live tank-pit page is on top of.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.action_lab._replay_core import ReplayResult
from tests.action_lab._replay_enemy_teleport import (
    EnemyTeleportReplayResult,
    replay_enemy_teleport_attempt,
)

from tankpit_bot.action_lab.enemy_teleport_types import EnemyTeleportAttemptResultDict
from tankpit_bot.action_lab.teleport import TeleportProbeError

REPO_ROOT = Path(__file__).resolve().parents[2]
ENEMY_CAPTURE = REPO_ROOT / "fuel_probe.capture_session.json"


@pytest.fixture()
def enemy_attempt() -> ReplayResult[EnemyTeleportAttemptResultDict]:
    """Replay one enemy-teleport attempt against the committed capture."""
    return replay_enemy_teleport_attempt(
        ENEMY_CAPTURE,
        acquisition_strategy="map_open",
        acquisition_timeout_ms=600_000,
        teleport_timeout_ms=600_000,
    )


def test_replay_consumes_recorded_frames(
    enemy_attempt: EnemyTeleportReplayResult,
) -> None:
    """Recorded frames feed the real probe through wait_for_timeout polls."""
    assert enemy_attempt.frames_fed > 0


def test_replay_dispatches_real_map_open_command(
    enemy_attempt: EnemyTeleportReplayResult,
) -> None:
    """The real probe issues a ``map_open`` command through real ``open_map``."""
    assert "map_open" in enemy_attempt.dispatched_commands


def test_replay_attempt_records_acquisition_strategy(
    enemy_attempt: EnemyTeleportReplayResult,
) -> None:
    """The real attempt result preserves the supplied acquisition strategy."""
    assert enemy_attempt.attempt["acquisition_strategy"] == "map_open"


def test_replay_attempt_reports_a_resolved_status(
    enemy_attempt: EnemyTeleportReplayResult,
) -> None:
    """The production attempt body reaches one of its declared terminal statuses."""
    assert enemy_attempt.attempt["status"] in (
        "landed_adjacent",
        "landed_not_adjacent",
        "no_enemy",
        "no_landing_tile",
        "acquisition_timeout",
        "teleport_timeout",
    )


def test_replay_attempt_records_fuel_before(
    enemy_attempt: EnemyTeleportReplayResult,
) -> None:
    """The attempt records the real fuel snapshot from decoded frames."""
    fuel_before = enemy_attempt.attempt["fuel_before"]
    assert fuel_before >= 0


def test_replay_raises_when_cdp_session_unavailable() -> None:
    """Real probe attempt fails fast when CDP is not attached."""
    with pytest.raises(TeleportProbeError, match="cdp session is unavailable"):
        replay_enemy_teleport_attempt(
            ENEMY_CAPTURE,
            acquisition_strategy="map_open",
            acquisition_timeout_ms=600_000,
            teleport_timeout_ms=600_000,
            omit_cdp=True,
        )


def test_replay_raises_on_acquisition_dispatch_failure() -> None:
    """Real probe raises ``enemy acquisition command dispatch failed`` end-to-end.

    Simulates a live WebSocket-send failure during the acquisition
    phase through the real probe attempt body.
    """
    expected_match = "enemy acquisition command dispatch failed"
    with pytest.raises(TeleportProbeError, match=expected_match):
        replay_enemy_teleport_attempt(
            ENEMY_CAPTURE,
            acquisition_strategy="nearest_enemy",
            acquisition_timeout_ms=600_000,
            teleport_timeout_ms=600_000,
            fail_command=lambda cmd: cmd == "nearest_enemy",
        )
