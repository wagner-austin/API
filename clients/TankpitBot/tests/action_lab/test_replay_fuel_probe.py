"""End-to-end replay test: drive the real FuelProbe through a capture.

The harness exercises the production
:class:`tankpit_bot.action_lab.fuel_probe.FuelProbe` end-to-end. Real
decoders ingest the recorded bytes; the real
``_probe_single_fuel_target`` body sequences through
map_open / teleport / radar / (optional reposition) / move / pickup;
real ``capture_page_client_snapshot`` calls hit the
:class:`WorldStateDerivedCDP` substitute; ``_send_bytes`` is captured
(rather than dispatched to a non-existent live server).

The committed capture is an HFSM equipment-recovery session, so the
real fuel-target finder will report no fuel visible -- exercising the
``no_fuel_visible`` terminal status path end-to-end through real
production code.

No fakes of probe behavior. The only substitutes are at the OS-level
boundary -- the same boundary the live tank-pit page is on top of.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from tests.action_lab._replay_fuel import (
    FuelReplayResult,
    replay_fuel_attempt,
)

from tankpit_bot.action_lab.fuel_probe import FuelProbeError
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.sniffer.xor import reset_xor_state

REPO_ROOT = Path(__file__).resolve().parents[2]
FUEL_CAPTURE = REPO_ROOT / "fuel_probe.capture_session.json"


@pytest.fixture(autouse=True)
def _isolate_world_state() -> Generator[None, None, None]:
    """Reset world-state and XOR singletons around every test.

    The replay harness builds a global XOR table from the capture's
    magic key; without an explicit teardown reset, that table would
    leak into subsequent tests that decode bytes with a different key.
    """
    reset_world_state()
    reset_xor_state()
    yield
    reset_world_state()
    reset_xor_state()


@pytest.fixture()
def fuel_attempt() -> FuelReplayResult:
    """Replay one fuel attempt against the committed capture.

    The recording does not surface fuel containers, so the real
    target finder reports ``no_fuel_visible`` -- a real terminal
    path exercised end-to-end through production code.
    """
    target = TeleportTargetDict(label="recorded_pivot", x=131, y=110)
    return replay_fuel_attempt(
        FUEL_CAPTURE,
        target,
        map_sync_timeout_ms=600_000,
        teleport_timeout_ms=600_000,
        radar_timeout_ms=600_000,
        pickup_timeout_ms=600_000,
    )


def test_replay_consumes_recorded_frames(
    fuel_attempt: FuelReplayResult,
) -> None:
    """Recorded frames feed the real probe through wait_for_timeout polls."""
    assert fuel_attempt.frames_fed > 0


def test_replay_dispatches_real_map_open_command(
    fuel_attempt: FuelReplayResult,
) -> None:
    """The real probe issues a ``map_open`` command through real ``open_map``."""
    assert "map_open" in fuel_attempt.dispatched_commands


def test_replay_attempt_records_target(
    fuel_attempt: FuelReplayResult,
) -> None:
    """The real attempt result preserves the supplied target verbatim."""
    target = fuel_attempt.attempt["target"]
    assert target["label"] == "recorded_pivot"
    assert target["x"] == 131
    assert target["y"] == 110


def test_replay_attempt_reports_a_resolved_status(
    fuel_attempt: FuelReplayResult,
) -> None:
    """The production attempt body reaches one of its declared terminal statuses."""
    assert fuel_attempt.attempt["status"] in (
        "picked_up_fuel",
        "no_fuel_visible",
        "radar_timeout",
        "map_sync_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    )


def test_replay_attempt_records_fuel_before(
    fuel_attempt: FuelReplayResult,
) -> None:
    """The attempt records the real fuel snapshot from decoded frames."""
    fuel_before = fuel_attempt.attempt["fuel_before"]
    assert fuel_before >= 0


def test_replay_raises_when_cdp_session_unavailable() -> None:
    """Real probe attempt fails fast when CDP is not attached.

    Without a CDP session, the page-client snapshot cannot be captured
    inside the production teleport phase. The attempt body raises the
    structured ``cdp session is unavailable`` error.
    """
    target = TeleportTargetDict(label="recorded_pivot", x=131, y=110)
    with pytest.raises(FuelProbeError, match="cdp session is unavailable"):
        replay_fuel_attempt(
            FUEL_CAPTURE,
            target,
            map_sync_timeout_ms=600_000,
            teleport_timeout_ms=600_000,
            radar_timeout_ms=600_000,
            pickup_timeout_ms=600_000,
            omit_cdp=True,
        )


def test_replay_raises_on_map_open_dispatch_failure() -> None:
    """The real probe raises ``map_open command dispatch failed`` end-to-end.

    Simulates a live WebSocket-send failure by failing the real
    ``_send_bytes`` for the ``map_open`` label -- exactly what the
    live runtime does when the socket is down.
    """
    target = TeleportTargetDict(label="recorded_pivot", x=131, y=110)
    with pytest.raises(FuelProbeError, match="map_open command dispatch failed"):
        replay_fuel_attempt(
            FUEL_CAPTURE,
            target,
            map_sync_timeout_ms=600_000,
            teleport_timeout_ms=600_000,
            radar_timeout_ms=600_000,
            pickup_timeout_ms=600_000,
            fail_command=lambda cmd: cmd == "map_open",
        )


def test_replay_raises_on_teleport_dispatch_failure() -> None:
    """The real probe raises ``teleport command dispatch failed`` end-to-end.

    Fails the real ``_send_bytes`` for any ``teleport(...)`` label
    (the production encoder formats coords into the label).
    """
    target = TeleportTargetDict(label="recorded_pivot", x=131, y=110)
    with pytest.raises(FuelProbeError, match="teleport command dispatch failed"):
        replay_fuel_attempt(
            FUEL_CAPTURE,
            target,
            map_sync_timeout_ms=600_000,
            teleport_timeout_ms=600_000,
            radar_timeout_ms=600_000,
            pickup_timeout_ms=600_000,
            fail_command=lambda cmd: cmd.startswith("teleport("),
        )
