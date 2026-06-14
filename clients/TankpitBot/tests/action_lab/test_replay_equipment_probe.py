"""End-to-end replay test: drive the real EquipmentProbe through a capture.

The harness exercises the production
:class:`tankpit_bot.action_lab.equipment_probe.EquipmentProbe`
end-to-end. Real decoders ingest the recorded bytes; the real
``_probe_single_equipment_target`` body sequences through
map_open / teleport / radar / (optional reposition) / move / pickup;
real ``capture_page_client_snapshot`` calls hit the
:class:`WorldStateDerivedCDP` substitute; ``_send_bytes`` is captured
(rather than dispatched to a non-existent live server).

No fakes of probe behavior. The only substitutes are at the OS-level
boundary -- the same boundary the live tank-pit page is on top of.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from tests.action_lab._replay_equipment import (
    EquipmentReplayResult,
    replay_equipment_attempt,
)

from tankpit_bot.action_lab.equipment_probe import EquipmentProbeError
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.sniffer.world_state import reset_world_state
from tankpit_bot.sniffer.xor import reset_xor_state

REPO_ROOT = Path(__file__).resolve().parents[2]
EQUIPMENT_CAPTURE = REPO_ROOT / "fuel_probe.capture_session.json"
"""The committed capture is named ``fuel_probe.*`` but the recording
was made during an HFSM ``RECOVER_EQUIPMENT`` session -- the bot does
equipment scans + pickups throughout, so it is the right capture for
the equipment-probe replay harness."""


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
def equipment_attempt() -> EquipmentReplayResult:
    """Replay one equipment attempt against the committed capture.

    Targets a tile the recording's session actually pivots toward;
    the production attempt body then sequences through map_open ->
    teleport -> radar -> (optional reposition) -> pickup as the real
    decoder updates world state from the recording.
    """
    target = TeleportTargetDict(label="recorded_pivot", x=131, y=110)
    return replay_equipment_attempt(
        EQUIPMENT_CAPTURE,
        target,
        map_sync_timeout_ms=600_000,
        teleport_timeout_ms=600_000,
        radar_timeout_ms=600_000,
        pickup_timeout_ms=600_000,
    )


def test_replay_consumes_recorded_frames(
    equipment_attempt: EquipmentReplayResult,
) -> None:
    """Recorded frames feed the real probe through wait_for_timeout polls."""
    assert equipment_attempt.frames_fed > 0


def test_replay_dispatches_real_map_open_command(
    equipment_attempt: EquipmentReplayResult,
) -> None:
    """The real probe issues a ``map_open`` command through real ``open_map``."""
    assert "map_open" in equipment_attempt.dispatched_commands


def test_replay_attempt_records_target(
    equipment_attempt: EquipmentReplayResult,
) -> None:
    """The real attempt result preserves the supplied target verbatim."""
    target = equipment_attempt.attempt["target"]
    assert target["label"] == "recorded_pivot"
    assert target["x"] == 131
    assert target["y"] == 110


def test_replay_attempt_reports_a_resolved_status(
    equipment_attempt: EquipmentReplayResult,
) -> None:
    """The production attempt body reaches one of its declared terminal statuses."""
    assert equipment_attempt.attempt["status"] in (
        "picked_up_equipment",
        "no_equipment_visible",
        "radar_timeout",
        "map_sync_timeout",
        "reposition_map_sync_timeout",
        "teleport_timeout",
        "reposition_teleport_timeout",
        "pickup_timeout",
    )


def test_replay_attempt_records_inventory_count_before(
    equipment_attempt: EquipmentReplayResult,
) -> None:
    """The attempt records the real inventory snapshot from decoded frames."""
    inventory_count_before = equipment_attempt.attempt["inventory_count_before"]
    assert inventory_count_before >= 0


def test_replay_raises_when_cdp_session_unavailable() -> None:
    """Real probe attempt fails fast when CDP is not attached.

    Without a CDP session, the page-client snapshot cannot be captured
    inside the production teleport phase. The attempt body raises the
    structured ``cdp session is unavailable`` error.
    """
    target = TeleportTargetDict(label="recorded_pivot", x=131, y=110)
    with pytest.raises(EquipmentProbeError, match="cdp session is unavailable"):
        replay_equipment_attempt(
            EQUIPMENT_CAPTURE,
            target,
            map_sync_timeout_ms=600_000,
            teleport_timeout_ms=600_000,
            radar_timeout_ms=600_000,
            pickup_timeout_ms=600_000,
            omit_cdp=True,
        )
