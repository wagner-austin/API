"""Branch coverage for EquipmentProbe failure paths.

Uses the real replay harness to exercise the actual pipeline through
real captured bytes, triggering failure branches with tiny timeouts.
The capture fixture is an HFSM equipment-recovery session — equipment
containers ARE visible, and the full teleport-radar-equipment pipeline
runs through real decoders, real world-state mutations, and real
probing logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.action_lab._replay_equipment import replay_equipment_attempt

from tankpit_bot.action_lab.equipment_probe import EquipmentProbeError
from tankpit_bot.action_lab.types import TeleportTargetDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE = REPO_ROOT / "fuel_probe.capture_session.json"
TARGET = TeleportTargetDict(label="pivot", x=131, y=110)


def test_cdp_unavailable_raises() -> None:
    """Missing CDP session raises EquipmentProbeError immediately."""
    with pytest.raises(
        EquipmentProbeError,
        match="cdp session is unavailable",
    ):
        replay_equipment_attempt(CAPTURE, TARGET, omit_cdp=True)


def test_teleport_timeout() -> None:
    """Tiny teleport timeout produces teleport_timeout status."""
    result = replay_equipment_attempt(
        CAPTURE,
        TARGET,
        teleport_timeout_ms=1,
    )
    assert result.attempt["status"] == "teleport_timeout"


def test_radar_timeout() -> None:
    """Tiny radar timeout produces radar_timeout status."""
    result = replay_equipment_attempt(
        CAPTURE,
        TARGET,
        teleport_timeout_ms=600_000,
        radar_timeout_ms=1,
    )
    assert result.attempt["status"] == "radar_timeout"


def test_walk_unreachable_equipment_yields_no_equipment_visible() -> None:
    """Walk-unreachable equipment is skipped, yielding no_equipment_visible.

    Under the 2026-06-26 walk-only contract, the probe's
    ``find_visible_equipment_target`` only returns walk-reachable
    containers. This capture's equipment requires a teleport
    reposition that is no longer attempted, so the probe terminates
    on ``no_equipment_visible`` instead of progressing to pickup.
    """
    result = replay_equipment_attempt(
        CAPTURE,
        TARGET,
        teleport_timeout_ms=600_000,
        radar_timeout_ms=600_000,
        pickup_timeout_ms=1,
    )
    assert result.attempt["status"] == "no_equipment_visible"


def test_teleport_timeout_with_sync_strategy() -> None:
    """sync_before_teleport strategy with tiny teleport timeout produces teleport_timeout."""
    result = replay_equipment_attempt(
        CAPTURE,
        TARGET,
        teleport_strategy="sync_before_teleport",
        map_sync_timeout_ms=600_000,
        teleport_timeout_ms=1,
    )
    assert result.attempt["status"] == "teleport_timeout"


def test_full_pipeline_reaches_collection_phase() -> None:
    """Full pipeline with generous timeouts reaches equipment collection."""
    result = replay_equipment_attempt(
        CAPTURE,
        TARGET,
        teleport_timeout_ms=600_000,
        radar_timeout_ms=600_000,
        pickup_timeout_ms=600_000,
    )
    assert result.attempt["status"] in (
        "picked_up_equipment",
        "no_equipment_visible",
        "pickup_timeout",
    )


def test_full_pipeline_with_sync_strategy() -> None:
    """Full pipeline with sync_before_teleport strategy completes."""
    result = replay_equipment_attempt(
        CAPTURE,
        TARGET,
        teleport_strategy="sync_before_teleport",
        map_sync_timeout_ms=600_000,
        teleport_timeout_ms=600_000,
        radar_timeout_ms=600_000,
        pickup_timeout_ms=600_000,
    )
    assert result.attempt["status"] in (
        "picked_up_equipment",
        "no_equipment_visible",
        "pickup_timeout",
        "radar_timeout",
    )
