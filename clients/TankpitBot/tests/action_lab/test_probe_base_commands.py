"""Tests for ProbeBase shoot, pickup_fuel, pickup_equipment methods."""

from __future__ import annotations

from tankpit_bot.action_lab.probe_base import ProbeBase


class _CommandRecorder(ProbeBase):
    """ProbeBase subclass that records dispatched commands instead of sending."""

    def __init__(self) -> None:
        self._dispatched: list[tuple[bytes, str]] = []
        self._commands_xor_table: bytes | None = None

    def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
        self._dispatched.append((data, cmd_name))
        return True


def test_shoot_dispatches_command() -> None:
    probe = _CommandRecorder()
    result = probe.shoot(10, 20, target_id=42)
    assert result is True
    assert len(probe._dispatched) == 1
    _, label = probe._dispatched[0]
    assert label == "shoot(10,20,id=42)"


def test_shoot_default_target_id() -> None:
    probe = _CommandRecorder()
    result = probe.shoot(5, 15)
    assert result is True
    _, label = probe._dispatched[0]
    assert label == "shoot(5,15,id=0)"


def test_pickup_fuel_dispatches_command() -> None:
    probe = _CommandRecorder()
    result = probe.pickup_fuel(30, 40)
    assert result is True
    assert len(probe._dispatched) == 1
    _, label = probe._dispatched[0]
    assert label == "pickup_fuel"


def test_pickup_equipment_dispatches_command() -> None:
    probe = _CommandRecorder()
    result = probe.pickup_equipment(50, 60)
    assert result is True
    assert len(probe._dispatched) == 1
    _, label = probe._dispatched[0]
    assert label == "pickup_equipment"


def test_request_inventory_dispatches_command() -> None:
    """The watch-dwell heartbeat command, framed as ``[len]['!'][2]['i']``."""
    probe = _CommandRecorder()
    result = probe.request_inventory()
    assert result is True
    assert len(probe._dispatched) == 1
    data, label = probe._dispatched[0]
    assert label == "inventory"
    assert data.endswith(b"\x02i")
