"""Tests for how the manager hands out child service ports.

Allocation is driven directly here rather than through spawns, because
the interesting cases need more concurrent children than there are
configured accounts — the manager refuses a second live bot on one
account, so the HTTP route cannot reach an exhausted range.

The property under test is that two live children never share a port.
Sharing one would mean the relay serving one bot's video under
another's name, which no caller could detect.
"""

from __future__ import annotations

import pytest

from tankpit_bot.service.constants import FLEET_CHILD_PORT_BASE, FLEET_CHILD_PORT_COUNT
from tankpit_bot.service.fleet_bot import _ManagedBot
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_manager import FleetManager
from tests.service._fleet_fixtures import FakeRecordStore, _FakeProcess


def _managed(instance: str, port: int, *, alive: bool) -> _ManagedBot:
    """Build a registry entry holding a controllable process double.

    Args:
        instance: Instance name.
        port: Service port the entry claims.
        alive: Whether the double reports as running.

    Returns:
        The registry entry.
    """
    process = _FakeProcess(pid=9000 + port)
    if not alive:
        process.returncode = 0
    return _ManagedBot(
        instance=instance,
        account=instance,
        role="fighter",
        room="Practice",
        troop="orange",
        doctrine="skirmish",
        kills=0,
        seconds=0,
        started_ms=1_788_000_000_000,
        service_port=port,
        process=process,
    )


def test_the_first_child_takes_the_bottom_of_the_range(records: FakeRecordStore) -> None:
    """An empty fleet allocates the base port."""
    _ = records
    manager = FleetManager()

    assert manager._allocate_service_port() == FLEET_CHILD_PORT_BASE


def test_allocation_fills_the_lowest_free_port(records: FakeRecordStore) -> None:
    """A gap left by a dead child is filled before the range grows.

    Lowest-free-first is what keeps a long-lived fleet from marching up
    the range and exhausting it while only two bots are ever running.
    """
    _ = records
    manager = FleetManager()
    manager._bots["alpha"] = _managed("alpha", FLEET_CHILD_PORT_BASE, alive=False)
    manager._bots["bravo"] = _managed("bravo", FLEET_CHILD_PORT_BASE + 1, alive=True)

    assert manager._allocate_service_port() == FLEET_CHILD_PORT_BASE


def test_only_live_children_reserve_a_port(records: FakeRecordStore) -> None:
    """A dead child's port returns to the pool immediately."""
    _ = records
    manager = FleetManager()
    for offset in range(FLEET_CHILD_PORT_COUNT):
        manager._bots[f"dead{offset}"] = _managed(
            f"dead{offset}", FLEET_CHILD_PORT_BASE + offset, alive=False
        )

    assert manager._allocate_service_port() == FLEET_CHILD_PORT_BASE


def test_an_exhausted_range_is_refused_rather_than_wrapped(records: FakeRecordStore) -> None:
    """Every port held by a live child is an error, not a reuse.

    Wrapping around would put two live children on one port, and the
    relay would serve whichever answered first — a confusion with no
    symptom at the caller.
    """
    _ = records
    manager = FleetManager()
    for offset in range(FLEET_CHILD_PORT_COUNT):
        manager._bots[f"live{offset}"] = _managed(
            f"live{offset}", FLEET_CHILD_PORT_BASE + offset, alive=True
        )

    with pytest.raises(FleetError, match="no free child service port"):
        manager._allocate_service_port()
