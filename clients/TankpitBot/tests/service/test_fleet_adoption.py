"""Adoption: a restarted manager finding the bots that kept playing."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import InvalidJsonError, JSONTypeError

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service._test_hooks import SpawnedProcessProtocol
from tankpit_bot.service.fleet_adoption import adopt_recorded_bots
from tankpit_bot.service.fleet_manager import FleetManager
from tankpit_bot.service.fleet_record import (
    FleetProcessRecordDict,
    process_record_path,
    write_process_record,
)
from tests.service._fleet_fixtures import FakeRecordStore, _FakeProcess, _FakeSpawner


class _FakeAdopter:
    """open_adopted_process double driven by an identity table.

    Attributes:
        living: ``(pid, created_at)`` pairs that are still running.
        asked: Every ``(pid, created_at)`` it was asked about.
    """

    def __init__(self, living: dict[int, float]) -> None:
        """Bind the adopter to the processes it should claim are alive.

        Args:
            living: Pid to creation time for each live process.
        """
        self.living = living
        self.asked: list[tuple[int, float]] = []

    def __call__(self, pid: int, created_at: float) -> SpawnedProcessProtocol | None:
        """Return a handle when the identity matches a live process.

        Args:
            pid: Recorded pid.
            created_at: Recorded creation time.

        Returns:
            A process double, or None when nothing matches.
        """
        self.asked.append((pid, created_at))
        if self.living.get(pid) != created_at:
            return None
        return _FakeProcess(pid=pid)


def _record(instance: str, pid: int, created_at: float) -> FleetProcessRecordDict:
    """Build one spawn record.

    Args:
        instance: Instance name.
        pid: Child process id.
        created_at: Child creation time.

    Returns:
        A fully populated record.
    """
    return FleetProcessRecordDict(
        instance=instance,
        account="Artax",
        role="fighter",
        room="World",
        troop="orange",
        kills=12,
        seconds=600,
        started_ms=1_788_000_000_000,
        pid=pid,
        created_at=created_at,
    )


def test_a_bot_still_running_is_re_attached_with_its_whole_identity(
    records: FakeRecordStore,
) -> None:
    """An adopted row carries every spawn parameter, not just the pid.

    The 2026-08-28 restart bug is the precedent for caring: a respawn
    that dropped the room silently relocated the bot to Practice while
    the row still said World.
    """
    write_process_record(_record("alpha", pid=4312, created_at=99.5))
    service_hooks.open_adopted_process = _FakeAdopter({4312: 99.5})

    adopted = adopt_recorded_bots()

    assert len(adopted) == 1
    row = adopted[0].report()
    assert row["instance"] == "alpha"
    assert row["account"] == "Artax"
    assert row["role"] == "fighter"
    assert row["room"] == "World"
    assert row["troop"] == "orange"
    assert row["kills"] == 12
    assert row["seconds"] == 600
    assert row["started_ms"] == 1_788_000_000_000
    assert row["pid"] == 4312
    assert row["alive"] is True


def test_a_bot_that_finished_unsupervised_is_dropped_and_its_record_cleared(
    records: FakeRecordStore,
) -> None:
    """Nothing running under the pid means the run is over."""
    write_process_record(_record("alpha", pid=4312, created_at=99.5))
    service_hooks.open_adopted_process = _FakeAdopter({})

    adopted = adopt_recorded_bots()

    assert adopted == []
    assert records.files == {}


def test_a_recycled_pid_is_not_mistaken_for_a_live_bot(
    records: FakeRecordStore,
) -> None:
    """Identity is the pid AND its creation time, compared exactly.

    Windows recycles pids. Adopting on the number alone would hand the
    fleet some unrelated program, and then refuse to restart the
    instance forever because its imaginary bot is always running.
    """
    write_process_record(_record("alpha", pid=4312, created_at=99.5))
    adopter = _FakeAdopter({4312: 12345.0})
    service_hooks.open_adopted_process = adopter

    adopted = adopt_recorded_bots()

    assert adopter.asked == [(4312, 99.5)]
    assert adopted == []
    assert records.files == {}


def test_every_recorded_instance_is_considered(records: FakeRecordStore) -> None:
    """A mixed fleet adopts the survivors and forgets the rest."""
    write_process_record(_record("alpha", pid=1, created_at=1.0))
    write_process_record(_record("bravo", pid=2, created_at=2.0))
    write_process_record(_record("charlie", pid=3, created_at=3.0))
    service_hooks.open_adopted_process = _FakeAdopter({1: 1.0, 3: 3.0})

    adopted = adopt_recorded_bots()

    assert [bot.instance for bot in adopted] == ["alpha", "charlie"]
    assert set(records.files) == {
        str(process_record_path("alpha")),
        str(process_record_path("charlie")),
    }


def test_a_corrupt_record_is_raised_not_skipped(records: FakeRecordStore) -> None:
    """Booting over corruption would silently forget a live tank.

    Records are written atomically, so a record that will not decode is
    real corruption rather than a torn write, and the honest response
    is to refuse to start rather than to pretend the fleet is empty.
    """
    records.files[str(process_record_path("alpha"))] = "{not json"

    with pytest.raises(InvalidJsonError):
        adopt_recorded_bots()


def test_a_record_missing_a_field_is_raised_not_skipped(
    records: FakeRecordStore,
) -> None:
    """Structural damage is surfaced with the same refusal."""
    records.files[str(process_record_path("alpha"))] = '{"instance": "alpha"}'

    with pytest.raises(JSONTypeError):
        adopt_recorded_bots()


def test_the_manager_registers_what_it_adopts(
    records: FakeRecordStore,
    spawner: _FakeSpawner,
) -> None:
    """After adopt, the surviving bots are ordinary registry members."""
    _ = spawner
    write_process_record(_record("alpha", pid=4312, created_at=99.5))
    write_process_record(_record("bravo", pid=4313, created_at=42.0))
    service_hooks.open_adopted_process = _FakeAdopter({4312: 99.5})

    manager = FleetManager()
    adopted = manager.adopt()

    assert adopted == ["alpha"]
    assert manager.live_instances() == ["alpha"]
    assert [row["instance"] for row in manager.report()] == ["alpha"]


def test_adopting_nothing_leaves_an_empty_registry(
    records: FakeRecordStore,
    spawner: _FakeSpawner,
) -> None:
    """A first-ever boot has no records and adopts nobody."""
    _ = spawner
    service_hooks.open_adopted_process = _FakeAdopter({})

    manager = FleetManager()

    assert manager.adopt() == []
    assert manager.report() == []


def test_an_adopted_bot_can_be_stopped_and_removed(
    records: FakeRecordStore,
    spawner: _FakeSpawner,
) -> None:
    """Adoption exists so a restarted manager can act, not just look.

    This is the whole point: before adoption, the only way to end one
    of these was to find its pid by hand.
    """
    _ = spawner
    write_process_record(_record("alpha", pid=4312, created_at=99.5))
    adopter = _FakeAdopter({4312: 99.5})
    service_hooks.open_adopted_process = adopter
    manager = FleetManager()
    manager.adopt()

    stops: list[str] = []

    def record_stop(path: Path, content: str) -> None:
        _ = content
        stops.append(str(path).replace("\\", "/"))

    saved = top_hooks.write_text
    top_hooks.write_text = record_stop
    try:
        manager.stop("alpha")
    finally:
        top_hooks.write_text = saved

    assert stops == ["runs/bot/alpha/STOP"]
    assert manager.report()[0]["alive"] is True
