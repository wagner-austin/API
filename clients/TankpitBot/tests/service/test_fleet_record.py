"""The spawn record: what a manager leaves behind for its successor."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError, load_json_str, narrow_json_to_dict

from tankpit_bot.service.fleet_record import (
    FleetProcessRecordDict,
    decode_process_record,
    encode_process_record,
    forget_process_record,
    process_record_path,
    read_process_record,
    recorded_instances,
    write_process_record,
)
from tests.service._fleet_fixtures import FakeRecordStore


def _record(instance: str = "alpha", pid: int = 4312) -> FleetProcessRecordDict:
    """Build one spawn record.

    Args:
        instance: Instance name.
        pid: Child process id.

    Returns:
        A fully populated record.
    """
    return FleetProcessRecordDict(
        instance=instance,
        account="Artax",
        role="gatherer",
        room="World",
        troop="orange",
        doctrine="skirmish",
        kills=30,
        seconds=2700,
        started_ms=1_788_000_000_000,
        pid=pid,
        created_at=1788265730.7457614,
    )


def test_the_record_path_is_inside_the_instances_own_run_directory() -> None:
    """Records are namespaced exactly like every other run artifact."""
    assert process_record_path("alpha") == Path("runs/bot/alpha/process.json")


def test_encode_then_decode_round_trips_every_field() -> None:
    """Nothing is lost between writing a record and reading it back."""
    original = _record()

    assert decode_process_record(encode_process_record(original)) == original


def test_the_creation_time_survives_full_float_precision() -> None:
    """Identity is an exact comparison, so the stored time must be exact.

    A record that rounded its creation time would stop matching the
    process it names, and the manager would refuse to adopt a bot that
    was still playing.
    """
    written: dict[str, str] = {}
    store = FakeRecordStore()
    originals = store.install()
    try:
        write_process_record(_record())
        written = dict(store.files)
        read_back = read_process_record("alpha")
    finally:
        store.restore(originals)

    raw = narrow_json_to_dict(load_json_str(next(iter(written.values()))))
    assert raw["created_at"] == 1788265730.7457614
    assert read_back["created_at"] == 1788265730.7457614


def test_a_record_is_written_read_and_forgotten() -> None:
    """The whole lifecycle of one record on the store."""
    store = FakeRecordStore()
    originals = store.install()
    try:
        write_process_record(_record())
        present = read_process_record("alpha")
        forget_process_record("alpha")
        with pytest.raises(OSError, match="no such record"):
            read_process_record("alpha")
    finally:
        store.restore(originals)

    assert present["pid"] == 4312
    assert store.files == {}


def test_forgetting_a_record_that_is_not_there_is_not_an_error() -> None:
    """Clearing an already-cleared record matches unlink(missing_ok)."""
    store = FakeRecordStore()
    originals = store.install()
    try:
        forget_process_record("never-existed")
    finally:
        store.restore(originals)

    assert store.files == {}


def test_recorded_instances_lists_every_instance_with_a_record() -> None:
    """The boot scan finds each instance directory holding a record."""
    store = FakeRecordStore()
    originals = store.install()
    try:
        write_process_record(_record(instance="alpha"))
        write_process_record(_record(instance="bravo"))
        found = recorded_instances()
    finally:
        store.restore(originals)

    assert found == ["alpha", "bravo"]


def test_recorded_instances_is_empty_when_nothing_was_ever_spawned() -> None:
    """A fresh working directory adopts nothing."""
    store = FakeRecordStore()
    originals = store.install()
    try:
        found = recorded_instances()
    finally:
        store.restore(originals)

    assert found == []


def test_decoding_refuses_a_role_that_is_not_a_fleet_role() -> None:
    """A corrupted role is surfaced, never coerced to a default."""
    data = encode_process_record(_record())
    data["role"] = "scout"

    with pytest.raises(JSONTypeError, match="not a fleet role"):
        decode_process_record(data)


@pytest.mark.parametrize(
    "field",
    [
        "instance",
        "account",
        "role",
        "room",
        "troop",
        "doctrine",
        "kills",
        "seconds",
        "started_ms",
        "pid",
    ],
)
def test_decoding_refuses_a_record_missing_any_field(field: str) -> None:
    """Every field is required; a partial record is corruption."""
    data = encode_process_record(_record())
    del data[field]

    with pytest.raises(JSONTypeError):
        decode_process_record(data)


def test_decoding_refuses_a_created_at_that_is_not_a_number() -> None:
    """The identity half of the record is validated like the rest."""
    data = encode_process_record(_record())
    data["created_at"] = "recently"

    with pytest.raises(JSONTypeError):
        decode_process_record(data)
