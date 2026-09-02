"""The ``GET /bots`` contract, round-tripped and validated."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.service.fleet_bot import FleetBotDict
from tankpit_bot.service.fleet_wire import (
    FleetSnapshotDict,
    decode_fleet_bot,
    decode_fleet_snapshot,
    encode_fleet_bot,
    encode_fleet_snapshot,
)


def _bot(instance: str = "alpha", returncode: int | None = None) -> FleetBotDict:
    """Build one report row.

    Args:
        instance: Instance name.
        returncode: Exit code, or None while running.

    Returns:
        A fully populated row.
    """
    return FleetBotDict(
        instance=instance,
        account="Artax",
        role="gatherer",
        room="World",
        troop="orange",
        pid=4312,
        alive=returncode is None,
        returncode=returncode,
        kills=30,
        seconds=2700,
        started_ms=1_788_000_000_000,
    )


def test_a_row_round_trips_through_the_wire() -> None:
    """Encoding then decoding one row changes nothing."""
    original = _bot()

    assert decode_fleet_bot(encode_fleet_bot(original)) == original


def test_a_finished_row_carries_its_exit_code() -> None:
    """An observed exit code survives the round trip."""
    assert decode_fleet_bot(encode_fleet_bot(_bot(returncode=3)))["returncode"] == 3


def test_a_missing_exit_code_decodes_as_absent_not_as_an_error() -> None:
    """A running bot -- and an adopted bot whose exit went unobserved.

    ``alive`` is the authoritative flag; ``returncode`` is the extra
    detail that may genuinely not exist.
    """
    data = encode_fleet_bot(_bot())

    assert data["returncode"] is None
    assert decode_fleet_bot(data)["returncode"] is None


def test_a_snapshot_round_trips_with_its_boot_and_drain_state() -> None:
    """The manager-level fields survive alongside the rows."""
    original = FleetSnapshotDict(
        boot="1788265730000",
        draining=True,
        bots=[_bot("alpha"), _bot("bravo", returncode=0)],
    )

    assert decode_fleet_snapshot(encode_fleet_snapshot(original)) == original


def test_an_empty_fleet_round_trips() -> None:
    """A manager with no bots is a snapshot, not an absence."""
    original = FleetSnapshotDict(boot="1", draining=False, bots=[])

    assert decode_fleet_snapshot(encode_fleet_snapshot(original)) == original


def test_decoding_refuses_a_role_that_is_not_a_fleet_role() -> None:
    """A row naming an unknown role is rejected, never coerced."""
    data = encode_fleet_bot(_bot())
    data["role"] = "scout"

    with pytest.raises(JSONTypeError, match="not a fleet role"):
        decode_fleet_bot(data)


def test_decoding_refuses_a_non_integer_exit_code() -> None:
    """Present-but-wrong is different from absent."""
    data = encode_fleet_bot(_bot())
    data["returncode"] = "crashed"

    with pytest.raises(JSONTypeError):
        decode_fleet_bot(data)


@pytest.mark.parametrize("field", ["boot", "draining", "bots"])
def test_decoding_refuses_a_snapshot_missing_any_field(field: str) -> None:
    """Every manager-level field is required."""
    data = encode_fleet_snapshot(FleetSnapshotDict(boot="1", draining=False, bots=[_bot()]))
    del data[field]

    with pytest.raises(JSONTypeError):
        decode_fleet_snapshot(data)


def test_decoding_refuses_a_snapshot_whose_rows_are_not_objects() -> None:
    """A malformed row fails the whole snapshot, not just itself."""
    with pytest.raises(JSONTypeError):
        decode_fleet_snapshot({"boot": "1", "draining": False, "bots": ["alpha"]})


@pytest.mark.parametrize(
    "field",
    ["instance", "account", "role", "room", "troop", "pid", "alive", "kills", "seconds"],
)
def test_decoding_refuses_a_row_missing_any_field(field: str) -> None:
    """A partial row is corruption, not a default."""
    data = encode_fleet_bot(_bot())
    del data[field]

    with pytest.raises(JSONTypeError):
        decode_fleet_bot(data)
