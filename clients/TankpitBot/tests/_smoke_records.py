"""Canned runtime-event records and the fake filesystem for the smoke tests."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
)
from scripts._test_hooks import (
    PathExistsProtocol,
    ReadTextProtocol,
)

from scripts import (
    _test_hooks,
    smoke,
)
from tests.conftest import FakeFileSystem


def _record_object(
    channel: str,
    message: str,
    timestamp: str = "2026-06-20T15:00:00",
    **fields: JSONValue,
) -> JSONObject:
    """Build a parsed-record JSON object directly (no serialisation)."""
    payload: JSONObject = {
        "timestamp": timestamp,
        "level": "INFO",
        "logger": "tankpit_bot.runtime.events",
        "mode": "bot",
        "channel": channel,
        "message": message,
    }
    payload.update(fields)
    return payload


def _record_raw(
    channel: str,
    message: str,
    timestamp: str = "2026-06-20T15:00:00",
    **fields: JSONValue,
) -> str:
    """Build the serialised JSONL line for a record."""
    return dump_json_str(_record_object(channel, message, timestamp, **fields))


def _smoke_record(
    line_no: int,
    channel: str,
    message: str,
    timestamp: str = "2026-06-20T15:00:00",
    **fields: JSONValue,
) -> smoke.SmokeRecord:
    """Build a SmokeRecord through the production decoder.

    Using ``_decode_smoke_record`` instead of the constructor keeps
    the helper aligned with how :func:`smoke.load_records` produces
    records in production.
    """
    parsed = _record_object(channel, message, timestamp, **fields)
    raw = dump_json_str(parsed)
    return smoke._decode_smoke_record(line_no=line_no, raw=raw, parsed=parsed)


def _login_records() -> list[smoke.SmokeRecord]:
    """A typical login sequence: INITIALIZING -> WAITING -> IDLE."""
    return [
        _smoke_record(1, "STATE", "INITIALIZING"),
        _smoke_record(2, "STATE", "INITIALIZING -> WAITING_FOR_POSITION"),
        _smoke_record(3, "STATE", "WAITING_FOR_POSITION -> IDLE"),
    ]


def _map_data_processed_record(line_no: int, timestamp: str) -> smoke.SmokeRecord:
    """A successful map_open WIRE_COMPLETE event."""
    return _smoke_record(
        line_no,
        "WIRE_COMPLETE",
        "map_open completed in 250ms via map_data_processed",
        timestamp=timestamp,
        action_kind="map_open",
        duration_ms=250,
        signal="map_data_processed",
    )


def _full_success_records(
    start: str = "2026-06-20T15:00:00",
) -> list[smoke.SmokeRecord]:
    """Build a record list that passes every assertion."""
    return [
        _smoke_record(1, "STATE", "INITIALIZING", timestamp=start),
        _smoke_record(2, "STATE", "INITIALIZING -> WAITING_FOR_POSITION", timestamp=start),
        _smoke_record(3, "STATE", "WAITING_FOR_POSITION -> IDLE", timestamp=start),
        _map_data_processed_record(4, start),
        _smoke_record(
            5,
            "AI",
            "HUNT score=0.8 target=(131,124)",
            timestamp=start,
            combat_target_x=131,
            combat_target_y=124,
        ),
        _smoke_record(
            6,
            "WIRE",
            "WIRE: shoot_at (131,124)",
            timestamp=start,
            action_kind="shoot",
        ),
    ]


def _full_success_jsonl(start: str = "2026-06-20T15:00:00") -> str:
    """Serialised JSONL string for a fully-passing run."""
    raws = [
        _record_raw("STATE", "INITIALIZING", timestamp=start),
        _record_raw("STATE", "INITIALIZING -> WAITING_FOR_POSITION", timestamp=start),
        _record_raw("STATE", "WAITING_FOR_POSITION -> IDLE", timestamp=start),
        _record_raw(
            "WIRE_COMPLETE",
            "map_open completed in 250ms via map_data_processed",
            timestamp=start,
            action_kind="map_open",
            duration_ms=250,
            signal="map_data_processed",
        ),
        _record_raw(
            "AI",
            "HUNT score=0.8 target=(131,124)",
            timestamp=start,
            combat_target_x=131,
            combat_target_y=124,
        ),
        _record_raw(
            "WIRE",
            "WIRE: shoot_at (131,124)",
            timestamp=start,
            action_kind="shoot",
        ),
    ]
    return "\n".join(raws) + "\n"


def _install_fake_filesystem() -> tuple[FakeFileSystem, PathExistsProtocol, ReadTextProtocol]:
    """Swap the real script hooks for a fake; return originals for restore.

    Returns:
        Tuple of ``(fake, original_path_exists, original_read_text)``.
        Callers MUST restore the originals in teardown.
    """
    fake = FakeFileSystem()
    original_path_exists: PathExistsProtocol = _test_hooks.path_exists
    original_read_text: ReadTextProtocol = _test_hooks.read_text
    _test_hooks.path_exists = fake.path_exists
    _test_hooks.read_text = fake.read_text
    return (fake, original_path_exists, original_read_text)
