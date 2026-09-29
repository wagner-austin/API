"""Runtime-event records and the fake filesystem for the smoke tests.

Action outcomes are never hand-built here. The smoke gate once read a
``WIRE_COMPLETE`` channel and a ``signal`` field for months after the
bot stopped writing them, and its tests stayed green because their
records were built from the same assumption (board task 14b91fb5). An
outcome line in these tests is the line the ledger's own emitter writes
to the events file; only its clock is re-stamped, the one field a
timing test has to control.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_str,
)
from scripts._test_hooks import (
    PathExistsProtocol,
    ReadTextProtocol,
)

from scripts import (
    _test_hooks,
    smoke,
)
from tankpit_bot.ledger.outcome.map_open import (
    emit_map_open_data_processed,
    emit_map_open_stall_timeout,
)
from tankpit_bot.ledger.outcome.move import emit_move_position_reached, emit_move_stall_timeout
from tankpit_bot.ledger.outcomes import is_action_outcome_event
from tankpit_bot.ledger.records import ActionOutcomeRecordDict
from tankpit_bot.ledger.service import LedgerService
from tankpit_bot.runtime_logging import configure_bot_runtime_logging
from tests.conftest import FakeFileSystem


def _emitted_outcome_object(
    fake_fs: FakeFileSystem,
    emit: Callable[[LedgerService], ActionOutcomeRecordDict],
    timestamp: str,
) -> JSONObject:
    """Emit one outcome through the real ledger emitter and read back its line.

    Args:
        fake_fs: The installed fake filesystem the runtime logger writes to.
        emit: A ledger outcome emitter, bound to its arguments.
        timestamp: The clock to stamp on the line.

    Returns:
        The parsed events-file line the emitter wrote, re-stamped.
    """
    artifacts = configure_bot_runtime_logging("20260929-000000")
    emit(LedgerService())
    written = fake_fs.get_written_files()[artifacts["latest_events_path"]]
    parsed = narrow_json_to_dict(load_json_str(written.splitlines()[-1]))
    if not is_action_outcome_event(require_str(parsed, "channel"), parsed):
        raise AssertionError(f"the emitter's last line is not an action outcome: {parsed}")
    parsed["timestamp"] = timestamp
    return parsed


def _emitted_outcome_record(
    fake_fs: FakeFileSystem,
    line_no: int,
    emit: Callable[[LedgerService], ActionOutcomeRecordDict],
    timestamp: str = "2026-06-20T15:00:00",
) -> smoke.SmokeRecord:
    """Decode the real emitter's line into a SmokeRecord, as ``load_records`` would.

    Args:
        fake_fs: The installed fake filesystem the runtime logger writes to.
        line_no: The record's line number.
        emit: A ledger outcome emitter, bound to its arguments.
        timestamp: The clock to stamp on the line.

    Returns:
        The decoded record.
    """
    parsed = _emitted_outcome_object(fake_fs, emit, timestamp)
    return smoke._decode_smoke_record(line_no=line_no, raw=dump_json_str(parsed), parsed=parsed)


def _map_open_resolved(ledger: LedgerService) -> ActionOutcomeRecordDict:
    """Record a map open resolved by its MAP_DATA answer, as the bot does.

    Args:
        ledger: The session ledger.

    Returns:
        The recorded outcome.
    """
    return emit_map_open_data_processed(ledger, duration_ms=250)


def _map_open_stalled(ledger: LedgerService) -> ActionOutcomeRecordDict:
    """Record a map open whose answer never came, as the stall clock does.

    Args:
        ledger: The session ledger.

    Returns:
        The recorded outcome.
    """
    return emit_map_open_stall_timeout(ledger, duration_ms=10000, timeout_ms=10000)


def _move_stalled(ledger: LedgerService) -> ActionOutcomeRecordDict:
    """Record a walk that stalled out, as the stall clock does.

    Args:
        ledger: The session ledger.

    Returns:
        The recorded outcome.
    """
    return emit_move_stall_timeout(
        ledger, duration_ms=5000, target_x=101, target_y=100, timeout_ms=5000
    )


def _move_reached(ledger: LedgerService) -> ActionOutcomeRecordDict:
    """Record a walk that reached its tile.

    Args:
        ledger: The session ledger.

    Returns:
        The recorded outcome.
    """
    return emit_move_position_reached(
        ledger, duration_ms=400, target_x=101, target_y=100, landed_x=101, landed_y=100
    )


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


def _full_success_records(
    fake_fs: FakeFileSystem,
    start: str = "2026-06-20T15:00:00",
) -> list[smoke.SmokeRecord]:
    """Build a record list that passes every assertion."""
    return [
        _smoke_record(1, "STATE", "INITIALIZING", timestamp=start),
        _smoke_record(2, "STATE", "INITIALIZING -> WAITING_FOR_POSITION", timestamp=start),
        _smoke_record(3, "STATE", "WAITING_FOR_POSITION -> IDLE", timestamp=start),
        _emitted_outcome_record(fake_fs, 4, _map_open_resolved, start),
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


def _full_success_jsonl(fake_fs: FakeFileSystem, start: str = "2026-06-20T15:00:00") -> str:
    """Serialised JSONL string for a fully-passing run."""
    raws = [
        _record_raw("STATE", "INITIALIZING", timestamp=start),
        _record_raw("STATE", "INITIALIZING -> WAITING_FOR_POSITION", timestamp=start),
        _record_raw("STATE", "WAITING_FOR_POSITION -> IDLE", timestamp=start),
        dump_json_str(_emitted_outcome_object(fake_fs, _map_open_resolved, start)),
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
