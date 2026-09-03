"""What the fleet remembers about a bot across its own restarts.

The fleet manager's registry is in memory, and its bots are child
processes that deliberately outlive it -- "an orchestrator dying can
never kill a live tank". Those two facts together meant a restarted
manager knew nothing: the tanks were still playing, still logging,
still burning fuel, and nothing could stop or even see them. The page
showed an empty fleet while five bots fought on.

This module is the missing half. Every spawn writes
``runs/bot/<instance>/process.json`` beside the run's other artifacts,
holding what it takes to find that child again and to describe it the
same way afterwards: its pid AND creation time (a pid alone is not an
identity -- see
:class:`~tankpit_bot.service._test_hooks.ProcessIdentityProtocol`),
plus the spawn parameters a restart would otherwise have to guess.

Records are written atomically. A reader therefore sees either the
previous complete record or the new complete record, never a torn one,
which is what lets the decode be strict: a record that fails to decode
is real corruption and says so, rather than being quietly skipped.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.fleetshare.types import FLEET_ROLES, FleetRole
from tankpit_bot.runtime_artifacts import bot_run_dir

#: File name of the per-instance spawn record, inside the instance's
#: own run directory so it is namespaced exactly like every other
#: artifact that run owns.
PROCESS_RECORD_NAME = "process.json"


class FleetProcessRecordDict(TypedDict):
    """One spawned bot, as recorded for a future manager to find.

    Attributes:
        instance: Validated instance name (artifact namespace).
        account: ``TANKPIT_ACCOUNT`` the child was spawned with
            (empty means the accounts.json default).
        role: Resolved fleet role the child was spawned with.
        room: ``TANKPIT_ROOM`` the child was spawned with (empty means
            the default Practice room).
        troop: Tank color name the child was spawned with (empty means
            the account's own default for that map).
        doctrine: Engagement doctrine the child was spawned with
            (empty means its own default, skirmish). Recorded so an
            ADOPTED bot still reports what it is fighting under -- a
            manager that inherits a bot and cannot say knows less
            about it than the one that spawned it.
        kills: Kill bound the child was spawned with (0 unbounded).
        seconds: Seconds bound the child was spawned with (0 unbounded).
        started_ms: Wall-clock spawn time, so an adopted row reports
            the same uptime the spawning manager would have.
        pid: Child process id.
        created_at: The child's process creation time in epoch
            seconds. Paired with ``pid`` this is an identity: pids are
            recycled, creation times are not.
        service_port: Port the child's own service is serving on.
            Recorded because an adopting manager needs it for two
            reasons: to relay ``/video`` to a bot it did not spawn, and
            to know the port is SPENT. Without it a fresh child could be
            handed a port a live adopted child already holds, and the
            two would serve each other's video.
    """

    instance: str
    account: str
    role: FleetRole
    room: str
    troop: str
    doctrine: str
    kills: int
    seconds: int
    started_ms: int
    pid: int
    created_at: float
    service_port: int


def process_record_path(instance: str) -> Path:
    """Return where one instance's spawn record lives.

    Args:
        instance: Validated instance name.

    Returns:
        ``runs/bot/<instance>/process.json``.
    """
    return bot_run_dir(instance) / PROCESS_RECORD_NAME


def _require_role(data: JSONObject, key: str) -> FleetRole:
    """Read a required field as a fleet role.

    Args:
        data: Decoded JSON object.
        key: Field name.

    Returns:
        The validated role.

    Raises:
        JSONTypeError: If the field is missing, not a string, or not
            one of the known fleet roles.
    """
    value = require_str(data, key)
    for known in FLEET_ROLES:
        if value == known:
            return known
    known_roles = ", ".join(FLEET_ROLES)
    raise JSONTypeError(f"field '{key}' is not a fleet role: {value!r} (one of: {known_roles})")


def encode_process_record(record: FleetProcessRecordDict) -> JSONObject:
    """Encode one spawn record for storage.

    Args:
        record: The record to encode.

    Returns:
        JSON-serializable object.
    """
    return {
        "instance": record["instance"],
        "account": record["account"],
        "role": record["role"],
        "room": record["room"],
        "troop": record["troop"],
        "doctrine": record["doctrine"],
        "kills": record["kills"],
        "seconds": record["seconds"],
        "started_ms": record["started_ms"],
        "pid": record["pid"],
        "created_at": record["created_at"],
        "service_port": record["service_port"],
    }


def decode_process_record(data: JSONObject) -> FleetProcessRecordDict:
    """Decode one spawn record, validating every field.

    Args:
        data: Decoded JSON object.

    Returns:
        The validated record.

    Raises:
        JSONTypeError: If any field is missing or of the wrong type.
    """
    return FleetProcessRecordDict(
        instance=require_str(data, "instance"),
        account=require_str(data, "account"),
        role=_require_role(data, "role"),
        room=require_str(data, "room"),
        troop=require_str(data, "troop"),
        doctrine=require_str(data, "doctrine"),
        kills=require_int(data, "kills"),
        seconds=require_int(data, "seconds"),
        started_ms=require_int(data, "started_ms"),
        pid=require_int(data, "pid"),
        created_at=require_float(data, "created_at"),
        service_port=require_int(data, "service_port"),
    )


def write_process_record(record: FleetProcessRecordDict) -> None:
    """Persist one spawn record atomically.

    Args:
        record: The record to write, under its own instance name.

    Returns:
        None.
    """
    top_hooks.replace_text(
        process_record_path(record["instance"]),
        dump_json_str(encode_process_record(record), indent=1),
    )


def read_process_record(instance: str) -> FleetProcessRecordDict:
    """Read one instance's spawn record.

    Args:
        instance: Validated instance name.

    Returns:
        The decoded record.

    Raises:
        OSError: If no record exists for the instance.
        InvalidJsonError: If the file is not valid JSON.
        JSONTypeError: If the record is missing fields or malformed.
    """
    raw = top_hooks.read_text(process_record_path(instance))
    return decode_process_record(narrow_json_to_dict(load_json_str(raw)))


def forget_process_record(instance: str) -> None:
    """Delete one instance's spawn record.

    Called when a bot is gone for good -- removed from the registry,
    or found already finished at adoption time. A record outliving its
    process would have the next manager re-check a pid forever.

    Args:
        instance: Validated instance name.

    Returns:
        None.
    """
    top_hooks.remove_file(process_record_path(instance))


def recorded_instances() -> list[str]:
    """List every instance that has a spawn record on disk.

    Returns:
        Instance names in sorted order; empty when no manager has
        spawned anything under this working directory.
    """
    root = bot_run_dir("")
    return [path.parent.name for path in top_hooks.glob_paths(root, f"*/{PROCESS_RECORD_NAME}")]


__all__ = [
    "PROCESS_RECORD_NAME",
    "FleetProcessRecordDict",
    "decode_process_record",
    "encode_process_record",
    "forget_process_record",
    "process_record_path",
    "read_process_record",
    "recorded_instances",
    "write_process_record",
]
