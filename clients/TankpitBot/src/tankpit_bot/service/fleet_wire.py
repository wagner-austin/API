"""The ``GET /bots`` contract, encoded once and decoded once.

The fleet has two clients: the control page, which reads this JSON in
the browser, and the lifecycle CLI (:mod:`tankpit_bot.service.fleet_control`),
which reads it in Python to decide whether a manager is up and whether
its bots have finished draining. Both sides of that contract live here
so the server cannot drift from the client that consumes it.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    require_bool,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot.fleetshare.types import FLEET_ROLES, FleetRole
from tankpit_bot.service.fleet_bot import FleetBotDict


class FleetSnapshotDict(TypedDict):
    """Everything ``GET /bots`` says about a manager and its fleet.

    Attributes:
        boot: The serving manager's boot identity. A client that sees
            this change knows every instance name it holds belongs to
            a manager that no longer exists.
        draining: Whether a shutdown is in progress — the manager is
            waiting for these bots to tear down and will then exit.
        bots: Every registered instance's current state.
    """

    boot: str
    draining: bool
    bots: list[FleetBotDict]


class SpawnRequestDict(TypedDict):
    """One ``POST /bots`` body, after parsing.

    A TypedDict rather than a tuple because six of its eight fields
    are strings and four of those are adjacent selectors — role, room,
    troop, doctrine. Positionally, transposing any two of them is
    silent: the bot spawns, joins somewhere, and plays a colour nobody
    asked for. Named fields make that a type error.

    Attributes:
        instance: Instance name, or ``""`` to derive it from the
            account.
        account: ``TANKPIT_ACCOUNT`` selector; ``""`` uses the
            accounts.json default.
        kills: Kill bound; ``0`` is unbounded.
        seconds: Seconds bound; ``0`` is unbounded.
        role: Fleet role selector; ``""`` means fighter.
        room: ``TANKPIT_ROOM`` selector; ``""`` keeps the child's
            default.
        troop: Tank colour name; ``""`` keeps the account's own
            last-played colour for that world.
        doctrine: ``TANKPIT_DOCTRINE`` selector; ``""`` means
            skirmish, which is the unset behaviour.
    """

    instance: str
    account: str
    kills: int
    seconds: int
    role: str
    room: str
    troop: str
    doctrine: str


def _require_role(data: JSONObject, key: str) -> FleetRole:
    """Read a required field as a fleet role.

    Args:
        data: Decoded JSON object.
        key: Field name.

    Returns:
        The validated role.

    Raises:
        JSONTypeError: If the field is missing, not a string, or not a
            known fleet role.
    """
    value = require_str(data, key)
    for known in FLEET_ROLES:
        if value == known:
            return known
    known_roles = ", ".join(FLEET_ROLES)
    raise JSONTypeError(f"field '{key}' is not a fleet role: {value!r} (one of: {known_roles})")


def _require_returncode(data: JSONObject, key: str) -> int | None:
    """Read the exit code, which is absent while a bot runs.

    Args:
        data: Decoded JSON object.
        key: Field name.

    Returns:
        The exit code, or ``None`` when the bot is still running or
        ended without this manager observing a code (an adopted bot
        whose exit nobody was holding a handle for).

    Raises:
        JSONTypeError: If the field is present but not an integer.
    """
    if data.get(key) is None:
        return None
    return require_int(data, key)


def encode_fleet_bot(bot: FleetBotDict) -> JSONObject:
    """Encode one report row for the HTTP surface.

    Args:
        bot: The report row.

    Returns:
        JSON-serializable object.
    """
    return {
        "instance": bot["instance"],
        "account": bot["account"],
        "role": bot["role"],
        "room": bot["room"],
        "troop": bot["troop"],
        "pid": bot["pid"],
        "alive": bot["alive"],
        "returncode": bot["returncode"],
        "kills": bot["kills"],
        "seconds": bot["seconds"],
        "started_ms": bot["started_ms"],
    }


def decode_fleet_bot(data: JSONObject) -> FleetBotDict:
    """Decode one report row, validating every field.

    Args:
        data: Decoded JSON object.

    Returns:
        The validated report row.

    Raises:
        JSONTypeError: If any field is missing or of the wrong type.
    """
    return FleetBotDict(
        instance=require_str(data, "instance"),
        account=require_str(data, "account"),
        role=_require_role(data, "role"),
        room=require_str(data, "room"),
        troop=require_str(data, "troop"),
        pid=require_int(data, "pid"),
        alive=require_bool(data, "alive"),
        returncode=_require_returncode(data, "returncode"),
        kills=require_int(data, "kills"),
        seconds=require_int(data, "seconds"),
        started_ms=require_int(data, "started_ms"),
    )


def encode_fleet_snapshot(snapshot: FleetSnapshotDict) -> JSONObject:
    """Encode the whole ``GET /bots`` payload.

    Args:
        snapshot: The snapshot to encode.

    Returns:
        JSON-serializable object.
    """
    rows: list[JSONValue] = [encode_fleet_bot(bot) for bot in snapshot["bots"]]
    return {
        "boot": snapshot["boot"],
        "draining": snapshot["draining"],
        "bots": rows,
    }


def decode_fleet_snapshot(data: JSONObject) -> FleetSnapshotDict:
    """Decode the whole ``GET /bots`` payload, validating every field.

    Args:
        data: Decoded JSON object.

    Returns:
        The validated snapshot.

    Raises:
        JSONTypeError: If any field is missing or of the wrong type.
    """
    return FleetSnapshotDict(
        boot=require_str(data, "boot"),
        draining=require_bool(data, "draining"),
        bots=[decode_fleet_bot(narrow_json_to_dict(row)) for row in require_list(data, "bots")],
    )


__all__ = [
    "FleetSnapshotDict",
    "SpawnRequestDict",
    "decode_fleet_bot",
    "decode_fleet_snapshot",
    "encode_fleet_bot",
    "encode_fleet_snapshot",
]
