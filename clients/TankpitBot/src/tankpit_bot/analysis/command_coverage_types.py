"""Data contract for the client-command coverage audit.

A command byte a real client sends and the sim does not handle is not a
fidelity nuance — it is a crash. ``SimServer.queue_command`` refuses any
kind outside ``SUPPORTED_KINDS``, and every client frame reaches it, so
an unmapped byte takes the server down on arrival.

That happened three times on 2026-09-03 alone (the keep-alive, the
enter-game and the inventory request), each found by a one-off sweep
rather than by a mechanism. This is the mechanism.

Every dict carries an encode/decode codec: the audit is written to an
artifact and read back, which is the boundary where the codec rule
binds ([[coding-standards]]).
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)

#: The sim decodes the byte, routes it, and answers it.
STATUS_HANDLED = "handled"

#: The sim decodes the byte to a named kind but has NO measured law, so
#: it refuses by name. Declared, not forgotten.
STATUS_DECLARED_UNMODELLED = "declared_unmodelled"

#: The sim does not map the byte at all: it decodes to ``other`` and
#: ``queue_command`` raises. A real client sending it kills the server.
STATUS_CRASHES = "crashes"


class CommandByteRowDict(TypedDict):
    """One client command byte, as the archive and the sim see it.

    Attributes:
        byte: The command byte the client sent.
        constant: The ``CMD_*`` constant naming it, or the empty string
            when no constant exists — itself a finding, since a byte
            with sends and no name has never been looked at.
        kind: The kind the sim's decoder resolves it to.
        sends: How many times the archive carries it.
        status: One of the ``STATUS_*`` values.
    """

    byte: int
    constant: str
    kind: str
    sends: int
    status: str


class CommandCoverageDict(TypedDict):
    """What a real client sends, against what the sim handles.

    Attributes:
        sessions: Archive sessions mined.
        rows: One entry per distinct command byte seen, descending by
            send count.
        unsent_constants: ``CMD_*`` constants defined in the protocol
            and never once sent in the archive. Not a defect — real
            client capabilities nobody in the corpus used — but the
            audit reports them so the gap between the protocol we have
            written down and the protocol we have OBSERVED stays
            visible.
    """

    sessions: int
    rows: list[CommandByteRowDict]
    unsent_constants: list[str]


def encode_command_byte_row(row: CommandByteRowDict) -> JSONObject:
    """Encode one command-byte row.

    Args:
        row: The row to encode.

    Returns:
        JSON object with every row field.
    """
    return {
        "byte": row["byte"],
        "constant": row["constant"],
        "kind": row["kind"],
        "sends": row["sends"],
        "status": row["status"],
    }


def decode_command_byte_row(data: JSONObject) -> CommandByteRowDict:
    """Decode one command-byte row with validation.

    Args:
        data: JSON object carrying the row fields.

    Returns:
        Validated row.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
    """
    return CommandByteRowDict(
        byte=require_int(data, "byte"),
        constant=require_str(data, "constant"),
        kind=require_str(data, "kind"),
        sends=require_int(data, "sends"),
        status=require_str(data, "status"),
    )


def encode_command_coverage(coverage: CommandCoverageDict) -> JSONObject:
    """Encode the whole coverage audit.

    Args:
        coverage: The audit to encode.

    Returns:
        JSON object with every audit field.
    """
    return {
        "sessions": coverage["sessions"],
        "rows": [encode_command_byte_row(row) for row in coverage["rows"]],
        "unsent_constants": list(coverage["unsent_constants"]),
    }


def decode_command_coverage(data: JSONObject) -> CommandCoverageDict:
    """Decode the whole coverage audit with validation.

    Args:
        data: JSON object carrying the audit fields.

    Returns:
        Validated audit.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
        TypeError: If an unsent-constant entry is not a string.
    """
    return CommandCoverageDict(
        sessions=require_int(data, "sessions"),
        rows=[
            decode_command_byte_row(narrow_json_to_dict(item))
            for item in require_list(data, "rows")
        ],
        unsent_constants=[_require_name(item) for item in require_list(data, "unsent_constants")],
    )


def _require_name(value: JSONValue) -> str:
    """Narrow one JSON list element to a constant name.

    Args:
        value: Candidate element from a decoded list.

    Returns:
        The name as a string.

    Raises:
        TypeError: If the element is not a string. A malformed artifact
            is not a name to coerce.
    """
    if not isinstance(value, str):
        raise TypeError(f"constant name must be a string, got {type(value).__name__}")
    return value


__all__ = [
    "STATUS_CRASHES",
    "STATUS_DECLARED_UNMODELLED",
    "STATUS_HANDLED",
    "CommandByteRowDict",
    "CommandCoverageDict",
    "decode_command_byte_row",
    "decode_command_coverage",
    "encode_command_byte_row",
    "encode_command_coverage",
]
