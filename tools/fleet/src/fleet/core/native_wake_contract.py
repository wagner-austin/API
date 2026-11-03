"""What a session host's ``codex-wake.py prepare`` prints, decoded (MCPs board task 562cfa7a).

One concern: the line the session host's half of the Codex wake answers
with, which MCPs ``packages/claude-hooks/codex_wake/deliver.py`` encodes
(``encode_answer``). The hooks run on the host's system interpreter and
this package on the hub's, so the line is the contract between two
programs, and each side owns its codec; a field either side renames fails
this decoder by name rather than reading as no wake.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_int,
    require_bool,
    require_list,
    require_str,
)


class WakeEvent(TypedDict):
    """One board row a wake carries.

    Attributes:
        at: Its instant, as the board printed it.
        kind: Its kind.
        author: The label that wrote it.
        task: The task it sits on, as the feed prints it.
    """

    at: str
    kind: str
    author: str
    task: str


class PreparedWake(TypedDict):
    """A pending wake.

    Attributes:
        key: The key to start the turn under.
        agent: The session's board label.
        new: True when this prepare selected it, False when an earlier one did.
        prepared_at: When it was selected.
        events: The rows it carries.
        message_ids: The board messages it carries.
        text: The turn's input.
    """

    key: str
    agent: str
    new: bool
    prepared_at: str
    events: list[WakeEvent]
    message_ids: list[int]
    text: str


class OpenDelivery(TypedDict):
    """An accepted wake whose turn has not been seen to end.

    Attributes:
        key: The wake's key.
        turn_id: The turn to read.
        agent: The session's board label when the wake was accepted.
    """

    key: str
    turn_id: str
    agent: str


class PrepareAnswer(TypedDict):
    """The whole line.

    Attributes:
        wake: The pending wake, or ``None``.
        open: The open deliveries.
    """

    wake: PreparedWake | None
    open: list[OpenDelivery]


def _event(value: JSONValue) -> WakeEvent:
    """Decode one carried row.

    Args:
        value: One element of ``events``.

    Returns:
        The row.

    Raises:
        JSONTypeError: When it is not an object of four text fields.
    """
    raw = narrow_json_to_dict(value)
    return WakeEvent(
        at=require_str(raw, "at"),
        kind=require_str(raw, "kind"),
        author=require_str(raw, "author"),
        task=require_str(raw, "task"),
    )


def _wake(value: JSONValue) -> PreparedWake | None:
    """Decode the ``wake`` field.

    Args:
        value: The field.

    Returns:
        The wake, or ``None`` for JSON null.

    Raises:
        JSONTypeError: A missing or mistyped field.
    """
    if value is None:
        return None
    raw = narrow_json_to_dict(value)
    return PreparedWake(
        key=require_str(raw, "key"),
        agent=require_str(raw, "agent"),
        new=require_bool(raw, "new"),
        prepared_at=require_str(raw, "prepared_at"),
        events=[_event(item) for item in require_list(raw, "events")],
        message_ids=[narrow_json_to_int(item) for item in require_list(raw, "message_ids")],
        text=require_str(raw, "text"),
    )


def decode_prepare(output: str) -> PrepareAnswer:
    """Decode ``prepare``'s standard output.

    Args:
        output: The one line it printed, with its newline.

    Returns:
        The answer.

    Raises:
        InvalidJsonError: When it is not JSON.
        JSONTypeError: When a field is missing or mistyped.
    """
    raw = narrow_json_to_dict(load_json_str(output))
    if "wake" not in raw:
        # A line without the field is another program's, never "no wake".
        raise JSONTypeError("codex-wake prepare: field 'wake' is missing")
    return PrepareAnswer(
        wake=_wake(raw["wake"]),
        open=[_open(item) for item in require_list(raw, "open")],
    )


def _open(value: JSONValue) -> OpenDelivery:
    """Decode one open delivery.

    Args:
        value: One element of ``open``.

    Returns:
        The delivery.

    Raises:
        JSONTypeError: When it is not an object of three text fields.
    """
    raw = narrow_json_to_dict(value)
    return OpenDelivery(
        key=require_str(raw, "key"),
        turn_id=require_str(raw, "turn_id"),
        agent=require_str(raw, "agent"),
    )


__all__ = ["OpenDelivery", "PrepareAnswer", "PreparedWake", "WakeEvent", "decode_prepare"]
