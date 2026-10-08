"""Stating the fleet audit's operator asks to the board (MCPs board task 1acbf53e).

The fleet audit (MCPs ``fleet-mcp/src/operator-asks.ts``) writes
``operator-asks.json`` beside the health journal on every run: the whole set
of asks it holds open for the operator under one producer (``fleet:disk``,
a node whose disk is below its floor). That file is the writer's contract
and :func:`decode_statement` mirrors ``encodeFleetAskStatement`` field for
field.

STATED ONLY WHEN IT CHANGED. The audit rewrites the file every twenty
minutes with a new ``at``, and the board's writer (``task_ask_sync``) keeps an
unchanged ask without a write, but every call is still a ledger checkin and
a tool call. So the bridge records the producer and asks it last stated, in
``operator-asks.json.fleet-health-wake-stated.json`` beside the file, and
states again only when they differ: a node going below its floor, coming
back above it, or the audit changing an ask's words.

THE MARKER IS WRITTEN AFTER THE BOARD ANSWERS, the journal cursor's order: a
crash between the two states the same set again, which the writer keeps
unchanged, rather than never stating a change.

AN ABSENT FILE IS SAID, NOT GUESSED AT: an audit that has not yet run since
it learned to write the file has stated nothing, so the bridge states
nothing and says so in its report line.
"""

from __future__ import annotations

import pathlib
from typing import Final

from platform_core.board import register_service_session
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_bytes,
    narrow_json_to_dict,
    require_bool,
    require_list,
    require_str,
)
from platform_core.mcp_client import McpCredentials, call_mcp_tool
from typing_extensions import TypedDict

from fleet_health_wake import _test_hooks
from fleet_health_wake.identity import HARNESS, IDENTITY, PURPOSE

#: The audit's statement file, beside the health journal in ``fleet-mcp/state``.
OPERATOR_ASKS_FILENAME: Final = "operator-asks.json"

#: The suffix of the marker recording what this bridge last stated.
STATED_SUFFIX: Final = ".fleet-health-wake-stated.json"

#: The board tool a producer's whole set of asks is stated with.
ASK_SYNC_TOOL: Final = "task_ask_sync"


class AskSpec(TypedDict):
    """One ask, as the audit states it and the board's writer takes it.

    Attributes:
        kind: The ask's kind, which decides whether the operator is texted
            or a board task is filed (MCPs board task db98562b); the audit
            states ``decision``, filed in its room.
        key: The producer's stable name for it.
        subjectTaskId: The task it is about, or None.
        subjectNode: The fleet node it is about, or None.
        urgent: Whether a texted kind is texted at once; False for a filed one.
        cause: The failure it stems from, shared by every ask it causes, or None.
        room: The room a filed kind naming no task is filed in, or None.
        title: A texted kind's one line on the operator's phone (MCPs board
            task e6514287), or None for a filed kind, which the audit's
            decision is.
        what: What happened, with no reading in it.
        why: Why it needs the operator.
        action: The one thing to do.
    """

    kind: str
    key: str
    subjectTaskId: str | None
    subjectNode: str | None
    urgent: bool
    cause: str | None
    room: str | None
    title: str | None
    what: str
    why: str
    action: str


class AskStatement(TypedDict):
    """One audit run's whole statement.

    Attributes:
        source: The producer the asks belong to.
        at: The snapshot's capture time it was made from.
        asks: Every ask open now.
    """

    source: str
    at: str
    asks: tuple[AskSpec, ...]


def _nullable_str(obj: JSONObject, key: str) -> str | None:
    """Read a field the writer always writes, as a string or null.

    Args:
        obj: The ask's object.
        key: The field.

    Returns:
        The string, or None for a JSON null.

    Raises:
        JSONTypeError: When the field is absent or neither a string nor null.
    """
    if key not in obj:
        raise JSONTypeError(f"operator ask field '{key}' is absent; the audit always writes it")
    value = obj[key]
    if value is None or isinstance(value, str):
        return value
    raise JSONTypeError(
        f"operator ask field '{key}' must be a string or null, got {type(value).__name__}"
    )


def decode_ask_spec(value: JSONValue) -> AskSpec:
    """Decode one ask.

    Args:
        value: The parsed JSON.

    Returns:
        The validated ask.

    Raises:
        JSONTypeError: When it is not an object or a field is absent or mistyped.
    """
    obj = narrow_json_to_dict(value)
    return AskSpec(
        kind=require_str(obj, "kind"),
        key=require_str(obj, "key"),
        subjectTaskId=_nullable_str(obj, "subjectTaskId"),
        subjectNode=_nullable_str(obj, "subjectNode"),
        urgent=require_bool(obj, "urgent"),
        cause=_nullable_str(obj, "cause"),
        room=_nullable_str(obj, "room"),
        title=_nullable_str(obj, "title"),
        what=require_str(obj, "what"),
        why=require_str(obj, "why"),
        action=require_str(obj, "action"),
    )


def decode_statement(value: JSONValue) -> AskStatement:
    """Decode the audit's statement.

    Args:
        value: The parsed JSON of ``operator-asks.json``.

    Returns:
        The validated statement.

    Raises:
        JSONTypeError: When it is not the shape ``encodeFleetAskStatement`` writes.
    """
    obj = narrow_json_to_dict(value)
    return AskStatement(
        source=require_str(obj, "source"),
        at=require_str(obj, "at"),
        asks=tuple(decode_ask_spec(ask) for ask in require_list(obj, "asks")),
    )


def encode_ask_spec(ask: AskSpec) -> JSONObject:
    """Encode one ask in the board's spelling.

    Args:
        ask: The ask.

    Returns:
        The arguments object ``task_ask_sync`` takes per ask.
    """
    return {
        "kind": ask["kind"],
        "key": ask["key"],
        "subjectTaskId": ask["subjectTaskId"],
        "subjectNode": ask["subjectNode"],
        "urgent": ask["urgent"],
        "cause": ask["cause"],
        "room": ask["room"],
        "title": ask["title"],
        "what": ask["what"],
        "why": ask["why"],
        "action": ask["action"],
    }


def stated_text(statement: AskStatement) -> str:
    """What the marker records: the producer and its asks, without the run's time.

    Args:
        statement: The statement.

    Returns:
        Compact JSON, equal for two statements exactly when stating either
        tells the board the same thing.
    """
    asks: list[JSONValue] = [encode_ask_spec(ask) for ask in statement["asks"]]
    return dump_json_str({"source": statement["source"], "asks": asks})


def stated_path(statement_path: pathlib.Path) -> pathlib.Path:
    """The marker beside the statement file.

    Args:
        statement_path: ``operator-asks.json``'s path.

    Returns:
        The marker's path.
    """
    return statement_path.with_name(statement_path.name + STATED_SUFFIX)


def state_asks(statement_path: pathlib.Path, credentials: McpCredentials) -> None:
    """State the audit's asks to the board when they differ from the last statement.

    Args:
        statement_path: ``operator-asks.json``'s path.
        credentials: Endpoint and both board secrets.

    Raises:
        AppError: The board refusing the checkin or the statement; the
            marker is then left as it was, so the next cycle states again.
        JSONTypeError: A statement or marker that does not decode.
        InvalidJsonError: A statement that is not JSON.
        OSError: A file that cannot be read or written.
    """
    if not _test_hooks.file_exists(statement_path):
        _test_hooks.emit(f"operator asks: no statement at {statement_path} yet; nothing stated")
        return
    statement = decode_statement(load_json_bytes(_test_hooks.read_bytes(statement_path)))
    text = stated_text(statement)
    marker = stated_path(statement_path)
    count = len(statement["asks"])
    source = statement["source"]
    if _test_hooks.file_exists(marker) and _test_hooks.read_bytes(marker).decode("utf-8") == text:
        _test_hooks.emit(f"operator asks: {count} open under {source}, unchanged; nothing stated")
        return
    register_service_session(
        _test_hooks.http_post, credentials, IDENTITY, harness=HARNESS, purpose=PURPOSE
    )
    asks: list[JSONValue] = [encode_ask_spec(ask) for ask in statement["asks"]]
    answer = call_mcp_tool(
        _test_hooks.http_post,
        credentials,
        ASK_SYNC_TOOL,
        {
            "agent": IDENTITY["agent"],
            "sessionId": IDENTITY["session_id"],
            "cwd": IDENTITY["cwd"],
            "source": source,
            "asks": asks,
        },
    )
    _test_hooks.write_text(marker, text)
    _test_hooks.emit(f"operator asks: stated {count} under {source}: {answer}")


__all__ = [
    "ASK_SYNC_TOOL",
    "OPERATOR_ASKS_FILENAME",
    "STATED_SUFFIX",
    "AskSpec",
    "AskStatement",
    "decode_ask_spec",
    "decode_statement",
    "encode_ask_spec",
    "state_asks",
    "stated_path",
    "stated_text",
]
