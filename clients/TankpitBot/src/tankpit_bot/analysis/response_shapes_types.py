"""Data contract for the response-shape differ.

A RESPONSE SHAPE is the ordered tuple of self-caused message tokens the
server answered one client command with. Comparing the distribution of
those shapes between the real archive and a sim archive is what turns
"is the sim faithful?" into a mechanical question:

* a shape the REAL server produces that the sim never does is a
  MISSING law;
* a shape only the sim produces is an INVENTED law — the class that
  survives a divergence-zero soak, because the sim and its own tests
  can agree about a server neither has asked
  ([[capture-differ]], and the teleport equipment grant settled
  2026-09-01).

Every dict carries an encode/decode codec so a diff can be written to
an artifact and read back with validation.
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

#: The real server produced this shape and the sim never did.
VERDICT_MISSING_LAW = "missing_law"

#: Only the sim produced this shape. The archive never shows it.
VERDICT_INVENTED_LAW = "invented_law"

#: The row is an invented law; the missing-side triage does not apply.
CAUSE_SIM_ONLY = "sim_only"

#: The live command drew NOTHING in its window. The real server is
#: asynchronous, so an answer that arrives after the next command has
#: opened its own window records against that one and leaves this
#: window silent. A tick-synchronous sim answers inside the same
#: batch and cannot produce a silent window for a command it handles,
#: so these rows are a property of the two servers' timing, not a law
#: the sim is missing.
CAUSE_LIVE_SILENT_WINDOW = "live_silent_window"

#: The sim corpus never sent this command KIND at all, so nothing
#: about its answers has been sampled. A coverage hole in the
#: scenarios, not a law gap — fix it by making the bot send the
#: command, not by changing the server.
CAUSE_COMMAND_NEVER_SENT = "command_never_sent"

#: The live shape contains a token the sim never emits for this
#: command kind. Either a genuine gap or a window-overlap artifact
#: (a live window catching a concurrent action's messages); the token
#: itself says which is worth checking.
CAUSE_TOKEN_NEVER_EMITTED = "token_never_emitted"

#: Every token in the live shape is one the sim DOES emit for this
#: command — only the combination is unseen. These are the readable
#: candidates: the sim demonstrably has the parts, so either it
#: assembles them differently or the corpus never hit the branch.
CAUSE_SHAPE_NEVER_ASSEMBLED = "shape_never_assembled"


class CommandWindowDict(TypedDict):
    """One client command paired with the shape it drew.

    Attributes:
        command_kind: The decoded client command's kind.
        shape: The ordered self-caused tokens the server answered
            with, empty when the command drew nothing.
        timestamp_ms: Capture time of the command that opened the
            window, carried so a window can be traced back.
    """

    command_kind: str
    shape: list[str]
    timestamp_ms: int


class ShapeCountDict(TypedDict):
    """How often one shape answered one command kind.

    Attributes:
        shape: The ordered self-caused tokens.
        count: Windows that produced exactly this shape.
    """

    shape: list[str]
    count: int


class CommandShapesDict(TypedDict):
    """One command kind's whole shape distribution.

    Attributes:
        command_kind: The command kind.
        windows: Total windows observed for this kind.
        shapes: Every distinct shape, descending by count.
    """

    command_kind: str
    windows: int
    shapes: list[ShapeCountDict]


class ShapeDivergenceDict(TypedDict):
    """One shape present on exactly one side of the diff.

    Attributes:
        command_kind: The command kind whose window produced it.
        shape: The ordered self-caused tokens.
        live_count: Windows in the real archive with this shape.
        sim_count: Windows in the sim archive with this shape.
        verdict: :data:`VERDICT_MISSING_LAW` when only the real server
            produced it, :data:`VERDICT_INVENTED_LAW` when only the
            sim did.
        cause: Why a MISSING row is one-sided — one of the ``CAUSE_``
            constants. Always :data:`CAUSE_SIM_ONLY` for an invented
            row. The field exists because a flat missing list mixes
            three unrelated phenomena and reads as one: on 2026-09-02
            it reported 208 rows of which 9 were a timing artifact the
            sim cannot reproduce by construction, 77 were commands the
            corpus never sent, and only 30 were readable as candidate
            gaps ([[capture-differ]]).
    """

    command_kind: str
    shape: list[str]
    live_count: int
    sim_count: int
    verdict: str
    cause: str


class ResponseShapeDiffDict(TypedDict):
    """The whole live-vs-sim comparison.

    Attributes:
        live_sessions: Real capture sessions mined.
        sim_sessions: Sim capture sessions mined.
        live_windows: Command windows paired in the real archive.
        sim_windows: Command windows paired in the sim archive.
        divergences: Every one-sided shape, missing laws first, each
            group descending by the count that makes it notable.
    """

    live_sessions: int
    sim_sessions: int
    live_windows: int
    sim_windows: int
    divergences: list[ShapeDivergenceDict]


def encode_command_window(window: CommandWindowDict) -> JSONObject:
    """Encode one paired command window.

    Args:
        window: The window to encode.

    Returns:
        JSON object with every window field.
    """
    return {
        "command_kind": window["command_kind"],
        "shape": list(window["shape"]),
        "timestamp_ms": window["timestamp_ms"],
    }


def decode_command_window(data: JSONObject) -> CommandWindowDict:
    """Decode one paired command window with validation.

    Args:
        data: JSON object carrying the window fields.

    Returns:
        Validated window.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
    """
    return CommandWindowDict(
        command_kind=require_str(data, "command_kind"),
        shape=[_require_token(item) for item in require_list(data, "shape")],
        timestamp_ms=require_int(data, "timestamp_ms"),
    )


def encode_shape_count(entry: ShapeCountDict) -> JSONObject:
    """Encode one shape tally.

    Args:
        entry: The tally to encode.

    Returns:
        JSON object with the shape and its count.
    """
    return {"shape": list(entry["shape"]), "count": entry["count"]}


def decode_shape_count(data: JSONObject) -> ShapeCountDict:
    """Decode one shape tally with validation.

    Args:
        data: JSON object carrying the tally fields.

    Returns:
        Validated tally.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
    """
    return ShapeCountDict(
        shape=[_require_token(item) for item in require_list(data, "shape")],
        count=require_int(data, "count"),
    )


def encode_command_shapes(entry: CommandShapesDict) -> JSONObject:
    """Encode one command kind's distribution.

    Args:
        entry: The distribution to encode.

    Returns:
        JSON object with every distribution field.
    """
    return {
        "command_kind": entry["command_kind"],
        "windows": entry["windows"],
        "shapes": [encode_shape_count(shape) for shape in entry["shapes"]],
    }


def decode_command_shapes(data: JSONObject) -> CommandShapesDict:
    """Decode one command kind's distribution with validation.

    Args:
        data: JSON object carrying the distribution fields.

    Returns:
        Validated distribution.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
    """
    return CommandShapesDict(
        command_kind=require_str(data, "command_kind"),
        windows=require_int(data, "windows"),
        shapes=[
            decode_shape_count(narrow_json_to_dict(item)) for item in require_list(data, "shapes")
        ],
    )


def encode_shape_divergence(entry: ShapeDivergenceDict) -> JSONObject:
    """Encode one one-sided shape.

    Args:
        entry: The divergence to encode.

    Returns:
        JSON object with every divergence field.
    """
    return {
        "command_kind": entry["command_kind"],
        "shape": list(entry["shape"]),
        "live_count": entry["live_count"],
        "sim_count": entry["sim_count"],
        "verdict": entry["verdict"],
        "cause": entry["cause"],
    }


def decode_shape_divergence(data: JSONObject) -> ShapeDivergenceDict:
    """Decode one one-sided shape with validation.

    Args:
        data: JSON object carrying the divergence fields.

    Returns:
        Validated divergence.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
    """
    return ShapeDivergenceDict(
        command_kind=require_str(data, "command_kind"),
        shape=[_require_token(item) for item in require_list(data, "shape")],
        live_count=require_int(data, "live_count"),
        sim_count=require_int(data, "sim_count"),
        verdict=require_str(data, "verdict"),
        cause=require_str(data, "cause"),
    )


def encode_response_shape_diff(diff: ResponseShapeDiffDict) -> JSONObject:
    """Encode the whole comparison.

    Args:
        diff: The diff to encode.

    Returns:
        JSON object with every diff field.
    """
    return {
        "live_sessions": diff["live_sessions"],
        "sim_sessions": diff["sim_sessions"],
        "live_windows": diff["live_windows"],
        "sim_windows": diff["sim_windows"],
        "divergences": [encode_shape_divergence(row) for row in diff["divergences"]],
    }


def decode_response_shape_diff(data: JSONObject) -> ResponseShapeDiffDict:
    """Decode the whole comparison with validation.

    Args:
        data: JSON object carrying the diff fields.

    Returns:
        Validated diff.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
    """
    return ResponseShapeDiffDict(
        live_sessions=require_int(data, "live_sessions"),
        sim_sessions=require_int(data, "sim_sessions"),
        live_windows=require_int(data, "live_windows"),
        sim_windows=require_int(data, "sim_windows"),
        divergences=[
            decode_shape_divergence(narrow_json_to_dict(item))
            for item in require_list(data, "divergences")
        ],
    )


def _require_token(value: JSONValue) -> str:
    """Narrow one JSON list element to a shape token.

    Args:
        value: Candidate element from a decoded shape list.

    Returns:
        The token as a string.

    Raises:
        TypeError: If the element is not a string. A shape is an
            ordered list of tokens; anything else is a malformed
            artifact, not a token to coerce.
    """
    if not isinstance(value, str):
        raise TypeError(f"shape token must be a string, got {type(value).__name__}")
    return value


__all__ = [
    "CAUSE_COMMAND_NEVER_SENT",
    "CAUSE_LIVE_SILENT_WINDOW",
    "CAUSE_SHAPE_NEVER_ASSEMBLED",
    "CAUSE_SIM_ONLY",
    "CAUSE_TOKEN_NEVER_EMITTED",
    "VERDICT_INVENTED_LAW",
    "VERDICT_MISSING_LAW",
    "CommandShapesDict",
    "CommandWindowDict",
    "ResponseShapeDiffDict",
    "ShapeCountDict",
    "ShapeDivergenceDict",
    "decode_command_shapes",
    "decode_command_window",
    "decode_response_shape_diff",
    "decode_shape_count",
    "decode_shape_divergence",
    "encode_command_shapes",
    "encode_command_window",
    "encode_response_shape_diff",
    "encode_shape_count",
    "encode_shape_divergence",
]
