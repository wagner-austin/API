"""The typed shapes of a capture-replay conformance run, and their codecs.

A conformance run (:mod:`tankpit_bot.validate.conformance`) replays every
archived capture through the sim one server tick at a time and compares
the self-caused wire the sim emits with the wire the real server sent in
the same tick. These are the records it produces: per-session totals, the
sessions it could not replay and why, per-command-group match rates, and
the divergences themselves with an example to go and look at. The report
is written to disk as JSON, so each record has an encoder and a
validating decoder.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    narrow_json_to_str,
    require_int,
    require_list,
    require_str,
)

from tankpit_bot.analysis.types import SessionSkipReason


class ReplaySkipReason(StrEnum):
    """Why a capture could not be replayed. A closed vocabulary.

    The first two are the archive scan's own reasons
    (:class:`~tankpit_bot.analysis.types.SessionSkipReason`), carried by
    value so a tally here reads the same as one there; the other three
    are what a replay needs beyond a decodable capture.
    """

    NO_MAGIC = SessionSkipReason.NO_MAGIC.value
    UNFRAMED_PAYLOAD = SessionSkipReason.UNFRAMED_PAYLOAD.value
    NO_ROOM = "no_room"
    """The capture never joined a known room, so it names no field."""
    FIELD_MISSING = "field_missing"
    """The joined room's field has no terrain minimap in this distribution."""
    NO_SELF = "no_self"
    """The capture never placed its own tank, so there is nothing to replay."""


class SessionResultDict(TypedDict):
    """One replayed capture's totals.

    Attributes:
        session: The capture's path, as given to the run.
        field: The field image the capture played (``field01.gif``).
        ticks: Server ticks compared (a tick the client sent a command into).
        matched: Ticks whose sim wire equalled the archive's.
        unmodelled: Ticks not compared because a command had no sim law.
    """

    session: str
    field: str
    ticks: int
    matched: int
    unmodelled: int


class SkippedSessionDict(TypedDict):
    """A capture the run could not replay.

    Attributes:
        session: The capture's path.
        reason: Why.
        detail: What exactly was missing.
    """

    session: str
    reason: ReplaySkipReason
    detail: str


class GroupTallyDict(TypedDict):
    """Match rate for one command group.

    Attributes:
        commands: The commands sent into the tick, ``+``-joined in send
            order (``teleport`` or ``scope+map_open``).
        ticks: Ticks with exactly this group.
        matched: Of those, ticks whose sim wire equalled the archive's.
    """

    commands: str
    ticks: int
    matched: int


class DivergenceDict(TypedDict):
    """One (group, live shape, sim shape) disagreement and where it happened.

    Attributes:
        commands: The command group.
        live: The self-caused tokens the real server sent that tick.
        sim: The self-caused tokens the sim emitted for the same tick.
        count: Ticks with exactly this disagreement.
        example_session: A capture holding one of them.
        example_timestamp_ms: The archived tick's time in that capture.
    """

    commands: str
    live: list[str]
    sim: list[str]
    count: int
    example_session: str
    example_timestamp_ms: int


class ConformanceReportDict(TypedDict):
    """A whole run.

    Attributes:
        sessions: Every replayed capture's totals, in run order.
        skipped: Every capture that could not be replayed.
        groups: Match rate per command group, most ticks first.
        divergences: Every disagreement, most frequent first.
    """

    sessions: list[SessionResultDict]
    skipped: list[SkippedSessionDict]
    groups: list[GroupTallyDict]
    divergences: list[DivergenceDict]


def _require_skip_reason(obj: JSONObject) -> ReplaySkipReason:
    """Read a skip reason and refuse anything outside the closed vocabulary.

    Args:
        obj: The skipped-session object.

    Returns:
        The reason.

    Raises:
        JSONTypeError: If ``reason`` is missing or not a known code.
    """
    reason = require_str(obj, "reason")
    for known in ReplaySkipReason:
        if known.value == reason:
            return known
    raise JSONTypeError(f"CONFORMANCE_SKIP_REASON: {reason!r} is not a skip reason")


def _require_tokens(obj: JSONObject, key: str) -> list[str]:
    """Read a list of shape tokens.

    Args:
        obj: The object holding them.
        key: The field name.

    Returns:
        The tokens.

    Raises:
        JSONTypeError: If the field is missing, not a list, or holds a
            non-string.
    """
    return [narrow_json_to_str(token) for token in require_list(obj, key)]


def encode_report(report: ConformanceReportDict) -> JSONObject:
    """Encode a run for writing.

    Args:
        report: The run.

    Returns:
        Its JSON object.
    """
    sessions: list[JSONValue] = [
        {
            "session": s["session"],
            "field": s["field"],
            "ticks": s["ticks"],
            "matched": s["matched"],
            "unmodelled": s["unmodelled"],
        }
        for s in report["sessions"]
    ]
    skipped: list[JSONValue] = [
        {"session": s["session"], "reason": s["reason"].value, "detail": s["detail"]}
        for s in report["skipped"]
    ]
    groups: list[JSONValue] = [
        {"commands": g["commands"], "ticks": g["ticks"], "matched": g["matched"]}
        for g in report["groups"]
    ]
    divergences: list[JSONValue] = []
    for d in report["divergences"]:
        live: list[JSONValue] = list(d["live"])
        sim: list[JSONValue] = list(d["sim"])
        divergences.append(
            {
                "commands": d["commands"],
                "live": live,
                "sim": sim,
                "count": d["count"],
                "example_session": d["example_session"],
                "example_timestamp_ms": d["example_timestamp_ms"],
            }
        )
    return {
        "sessions": sessions,
        "skipped": skipped,
        "groups": groups,
        "divergences": divergences,
    }


def decode_report(data: JSONObject) -> ConformanceReportDict:
    """Decode and validate a written run.

    Args:
        data: The JSON object :func:`encode_report` wrote.

    Returns:
        The run.

    Raises:
        JSONTypeError: If any field is missing or of the wrong type.
    """
    sessions: list[SessionResultDict] = []
    for raw in require_list(data, "sessions"):
        obj = narrow_json_to_dict(raw)
        sessions.append(
            SessionResultDict(
                session=require_str(obj, "session"),
                field=require_str(obj, "field"),
                ticks=require_int(obj, "ticks"),
                matched=require_int(obj, "matched"),
                unmodelled=require_int(obj, "unmodelled"),
            )
        )
    skipped: list[SkippedSessionDict] = []
    for raw in require_list(data, "skipped"):
        obj = narrow_json_to_dict(raw)
        skipped.append(
            SkippedSessionDict(
                session=require_str(obj, "session"),
                reason=_require_skip_reason(obj),
                detail=require_str(obj, "detail"),
            )
        )
    groups: list[GroupTallyDict] = []
    for raw in require_list(data, "groups"):
        obj = narrow_json_to_dict(raw)
        groups.append(
            GroupTallyDict(
                commands=require_str(obj, "commands"),
                ticks=require_int(obj, "ticks"),
                matched=require_int(obj, "matched"),
            )
        )
    divergences: list[DivergenceDict] = []
    for raw in require_list(data, "divergences"):
        obj = narrow_json_to_dict(raw)
        divergences.append(
            DivergenceDict(
                commands=require_str(obj, "commands"),
                live=_require_tokens(obj, "live"),
                sim=_require_tokens(obj, "sim"),
                count=require_int(obj, "count"),
                example_session=require_str(obj, "example_session"),
                example_timestamp_ms=require_int(obj, "example_timestamp_ms"),
            )
        )
    return ConformanceReportDict(
        sessions=sessions, skipped=skipped, groups=groups, divergences=divergences
    )


__all__ = [
    "ConformanceReportDict",
    "DivergenceDict",
    "GroupTallyDict",
    "ReplaySkipReason",
    "SessionResultDict",
    "SkippedSessionDict",
    "decode_report",
    "encode_report",
]
