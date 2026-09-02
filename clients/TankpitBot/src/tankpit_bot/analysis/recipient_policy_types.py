"""Data contract for the recipient-policy sweep.

The sweep answers one question per message family: which connections
receive it. A single-client sim cannot tell "broadcast to the room"
from "send to this client" — both produce identical output — so every
unconditional emission is an undecided ruling until the archive decides
it ([[recipient-policy]]).

Every dict here carries an encode/decode codec so a sweep result can be
written to an artifact and read back with validation.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    narrow_json_to_dict,
    require_bool,
    require_int,
    require_list,
    require_str,
)

#: A family is BROADCAST once the archive shows it arriving without the
#: recipient having triggered it, or naming an actor that is not the
#: recipient. Absent either, it is per-recipient.
VERDICT_BROADCAST = "broadcast"
VERDICT_PER_RECIPIENT = "per_recipient"

#: The zero-trigger test asks whether a family answers the client's own
#: COMMAND. For a family no command triggers -- 0x74 rides the join
#: burst ([[recipient-policy]]) -- a zero-trigger arrival says only
#: "not command-triggered", which is not the recipient question. The
#: sweep reports that rather than emitting a verdict it cannot support.
VERDICT_UNDETERMINED = "undetermined"


class FamilyCountDict(TypedDict):
    """One family's tallies within ONE capture session.

    Attributes:
        label: Family label, e.g. ``"0x42 BuildPickup"``.
        received: Times the family arrived in this session.
        own_triggers: Times the client sent the command that would
            trigger it if the family were per-recipient.
        foreign_actor_hits: Times the family named an actor tank other
            than this session's own tank. Zero for families that carry
            no tank id.
    """

    label: str
    received: int
    own_triggers: int
    foreign_actor_hits: int


class SessionEvidenceDict(TypedDict):
    """One capture session's contribution to the sweep.

    Attributes:
        session_id: The capture's session id.
        own_tank_id: The tank id the session's first 0x21 names — the
            archive convention for the player's own tank
            ([[session-state-deglobalisation]]). Zero when the session
            carried no 0x21.
        identified_own_tank: Whether a first 0x21 was seen at all. The
            foreign-actor test is meaningless without it, so it is
            recorded rather than inferred from a zero id.
        families: Per-family tallies, in family-table order.
        framing_errors: Payloads that failed frame splitting.
        undecodable_frames: Frames NO decoder claims — an unknown
            message type. Counted, never silently dropped: an
            unreported skip would understate every ``received`` tally
            it hides.
        malformed_frames: Frames of a KNOWN type whose body failed the
            decoder's validation. Distinct from ``undecodable_frames``
            because the two say different things about the archive —
            the same split ``capture.protocol_census`` draws between
            ``unsupported`` and ``short_or_invalid``. Measured
            2026-09-01: 101 of 262,588 received frames, every one a
            short 0x41 Deactivation.
    """

    session_id: str
    own_tank_id: int
    identified_own_tank: bool
    families: list[FamilyCountDict]
    framing_errors: int
    undecodable_frames: int
    malformed_frames: int


class FamilyEvidenceDict(TypedDict):
    """One family's rolled-up evidence and the verdict it supports.

    Attributes:
        label: Family label.
        trigger_kind: The client command kind that would trigger this
            family if it were per-recipient.
        received: Times the family arrived across every swept session.
        own_triggers: Times the client sent the triggering command
            across every swept session.
        zero_trigger_sessions: Sessions that received the family having
            sent ZERO triggering commands. Any non-zero count is
            positive proof of broadcast.
        foreign_actor_hits: Times the family named an actor other than
            the receiving session's own tank. Also positive proof.
        verdict: :data:`VERDICT_BROADCAST` or
            :data:`VERDICT_PER_RECIPIENT`.
    """

    label: str
    trigger_kind: str
    received: int
    own_triggers: int
    zero_trigger_sessions: int
    foreign_actor_hits: int
    verdict: str


class RecipientPolicyDict(TypedDict):
    """The whole sweep, across every session examined.

    Attributes:
        sessions_examined: Sessions handed to the sweep.
        sessions_decoded: Sessions that carried a magic and were swept.
        sessions_without_magic: Sessions skipped for want of a magic —
            they cannot be XOR-decoded, and counting them keeps the
            denominator honest.
        framing_errors: Total payloads that failed frame splitting.
        undecodable_frames: Total frames of an unknown type.
        malformed_frames: Total frames of a known type whose body
            failed validation.
        families: Per-family evidence, in family-table order.
    """

    sessions_examined: int
    sessions_decoded: int
    sessions_without_magic: int
    framing_errors: int
    undecodable_frames: int
    malformed_frames: int
    families: list[FamilyEvidenceDict]


def encode_family_count(entry: FamilyCountDict) -> JSONObject:
    """Encode one per-session family tally.

    Args:
        entry: Tally to encode.

    Returns:
        JSON object with every tally field.
    """
    return {
        "label": entry["label"],
        "received": entry["received"],
        "own_triggers": entry["own_triggers"],
        "foreign_actor_hits": entry["foreign_actor_hits"],
    }


def decode_family_count(data: JSONObject) -> FamilyCountDict:
    """Decode one per-session family tally with validation.

    Args:
        data: JSON object carrying the tally fields.

    Returns:
        Validated tally.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
    """
    return FamilyCountDict(
        label=require_str(data, "label"),
        received=require_int(data, "received"),
        own_triggers=require_int(data, "own_triggers"),
        foreign_actor_hits=require_int(data, "foreign_actor_hits"),
    )


def encode_session_evidence(entry: SessionEvidenceDict) -> JSONObject:
    """Encode one session's contribution.

    Args:
        entry: Session evidence to encode.

    Returns:
        JSON object with every evidence field.
    """
    return {
        "session_id": entry["session_id"],
        "own_tank_id": entry["own_tank_id"],
        "identified_own_tank": entry["identified_own_tank"],
        "families": [encode_family_count(family) for family in entry["families"]],
        "framing_errors": entry["framing_errors"],
        "undecodable_frames": entry["undecodable_frames"],
        "malformed_frames": entry["malformed_frames"],
    }


def decode_session_evidence(data: JSONObject) -> SessionEvidenceDict:
    """Decode one session's contribution with validation.

    Args:
        data: JSON object carrying the evidence fields.

    Returns:
        Validated session evidence.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
    """
    return SessionEvidenceDict(
        session_id=require_str(data, "session_id"),
        own_tank_id=require_int(data, "own_tank_id"),
        identified_own_tank=require_bool(data, "identified_own_tank"),
        families=[
            decode_family_count(narrow_json_to_dict(item))
            for item in require_list(data, "families")
        ],
        framing_errors=require_int(data, "framing_errors"),
        undecodable_frames=require_int(data, "undecodable_frames"),
        malformed_frames=require_int(data, "malformed_frames"),
    )


def encode_family_evidence(entry: FamilyEvidenceDict) -> JSONObject:
    """Encode one family's rolled-up evidence.

    Args:
        entry: Family evidence to encode.

    Returns:
        JSON object with every evidence field.
    """
    return {
        "label": entry["label"],
        "trigger_kind": entry["trigger_kind"],
        "received": entry["received"],
        "own_triggers": entry["own_triggers"],
        "zero_trigger_sessions": entry["zero_trigger_sessions"],
        "foreign_actor_hits": entry["foreign_actor_hits"],
        "verdict": entry["verdict"],
    }


def decode_family_evidence(data: JSONObject) -> FamilyEvidenceDict:
    """Decode one family's rolled-up evidence with validation.

    Args:
        data: JSON object carrying the evidence fields.

    Returns:
        Validated family evidence.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
    """
    return FamilyEvidenceDict(
        label=require_str(data, "label"),
        trigger_kind=require_str(data, "trigger_kind"),
        received=require_int(data, "received"),
        own_triggers=require_int(data, "own_triggers"),
        zero_trigger_sessions=require_int(data, "zero_trigger_sessions"),
        foreign_actor_hits=require_int(data, "foreign_actor_hits"),
        verdict=require_str(data, "verdict"),
    )


def encode_recipient_policy(result: RecipientPolicyDict) -> JSONObject:
    """Encode the whole sweep result.

    Args:
        result: Sweep result to encode.

    Returns:
        JSON object with every sweep field.
    """
    return {
        "sessions_examined": result["sessions_examined"],
        "sessions_decoded": result["sessions_decoded"],
        "sessions_without_magic": result["sessions_without_magic"],
        "framing_errors": result["framing_errors"],
        "undecodable_frames": result["undecodable_frames"],
        "malformed_frames": result["malformed_frames"],
        "families": [encode_family_evidence(family) for family in result["families"]],
    }


def decode_recipient_policy(data: JSONObject) -> RecipientPolicyDict:
    """Decode the whole sweep result with validation.

    Args:
        data: JSON object carrying the sweep fields.

    Returns:
        Validated sweep result.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
    """
    return RecipientPolicyDict(
        sessions_examined=require_int(data, "sessions_examined"),
        sessions_decoded=require_int(data, "sessions_decoded"),
        sessions_without_magic=require_int(data, "sessions_without_magic"),
        framing_errors=require_int(data, "framing_errors"),
        undecodable_frames=require_int(data, "undecodable_frames"),
        malformed_frames=require_int(data, "malformed_frames"),
        families=[
            decode_family_evidence(narrow_json_to_dict(item))
            for item in require_list(data, "families")
        ],
    )


__all__ = [
    "VERDICT_BROADCAST",
    "VERDICT_PER_RECIPIENT",
    "FamilyCountDict",
    "FamilyEvidenceDict",
    "RecipientPolicyDict",
    "SessionEvidenceDict",
    "decode_family_count",
    "decode_family_evidence",
    "decode_recipient_policy",
    "decode_session_evidence",
    "encode_family_count",
    "encode_family_evidence",
    "encode_recipient_policy",
    "encode_session_evidence",
]
