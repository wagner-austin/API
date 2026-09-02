"""Mine the archive for each message family's recipient policy.

Which connections receive a given server message? A single-client sim
cannot tell "broadcast to the room" from "send to this client" — both
produce identical output — so every unconditional emission is an
undecided ruling until the archive decides it. Getting one wrong builds
a server that looks correct at one client and leaks another player's
private receipts at two ([[recipient-policy]]).

Two tests, either sufficient to establish broadcast:

* **Zero-trigger arrival.** The family arrives in a session where the
  client sent ZERO of the command that would trigger it. A
  per-recipient family cannot do this.
* **Foreign actor.** The family names an actor tank other than the
  receiving session's own tank. Only families carrying a tank id can
  be tested this way; today that is 0x42 BuildPickup.

Absent either across the whole archive, the family is per-recipient.
The own tank is the session's first 0x21 TankInfo, which is the archive
convention the audit validators key self-attribution on
([[session-state-deglobalisation]]).

The zero-trigger test has a limit worth stating, because ignoring it
produces a confidently wrong answer. It asks whether a family answers
the client's own COMMAND, which is the recipient question only for
families a command can trigger. 0x74 rides the JOIN burst instead, so
its 325 zero-trigger sessions say "not command-triggered", not
"broadcast" — and this sweep reports :data:`VERDICT_UNDETERMINED` for
it rather than a verdict the evidence does not support. Its policy is
settled structurally in [[recipient-policy]].

The session walk is NOT re-implemented here: :mod:`analysis.scan` owns
the load-XOR-split-decode pipeline and its typed skips, and this module
consumes them.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

from tankpit_bot.analysis.recipient_policy_types import (
    VERDICT_BROADCAST,
    VERDICT_PER_RECIPIENT,
    VERDICT_UNDETERMINED,
    FamilyCountDict,
    FamilyEvidenceDict,
    RecipientPolicyDict,
    SessionEvidenceDict,
)
from tankpit_bot.analysis.scan import scan_archive
from tankpit_bot.analysis.types import DecodedFrameDict, ScannedSessionDict
from tankpit_bot.protocol import try_decode_binary_message
from tankpit_bot.protocol.commands import COMMAND_PREFIX
from tankpit_bot.sim.commands import decode_client_command
from tankpit_bot.wire.helpers import DecodeError

#: Declared ``Final`` so mypy carries the literal type: the
#: ``BinaryMessage`` union is discriminated on ``msg_type``, and
#: narrowing it to the one member that owns ``tank_id`` needs a literal
#: on the right of the comparison. A plain ``int`` would force a cast,
#: which the codebase does not permit.
_TANK_INFO: Final = 0x21
_BUILD_PICKUP: Final = 0x42

#: Each family, its label, and the client command kind that would
#: trigger it if the family were per-recipient. Order is the report
#: order and the order of every ``families`` list this module builds.
#:
#: An EMPTY trigger means no client command produces the family, so the
#: zero-trigger test cannot speak to it. 0x74 is the case: it rides the
#: join burst, one per session, and 325 of 341 sessions receive it
#: having sent no toggle -- which says "not command-triggered", NOT
#: "broadcast". Its recipient policy is settled structurally instead
#: (no tank id, describes the recipient's own loadout) in
#: [[recipient-policy]], which is reasoning this sweep cannot do.
FAMILY_TABLE: tuple[tuple[int | str, str, str], ...] = (
    (0x42, "0x42 BuildPickup", "block"),
    (0x4A, "0x4A TerrainUpdate", "block"),
    (0x74, "0x74 EquipmentToggle", ""),
    (0x4F, "0x4F RadarScanResult", "radar"),
    (0x46, "0x46 RadarResult", "radar"),
    (0x4C, "0x4C MapData", "map_open"),
    ("teleport_landed", "TeleportLanded", "teleport"),
)


def _own_tank_id(frames: list[DecodedFrameDict]) -> int | None:
    """Find the tank id the session's FIRST 0x21 TankInfo names.

    The archive convention is that a session's first TankInfo names the
    player's own tank — the same convention ``validate.wire_timeline``
    keys self-attribution on.

    Args:
        frames: Every decoded frame of one session, in capture order.

    Returns:
        The own tank id, or None when the session carried no decodable
        0x21 at all. None disables the foreign-actor test rather than
        silently comparing against a fabricated id.
    """
    for frame in frames:
        if frame["direction"] != "received":
            continue
        try:
            message = try_decode_binary_message(frame["msg_type"], frame["body"])
        except DecodeError:
            # A short frame of a known type is archive noise, not the
            # identity: 101 of the archive's 262,588 received frames
            # are short 0x41s, and any one of them can precede the
            # roster dump. :func:`sweep_session` counts them.
            continue
        if message is None:
            continue
        # The DECODED type decides, not the frame's leading byte: a
        # pre-filter on the byte would make this test permanently true
        # and leave a branch no test could reach.
        if message["msg_type"] == _TANK_INFO:
            return message["tank_id"]
    return None


def sweep_session(scanned: ScannedSessionDict) -> SessionEvidenceDict:
    """Tally one decoded session's recipient-policy evidence.

    Args:
        scanned: One session's decoded frames, from
            :func:`analysis.scan.scan_session`.

    Returns:
        The session's per-family tallies, its own tank id, and the two
        skip counts. Both are COUNTED, never dropped silently — an
        unreported skip would understate every ``received`` tally it
        hides.

        The two are kept apart because they say different things about
        the archive, the same split ``capture.protocol_census`` draws:
        an unknown message type means the decoder has a gap, while a
        known type whose body fails validation means the capture holds
        a short frame. Catching the validation error here classifies
        it; it does not soften anything, and a census that died on the
        101 short 0x41s in the archive would measure nothing at all.
    """
    own_id = _own_tank_id(scanned["frames"])
    received: Counter[int | str] = Counter()
    triggers: Counter[str] = Counter()
    foreign: Counter[int | str] = Counter()
    undecodable = 0
    malformed = 0

    for frame in scanned["frames"]:
        if frame["direction"] == "sent":
            if frame["msg_type"] != COMMAND_PREFIX:
                continue
            triggers[decode_client_command(frame["body"])["kind"]] += 1
            continue
        try:
            message = try_decode_binary_message(frame["msg_type"], frame["body"])
        except DecodeError:
            malformed += 1
            continue
        if message is None:
            undecodable += 1
            continue
        received[message["msg_type"]] += 1
        # Tested against the literal directly: narrowing the union
        # through an intermediate variable does not propagate to
        # ``message``, and ``tank_id`` exists on one member only.
        if (
            message["msg_type"] == _BUILD_PICKUP
            and own_id is not None
            and message["tank_id"] != own_id
        ):
            foreign[_BUILD_PICKUP] += 1

    families = [
        FamilyCountDict(
            label=label,
            received=received[key],
            own_triggers=triggers[trigger],
            foreign_actor_hits=foreign[key],
        )
        for key, label, trigger in FAMILY_TABLE
    ]
    return SessionEvidenceDict(
        session_id=scanned["session_id"],
        own_tank_id=0 if own_id is None else own_id,
        identified_own_tank=own_id is not None,
        families=families,
        framing_errors=0,
        undecodable_frames=undecodable,
        malformed_frames=malformed,
    )


def _verdict(trigger: str, zero_trigger_sessions: int, foreign_actor_hits: int) -> str:
    """Rule on one family from its evidence.

    A foreign actor settles broadcast outright — it is direct proof
    another connection's action reached this one. Otherwise a
    zero-trigger arrival settles broadcast, but ONLY for a family a
    client command can trigger: with no such command the test measures
    nothing about recipients and the family is left undetermined for a
    human to settle structurally.

    Args:
        trigger: The triggering client command kind, empty when no
            command produces the family.
        zero_trigger_sessions: Sessions that received it having sent no
            triggering command.
        foreign_actor_hits: Arrivals naming another tank as the actor.

    Returns:
        One of :data:`VERDICT_BROADCAST`, :data:`VERDICT_PER_RECIPIENT`
        or :data:`VERDICT_UNDETERMINED`.
    """
    if foreign_actor_hits > 0:
        return VERDICT_BROADCAST
    if not trigger:
        return VERDICT_UNDETERMINED
    if zero_trigger_sessions > 0:
        return VERDICT_BROADCAST
    return VERDICT_PER_RECIPIENT


def merge_session_evidence(
    entries: list[SessionEvidenceDict],
    *,
    sessions_examined: int,
    sessions_without_magic: int,
    framing_errors: int,
) -> RecipientPolicyDict:
    """Roll per-session tallies up into the archive-wide verdicts.

    A family is :data:`VERDICT_BROADCAST` the moment ANY session
    received it having sent zero triggering commands, or ANY session
    saw it name a foreign actor. One positive sample settles broadcast;
    per-recipient is the verdict only when the whole archive is silent.

    Args:
        entries: One entry per decoded session, each carrying its
            families in :data:`FAMILY_TABLE` order.
        sessions_examined: Sessions handed to the sweep, decoded or not.
        sessions_without_magic: Sessions skipped for want of an XOR
            magic — kept so the denominator stays honest.
        framing_errors: Sessions skipped for a framing-contract
            violation.

    Returns:
        The archive-wide evidence and verdict for every family.
    """
    families: list[FamilyEvidenceDict] = []
    for index, (_key, label, trigger) in enumerate(FAMILY_TABLE):
        received = 0
        own_triggers = 0
        zero_trigger_sessions = 0
        foreign_actor_hits = 0
        for entry in entries:
            tally = entry["families"][index]
            received += tally["received"]
            own_triggers += tally["own_triggers"]
            foreign_actor_hits += tally["foreign_actor_hits"]
            if tally["received"] > 0 and tally["own_triggers"] == 0:
                zero_trigger_sessions += 1
        families.append(
            FamilyEvidenceDict(
                label=label,
                trigger_kind=trigger,
                received=received,
                own_triggers=own_triggers,
                zero_trigger_sessions=zero_trigger_sessions,
                foreign_actor_hits=foreign_actor_hits,
                verdict=_verdict(trigger, zero_trigger_sessions, foreign_actor_hits),
            )
        )
    return RecipientPolicyDict(
        sessions_examined=sessions_examined,
        sessions_decoded=len(entries),
        sessions_without_magic=sessions_without_magic,
        framing_errors=framing_errors,
        undecodable_frames=sum(entry["undecodable_frames"] for entry in entries),
        malformed_frames=sum(entry["malformed_frames"] for entry in entries),
        families=families,
    )


def analyze_recipient_policy(directories: list[Path]) -> RecipientPolicyDict:
    """Sweep every capture session under the given directories.

    Args:
        directories: Directories holding ``*.capture_session.json``
            files, swept in the order given.

    Returns:
        The archive-wide recipient-policy evidence and verdicts.

    Raises:
        OSError: If a session file cannot be read.
        InvalidJsonError: If a session file is not valid JSON.
        JSONTypeError: If a session file is not a capture session.
    """
    entries: list[SessionEvidenceDict] = []
    examined = 0
    without_magic = 0
    framing_errors = 0
    for directory in directories:
        for result in scan_archive(directory):
            examined += 1
            if result["kind"] == "scanned":
                entries.append(sweep_session(result))
            elif result["reason"] == "unframed_payload":
                framing_errors += 1
            else:
                without_magic += 1
    return merge_session_evidence(
        entries,
        sessions_examined=examined,
        sessions_without_magic=without_magic,
        framing_errors=framing_errors,
    )


def format_recipient_policy(result: RecipientPolicyDict) -> str:
    """Format the sweep as a readable report.

    Args:
        result: The sweep result.

    Returns:
        Multi-line human-readable summary, verdicts included.
    """
    lines = [
        f"sessions_examined={result['sessions_examined']}",
        f"sessions_decoded={result['sessions_decoded']}",
        f"sessions_without_magic={result['sessions_without_magic']}",
        f"framing_errors={result['framing_errors']}",
        f"undecodable_frames={result['undecodable_frames']}",
        f"malformed_frames={result['malformed_frames']}",
        "families:",
    ]
    for family in result["families"]:
        lines.append(
            f"  {family['label']} -> {family['verdict']}"
            f" (received={family['received']}"
            f" trigger={family['trigger_kind']} x{family['own_triggers']}"
            f" zero_trigger_sessions={family['zero_trigger_sessions']}"
            f" foreign_actor_hits={family['foreign_actor_hits']})"
        )
    return "\n".join(lines)


__all__ = [
    "FAMILY_TABLE",
    "analyze_recipient_policy",
    "format_recipient_policy",
    "merge_session_evidence",
    "sweep_session",
]
