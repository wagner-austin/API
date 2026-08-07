"""Capture replay cross-validation: raw wire bytes vs what the run acted on.

The check that falsified two month-old protocol claims (2026-07-19):
replaying the June 10 captures through the current decoder revealed 20
own-kill 0x41 deactivations and 21 0x52 empty-container errors that
the June-era decoder could not unwrap from their 0x2E envelopes -- the
"0x41 never fires for own kills" and "the wire is silent on failed
pickups" claims were decoder blind spots codified as server behavior.

This module makes that replay a standing audit:

* every received frame is decoded with the CURRENT decoder and the
  message census is compared against what the live run ingested --
  a wire message the bot received but did not act on is exactly the
  June class of bug, caught the day it appears;
* undecodable frames and unknown 0x2E subtypes are surfaced as the
  canary for the NEXT blind spot;
* the DOM game-log witness (recorded but never acted on) is diffed
  against the wire: a banner the client rendered with no matching wire
  message means the decoder is missing something the client can see.
"""

from __future__ import annotations

from typing import Literal

from platform_core.logging import get_logger

from tankpit_bot import protocol
from tankpit_bot.capture.frames import split_payload_frames
from tankpit_bot.capture.xor import XorStaticKeyUnavailableError, build_session_xor_table
from tankpit_bot.diagnostics.event_stream import scan_diagnostic_records
from tankpit_bot.diagnostics.run_audit_types import FindingDict, make_finding
from tankpit_bot.protocol.decoders import try_decode_plaintext_ack
from tankpit_bot.protocol.framing import FramingError
from tankpit_bot.runtime_logging import RuntimeEventRecordDict
from tankpit_bot.sniffer.constants import MSG_MIN_LENGTHS
from tankpit_bot.types import CaptureSession
from tankpit_bot.validate.fight_timeline import extract_human_episodes
from tankpit_bot.validate.shadow_timeline import extract_shadow_timeline
from tankpit_bot.wire.helpers import DecodeError

log = get_logger(__name__)

_KILL_BANNER_SUFFIX = "has been deactivated by you"
_EMPTY_CONTAINER_TEXT = "Empty container"
_BLOCKED_MOVE_TEXT = "You can't go there!"

_SUPERVISOR_EMPTY_CONTAINER = 4
_SUPERVISOR_CANT_GO = 1


def _xor_with_table(body: bytes, table: bytes) -> bytes:
    """XOR-decode a message body against a local table (skip type byte).

    Mirrors :func:`tankpit_bot.capture.xor.xor_decode_body` at
    ``offset=1``, but passes bytes beyond the table through in the
    clear instead of raising — an audit reads whatever the archive
    holds. Folding the two is tracked as its own step
    ([[session-state-deglobalisation]]).

    Args:
        body: Raw message body bytes including the leading type byte.
        table: XOR table built from the capture's magic.

    Returns:
        Decoded bytes without the type byte.
    """
    decoded = bytearray(len(body) - 1)
    for index in range(len(decoded)):
        if index < len(table):
            decoded[index] = body[index + 1] ^ table[index]
        else:
            decoded[index] = body[index + 1]
    return bytes(decoded)


def _replay_frame_bodies(capture: CaptureSession) -> list[bytes]:
    """Collect the received frame bodies the XOR replay should decode.

    Plaintext toggle acks travel un-XORed and carry no replay-relevant
    state — they are discriminated here, pre-XOR, so they never reach
    the decoder as garbage.

    Args:
        capture: Decoded capture session.

    Returns:
        Raw frame bodies in capture order, acks excluded.
    """
    bodies: list[bytes] = []
    for message in capture["messages"]:
        if message["direction"] != "received":
            continue
        # An audit reports what it cannot read rather than quietly
        # reading less — the private walk this replaced dropped a torn
        # tail without a word ([[session-state-deglobalisation]]).
        try:
            frames = split_payload_frames(message["payload"])
        except FramingError as error:
            log.warning("replay audit: skipping unparseable payload: %s", error)
            continue
        for body in frames:
            if try_decode_plaintext_ack(body) is None:
                bodies.append(body)
    return bodies


def _replay_received(
    capture: CaptureSession, table: bytes
) -> tuple[list[tuple[int, int]], list[int], dict[int, int], int]:
    """Decode every received frame with the current decoder.

    Args:
        capture: Decoded capture session.
        table: XOR table built from the capture's magic.

    Returns:
        ``(deactivations, supervisor_codes, unknown_subtypes,
        decode_errors)`` where ``deactivations`` is ``(victim_id,
        killer_id)`` per wire 0x41, ``supervisor_codes`` is the 0x52
        ``error_code`` list, ``unknown_subtypes`` maps undecoded 0x2E
        subtypes to counts, and ``decode_errors`` counts frames the
        decoder raised on.
    """
    deactivations: list[tuple[int, int]] = []
    supervisor_codes: list[int] = []
    unknown_subtypes: dict[int, int] = {}
    decode_errors = 0
    for body in _replay_frame_bodies(capture):
        decoded_data = _xor_with_table(body, table)
        if len(decoded_data) == 0:
            continue
        # Mirror the live router: only types the sniffer itself
        # would decode go through the decoder. Everything else is
        # outside the bot's decode surface by design (text routes,
        # lobby traffic) and is not a replay finding.
        min_len = MSG_MIN_LENGTHS.get(body[0])
        if min_len is None or len(decoded_data) < min_len:
            continue
        try:
            result = protocol.decode_message(body[0], decoded_data)
        except DecodeError as error:
            log.warning(
                "replay decode failure: type=0x%02X len=%d: %s",
                body[0],
                len(decoded_data),
                error,
            )
            decode_errors += 1
            continue
        match result:
            case {"msg_type": 0x41, "victim_id": int(victim_id), "killer_id": int(killer_id)}:
                deactivations.append((victim_id, killer_id))
            case {"msg_type": 0x52, "error_code": int(error_code)}:
                supervisor_codes.append(error_code)
            case {"msg_type": "unknown_container", "subtype": int(subtype)}:
                unknown_subtypes[subtype] = unknown_subtypes.get(subtype, 0) + 1
            case _:
                pass
    return (deactivations, supervisor_codes, unknown_subtypes, decode_errors)


def _own_tank_id(records: list[RuntimeEventRecordDict]) -> int:
    """Return the run's own tank id from the identity diagnostic, or -1."""
    _, identities = scan_diagnostic_records(records, "tank_identity")
    for record in identities:
        tank_id = record["fields"].get("tank_id")
        if isinstance(tank_id, int) and not isinstance(tank_id, bool):
            return tank_id
    return -1


def _channel_diff_finding(
    check: Literal["deactivation_channel_diff", "supervisor_channel_diff"],
    label: str,
    wire_count: int,
    ledger_count: int,
) -> FindingDict:
    """Build the wire-vs-ledger verdict for one message channel.

    Args:
        check: Which channel-diff check is reporting.
        label: Human name of the channel.
        wire_count: Messages found by replaying the raw capture.
        ledger_count: Messages the live run ingested per its ledger.

    Returns:
        A critical finding on mismatch (the June class of bug: bytes
        arrived that the run did not act on, or the run acted on
        messages the capture never carried), info on match.
    """
    if wire_count != ledger_count:
        return make_finding(
            check,
            "critical",
            f"{label}: capture replay found {wire_count} but the run "
            f"ingested {ledger_count} -- decode/dispatch gap",
            wire=wire_count,
            ledger=ledger_count,
        )
    return make_finding(
        check,
        "info",
        f"{label}: wire and ledger agree ({wire_count})",
        wire=wire_count,
        ledger=ledger_count,
    )


def _dom_witness_findings(
    capture: CaptureSession,
    own_deactivations: int,
    supervisor_codes: list[int],
) -> list[FindingDict]:
    """Diff the DOM game-log witness against the wire replay.

    Args:
        capture: Decoded capture session (carries the witness entries).
        own_deactivations: Wire 0x41 count attributed to the own tank.
        supervisor_codes: Replayed 0x52 error codes.

    Returns:
        A critical finding per banner class the client rendered more
        often than the wire explains -- the canary for a decoder blind
        spot the client can see.
    """
    kill_banners = 0
    empty_banners = 0
    blocked_banners = 0
    for entry in capture["game_log"]:
        if entry["text"].endswith(_KILL_BANNER_SUFFIX):
            kill_banners += 1
        elif entry["text"] == _EMPTY_CONTAINER_TEXT:
            empty_banners += 1
        elif entry["text"] == _BLOCKED_MOVE_TEXT:
            blocked_banners += 1
    pairs = [
        ("kill banner", kill_banners, own_deactivations, "0x41 own-kill deactivations"),
        (
            "empty-container banner",
            empty_banners,
            sum(1 for code in supervisor_codes if code == _SUPERVISOR_EMPTY_CONTAINER),
            "0x52 code-4 errors",
        ),
        (
            "blocked-move banner",
            blocked_banners,
            sum(1 for code in supervisor_codes if code == _SUPERVISOR_CANT_GO),
            "0x52 code-1 errors",
        ),
    ]
    findings: list[FindingDict] = []
    for label, banner_count, wire_count, wire_label in pairs:
        if banner_count > wire_count:
            findings.append(
                make_finding(
                    "dom_witness_diff",
                    "critical",
                    f"the client rendered {banner_count} {label}(s) but the "
                    f"wire carried only {wire_count} {wire_label} -- the "
                    "decoder is missing something the client can see",
                    banners=banner_count,
                    wire=wire_count,
                )
            )
        elif banner_count > 0:
            findings.append(
                make_finding(
                    "dom_witness_diff",
                    "info",
                    f"{label}s consistent with the wire "
                    f"({banner_count} banner(s), {wire_count} wire message(s))",
                    banners=banner_count,
                    wire=wire_count,
                )
            )
    return findings


_TURRET_STREAK_THRESHOLD = 4
"""Consecutive own shots from ONE tile, while the human is firing,
that flag the turret behavior. The 2026-08-03 nope fight measured a
streak of 6 with a hit taken every tick; 4 keeps two-shot trades and
repositioning duels out of the warning."""


def _human_episode_findings(capture: CaptureSession) -> list[FindingDict]:
    """Surface every human engagement as first-class findings.

    The nope fight hid inside a rejection stream for three read
    passes because no audit layer named the humans. One INFO finding
    per engaged human; a WARNING when the episode shows the turret
    exchange (a stationary own-shot streak at or past
    ``_TURRET_STREAK_THRESHOLD`` against an actively firing human).

    Args:
        capture: Decoded capture session for the run.

    Returns:
        Episode findings, first-shot order.
    """
    episodes = extract_human_episodes(extract_shadow_timeline(capture))
    findings: list[FindingDict] = []
    for episode in episodes:
        findings.append(
            make_finding(
                "human_episode",
                "info",
                (
                    f"human {episode['name']}: {episode['shots_by_human']} shots taken, "
                    f"{episode['our_shots_in_window']} returned, "
                    f"{episode['kills_of_human']} kill(s) of them, "
                    f"{episode['deaths_to_human']} death(s) to them"
                ),
                tank_id=episode["tank_id"],
                first_shot_ms=episode["first_shot_ms"],
                last_shot_ms=episode["last_shot_ms"],
                max_stationary_streak=episode["max_stationary_streak"],
            )
        )
        if (
            episode["max_stationary_streak"] >= _TURRET_STREAK_THRESHOLD
            and episode["shots_by_human"] >= _TURRET_STREAK_THRESHOLD
        ):
            findings.append(
                make_finding(
                    "turret_exchange",
                    "warning",
                    (
                        f"stood on one tile for {episode['max_stationary_streak']} "
                        f"consecutive shots while {episode['name']} was firing -- "
                        "stationary trading gives a human 100% uptime"
                    ),
                    tank_id=episode["tank_id"],
                    max_stationary_streak=episode["max_stationary_streak"],
                )
            )
    return findings


def audit_capture(
    capture: CaptureSession,
    records: list[RuntimeEventRecordDict],
) -> list[FindingDict]:
    """Replay a capture through the current decoder and diff every channel.

    Args:
        capture: Decoded capture session for the run.
        records: The run's decoded event records (the ledger side).

    Returns:
        Findings: decode errors, unknown subtypes, wire-vs-ledger
        channel diffs, and DOM-witness diffs.
    """
    magic = capture["magic"]
    if magic is None:
        return [
            make_finding(
                "capture_unreadable",
                "warning",
                "capture carries no XOR magic -- replay audit skipped",
            )
        ]
    # An audit reports conditions rather than raising them: a missing
    # key means this capture cannot be replayed, which is a finding,
    # not a crash. Everywhere else the same condition is now fatal
    # ([[session-state-deglobalisation]]).
    try:
        table = build_session_xor_table(magic)
    except XorStaticKeyUnavailableError as error:
        log.warning("replay audit skipped: %s", error)
        return [
            make_finding(
                "capture_unreadable",
                "warning",
                "XOR static key file missing -- replay audit skipped",
            )
        ]
    deactivations, supervisor_codes, unknown_subtypes, decode_errors = _replay_received(
        capture, table
    )
    findings: list[FindingDict] = []
    if decode_errors > 0:
        findings.append(
            make_finding(
                "decode_error",
                "critical",
                f"{decode_errors} received frame(s) crashed the current "
                "decoder -- the wire carries a shape the decoder rejects",
                count=decode_errors,
            )
        )
    if unknown_subtypes:
        rendered = ",".join(
            f"0x{subtype:02X}x{count}" for subtype, count in sorted(unknown_subtypes.items())
        )
        findings.append(
            make_finding(
                "unknown_container_subtypes",
                "warning",
                f"{sum(unknown_subtypes.values())} 0x2E message(s) fell "
                "through to unknown_container -- undecoded wire channels "
                "(the June-blind-spot canary)",
                subtypes=rendered,
            )
        )
    _, ledger_deacts = scan_diagnostic_records(records, "tank_deactivated")
    ledger_deact_count = sum(
        1 for record in ledger_deacts if record["fields"].get("origin") == "protocol_0x41"
    )
    findings.append(
        _channel_diff_finding(
            "deactivation_channel_diff",
            "0x41 deactivations",
            len(deactivations),
            ledger_deact_count,
        )
    )
    _, ledger_errors = scan_diagnostic_records(records, "command_error")
    findings.append(
        _channel_diff_finding(
            "supervisor_channel_diff",
            "0x52 command errors",
            len(supervisor_codes),
            len(ledger_errors),
        )
    )
    own_id = _own_tank_id(records)
    own_deactivations = sum(1 for _, killer_id in deactivations if killer_id == own_id)
    findings.extend(_dom_witness_findings(capture, own_deactivations, supervisor_codes))
    findings.extend(_human_episode_findings(capture))
    return findings


__all__ = [
    "audit_capture",
]
