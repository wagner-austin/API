"""Encoder round-trip validator: encode(decode(x)) == x over the archive.

Phase 4 step (a) acceptance instrument (wiki [[physics-module-roadmap]]):
every binary server message in every archived capture must re-encode
byte-identically through ``protocol.encoders``. A mismatch means an
encoder and its decoder disagree about a wire layout — exactly the
drift the simulator's fake server must never have.

Lobby text frames (``is_text_message`` types: room listings, profile
rows) are not part of the binary protocol and are skipped; frames that
fail binary decoding are counted as invalid, not judged.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger

from tankpit_bot.capture.xor import build_session_xor_table, xor_decode_body
from tankpit_bot.protocol import (
    is_text_message,
    try_decode_binary_message,
    try_decode_plaintext_ack,
)
from tankpit_bot.protocol.encoders import (
    encode_envelope_body,
    encode_message_payload,
    encode_plaintext_ack,
)
from tankpit_bot.sniffer.constants import MSG_MIN_LENGTHS
from tankpit_bot.types import CaptureSession, decode_capture_session
from tankpit_bot.validate.types import ClaimEvidenceDict
from tankpit_bot.validate.wire_timeline import _split_frame_bodies
from tankpit_bot.wire.helpers import DecodeError

log = get_logger(__name__)

_ENVELOPE_TYPE = 0x2E


class _Tally:
    """Per-family exact/mismatch counters with one preserved diff."""

    def __init__(self) -> None:
        """Initialize empty counters."""
        self.exact: Counter[str] = Counter()
        self.mismatches: Counter[str] = Counter()
        self.first_diff: dict[str, str] = {}
        self.invalid_frames = 0

    def record(self, family: str, wire: bytes, encoded: bytes) -> None:
        """Compare one message's wire payload against its re-encoding.

        Args:
            family: Decoded msg_type rendered as a string.
            wire: The XOR-decoded payload the decoder consumed.
            encoded: The encoder's reproduction.
        """
        if encoded == wire:
            self.exact[family] += 1
        else:
            self.mismatches[family] += 1
            if family not in self.first_diff:
                self.first_diff[family] = f"want={wire.hex()} got={encoded.hex()}"


def _record_plaintext_ack(tally: _Tally, body: bytes) -> bool:
    """Round-trip a plaintext toggle ack, or return False when not one.

    The acks travel un-XORed (raw ``"A0"``-style echoes), so the raw
    two-byte body is compared against the raw-frame encoder.

    Args:
        tally: Accumulator shared across sessions.
        body: Raw frame body, NOT XOR-decoded.

    Returns:
        True if the body was a plaintext ack and was recorded.
    """
    ack = try_decode_plaintext_ack(body)
    if ack is None:
        return False
    tally.record(str(ack["msg_type"]), body, encode_plaintext_ack(ack))
    return True


def _roundtrip_session(session: CaptureSession, tally: _Tally) -> None:
    """Round-trip every received binary message of one session.

    Args:
        session: Loaded capture session (must have a magic key).
        tally: Accumulator shared across sessions.
    """
    magic = session["magic"]
    if magic is None:
        return
    xor_table = build_session_xor_table(magic)
    for msg in session["messages"]:
        if msg["direction"] != "received":
            continue
        for body in _split_frame_bodies(msg["payload"]):
            _roundtrip_body(body, xor_table, tally)


def _roundtrip_body(body: bytes, xor_table: bytes, tally: _Tally) -> None:
    """Round-trip one received frame body into the tally.

    Args:
        body: Raw frame body (msg_type byte + XOR-encoded rest, or a
            plaintext ack).
        xor_table: The owning session's XOR table.
        tally: Accumulator shared across sessions.
    """
    msg_type = body[0]
    if _record_plaintext_ack(tally, body):
        return
    if msg_type not in MSG_MIN_LENGTHS or is_text_message(msg_type):
        return
    payload = xor_decode_body(body, xor_table, offset=1)
    if len(payload) < MSG_MIN_LENGTHS[msg_type]:
        tally.invalid_frames += 1
        return
    try:
        message = try_decode_binary_message(msg_type, payload)
    except DecodeError as error:
        log.debug("roundtrip: undecodable 0x%02X frame: %s", msg_type, error)
        tally.invalid_frames += 1
        return
    if message is None:
        tally.invalid_frames += 1
        return
    if msg_type == _ENVELOPE_TYPE:
        encoded = encode_envelope_body(message)
    else:
        encoded = encode_message_payload(message)
    tally.record(str(message["msg_type"]), payload, encoded)


def collect_roundtrip_evidence(runs_root: Path) -> list[ClaimEvidenceDict]:
    """Round-trip the whole archive and report per-family evidence.

    Args:
        runs_root: Root of the runs tree (``runs/``).

    Returns:
        One evidence record per message family seen, ordered by family,
        plus a trailing ``invalid-frames`` record counting bodies that
        failed binary decoding (they are not judged).
    """
    tally = _Tally()
    for capture_path in sorted(runs_root.glob("*/*.capture_session.json")):
        if capture_path.name.startswith("latest"):
            continue
        session = decode_capture_session(
            narrow_json_to_dict(load_json_str(capture_path.read_text(encoding="utf-8")))
        )
        _roundtrip_session(session, tally)
    families = sorted(set(tally.exact) | set(tally.mismatches))
    evidence = [
        ClaimEvidenceDict(
            claim_id=f"roundtrip-{family}",
            samples=tally.exact[family] + tally.mismatches[family],
            exact=tally.exact[family],
            mismatches=tally.mismatches[family],
            detail=tally.first_diff.get(family, "byte-identical"),
        )
        for family in families
    ]
    evidence.append(
        ClaimEvidenceDict(
            claim_id="invalid-frames",
            samples=tally.invalid_frames,
            exact=0,
            mismatches=0,
            detail="bodies that failed binary decoding (not judged)",
        )
    )
    return evidence


def run_roundtrip(runs_root: Path) -> int:
    """Run the round-trip suite and print the per-family report.

    Args:
        runs_root: Root of the runs tree.

    Returns:
        Process exit code: 0 when every family round-trips with at
        least one sample, 1 on any mismatch or an empty archive.
    """
    evidence = collect_roundtrip_evidence(runs_root)
    total_samples = 0
    total_mismatches = 0
    for record in evidence:
        if record["claim_id"] == "invalid-frames":
            sys.stdout.write(f"{record['claim_id']:>28}  skipped={record['samples']}\n")
            continue
        total_samples += record["samples"]
        total_mismatches += record["mismatches"]
        verdict = "PASS" if record["mismatches"] == 0 else "FAIL"
        sys.stdout.write(
            f"{record['claim_id']:>28}  samples={record['samples']:<7} "
            f"exact={record['exact']:<7} mismatches={record['mismatches']:<5} {verdict}\n"
        )
        if record["mismatches"] > 0:
            sys.stdout.write(f"{'':>28}  first diff: {record['detail']}\n")
    sys.stdout.write(f"TOTAL: {total_samples} messages, {total_mismatches} mismatches\n")
    if total_samples == 0:
        sys.stdout.write("FAIL: no binary messages found in the archive\n")
        return 1
    return 0 if total_mismatches == 0 else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``tankpit-roundtrip``.

    Args:
        argv: Argument list, or None for ``sys.argv``.

    Returns:
        Process exit code.
    """
    args = list(argv) if argv is not None else list(sys.argv[1:])
    runs_root = Path("runs")
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--runs-dir" and index + 1 < len(args):
            runs_root = Path(args[index + 1])
            index += 2
        else:
            index += 1
    return run_roundtrip(runs_root)


__all__ = [
    "collect_roundtrip_evidence",
    "main",
    "run_roundtrip",
]
