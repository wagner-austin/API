"""Message statistics building utilities.

This module provides functions for analyzing captured WebSocket sessions
and generating message statistics.
"""

from __future__ import annotations

from collections import Counter

from platform_core.logging import get_logger

from tankpit_bot.capture.signature import extract_message_signature, identify_message
from tankpit_bot.capture.xor import XorStaticKeyUnavailableError, build_session_xor_table
from tankpit_bot.container import DecodeLevel
from tankpit_bot.types import CaptureSession, MessageStats, UnknownMessageEntry

log = get_logger(__name__)


def empty_message_stats() -> MessageStats:
    """Return empty MessageStats.

    Returns:
        MessageStats with zero counts and empty dictionaries.
    """
    return MessageStats(decoded={}, unknown={}, total_received=0, decode_coverage="0%")


def build_message_stats(session: CaptureSession) -> MessageStats:
    """Build message statistics from captured session.

    Uses identify_container_type from container_decoder to determine message
    types and get_decode_level for understanding levels.

    Args:
        session: The capture session to analyze.

    Returns:
        MessageStats with decoded vs unknown breakdown.
    """
    magic = session.get("magic")
    if not magic:
        return empty_message_stats()

    # Same contract as the magic guard above: a capture that cannot be
    # deciphered yields empty stats, not an error. This used to read
    # xor_static_key.txt through a private copy of the path expression
    # ([[session-state-deglobalisation]]).
    try:
        xor_table = build_session_xor_table(magic)
    except XorStaticKeyUnavailableError as error:
        log.warning("message stats empty: %s", error)
        return empty_message_stats()

    decoded_counts: Counter[str] = Counter()
    unknown_counts: Counter[str] = Counter()
    unknown_samples: dict[str, list[str]] = {}
    level_counts: Counter[DecodeLevel] = Counter()

    for msg in session["messages"]:
        if msg["direction"] != "received":
            continue

        decoded = extract_message_signature(msg["payload"], xor_table)
        if decoded is None:
            continue

        result = identify_message(decoded)
        if result is not None:
            name, level = result
            decoded_counts[f"len={len(decoded):02d} {name}"] += 1
            level_counts[level] += 1
        else:
            len_key = f"len={len(decoded):02d}"
            unknown_counts[len_key] += 1
            if len_key not in unknown_samples:
                unknown_samples[len_key] = []
            if len(unknown_samples[len_key]) < 3:
                unknown_samples[len_key].append(decoded[:20].hex())

    total = sum(decoded_counts.values()) + sum(unknown_counts.values())
    decoded_total = sum(decoded_counts.values())

    if total > 0:
        sig_coverage = 100 * decoded_total // total
        weighted = (
            level_counts[DecodeLevel.FULL] * DecodeLevel.FULL
            + level_counts[DecodeLevel.PARTIAL] * DecodeLevel.PARTIAL
            + level_counts[DecodeLevel.IDENTIFIED] * DecodeLevel.IDENTIFIED
        )
        understanding = weighted // total
        coverage = f"{sig_coverage}% sig, {understanding}% understood"
    else:
        coverage = "0%"

    unknown_dict: dict[str, UnknownMessageEntry] = {
        k: UnknownMessageEntry(count=v, samples=unknown_samples.get(k, []))
        for k, v in unknown_counts.items()
    }

    return MessageStats(
        decoded=dict(decoded_counts),
        unknown=unknown_dict,
        total_received=total,
        decode_coverage=coverage,
    )


__all__ = [
    "build_message_stats",
    "empty_message_stats",
]
