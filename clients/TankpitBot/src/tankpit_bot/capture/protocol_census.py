"""Protocol-level capture census utilities.

This module analyzes saved capture sessions using the same frame-splitting and
XOR decode path as the live sniffer. It separates fully decoded packets from
short/invalid packets and unsupported packet types so protocol gaps can be
investigated without relying on container-oriented summaries.
"""

from __future__ import annotations

from collections import Counter
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_list,
    require_str,
)
from platform_core.logging import get_logger

from tankpit_bot.browser.types import TEXT_MESSAGE_TYPES
from tankpit_bot.capture.xor import build_session_xor_table, decode_base64_safe
from tankpit_bot.protocol import try_decode_binary_message
from tankpit_bot.protocol.framing import FramingError, split_frames
from tankpit_bot.types import CaptureSession
from tankpit_bot.wire.helpers import DecodeError

log = get_logger(__name__)


class ProtocolCountDict(TypedDict):
    """Count for a decoded protocol message class.

    Attributes:
        label: Human-readable packet label.
        count: Number of matching packets.
    """

    label: str
    count: int


class ProtocolSampleDict(TypedDict):
    """Sample entry for undecoded protocol packets.

    Attributes:
        label: Packet signature label.
        count: Number of matching packets.
        sample_body_hex: Sample raw body bytes in hex.
        sample_decoded_hex: Sample XOR-decoded bytes in hex.
    """

    label: str
    count: int
    sample_body_hex: str
    sample_decoded_hex: str


class ProtocolCensusDict(TypedDict):
    """Protocol census derived from a capture session.

    Attributes:
        received_message_count: Number of received WebSocket messages examined.
        received_frame_count: Number of received logical frames examined.
        text_frame_count: Number of text frames observed.
        decoded_binary_frame_count: Number of binary frames decoded successfully.
        short_or_invalid_frame_count: Number of known packet families that failed validation.
        unsupported_frame_count: Number of packets with no decoder match.
        framing_error_count: Number of payloads that failed frame splitting.
        decoded: Sorted decoded packet counts.
        short_or_invalid: Sorted short/invalid packet samples.
        unsupported: Sorted unsupported packet samples.
    """

    received_message_count: int
    received_frame_count: int
    text_frame_count: int
    decoded_binary_frame_count: int
    short_or_invalid_frame_count: int
    unsupported_frame_count: int
    framing_error_count: int
    decoded: list[ProtocolCountDict]
    short_or_invalid: list[ProtocolSampleDict]
    unsupported: list[ProtocolSampleDict]


class _ProtocolCensusAccumulatorDict(TypedDict):
    """Mutable accumulator for protocol census construction."""

    decoded_counts: Counter[str]
    short_counts: Counter[str]
    unsupported_counts: Counter[str]
    short_samples: dict[str, ProtocolSampleDict]
    unsupported_samples: dict[str, ProtocolSampleDict]
    received_message_count: int
    received_frame_count: int
    text_frame_count: int
    decoded_binary_frame_count: int
    short_or_invalid_frame_count: int
    unsupported_frame_count: int
    framing_error_count: int


def encode_protocol_count(entry: ProtocolCountDict) -> JSONObject:
    """Encode a protocol count entry to JSON.

    Args:
        entry: Protocol count entry to encode.

    Returns:
        JSON object representation.
    """
    return {"label": entry["label"], "count": entry["count"]}


def decode_protocol_count(data: JSONObject) -> ProtocolCountDict:
    """Decode a protocol count entry from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated protocol count entry.
    """
    return ProtocolCountDict(label=require_str(data, "label"), count=require_int(data, "count"))


def encode_protocol_sample(entry: ProtocolSampleDict) -> JSONObject:
    """Encode a protocol sample entry to JSON.

    Args:
        entry: Protocol sample entry to encode.

    Returns:
        JSON object representation.
    """
    return {
        "label": entry["label"],
        "count": entry["count"],
        "sample_body_hex": entry["sample_body_hex"],
        "sample_decoded_hex": entry["sample_decoded_hex"],
    }


def decode_protocol_sample(data: JSONObject) -> ProtocolSampleDict:
    """Decode a protocol sample entry from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated protocol sample entry.
    """
    return ProtocolSampleDict(
        label=require_str(data, "label"),
        count=require_int(data, "count"),
        sample_body_hex=require_str(data, "sample_body_hex"),
        sample_decoded_hex=require_str(data, "sample_decoded_hex"),
    )


def encode_protocol_census(result: ProtocolCensusDict) -> JSONObject:
    """Encode protocol census to JSON.

    Args:
        result: Protocol census result.

    Returns:
        JSON object representation.
    """
    decoded_json: list[JSONValue] = [encode_protocol_count(entry) for entry in result["decoded"]]
    short_json: list[JSONValue] = [
        encode_protocol_sample(entry) for entry in result["short_or_invalid"]
    ]
    unsupported_json: list[JSONValue] = [
        encode_protocol_sample(entry) for entry in result["unsupported"]
    ]
    return {
        "received_message_count": result["received_message_count"],
        "received_frame_count": result["received_frame_count"],
        "text_frame_count": result["text_frame_count"],
        "decoded_binary_frame_count": result["decoded_binary_frame_count"],
        "short_or_invalid_frame_count": result["short_or_invalid_frame_count"],
        "unsupported_frame_count": result["unsupported_frame_count"],
        "framing_error_count": result["framing_error_count"],
        "decoded": decoded_json,
        "short_or_invalid": short_json,
        "unsupported": unsupported_json,
    }


def decode_protocol_census(data: JSONObject) -> ProtocolCensusDict:
    """Decode protocol census from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated protocol census result.

    Raises:
        JSONTypeError: If any nested entry is not an object.
    """
    raw_decoded = require_list(data, "decoded")
    decoded_entries: list[ProtocolCountDict] = []
    for index, entry in enumerate(raw_decoded):
        if not isinstance(entry, dict):
            raise JSONTypeError(f"decoded[{index}] must be an object")
        decoded_entries.append(decode_protocol_count(entry))

    raw_short = require_list(data, "short_or_invalid")
    short_entries: list[ProtocolSampleDict] = []
    for index, entry in enumerate(raw_short):
        if not isinstance(entry, dict):
            raise JSONTypeError(f"short_or_invalid[{index}] must be an object")
        short_entries.append(decode_protocol_sample(entry))

    raw_unsupported = require_list(data, "unsupported")
    unsupported_entries: list[ProtocolSampleDict] = []
    for index, entry in enumerate(raw_unsupported):
        if not isinstance(entry, dict):
            raise JSONTypeError(f"unsupported[{index}] must be an object")
        unsupported_entries.append(decode_protocol_sample(entry))

    return ProtocolCensusDict(
        received_message_count=require_int(data, "received_message_count"),
        received_frame_count=require_int(data, "received_frame_count"),
        text_frame_count=require_int(data, "text_frame_count"),
        decoded_binary_frame_count=require_int(data, "decoded_binary_frame_count"),
        short_or_invalid_frame_count=require_int(data, "short_or_invalid_frame_count"),
        unsupported_frame_count=require_int(data, "unsupported_frame_count"),
        framing_error_count=require_int(data, "framing_error_count"),
        decoded=decoded_entries,
        short_or_invalid=short_entries,
        unsupported=unsupported_entries,
    )


def _sort_protocol_count(entry: ProtocolCountDict) -> tuple[int, str]:
    """Return sort key for decoded protocol counts.

    Args:
        entry: Decoded count entry.

    Returns:
        Sort key ordering highest counts first.
    """
    return (-entry["count"], entry["label"])


def _sort_protocol_sample(entry: ProtocolSampleDict) -> tuple[int, str]:
    """Return sort key for undecoded protocol samples.

    Args:
        entry: Sample entry to sort.

    Returns:
        Sort key ordering highest counts first.
    """
    return (-entry["count"], entry["label"])


def _message_label(msg_type_value: int | str | None) -> str:
    """Format decoded message label from a protocol decoder result.

    Args:
        msg_type_value: ``msg_type`` field from a decoded protocol message.

    Returns:
        Human-readable label.
    """
    if isinstance(msg_type_value, str):
        return msg_type_value
    if isinstance(msg_type_value, int):
        return f"0x{msg_type_value:02X}"
    return "unknown"


def _packet_label(msg_type: int, frame_length: int) -> str:
    """Format packet label for undecoded entries.

    Args:
        msg_type: Raw protocol type byte from the frame body.
        frame_length: Raw body length including the type byte.

    Returns:
        Packet label string.
    """
    return f"0x{msg_type:02X} len={frame_length}"


def _build_census_accumulator() -> _ProtocolCensusAccumulatorDict:
    """Create an empty protocol census accumulator.

    Returns:
        Mutable accumulator with zeroed counters.
    """
    return _ProtocolCensusAccumulatorDict(
        decoded_counts=Counter(),
        short_counts=Counter(),
        unsupported_counts=Counter(),
        short_samples={},
        unsupported_samples={},
        received_message_count=0,
        received_frame_count=0,
        text_frame_count=0,
        decoded_binary_frame_count=0,
        short_or_invalid_frame_count=0,
        unsupported_frame_count=0,
        framing_error_count=0,
    )


def _record_protocol_sample(
    samples: dict[str, ProtocolSampleDict],
    label: str,
    body: bytes,
    decoded_data: bytes,
) -> None:
    """Record a first sample for a packet label.

    Args:
        samples: Mapping of packet labels to first-observed sample.
        label: Packet label string.
        body: Raw frame body bytes.
        decoded_data: XOR-decoded bytes after the type byte.
    """
    if label in samples:
        return
    samples[label] = ProtocolSampleDict(
        label=label,
        count=0,
        sample_body_hex=body.hex(),
        sample_decoded_hex=decoded_data.hex(),
    )


def _classify_frame(
    body: bytes,
    xor_table: bytes,
    acc: _ProtocolCensusAccumulatorDict,
) -> None:
    """Classify one received logical frame.

    Args:
        body: Raw logical frame body.
        xor_table: Session XOR table.
        acc: Mutable census accumulator.
    """
    if len(body) == 0:
        return

    acc["received_frame_count"] += 1
    msg_type = body[0]
    if msg_type in TEXT_MESSAGE_TYPES:
        acc["text_frame_count"] += 1
        return

    decoded_data = _xor_decode_frame_body(body, xor_table)
    try:
        parsed = try_decode_binary_message(msg_type, decoded_data)
    except DecodeError as exc:
        acc["short_or_invalid_frame_count"] += 1
        label = _packet_label(msg_type, len(body))
        acc["short_counts"][label] += 1
        _record_protocol_sample(acc["short_samples"], label, body, decoded_data)
        log.warning(
            "Short or invalid protocol packet during census: %s (%s)",
            label,
            exc,
        )
        return

    if parsed is None:
        acc["unsupported_frame_count"] += 1
        label = _packet_label(msg_type, len(body))
        acc["unsupported_counts"][label] += 1
        _record_protocol_sample(acc["unsupported_samples"], label, body, decoded_data)
        return

    acc["decoded_binary_frame_count"] += 1
    acc["decoded_counts"][_message_label(parsed["msg_type"])] += 1


def _accumulate_message(
    payload: str,
    xor_table: bytes,
    acc: _ProtocolCensusAccumulatorDict,
) -> None:
    """Accumulate one received capture message into the census.

    Args:
        payload: Base64-encoded WebSocket payload.
        xor_table: Session XOR table.
        acc: Mutable census accumulator.
    """
    raw_bytes = decode_base64_safe(payload)
    if raw_bytes is None:
        return
    try:
        frames = split_frames(raw_bytes)
    except FramingError as exc:
        acc["framing_error_count"] += 1
        log.warning("Skipping malformed capture payload during protocol census: %s", exc)
        return

    for body in frames:
        _classify_frame(body, xor_table, acc)


def _sample_entries(
    counts: Counter[str],
    samples: dict[str, ProtocolSampleDict],
) -> list[ProtocolSampleDict]:
    """Build sorted sample entries from counters and first samples.

    Args:
        counts: Packet counts keyed by label.
        samples: First sample keyed by label.

    Returns:
        Sorted sample entry list.
    """
    entries = [
        ProtocolSampleDict(
            label=samples[label]["label"],
            count=count,
            sample_body_hex=samples[label]["sample_body_hex"],
            sample_decoded_hex=samples[label]["sample_decoded_hex"],
        )
        for label, count in counts.items()
    ]
    entries.sort(key=_sort_protocol_sample)
    return entries


def analyze_protocol_census(session: CaptureSession) -> ProtocolCensusDict:
    """Analyze protocol coverage from a captured session.

    Args:
        session: Capture session to analyze.

    Returns:
        Protocol census derived from received frames.

    Raises:
        ValueError: If the session has no magic key.
    """
    magic = session["magic"]
    if magic is None:
        raise ValueError("Capture session has no magic key")

    xor_table = build_session_xor_table(magic)

    acc = _build_census_accumulator()

    for message in session["messages"]:
        if message["direction"] != "received":
            continue
        acc["received_message_count"] += 1
        _accumulate_message(message["payload"], xor_table, acc)

    decoded_entries = [
        ProtocolCountDict(label=label, count=count)
        for label, count in acc["decoded_counts"].items()
    ]
    decoded_entries.sort(key=_sort_protocol_count)

    return ProtocolCensusDict(
        received_message_count=acc["received_message_count"],
        received_frame_count=acc["received_frame_count"],
        text_frame_count=acc["text_frame_count"],
        decoded_binary_frame_count=acc["decoded_binary_frame_count"],
        short_or_invalid_frame_count=acc["short_or_invalid_frame_count"],
        unsupported_frame_count=acc["unsupported_frame_count"],
        framing_error_count=acc["framing_error_count"],
        decoded=decoded_entries,
        short_or_invalid=_sample_entries(acc["short_counts"], acc["short_samples"]),
        unsupported=_sample_entries(acc["unsupported_counts"], acc["unsupported_samples"]),
    )


def _xor_decode_frame_body(body: bytes, xor_table: bytes) -> bytes:
    """XOR decode a binary frame body without its leading type byte.

    Args:
        body: Raw frame body including the type byte.
        xor_table: Session XOR table.

    Returns:
        XOR-decoded bytes after the type byte.
    """
    if len(body) < 2:
        return b""
    decoded = bytearray(len(body) - 1)
    for index in range(len(decoded)):
        decoded[index] = body[index + 1] ^ xor_table[index]
    return bytes(decoded)


def format_protocol_census(result: ProtocolCensusDict) -> str:
    """Format protocol census as readable text.

    Args:
        result: Protocol census result.

    Returns:
        Multi-line human-readable summary.
    """
    lines = [
        f"received_messages={result['received_message_count']}",
        f"received_frames={result['received_frame_count']}",
        f"text_frames={result['text_frame_count']}",
        f"decoded_binary_frames={result['decoded_binary_frame_count']}",
        f"short_or_invalid_frames={result['short_or_invalid_frame_count']}",
        f"unsupported_frames={result['unsupported_frame_count']}",
        f"framing_errors={result['framing_error_count']}",
    ]

    if result["decoded"]:
        lines.append("decoded:")
        for entry in result["decoded"]:
            lines.append(f"  {entry['label']} x{entry['count']}")

    if result["short_or_invalid"]:
        lines.append("short_or_invalid:")
        for entry in result["short_or_invalid"]:
            lines.append(
                "  "
                + f"{entry['label']} x{entry['count']} "
                + f"body={entry['sample_body_hex']} decoded={entry['sample_decoded_hex']}"
            )

    if result["unsupported"]:
        lines.append("unsupported:")
        for entry in result["unsupported"]:
            lines.append(
                "  "
                + f"{entry['label']} x{entry['count']} "
                + f"body={entry['sample_body_hex']} decoded={entry['sample_decoded_hex']}"
            )

    return "\n".join(lines)


__all__ = [
    "ProtocolCensusDict",
    "ProtocolCountDict",
    "ProtocolSampleDict",
    "analyze_protocol_census",
    "decode_protocol_census",
    "decode_protocol_count",
    "decode_protocol_sample",
    "encode_protocol_census",
    "encode_protocol_count",
    "encode_protocol_sample",
    "format_protocol_census",
]
