"""Codec for observation records.

The only record whose body is a list of *values* rather than digests or
verdicts, and the only one that can get large — a 64 by 64 depth buffer over two
worlds is 8,192 numbers.

One value per line rather than one long row. A file of eight thousand
tab-separated fields is not diff-able, and the whole reason this record exists is
so two environments' outputs can be put side by side. Line-per-value also means
a difference reads as a line difference in any tool.

Values are stored in the exact hexadecimal form the other float-bearing codecs
use. A magnitude computed from rounded values would be a magnitude of the
rounding.
"""

from __future__ import annotations

from navprobe.codecs.scene import encode_float_field
from navprobe.records import ObservationRecord
from navprobe.wireformat import (
    SEPARATOR,
    WireFormatError,
    header_line,
    join_document,
    require_non_negative_field,
    require_text_field,
    split_document,
    split_header_line,
)

#: Banner identifying an encoded observation record.
OBSERVATION_BANNER = "navprobe-observation/1"

#: Leading token marking one observed value.
VALUE_TAG = "value"

#: Header lines an encoded observation record occupies, before its values.
OBSERVATION_HEADER_FIELD_COUNT = 4


def require_float_field(raw: str, field: str) -> float:
    """Convert a hexadecimal token to a float of any sign.

    Unlike every other float in this package, an observed value has no range at
    all: a position may be negative, a depth may be zero, and neither is a
    construction error. The only thing refused is a token that is not a float.

    Args:
        raw: The token to convert.
        field: Field name, used in the error message.

    Returns:
        The token as a float.

    Raises:
        WireFormatError: When the token is not a hexadecimal float.
    """
    if not raw.startswith(("0x", "-0x", "inf", "-inf", "nan")):
        raise WireFormatError(
            "NP-WIRE-014", f"field {field!r} must be a hexadecimal float, got {raw!r}"
        )
    return float.fromhex(raw)


def encode_observation_record(record: ObservationRecord) -> str:
    """Encode an observation record to its text form.

    Args:
        record: The record to encode.

    Returns:
        The encoded text, newline-terminated.
    """
    return join_document(
        [
            OBSERVATION_BANNER,
            header_line("label", record["label"]),
            header_line("seed", str(record["seed"])),
            header_line("step_count", str(record["step_count"])),
            header_line("value_count", str(len(record["values"]))),
            *(
                f"{VALUE_TAG}{SEPARATOR}{index}{SEPARATOR}{encode_float_field(value)}"
                for index, value in enumerate(record["values"])
            ),
        ]
    )


def decode_value_row(line: str, position: int) -> float:
    """Decode one observed value.

    Args:
        line: The line to decode.
        position: The row's position, which its declared index must match.

    Returns:
        The value.

    Raises:
        WireFormatError: When the row is malformed, or declares an index other
            than its position. Values must be contiguous and in order, because
            element order is the observation's contract and a reordered file
            would compare position three against position four.
    """
    parts = line.split(SEPARATOR)
    if len(parts) != 3 or parts[0] != VALUE_TAG:
        raise WireFormatError(
            "NP-WIRE-018",
            f"value row {position} must be {VALUE_TAG!r} followed by an index and a "
            f"value, got {line!r}",
        )
    index = require_non_negative_field(parts[1], f"value[{position}].index")
    if index != position:
        raise WireFormatError(
            "NP-WIRE-019",
            f"value row {position} declares index {index}; values must be contiguous and in order",
        )
    return require_float_field(parts[2], f"value[{position}]")


def decode_observation_record(text: str) -> ObservationRecord:
    """Decode an observation record from its text form.

    Args:
        text: The encoded record.

    Returns:
        The decoded record.

    Raises:
        WireFormatError: When the banner is absent or belongs to another record
            type, a header field is missing or malformed, a value row is
            malformed, or the declared count disagrees with the rows present.
    """
    header, body = split_document(text, OBSERVATION_BANNER, OBSERVATION_HEADER_FIELD_COUNT)
    label = require_text_field(split_header_line(header[0], "label"), "label")
    seed = require_non_negative_field(split_header_line(header[1], "seed"), "seed")
    step_count = require_non_negative_field(
        split_header_line(header[2], "step_count"), "step_count"
    )
    value_count = require_non_negative_field(
        split_header_line(header[3], "value_count"), "value_count"
    )
    if len(body) != value_count:
        raise WireFormatError(
            "NP-WIRE-020",
            f"record declares {value_count} values but carries {len(body)}",
        )
    return ObservationRecord(
        label=label,
        seed=seed,
        step_count=step_count,
        values=tuple(decode_value_row(line, position) for position, line in enumerate(body)),
    )


__all__ = [
    "OBSERVATION_BANNER",
    "OBSERVATION_HEADER_FIELD_COUNT",
    "VALUE_TAG",
    "decode_observation_record",
    "decode_value_row",
    "encode_observation_record",
    "require_float_field",
]
