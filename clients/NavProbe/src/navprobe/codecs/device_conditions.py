"""Codec for the conditions any device-scoped measurement run was taken under.

Every report this package emits from a GPU carries the same four facts, and they
are encoded here once. A sweep report and a scaling report that each spelled out
their own mode and device fields would be two declarations of one layout, free
to drift; a reader comparing a sweep on one card against a ladder on another
would then be comparing headers that only look alike.

The fields are header lines rather than a row because they describe the whole
document, and because a reader that wants only "which card was this" should not
have to parse the body to find out.
"""

from __future__ import annotations

from navprobe.records import DeviceRunConditions
from navprobe.wireformat import (
    header_line,
    require_non_negative_field,
    require_text_field,
    split_header_line,
)

#: Header lines a :class:`navprobe.records.DeviceRunConditions` occupies.
DEVICE_CONDITIONS_FIELD_COUNT = 4


def encode_device_conditions(conditions: DeviceRunConditions) -> tuple[str, ...]:
    """Encode run conditions to their header lines.

    Args:
        conditions: The conditions to encode.

    Returns:
        Exactly :data:`DEVICE_CONDITIONS_FIELD_COUNT` header lines in fixed
        order.
    """
    return (
        header_line("mode", conditions["mode"]),
        header_line("device", conditions["device"]),
        header_line("device_request", conditions["device_request"]),
        header_line("max_records", str(conditions["max_records"])),
    )


def decode_device_conditions(lines: tuple[str, ...]) -> DeviceRunConditions:
    """Decode run conditions from their header lines.

    Args:
        lines: Exactly :data:`DEVICE_CONDITIONS_FIELD_COUNT` header lines, as
            sliced by :func:`navprobe.wireformat.split_document`, which has
            already guaranteed the count.

    Returns:
        The decoded conditions.

    Raises:
        WireFormatError: When a line is malformed, carries the wrong key, or
            holds a value outside its field's range.
    """
    return DeviceRunConditions(
        mode=require_text_field(split_header_line(lines[0], "mode"), "mode"),
        device=require_text_field(split_header_line(lines[1], "device"), "device"),
        device_request=require_text_field(
            split_header_line(lines[2], "device_request"), "device_request"
        ),
        max_records=require_non_negative_field(
            split_header_line(lines[3], "max_records"), "max_records"
        ),
    )


__all__ = [
    "DEVICE_CONDITIONS_FIELD_COUNT",
    "decode_device_conditions",
    "encode_device_conditions",
]
