"""Codec for scene specifications.

A scene spec is the one record with float fields, which needs deciding rather
than defaulting: a float written at whatever precision ``str`` chooses and read
back by ``float`` round-trips exactly in CPython, but "exactly" resting on a
repr implementation is a weaker guarantee than this package makes anywhere else.

So floats are encoded through :func:`float.hex`, which is exact by
specification, and decoded through :func:`float.fromhex`. The file stops being
pleasant to read at that field and starts being correct, which is the right
trade for a value that decides which scene a published measurement refers to.
"""

from __future__ import annotations

from navprobe.records import SceneSpec
from navprobe.wireformat import (
    WireFormatError,
    header_line,
    join_document,
    require_no_body,
    require_positive_field,
    split_document,
    split_header_line,
)

#: Banner identifying an encoded scene specification.
SCENE_BANNER = "navprobe-scene/1"

#: Header lines an encoded scene specification occupies.
SCENE_HEADER_FIELD_COUNT = 5


def encode_float_field(value: float) -> str:
    """Encode a float exactly.

    Args:
        value: The value to encode.

    Returns:
        The value's hexadecimal form, which round-trips without loss.
    """
    return value.hex()


def require_positive_float_field(raw: str, field: str) -> float:
    """Convert a hexadecimal token to a float greater than zero.

    Args:
        raw: The token to convert.
        field: Field name, used in the error message.

    Returns:
        The token as a float.

    Raises:
        WireFormatError: When the token is not a hexadecimal float, or is not
            positive. Every float in a scene is a length or a duration, and
            neither is meaningful at zero or below.
    """
    if not raw.startswith(("0x", "-0x", "inf", "-inf", "nan")):
        raise WireFormatError(
            "NP-WIRE-014",
            f"field {field!r} must be a hexadecimal float, got {raw!r}",
        )
    value = float.fromhex(raw)
    if not value > 0.0:
        raise WireFormatError(
            "NP-WIRE-015", f"field {field!r} must be greater than zero, got {value}"
        )
    return value


def encode_scene_spec(spec: SceneSpec) -> str:
    """Encode a scene specification to its text form.

    Args:
        spec: The specification to encode.

    Returns:
        The encoded text, newline-terminated.
    """
    return join_document(
        [
            SCENE_BANNER,
            header_line("body_count", str(spec["body_count"])),
            header_line("lattice_width", str(spec["lattice_width"])),
            header_line("spacing", encode_float_field(spec["spacing"])),
            header_line("radius", encode_float_field(spec["radius"])),
            header_line("timestep", encode_float_field(spec["timestep"])),
        ]
    )


def decode_scene_spec(text: str) -> SceneSpec:
    """Decode a scene specification from its text form.

    Args:
        text: The encoded specification.

    Returns:
        The decoded specification.

    Raises:
        WireFormatError: When the banner is absent or belongs to another record
            type, a header field is missing or malformed, or lines trail the
            header.
    """
    header, body = split_document(text, SCENE_BANNER, SCENE_HEADER_FIELD_COUNT)
    require_no_body(body, SCENE_BANNER)
    return SceneSpec(
        body_count=require_positive_field(split_header_line(header[0], "body_count"), "body_count"),
        lattice_width=require_positive_field(
            split_header_line(header[1], "lattice_width"), "lattice_width"
        ),
        spacing=require_positive_float_field(split_header_line(header[2], "spacing"), "spacing"),
        radius=require_positive_float_field(split_header_line(header[3], "radius"), "radius"),
        timestep=require_positive_float_field(split_header_line(header[4], "timestep"), "timestep"),
    )


def scene_fields(spec: SceneSpec) -> tuple[str, ...]:
    """Encode a scene's fields for embedding in another record's row.

    A sweep carries one scene per row rather than one per document, so its codec
    needs the fields without the banner wrapping them.

    Args:
        spec: The specification to encode.

    Returns:
        The five field values, in the same fixed order the document form uses.
    """
    return (
        str(spec["body_count"]),
        str(spec["lattice_width"]),
        encode_float_field(spec["spacing"]),
        encode_float_field(spec["radius"]),
        encode_float_field(spec["timestep"]),
    )


def decode_scene_fields(fields: tuple[str, ...], position: int) -> SceneSpec:
    """Decode a scene from fields embedded in another record's row.

    Args:
        fields: Exactly five field values in the order :func:`scene_fields`
            emits them.
        position: The row's position, used in error messages.

    Returns:
        The decoded specification.

    Raises:
        WireFormatError: When a field is missing or outside its range.
    """
    return SceneSpec(
        body_count=require_positive_field(fields[0], f"entry[{position}].body_count"),
        lattice_width=require_positive_field(fields[1], f"entry[{position}].lattice_width"),
        spacing=require_positive_float_field(fields[2], f"entry[{position}].spacing"),
        radius=require_positive_float_field(fields[3], f"entry[{position}].radius"),
        timestep=require_positive_float_field(fields[4], f"entry[{position}].timestep"),
    )


#: Fields a scene occupies when embedded in another record's row.
SCENE_FIELD_COUNT = 5


__all__ = [
    "SCENE_BANNER",
    "SCENE_FIELD_COUNT",
    "SCENE_HEADER_FIELD_COUNT",
    "decode_scene_fields",
    "decode_scene_spec",
    "encode_float_field",
    "encode_scene_spec",
    "require_positive_float_field",
    "scene_fields",
]
