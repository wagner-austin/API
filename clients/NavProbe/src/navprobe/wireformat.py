"""Primitives every record codec is built from.

Probe records are flat: a handful of scalars and, for a run, a list of
``(index, digest)`` rows. They are not nested documents, so they are not encoded
as ones. A tab-separated line format parses exactly with string operations,
keeps records diff-able, and needs no JSON parser.

That choice also keeps the decode path typed end to end. A JSON decoder hands
back a value of unknown shape that every field must then be narrowed out of;
splitting text hands back ``str``, and each field is converted by a ``require_*``
helper that names the field it failed on.

This module owns the pieces shared across record types — the separator, the
scalar tokens, the ``require_*`` family, and the document/header splitting —
and nothing else. Each record type's own codec lives in :mod:`navprobe.codecs`,
one module per record, and none of them may reimplement anything found here.
"""

from __future__ import annotations

from navprobe import NavProbeError

#: Field separator. Tab rather than space because a label may contain spaces.
SEPARATOR = "\t"

#: Token encoding an absent optional value. Spelled explicitly rather than as an
#: empty field so that "absent" and "malformed" cannot be confused.
NONE_TOKEN = "none"

#: Token encoding boolean true.
TRUE_TOKEN = "true"

#: Token encoding boolean false.
FALSE_TOKEN = "false"


class WireFormatError(NavProbeError):
    """A record could not be decoded from its text form.

    Raised by this module and by every codec in :mod:`navprobe.codecs`, so a
    caller catches one type regardless of which record it was reading.

    Args:
        code: Stable identifier in the ``NP-WIRE-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


def require_int_field(raw: str, field: str) -> int:
    """Convert a token to an integer.

    Args:
        raw: The token to convert.
        field: Field name, used in the error message.

    Returns:
        The token as an integer.

    Raises:
        WireFormatError: When the token is not a base-ten integer.
    """
    if not (raw.lstrip("-").isdigit() and raw not in {"-", ""}):
        raise WireFormatError("NP-WIRE-001", f"field {field!r} must be an integer, got {raw!r}")
    return int(raw)


def require_non_negative_field(raw: str, field: str) -> int:
    """Convert a token to an integer of zero or greater.

    Args:
        raw: The token to convert.
        field: Field name, used in the error message.

    Returns:
        The token as a non-negative integer.

    Raises:
        WireFormatError: When the token is not an integer, or is negative.
    """
    value = require_int_field(raw, field)
    if value < 0:
        raise WireFormatError(
            "NP-WIRE-002", f"field {field!r} must be zero or greater, got {value}"
        )
    return value


def require_positive_field(raw: str, field: str) -> int:
    """Convert a token to an integer greater than zero.

    Args:
        raw: The token to convert.
        field: Field name, used in the error message.

    Returns:
        The token as a positive integer.

    Raises:
        WireFormatError: When the token is not an integer, or is below one.
    """
    value = require_int_field(raw, field)
    if value < 1:
        raise WireFormatError(
            "NP-WIRE-003", f"field {field!r} must be greater than zero, got {value}"
        )
    return value


def require_text_field(raw: str, field: str) -> str:
    """Validate a token that must be non-empty text.

    Args:
        raw: The token to validate.
        field: Field name, used in the error message.

    Returns:
        The token unchanged.

    Raises:
        WireFormatError: When the token is empty. An empty label or digest is
            always a construction bug rather than a meaningful value.
    """
    if raw == "":
        raise WireFormatError("NP-WIRE-004", f"field {field!r} must not be empty")
    return raw


def require_bool_field(raw: str, field: str) -> bool:
    """Convert a token to a boolean.

    Only the two spelled tokens are accepted. Treating any other value as false
    would let a typo read as a negative determinism verdict, which is the one
    result this instrument must never invent.

    Args:
        raw: The token to convert.
        field: Field name, used in the error message.

    Returns:
        The token as a boolean.

    Raises:
        WireFormatError: When the token is neither :data:`TRUE_TOKEN` nor
            :data:`FALSE_TOKEN`.
    """
    if raw not in {TRUE_TOKEN, FALSE_TOKEN}:
        raise WireFormatError(
            "NP-WIRE-012",
            f"field {field!r} must be {TRUE_TOKEN!r} or {FALSE_TOKEN!r}, got {raw!r}",
        )
    return raw == TRUE_TOKEN


def require_optional_non_negative_field(raw: str, field: str) -> int | None:
    """Convert a token to a non-negative integer or to an explicit absence.

    Args:
        raw: The token to convert.
        field: Field name, used in the error message.

    Returns:
        ``None`` when the token is :data:`NONE_TOKEN`, otherwise the token as a
        non-negative integer.

    Raises:
        WireFormatError: When the token is neither :data:`NONE_TOKEN` nor a
            non-negative integer.
    """
    if raw == NONE_TOKEN:
        return None
    return require_non_negative_field(raw, field)


def encode_optional_int(value: int | None) -> str:
    """Encode an optional integer to its token.

    Args:
        value: The value to encode.

    Returns:
        :data:`NONE_TOKEN` when ``value`` is ``None``, otherwise its decimal
        form.
    """
    if value is None:
        return NONE_TOKEN
    return str(value)


def encode_bool(value: bool) -> str:
    """Encode a boolean to its token.

    Args:
        value: The value to encode.

    Returns:
        :data:`TRUE_TOKEN` or :data:`FALSE_TOKEN`.
    """
    if value:
        return TRUE_TOKEN
    return FALSE_TOKEN


def encode_float_field(value: float) -> str:
    """Encode a float exactly.

    Args:
        value: The value to encode.

    Returns:
        The value's hexadecimal form, which round-trips without loss.
    """
    return value.hex()


def require_hexadecimal_float(raw: str, field: str) -> float:
    """Convert a hexadecimal token to a float.

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
            "NP-WIRE-014",
            f"field {field!r} must be a hexadecimal float, got {raw!r}",
        )
    return float.fromhex(raw)


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
    value = require_hexadecimal_float(raw, field)
    if not value > 0.0:
        raise WireFormatError(
            "NP-WIRE-015", f"field {field!r} must be greater than zero, got {value}"
        )
    return value


def require_float_field(raw: str, field: str) -> float:
    """Convert a hexadecimal token to a float of any sign.

    The unconstrained variant. An observed value has no range at all: a
    position may be negative, a depth may be zero, and neither is a
    construction error. The only thing refused is a token that is not a float.

    Args:
        raw: The token to convert.
        field: Field name, used in the error message.

    Returns:
        The token as a float.

    Raises:
        WireFormatError: When the token is not a hexadecimal float.
    """
    return require_hexadecimal_float(raw, field)


def require_non_negative_float_field(raw: str, field: str) -> float:
    """Convert a hexadecimal token to a float of zero or greater.

    The variant a *measurement* needs rather than a scene parameter. A spread
    is a range and cannot be below zero; an elapsed time may legitimately be
    zero where a spacing or a radius may not.

    Args:
        raw: The token to convert.
        field: Field name, used in the error message.

    Returns:
        The token as a float.

    Raises:
        WireFormatError: When the token is not a hexadecimal float, or is
            negative.
    """
    value = require_hexadecimal_float(raw, field)
    if value < 0.0:
        raise WireFormatError(
            "NP-WIRE-016", f"field {field!r} must be zero or greater, got {value}"
        )
    return value


def header_line(key: str, value: str) -> str:
    """Build one ``key<TAB>value`` header line.

    Args:
        key: The field name.
        value: The field's encoded value.

    Returns:
        The joined line, without a trailing newline.
    """
    return f"{key}{SEPARATOR}{value}"


def split_header_line(line: str, expected_key: str) -> str:
    """Split one ``key<TAB>value`` header line and return its value.

    Args:
        line: The line to split.
        expected_key: The key this line must carry, which pins field order.

    Returns:
        The line's value.

    Raises:
        WireFormatError: When the line is not a two-token pair, or carries a
            different key than the format's fixed order requires.
    """
    parts = line.split(SEPARATOR)
    if len(parts) != 2:
        raise WireFormatError(
            "NP-WIRE-005",
            f"header line for {expected_key!r} must have two tab-separated tokens, "
            f"got {len(parts)}",
        )
    key, value = parts
    if key != expected_key:
        raise WireFormatError("NP-WIRE-006", f"expected header field {expected_key!r}, got {key!r}")
    return value


def split_document(
    text: str, banner: str, header_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split encoded text into its header lines and its body lines.

    Shared by every record decoder so the banner check, the minimum line count,
    and the header/body boundary are stated once. Guarantees the returned header
    tuple has exactly ``header_count`` entries, which is what lets each record's
    field decoder index its slice without re-checking the length.

    Args:
        text: The encoded record.
        banner: The banner this record type must carry.
        header_count: Number of header lines this record type declares.

    Returns:
        The header lines and the body lines that follow them.

    Raises:
        WireFormatError: When the banner is absent or belongs to a different
            record type, or when the text carries fewer lines than the header
            alone requires.
    """
    lines = text.strip("\n").split("\n")
    if lines[0] != banner:
        raise WireFormatError(
            "NP-WIRE-009", f"expected banner {banner!r} on the first line, got {lines[0]!r}"
        )
    if len(lines) < header_count + 1:
        raise WireFormatError(
            "NP-WIRE-010",
            f"record needs a banner and {header_count} header lines, got {len(lines)} lines",
        )
    return tuple(lines[1 : header_count + 1]), tuple(lines[header_count + 1 :])


def require_no_body(body: tuple[str, ...], banner: str) -> None:
    """Refuse trailing lines after a record type that declares no body.

    Args:
        body: The lines found after the header.
        banner: The record's banner, used in the error message.

    Raises:
        WireFormatError: When any line follows the header. Trailing content
            means the text is not the record it claims to be, and ignoring it
            would let two different documents decode to the same value.
    """
    if body:
        raise WireFormatError(
            "NP-WIRE-013",
            f"{banner!r} declares no rows but carries {len(body)} trailing lines",
        )


def join_document(lines: list[str]) -> str:
    """Join encoded lines into a finished document.

    Args:
        lines: The banner, header lines, and any body rows, in order.

    Returns:
        The joined text, newline-terminated as a text file should be.
    """
    return "\n".join(lines) + "\n"


__all__ = [
    "FALSE_TOKEN",
    "NONE_TOKEN",
    "SEPARATOR",
    "TRUE_TOKEN",
    "WireFormatError",
    "encode_bool",
    "encode_float_field",
    "encode_optional_int",
    "header_line",
    "join_document",
    "require_bool_field",
    "require_float_field",
    "require_hexadecimal_float",
    "require_int_field",
    "require_no_body",
    "require_non_negative_field",
    "require_non_negative_float_field",
    "require_optional_non_negative_field",
    "require_positive_field",
    "require_positive_float_field",
    "require_text_field",
    "split_document",
    "split_header_line",
]
