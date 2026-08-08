"""The doctrine's file form: one ``field value`` line per knob.

Split from :mod:`rw_bot.policy.doctrine` when the schema's growth pushed that
module past the size cap -- the payload codec and the schema stay there; what
lives here is only the disk format a preset is written in and read from. The
split is a real seam: everything below concerns lines, comments and the
field-kind tables, and nothing below knows what any field means.

The five error codes here are the file form's own -- a malformed line, an
unknown or repeated field, a value of the wrong shape. What a field's VALUE
may be is :func:`~rw_bot.policy.doctrine.decode_doctrine`'s question, asked
after this module has turned lines into a payload.
"""

from __future__ import annotations

from collections.abc import Sequence

from rw_bot.policy.doctrine import (
    DOCTRINE_FIELDS,
    FLAG_FIELDS,
    INT_FIELDS,
    STR_FIELDS,
    Doctrine,
    DoctrineError,
)
from rw_bot.policy.doctrine_codecs import decode_doctrine, encode_doctrine

_FIELD_SHAPE = "RW-DOCTRINE-001"
_UNKNOWN_FIELD = "RW-DOCTRINE-002"
_NOT_A_NUMBER = "RW-DOCTRINE-003"
_NOT_A_FLAG = "RW-DOCTRINE-004"
_REPEATED_FIELD = "RW-DOCTRINE-005"


def parse_doctrine_lines(lines: Sequence[str]) -> Doctrine:
    """Read a doctrine file: one ``field value`` pair per line.

    Blank lines and ``#`` comments are skipped, so a preset can record why its
    values are what they are beside the values themselves. Fields may appear in
    any order; each exactly once.

    Args:
        lines: The file's lines, without newlines.

    Returns:
        The doctrine it describes.

    Raises:
        DoctrineError: When a line is malformed, names an unknown field,
            repeats one, or carries a value of the wrong shape.
        DecodeError: When a field is absent or out of range.
    """
    payload: dict[str, str | int | float | bool] = {}
    for line in lines:
        bare = line.strip()
        if not bare or bare.startswith("#"):
            continue
        field, _, raw = bare.partition(" ")
        raw = raw.strip()
        if not raw:
            raise DoctrineError(_FIELD_SHAPE, f"a doctrine line is 'field value', got {line!r}")
        if field in payload:
            raise DoctrineError(_REPEATED_FIELD, f"field {field!r} appears twice")
        if field in STR_FIELDS:
            payload[field] = raw
        elif field in INT_FIELDS:
            try:
                payload[field] = int(raw)
            except ValueError as error:
                raise DoctrineError(
                    _NOT_A_NUMBER, f"field {field!r} must be a whole number, got {raw!r}"
                ) from error
        elif field in FLAG_FIELDS:
            if raw not in ("0", "1"):
                raise DoctrineError(_NOT_A_FLAG, f"field {field!r} must be 0 or 1, got {raw!r}")
            payload[field] = raw == "1"
        else:
            raise DoctrineError(
                _UNKNOWN_FIELD,
                f"field {field!r} is not one of {', '.join(DOCTRINE_FIELDS)}",
            )
    return decode_doctrine(payload)


def format_doctrine(doctrine: Doctrine) -> tuple[str, ...]:
    """Render a doctrine as the lines :func:`parse_doctrine_lines` reads.

    What a probe or a test writes when it needs a preset on disk, so the two
    formats cannot drift.

    Args:
        doctrine: The doctrine to render.

    Returns:
        One line per field, in :data:`~rw_bot.policy.doctrine.DOCTRINE_FIELDS`
        order.
    """
    flat = encode_doctrine(doctrine)
    rendered: list[str] = []
    for field in DOCTRINE_FIELDS:
        value = flat[field]
        if isinstance(value, bool):
            rendered.append(f"{field} {int(value)}")
        else:
            rendered.append(f"{field} {value}")
    return tuple(rendered)


__all__ = ["format_doctrine", "parse_doctrine_lines"]
