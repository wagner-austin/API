"""A strict reader for the agent's newline-delimited JSON.

The standard library is not used here, and the reason is a hard constraint
rather than a preference. This package is type-checked with
``disallow_any_expr``, under which ``json.loads`` is unusable: its return type
is ``Any``, every expression touching that value is an error, ``isinstance``
narrowing does not help because the expression itself is rejected, and
suppressions are banned. Parsing the format directly is what keeps the consumer
fully typed end to end.

That is affordable only because the producer is constrained to match. Every
line the agent writes is a flat JSON object whose values are strings, numbers or
booleans — no nesting, no arrays, no null (see ``StateStream`` on the agent
side). This module implements exactly that grammar and rejects everything else,
so a producer that starts nesting fails loudly here instead of being silently
half-read.

Nothing is coerced. ``"4250"`` is a string, not a number, and a duplicate key is
an error rather than a last-one-wins merge — the same discipline the
``require_*`` validators apply one layer up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot import RwBotError

_NOT_OBJECT = "RW-NDJSON-001"
_UNEXPECTED = "RW-NDJSON-002"
_BAD_STRING = "RW-NDJSON-003"
_BAD_NUMBER = "RW-NDJSON-004"
_DUPLICATE_KEY = "RW-NDJSON-005"
_TRAILING = "RW-NDJSON-006"

_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class NdjsonError(RwBotError):
    """A line did not match the flat-object grammar the agent emits.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description, including the offending offset.
    """


def parse_object(text: str) -> dict[str, str | int | float | bool]:
    """Parse one NDJSON line into a flat mapping.

    Args:
        text: A single line, without its newline terminator.

    Returns:
        The object's fields, with JSON strings, numbers and booleans narrowed to
        ``str``, ``int``/``float`` and ``bool`` respectively.

    Raises:
        NdjsonError: ``RW-NDJSON-001`` when the line is not an object,
            ``RW-NDJSON-002`` on an unexpected character, ``RW-NDJSON-003`` on a
            malformed string, ``RW-NDJSON-004`` on a malformed number,
            ``RW-NDJSON-005`` on a duplicate key, ``RW-NDJSON-006`` when
            anything follows the closing brace.
    """
    fields: dict[str, str | int | float | bool] = {}
    index = _skip_space(text, 0)
    index = _expect(text, index, "{", _NOT_OBJECT)
    index = _skip_space(text, index)

    if _peek(text, index) == "}":
        index = _skip_space(text, index + 1)
        _require_end(text, index)
        return fields

    while True:
        index = _skip_space(text, index)
        key, index = _read_string(text, index)
        if key in fields:
            raise NdjsonError(_DUPLICATE_KEY, f"duplicate key {key!r} at offset {index}")
        index = _skip_space(text, index)
        index = _expect(text, index, ":", _UNEXPECTED)
        index = _skip_space(text, index)
        value, index = _read_value(text, index)
        fields[key] = value
        index = _skip_space(text, index)
        char = _peek(text, index)
        if char == ",":
            index += 1
            continue
        if char == "}":
            index = _skip_space(text, index + 1)
            _require_end(text, index)
            return fields
        raise NdjsonError(_UNEXPECTED, f"expected ',' or '}}' at offset {index}, got {char!r}")


def _read_value(text: str, index: int) -> tuple[str | int | float | bool, int]:
    """Read one scalar value.

    Args:
        text: The line being parsed.
        index: Offset of the value's first character.

    Returns:
        The value and the offset just past it.

    Raises:
        NdjsonError: ``RW-NDJSON-002`` when the value is not a scalar this
            grammar admits, including nested objects and arrays.
    """
    char = _peek(text, index)
    if char == '"':
        return _read_string(text, index)
    if text.startswith("true", index):
        return True, index + 4
    if text.startswith("false", index):
        return False, index + 5
    if char == "-" or char.isdigit():
        return _read_number(text, index)
    raise NdjsonError(
        _UNEXPECTED,
        f"unexpected value at offset {index}: {char!r}; this grammar admits only "
        "strings, numbers and booleans",
    )


def _read_string(text: str, index: int) -> tuple[str, int]:
    """Read a quoted string, resolving escapes.

    Args:
        text: The line being parsed.
        index: Offset of the opening quote.

    Returns:
        The decoded string and the offset just past the closing quote.

    Raises:
        NdjsonError: ``RW-NDJSON-003`` when the string is unterminated or an
            escape is malformed.
    """
    index = _expect(text, index, '"', _BAD_STRING)
    out: list[str] = []
    while index < len(text):
        char = text[index]
        if char == '"':
            return "".join(out), index + 1
        if char != "\\":
            out.append(char)
            index += 1
            continue
        index += 1
        if index >= len(text):
            raise NdjsonError(_BAD_STRING, f"escape runs past end of line at offset {index}")
        marker = text[index]
        if marker == "u":
            out.append(_read_unicode_escape(text, index + 1))
            index += 5
            continue
        if marker not in _ESCAPES:
            raise NdjsonError(_BAD_STRING, f"unknown escape {marker!r} at offset {index}")
        out.append(_ESCAPES[marker])
        index += 1
    raise NdjsonError(_BAD_STRING, f"unterminated string at offset {index}")


def _read_unicode_escape(text: str, index: int) -> str:
    """Decode the four hex digits of a ``\\uXXXX`` escape.

    Args:
        text: The line being parsed.
        index: Offset of the first hex digit.

    Returns:
        The single decoded character.

    Raises:
        NdjsonError: ``RW-NDJSON-003`` when fewer than four hex digits follow.
    """
    digits = text[index : index + 4]
    if len(digits) != 4 or any(d not in "0123456789abcdefABCDEF" for d in digits):
        raise NdjsonError(_BAD_STRING, f"malformed \\u escape at offset {index}: {digits!r}")
    return chr(int(digits, 16))


def _read_number(text: str, index: int) -> tuple[int | float, int]:
    """Read a number, returning ``int`` when it has no fractional or exponent part.

    Args:
        text: The line being parsed.
        index: Offset of the number's first character.

    Returns:
        The value and the offset just past it.

    Raises:
        NdjsonError: ``RW-NDJSON-004`` when the run of characters is not a
            number Python can read exactly.
    """
    end = index
    if _peek(text, end) == "-":
        end += 1
    while end < len(text) and (text[end].isdigit() or text[end] in ".eE+-"):
        end += 1
    literal = text[index:end]
    is_integer = "." not in literal and "e" not in literal and "E" not in literal
    try:
        return (int(literal) if is_integer else float(literal)), end
    except ValueError as error:
        raise NdjsonError(_BAD_NUMBER, f"malformed number {literal!r} at offset {index}") from error


def _skip_space(text: str, index: int) -> int:
    """Advance past insignificant whitespace.

    Args:
        text: The line being parsed.
        index: Offset to start from.

    Returns:
        The offset of the next non-space character.
    """
    while index < len(text) and text[index] in " \t":
        index += 1
    return index


def _peek(text: str, index: int) -> str:
    """Return the character at an offset, or the empty string past the end.

    Args:
        text: The line being parsed.
        index: Offset to read.

    Returns:
        A one-character string, or ``""`` at end of line.
    """
    return text[index] if index < len(text) else ""


def _expect(text: str, index: int, char: str, code: str) -> int:
    """Consume one required character.

    Args:
        text: The line being parsed.
        index: Offset to read.
        char: The character required there.
        code: Error code to raise under when it is absent.

    Returns:
        The offset just past the consumed character.

    Raises:
        NdjsonError: Under ``code`` when the character is not present.
    """
    if _peek(text, index) != char:
        raise NdjsonError(code, f"expected {char!r} at offset {index}, got {_peek(text, index)!r}")
    return index + 1


def _require_end(text: str, index: int) -> None:
    """Assert that nothing but whitespace follows.

    Args:
        text: The line being parsed.
        index: Offset just past the closing brace.

    Raises:
        NdjsonError: ``RW-NDJSON-006`` when any content remains. Two objects on
            one line would otherwise be read as one, which is exactly the
            corruption newline-delimiting exists to prevent.
    """
    if index != len(text):
        raise NdjsonError(
            _TRAILING, f"unexpected content after the object at offset {index}: {text[index:]!r}"
        )


__all__ = ["NdjsonError", "parse_object"]


_STRING_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _render_string(value: str) -> str:
    """Render one JSON string literal.

    Args:
        value: The string to render.

    Returns:
        The quoted, escaped literal.
    """
    rendered: list[str] = ['"']
    for character in value:
        if character in _STRING_ESCAPES:
            rendered.append(_STRING_ESCAPES[character])
        elif ord(character) < 0x20:
            rendered.append(f"\\u{ord(character):04x}")
        else:
            rendered.append(character)
    rendered.append('"')
    return "".join(rendered)


def render_json(
    payload: Mapping[
        str,
        str | int | bool | None | Sequence[str] | Sequence[Mapping[str, str | int | bool | None]],
    ],
) -> str:
    """Render one response object as JSON.

    The emit-side sibling of :func:`rw_bot.wire.ndjson.parse_object`:
    the value grammar is exactly what the fleet's responses carry —
    flat scalars, a list of strings (the report lines), or a list of
    flat objects (the match rows). Anything else is a programming
    error, not data.

    Args:
        payload: The response object.

    Returns:
        Its JSON text.
    """
    parts: list[str] = []
    for key, value in payload.items():
        parts.append(f"{_render_string(key)}: {_render_value(value)}")
    return "{" + ", ".join(parts) + "}"


def _render_value(
    value: str
    | int
    | bool
    | None
    | Sequence[str]
    | Sequence[Mapping[str, str | int | bool | None]],
) -> str:
    """Render one value of the shapes the fleet serves.

    Args:
        value: A flat scalar, a list of strings, or a list of flat
            objects.

    Returns:
        Its JSON rendering.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _render_string(value)
    items: list[str] = []
    for item in value:
        items.append(_render_string(item) if isinstance(item, str) else render_json(item))
    return "[" + ", ".join(items) + "]"
