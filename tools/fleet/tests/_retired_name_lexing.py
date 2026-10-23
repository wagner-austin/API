"""Where a tracked file's comments and string literals sit, for the retired
machine name scan (test_retired_machine_names.py, MCPs board task 3f37b2a7).

This is MCPs ``mcp-shared/tests/architecture/_retired-name-lexing.ts`` (board
task 88f5676c) in this repository's language, and it reads each language the
way that module does where this repository has the same means:

- PYTHON goes through Python's own :mod:`tokenize`, in process, so a
  docstring is a string statement as Python reads one and a ``#`` inside a
  string is no comment.
- JSON is read by :func:`_json_spans`, which tells a key from a value the way
  the grammar does. A key is a name, never a literal, and the values of
  :data:`JSON_PROSE_KEYS` are prose: ``_comment`` and ``$comment`` as in MCPs,
  and ``notes``, the free text each roster file under
  ``tools/fleet/registry/machines`` keeps its history in, which no program
  here interprets (``fleet.contracts.machine`` leaves it to MCPs' decoder,
  which requires only that it is a string).
- The line comment languages (shell, YAML, TOML, INI, Makefile, PowerShell,
  SQL, git's attribute and ignore files) are read by :func:`_marker_spans`,
  which honours their quotes, their block comments and PowerShell's
  here-strings, so a ``#`` inside any of those strings is no comment.
- Markdown, and anything under a wiki, is prose throughout.
- Anything else is DATA: it has neither comments nor literals, so every line
  of it naming the name must be accounted for by a kept class. That includes
  the TypeScript, Rust and Java this repository carries, which MCPs reads with
  the TypeScript compiler and this repository has no lexer for: a name there
  is held to a class even in a comment, which is strict rather than lenient.
"""

from __future__ import annotations

import io
import re
import tokenize
from collections.abc import Mapping
from enum import StrEnum
from typing import Final, NamedTuple


class Span(NamedTuple):
    """A half-open range of offsets into a file's text."""

    start: int
    end: int


class Lexer(StrEnum):
    """How a file is read."""

    PYTHON = "python"
    JSON = "json"
    HASH = "hash"
    INI = "ini"
    POWERSHELL = "powershell"
    SQL = "sql"
    PROSE = "prose"
    DATA = "data"


class Lexed(NamedTuple):
    """What a file's lexer found.

    Attributes:
        lexer: The lexer that read it.
        comments: Comments, docstrings and prose: text no program reads as code.
        strings: String literals, quotes and prefixes included.
    """

    lexer: Lexer
    comments: tuple[Span, ...]
    strings: tuple[Span, ...]


#: The JSON keys whose values are prose rather than data.
JSON_PROSE_KEYS: Final[frozenset[str]] = frozenset({"_comment", "$comment", "notes"})

#: File names, without an extension, whose lines comment with ``#``.
HASH_NAMES: Final[frozenset[str]] = frozenset(
    {
        "Makefile",
        "Dockerfile",
        "pre-push",
        "pre-commit",
        "commit-msg",
        ".gitattributes",
        ".gitignore",
        ".dockerignore",
    }
)

#: Extensions read as line comment languages with ``#``.
HASH_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {"sh", "bash", "yml", "yaml", "toml", "example", "mk", "cfg", "conf"}
)

#: The lexer each extension other than Markdown's names.
_BY_EXTENSION: Final[dict[str, Lexer]] = {
    "py": Lexer.PYTHON,
    "json": Lexer.JSON,
    "ps1": Lexer.POWERSHELL,
    "psm1": Lexer.POWERSHELL,
    "psd1": Lexer.POWERSHELL,
    "sql": Lexer.SQL,
    "ini": Lexer.INI,
    **dict.fromkeys(HASH_EXTENSIONS, Lexer.HASH),
}


def lexer_for(path: str) -> Lexer:
    """Choose a file's lexer from its path.

    Args:
        path: The repository-relative path.

    Returns:
        The lexer; data for an extension no rule names.
    """
    name = path[path.rfind("/") + 1 :]
    dot = name.rfind(".")
    extension = name[dot + 1 :].lower() if dot > 0 else ""
    if extension == "md" or path.startswith("wiki/") or "/wiki/" in path:
        return Lexer.PROSE
    if extension in _BY_EXTENSION:
        return _BY_EXTENSION[extension]
    if (dot <= 0 and name in HASH_NAMES) or name.startswith(".env"):
        return Lexer.HASH
    return Lexer.DATA


def line_starts_of(text: str) -> tuple[int, ...]:
    """The offset every line of a text starts at.

    Args:
        text: The text.

    Returns:
        One offset per line, the first 0.
    """
    return (0, *(match.end() for match in re.finditer("\n", text)))


#: The token names that open and close an f-string or t-string, which
#: Python 3.12 and later split into pieces and 3.11 gives as one STRING.
_PIECE_OPEN: Final[frozenset[str]] = frozenset({"FSTRING_START", "TSTRING_START"})
_PIECE_CLOSE: Final[frozenset[str]] = frozenset({"FSTRING_END", "TSTRING_END"})

#: The tokens after which a new statement starts.
_STATEMENT_STARTS: Final[frozenset[int]] = frozenset(
    {tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}
)


def _python_tokens(text: str) -> list[tuple[int, Span]]:
    """Each token Python reads, with an f-string's or t-string's pieces joined.

    Args:
        text: The file's text.

    Returns:
        Each token's type and span; a split f-string or t-string is one
        ``STRING`` from its start to its end, and a comment inside one is kept.

    Raises:
        tokenize.TokenError: When the text is not Python tokenize can read.
    """
    starts = line_starts_of(text)
    joined: list[tuple[int, Span]] = []
    depth = 0
    piece_start = 0
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        name = tokenize.tok_name[token.type]
        (start_row, start_column), (end_row, end_column) = token.start, token.end
        span = Span(starts[start_row - 1] + start_column, starts[end_row - 1] + end_column)
        if name in _PIECE_OPEN:
            piece_start = span.start if depth == 0 else piece_start
            depth += 1
        elif name in _PIECE_CLOSE:
            depth -= 1
            if depth == 0:
                joined.append((tokenize.STRING, Span(piece_start, span.end)))
        elif depth == 0 or token.type == tokenize.COMMENT:
            joined.append((token.type, span))
    return joined


def _python_spans(text: str) -> tuple[list[Span], list[Span]]:
    """Python's own reading of a file's comments, docstrings and literals.

    A docstring is a statement that is only strings, which is how Python
    reads one, so a string statement anywhere in the module counts.

    Args:
        text: The file's text.

    Returns:
        The comments, docstrings included, and the string literals.
    """
    comments: list[Span] = []
    strings: list[Span] = []
    statement: list[Span] = []
    at_start = True
    for kind, span in _python_tokens(text):
        if kind == tokenize.COMMENT:
            comments.append(span)
        elif kind == tokenize.STRING:
            strings.append(span)
            statement.append(span)
        elif kind != tokenize.NL:
            if kind in (tokenize.NEWLINE, tokenize.ENDMARKER) and at_start and statement:
                comments.extend(statement)
            at_start = kind in _STATEMENT_STARTS
            statement = []
    return comments, strings


def _string_end(text: str, start: int) -> int:
    """Where a JSON string opening at ``start`` ends.

    Args:
        text: The text.
        start: The offset of its opening quote.

    Returns:
        The offset after its closing quote, or the text's end when it has none.
    """
    i = start + 1
    while i < len(text) and text[i] != '"':
        i += 2 if text[i] == "\\" else 1
    return min(i + 1, len(text))


def _skip_space(text: str, at: int) -> int:
    """The first offset at or after ``at`` that is not whitespace.

    Args:
        text: The text.
        at: Where to start.

    Returns:
        That offset, or the text's length.
    """
    while at < len(text) and text[at].isspace():
        at += 1
    return at


def _value_end(text: str, start: int) -> int:
    """Where a JSON value opening at ``start`` ends.

    Args:
        text: The text.
        start: The offset of its first character.

    Returns:
        The offset after it: after its closing quote or bracket, or before the
        separator ending a scalar.
    """
    if start < len(text) and text[start] == '"':
        return _string_end(text, start)
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == '"':
            i = _string_end(text, i)
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}" and depth > 0:
            depth -= 1
            if depth == 0:
                return i + 1
        elif depth == 0 and (ch in ",]}" or ch.isspace()):
            return i
        i += 1
    return i


def _json_spans(text: str) -> tuple[list[Span], list[Span]]:
    """A JSON file's prose values and its string literals.

    Args:
        text: The file's text.

    Returns:
        The values of :data:`JSON_PROSE_KEYS`, and every string that is not a
        key.
    """
    comments: list[Span] = []
    strings: list[Span] = []
    i = 0
    while i < len(text):
        if text[i] != '"':
            i += 1
            continue
        end = _string_end(text, i)
        after = _skip_space(text, end)
        if after < len(text) and text[after] == ":":
            if text[i + 1 : end - 1] in JSON_PROSE_KEYS:
                value = _skip_space(text, after + 1)
                comments.append(Span(value, _value_end(text, value)))
        else:
            strings.append(Span(i, end))
        i = end
    return comments, strings


class _Marker(NamedTuple):
    """One line comment marker, and whether whitespace or a line start must precede it."""

    text: str
    bounded: bool


class _MarkerLanguage(NamedTuple):
    """How one line comment language writes comments and strings.

    Attributes:
        markers: Its line comment markers.
        block: Its block comment's opening and closing text, or None.
        quotes: Its quote characters; a backslash escapes inside a double quote.
        here_strings: Each multi-line string's opening and closing text; it
            runs from the opening to the closing at the start of a later line.
    """

    markers: tuple[_Marker, ...]
    block: tuple[str, str] | None
    quotes: tuple[str, ...]
    here_strings: tuple[tuple[str, str], ...]


#: The line comment languages.
_MARKER_LANGUAGES: Final[dict[Lexer, _MarkerLanguage]] = {
    Lexer.HASH: _MarkerLanguage((_Marker("#", True),), None, ("'", '"'), ()),
    Lexer.INI: _MarkerLanguage((_Marker(";", True), _Marker("#", True)), None, ('"',), ()),
    Lexer.POWERSHELL: _MarkerLanguage(
        (_Marker("#", True),), ("<#", "#>"), ("'", '"'), (('@"', '"@'), ("@'", "'@"))
    ),
    Lexer.SQL: _MarkerLanguage((_Marker("--", False),), ("/*", "*/"), ("'",), ()),
}


def _closed_at(text: str, start: int, close: str) -> int:
    """Where a construct closed by ``close`` ends, searching from ``start``.

    Args:
        text: The text.
        start: Where the search begins.
        close: The closing text.

    Returns:
        The offset after the closing text, or the text's end when it has none.
    """
    at = text.find(close, start)
    return len(text) if at < 0 else at + len(close)


def _quote_end(text: str, start: int) -> tuple[int, bool]:
    """Where a quote opening at ``start`` ends, which is never past its line.

    Args:
        text: The text.
        start: The offset of the opening quote.

    Returns:
        The offset after the closing quote and True, or the line's end and
        False when the line closes no quote.
    """
    quote = text[start]
    i = start + 1
    while i < len(text) and text[i] != "\n":
        if text[i] == quote:
            return i + 1, True
        i += 2 if text[i] == "\\" and quote == '"' else 1
    return i, False


def _marker_step(
    text: str, i: int, language: _MarkerLanguage, comments: list[Span], strings: list[Span]
) -> int:
    """Read what starts at one offset of a line comment language.

    Args:
        text: The file's text.
        i: The offset, outside any comment or string.
        language: How the language writes comments and strings.
        comments: The comments so far, extended by one that starts here.
        strings: The strings so far, extended by one that starts here.

    Returns:
        The offset to read next.
    """
    here = next((h for h in language.here_strings if text.startswith(h[0], i)), None)
    if here is not None:
        end = _closed_at(text, i + len(here[0]), f"\n{here[1]}")
        strings.append(Span(i, end))
        return end
    if language.block is not None and text.startswith(language.block[0], i):
        end = _closed_at(text, i + len(language.block[0]), language.block[1])
        comments.append(Span(i, end))
        return end
    bounded = i == 0 or text[i - 1].isspace()
    if any(text.startswith(m.text, i) and (bounded or not m.bounded) for m in language.markers):
        line_end = text.find("\n", i)
        end = len(text) if line_end < 0 else line_end
        comments.append(Span(i, end))
        return end
    if text[i] in language.quotes:
        end, closed = _quote_end(text, i)
        if closed:
            strings.append(Span(i, end))
        return end
    return i + 1


def _marker_spans(text: str, language: _MarkerLanguage) -> tuple[list[Span], list[Span]]:
    """Comments and literals of a line comment language.

    Args:
        text: The file's text.
        language: How it writes them.

    Returns:
        The comments and the strings. A quote does not run past its line, so
        one apostrophe in a comment-free line cannot swallow the rest of the
        file; a block comment or here-string with no close runs to the end.
    """
    comments: list[Span] = []
    strings: list[Span] = []
    i = 0
    while i < len(text):
        i = _marker_step(text, i, language, comments, strings)
    return comments, strings


def lex(path: str, text: str) -> Lexed:
    """Lex one file.

    Args:
        path: Its repository-relative path, which picks the lexer.
        text: Its text.

    Returns:
        Its lexer and spans.
    """
    lexer = lexer_for(path)
    if lexer is Lexer.PROSE:
        return Lexed(lexer, (Span(0, len(text)),), ())
    if lexer is Lexer.DATA:
        return Lexed(lexer, (), ())
    if lexer is Lexer.PYTHON:
        comments, strings = _python_spans(text)
    elif lexer is Lexer.JSON:
        comments, strings = _json_spans(text)
    else:
        comments, strings = _marker_spans(text, _MARKER_LANGUAGES[lexer])
    return Lexed(lexer, tuple(comments), tuple(strings))


def lex_all(texts: Mapping[str, str]) -> dict[str, Lexed]:
    """Lex every file.

    Args:
        texts: Each file's path and text.

    Returns:
        Each file's lexer and spans, by path.
    """
    return {path: lex(path, text) for path, text in texts.items()}


def within(spans: tuple[Span, ...], offset: int) -> bool:
    """Whether an offset lies inside any of the spans.

    Args:
        spans: The spans.
        offset: The offset.

    Returns:
        True when a span holds it.
    """
    return any(span.start <= offset < span.end for span in spans)
