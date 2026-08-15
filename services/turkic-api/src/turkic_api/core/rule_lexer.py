"""Turning rule-file text into tokens.

The lexer owns everything that depends on how a rule file is *written* rather
than on what it means: quoting, escapes, comments, character sets, and where one
statement ends and the next begins. The parser above it never looks at a
character again.

Four details here are ICU's, not conveniences:

* ``'`` opens a quoted literal that runs to the next ``'``, so a ``;`` or ``>``
  inside one is an ordinary character. ``ar_lat.rules`` once shipped two bare
  apostrophes on one line, which paired with each other and swallowed the rule
  between them; the file now writes ``''``.
* ``''`` is one apostrophe.
* Inside ``[...]`` quoting does not apply, which is how ``$Apo`` in
  ``uz_ipa.rules`` contains a bare ``'``.
* A ``[...]`` set may open with ``^`` to accept everything it does not name,
  and may name a ``$Variable`` among its members.
"""

from __future__ import annotations

from typing import Final, Literal, NamedTuple

from turkic_api.core.rule_errors import fail

_OPERATORS: Final = frozenset("{}>=;")
_NAME_EXTRA: Final = "_"


class Token(NamedTuple):
    """One lexical unit of a rule file.

    Attributes:
        kind (Literal): Which kind of unit this is.
        offset (int): Offset into the source, used to report a line number.
            Named ``offset`` rather than ``index`` because ``tuple.index`` is
            a method and a NamedTuple field cannot shadow it.
        text (str): The literal character for ``char``, the operator for
            ``op``, the variable name for ``var``, empty otherwise.
        members (frozenset[str]): The characters a ``set`` names literally,
            empty otherwise.
        refs (tuple[str, ...]): Names of the variables a ``set`` references
            with ``$Name``, whose contents the parser adds to ``members``.
            Empty otherwise.
        negated (bool): Whether a ``set`` opened with ``^``, so that it
            accepts every character it does *not* name. False otherwise.
    """

    kind: Literal["char", "set", "var", "op", "anchor_start"]
    offset: int
    text: str
    members: frozenset[str]
    refs: tuple[str, ...] = ()
    negated: bool = False


def _read_name(text: str, start: int) -> tuple[str, int]:
    """Read a variable name beginning at ``start``.

    Args:
        text: Whole rule file.
        start: Offset of the first name character.

    Returns:
        The name, which is empty when no name is present, and the offset
        just past it.
    """
    end = start
    while end < len(text) and (text[end].isalnum() or text[end] in _NAME_EXTRA):
        end += 1
    return text[start:end], end


def _read_quote(text: str, start: int) -> tuple[str, int]:
    """Read a ``'``-quoted literal run beginning at ``start``.

    Args:
        text: Whole rule file.
        start: Offset of the opening quote.

    Returns:
        The literal characters and the offset just past the closing quote.
        An empty run means ``''``, which ICU reads as one apostrophe.

    Raises:
        RuleParseError: If the quote is never closed.
    """
    close = text.find("'", start + 1)
    if close == -1:
        raise fail(text, start, "unterminated quoted literal")
    body = text[start + 1 : close]
    return (body if body else "'"), close + 1


def _read_set_item(text: str, start: int, members: set[str], refs: list[str]) -> int:
    """Consume one item inside a ``[...]`` set and return the next offset.

    Args:
        text: Whole rule file.
        start: Offset of the item, which is inside the brackets.
        members: Accumulator for literal characters, added to in place.
        refs: Accumulator for ``$Name`` references, appended to in place.

    Returns:
        The offset just past what was consumed.

    Raises:
        RuleParseError: On a nested set, a trailing backslash, or a ``$``
            with no name after it.
    """
    ch = text[start]
    if ch == "[":
        raise fail(text, start, "nested character set")
    if ch == "\\":
        if start + 1 >= len(text):
            raise fail(text, start, "trailing backslash in character set")
        members.add(text[start + 1])
        return start + 2
    if ch == "$":
        name, nxt = _read_name(text, start + 1)
        if not name:
            raise fail(text, start, "'$' is not followed by a variable name")
        refs.append(name)
        return nxt
    if not ch.isspace():
        members.add(ch)
    return start + 1


def _read_set(text: str, start: int) -> tuple[frozenset[str], tuple[str, ...], bool, int]:
    """Read a ``[...]`` character set beginning at ``start``.

    Whitespace inside the brackets separates members and is discarded.
    Quoting does not apply inside a set; ``\\x`` still escapes ``x``. A
    leading ``^`` negates the set, and ``$Name`` inside the brackets names
    a variable whose members belong to the set; the parser resolves it,
    because the lexer does not know what has been defined.

    Args:
        text: Whole rule file.
        start: Offset of the opening bracket.

    Returns:
        The literal members, the variable names referenced, whether the
        set is negated, and the offset just past the closing bracket.

    Raises:
        RuleParseError: If the set is unterminated, nested, or names
            nothing at all.
    """
    members: set[str] = set()
    refs: list[str] = []
    i = start + 1
    negated = i < len(text) and text[i] == "^"
    if negated:
        i += 1
    while i < len(text) and text[i] != "]":
        i = _read_set_item(text, i, members, refs)
    if i >= len(text):
        raise fail(text, start, "unterminated character set")
    if not members and not refs:
        raise fail(text, start, "empty character set")
    return frozenset(members), tuple(refs), negated, i + 1


def _skip_trivia(text: str, start: int) -> int:
    """Return the offset of the next meaningful character at or after ``start``.

    Whitespace is insignificant outside quotes and sets, and ``#`` begins
    a comment that runs to the end of the line.
    """
    i = start
    while i < len(text):
        if text[i].isspace():
            i += 1
        elif text[i] == "#":
            newline = text.find("\n", i)
            i = len(text) if newline == -1 else newline
        else:
            break
    return i


def _lex_one(text: str, start: int, tokens: list[Token]) -> int:
    """Append the token beginning at ``start`` and return the next offset.

    Args:
        text: Whole rule file.
        start: Offset of a meaningful character.
        tokens: Accumulator appended to in place.

    Returns:
        The offset just past what was consumed.

    Raises:
        RuleParseError: On a trailing backslash or a bare ``$``.
    """
    ch = text[start]
    if ch == "'":
        literal, nxt = _read_quote(text, start)
        tokens.extend(Token("char", start, c, frozenset()) for c in literal)
        return nxt
    if ch == "\\":
        if start + 1 >= len(text):
            raise fail(text, start, "trailing backslash")
        tokens.append(Token("char", start, text[start + 1], frozenset()))
        return start + 2
    if ch == "[":
        members, refs, negated, nxt = _read_set(text, start)
        tokens.append(Token("set", start, "", members, refs, negated))
        return nxt
    if ch == "$":
        name, nxt = _read_name(text, start + 1)
        if not name:
            raise fail(text, start, "'$' is not followed by a variable name")
        tokens.append(Token("var", start, name, frozenset()))
        return nxt
    if text.startswith("::", start):
        tokens.append(Token("op", start, "::", frozenset()))
        return start + 2
    if ch == "^":
        tokens.append(Token("anchor_start", start, ch, frozenset()))
        return start + 1
    if ch in _OPERATORS:
        tokens.append(Token("op", start, ch, frozenset()))
        return start + 1
    tokens.append(Token("char", start, ch, frozenset()))
    return start + 1


def tokenize(text: str) -> list[Token]:
    """Lex a whole rule file into tokens, discarding comments and whitespace."""
    tokens: list[Token] = []
    i = 0
    while True:
        i = _skip_trivia(text, i)
        if i >= len(text):
            return tokens
        i = _lex_one(text, i, tokens)


def split_statements(tokens: list[Token]) -> list[list[Token]]:
    """Group tokens into statements, discarding the ``;`` separators."""
    statements: list[list[Token]] = []
    current: list[Token] = []
    for token in tokens:
        if token.kind == "op" and token.text == ";":
            if current:
                statements.append(current)
            current = []
        else:
            current.append(token)
    if current:
        statements.append(current)
    return statements


def is_op(token: Token, value: str) -> bool:
    """Return whether ``token`` is the operator ``value``."""
    return token.kind == "op" and token.text == value


__all__ = [
    "Token",
    "is_op",
    "split_statements",
    "tokenize",
]
