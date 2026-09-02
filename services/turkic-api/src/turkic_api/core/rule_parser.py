"""Turning tokens into rules.

The parser owns what a statement *means*: which elements are the before
context, which are the input, which are the after context, and what a variable
or a directive does. Every element it produces is a :class:`MatchSet`, so the
engine below it needs no notion of literals, sets or variables — only of whether
one accepts the character at a position.

It refuses more than it accepts. A rule that is parsed but misunderstood
transliterates silently and wrongly, which costs far more to find than a file
that will not load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, NamedTuple

from turkic_api.core.rule_errors import fail
from turkic_api.core.rule_lexer import Token, is_op, split_statements, tokenize

_RULES_DIR: Final[Path] = Path(__file__).with_suffix("").parent / "rules"
_NFC_DIRECTIVE: Final = "NFC"


class MatchSet(NamedTuple):
    """The characters one element of a pattern accepts.

    Attributes:
        members: The characters named literally, plus the contents of any
            variables the set referenced.
        negated: Whether the set was written ``[^...]``, in which case it
            accepts every character it does not name.
    """

    members: frozenset[str]
    negated: bool

    def accepts(self, char: str | None) -> bool:
        """Report whether this element matches one position.

        ``None`` means the position is past an end of the text. ICU treats
        that as matching a negated set and not matching a positive one, so
        ``a } [^b] > X`` rewrites a lone "a" while ``a } [b] > X`` does
        not. That was measured against PyICU 78.3, not inferred.

        Args:
            char: The character at the position, or None at a boundary.

        Returns:
            Whether the element accepts what is there.
        """
        if char is None:
            return self.negated
        return (char in self.members) != self.negated


class Rule(NamedTuple):
    """One transform rule, with every element reduced to a set of characters.

    Attributes:
        before: Elements matched right-anchored against the converted
            output, in order.
        anchor_start: Whether the rule only applies where nothing has been
            emitted yet. Like ``before``, it reads the converted output, so
            a preceding run of deletions leaves the position initial.
        source: Elements matched against the source at the cursor. Never
            empty; its length is what the cursor advances by.
        after: Elements matched against the source following ``source``.
        output: Replacement text. Empty means deletion.
    """

    before: tuple[MatchSet, ...]
    anchor_start: bool
    source: tuple[MatchSet, ...]
    after: tuple[MatchSet, ...]
    output: str


class RuleSet(NamedTuple):
    """A parsed rule file.

    Attributes:
        rules: Every rule, in the order written.
        normalize_nfc: Whether the file ended with ``:: NFC ;``.
    """

    rules: tuple[Rule, ...]
    normalize_nfc: bool


def _defined(
    name: str,
    macros: dict[str, frozenset[str]],
    offset: int,
    text: str,
) -> frozenset[str]:
    """Return the contents of a variable, refusing an undefined one.

    Args:
        name: Variable name, without the ``$``.
        macros: Variables defined so far.
        offset: Offset of the reference, for error reporting.
        text: Whole rule file, for error reporting.

    Returns:
        The characters the variable names.

    Raises:
        RuleParseError: If the variable has not been defined.
    """
    members = macros.get(name)
    if members is None:
        raise fail(text, offset, f"variable '${name}' is not defined")
    return members


def _elements(
    tokens: list[Token],
    macros: dict[str, frozenset[str]],
    text: str,
) -> tuple[MatchSet, ...]:
    """Reduce pattern tokens to one character set per element.

    Args:
        tokens: Tokens of one pattern section.
        macros: Variables defined so far.
        text: Whole rule file, for error reporting.

    Returns:
        One set per element, in order.

    Raises:
        RuleParseError: On an operator or anchor inside a section, or a
            reference to a variable that has not been defined.
    """
    elements: list[MatchSet] = []
    for token in tokens:
        if token.kind == "char":
            elements.append(MatchSet(frozenset({token.text}), False))
        elif token.kind == "set":
            members = set(token.members)
            for name in token.refs:
                members |= _defined(name, macros, token.offset, text)
            elements.append(MatchSet(frozenset(members), token.negated))
        elif token.kind == "var":
            elements.append(MatchSet(_defined(token.text, macros, token.offset, text), False))
        else:
            raise fail(text, token.offset, f"unexpected {token.text!r} in a rule pattern")
    return tuple(elements)


def _output_of(tokens: list[Token], text: str) -> str:
    """Join output tokens into replacement text.

    Args:
        tokens: Tokens to the right of ``>``.
        text: Whole rule file, for error reporting.

    Returns:
        The replacement, empty for a deletion.

    Raises:
        RuleParseError: If the output side is anything but literal
            characters. Sets and variables describe several characters at
            once, so they cannot name what to emit.
    """
    for token in tokens:
        if token.kind != "char":
            raise fail(text, token.offset, "a rule output must be literal characters")
    return "".join(token.text for token in tokens)


def _find_op(tokens: list[Token], value: str, text: str) -> int:
    """Return the index of the sole ``value`` operator, or -1 when absent.

    Raises:
        RuleParseError: If the operator appears more than once.
    """
    found = [i for i, token in enumerate(tokens) if is_op(token, value)]
    if len(found) > 1:
        raise fail(text, tokens[found[1]].offset, f"more than one {value!r} in one rule")
    return found[0] if found else -1


def _split_pattern(
    tokens: list[Token],
    text: str,
) -> tuple[list[Token], list[Token], list[Token]]:
    """Split a pattern into before-context, source, and after-context.

    Args:
        tokens: Pattern tokens, with any leading ``^`` already removed.
        text: Whole rule file, for error reporting.

    Returns:
        The three sections, any of which but the source may be empty.

    Raises:
        RuleParseError: If ``}`` precedes ``{``, or the source is empty.
    """
    brace_open = _find_op(tokens, "{", text)
    brace_close = _find_op(tokens, "}", text)
    if brace_open != -1 and brace_close != -1 and brace_close < brace_open:
        raise fail(text, tokens[brace_close].offset, "'}' appears before '{'")

    before = tokens[:brace_open] if brace_open != -1 else []
    rest = tokens[brace_open + 1 :] if brace_open != -1 else tokens
    cut = _find_op(rest, "}", text)
    source = rest[:cut] if cut != -1 else rest
    after = rest[cut + 1 :] if cut != -1 else []
    if not source:
        raise fail(text, tokens[0].offset, "rule matches nothing")
    return before, source, after


def _parse_rule(
    tokens: list[Token],
    macros: dict[str, frozenset[str]],
    text: str,
) -> Rule:
    """Parse one transform statement.

    Args:
        tokens: Statement tokens, without the trailing ``;``.
        macros: Variables defined so far.
        text: Whole rule file, for error reporting.

    Returns:
        The rule.

    Raises:
        RuleParseError: If the statement has no ``>``, or a ``^`` anywhere
            but at the front.
    """
    arrow = _find_op(tokens, ">", text)
    if arrow == -1:
        raise fail(text, tokens[0].offset, "statement is not a rule and has no '>'")

    pattern = tokens[:arrow]
    anchor_start = bool(pattern) and pattern[0].kind == "anchor_start"
    if anchor_start:
        pattern = pattern[1:]
    for token in pattern:
        if token.kind == "anchor_start":
            raise fail(text, token.offset, "'^' is only allowed at the start of a rule")
    if not pattern:
        raise fail(text, tokens[arrow].offset, "rule matches nothing")

    before, source, after = _split_pattern(pattern, text)
    return Rule(
        before=_elements(before, macros, text),
        anchor_start=anchor_start,
        source=_elements(source, macros, text),
        after=_elements(after, macros, text),
        output=_output_of(tokens[arrow + 1 :], text),
    )


def _parse_macro(
    tokens: list[Token],
    macros: dict[str, frozenset[str]],
    text: str,
) -> None:
    """Record a ``$Name = [ chars ]`` definition.

    A variable holds the characters it names, so a negated definition is
    refused rather than stored with its negation dropped. Nothing in the
    vendored files writes one, and a variable whose meaning silently
    inverted would be the class of defect this parser exists to prevent.

    Raises:
        RuleParseError: If the right-hand side is not exactly one set, or
            that set is negated.
    """
    if len(tokens) != 3 or tokens[2].kind != "set":
        raise fail(text, tokens[0].offset, "a variable must be defined as one '[...]' set")
    if tokens[2].negated:
        raise fail(text, tokens[2].offset, "a variable cannot be defined as a negated set")
    members = set(tokens[2].members)
    for name in tokens[2].refs:
        members |= _defined(name, macros, tokens[2].offset, text)
    macros[tokens[0].text] = frozenset(members)


def _parse_directive(tokens: list[Token], text: str) -> None:
    """Check that a ``::`` directive is one this interpreter implements.

    Only ``:: NFC`` is implemented, because it is the only directive the
    vendored files use. Anything else changes what the rules mean, so it
    is refused rather than ignored.

    Raises:
        RuleParseError: On any directive other than ``NFC``.
    """
    name = "".join(token.text for token in tokens[1:] if token.kind == "char")
    if name.upper() != _NFC_DIRECTIVE:
        raise fail(text, tokens[0].offset, f"unsupported directive '::{name}'")


def parse_rules(text: str) -> RuleSet:
    """Parse rule-file source into a rule set.

    Args:
        text: Contents of a rule file.

    Returns:
        The rules in the order written, and whether to normalise the
        output to NFC.

    Raises:
        RuleParseError: On any syntax this interpreter does not accept, or
            on a statement following ``:: NFC``. ICU would apply such a
            statement *after* normalising, which is not what a file
            ending in ``:: NFC ;`` means, so the ambiguity is refused.
    """
    macros: dict[str, frozenset[str]] = {}
    rules: list[Rule] = []
    normalize_nfc = False

    for statement in split_statements(tokenize(text)):
        if normalize_nfc:
            raise fail(text, statement[0].offset, "statement follows ':: NFC'")
        if is_op(statement[0], "::"):
            _parse_directive(statement, text)
            normalize_nfc = True
        elif statement[0].kind == "var" and len(statement) > 1 and is_op(statement[1], "="):
            _parse_macro(statement, macros, text)
        else:
            rules.append(_parse_rule(statement, macros, text))

    return RuleSet(rules=tuple(rules), normalize_nfc=normalize_nfc)


def load_rules(name: str) -> RuleSet:
    """Parse one of the vendored rule files.

    Args:
        name: File name within the packaged rules directory, including
            the ``.rules`` suffix.

    Returns:
        The parsed rule set.

    Raises:
        FileNotFoundError: When no such rule file is packaged.
        RuleParseError: When the file cannot be parsed.
    """
    return parse_rules((_RULES_DIR / name).read_text(encoding="utf-8"))


__all__ = [
    "MatchSet",
    "Rule",
    "RuleSet",
    "load_rules",
    "parse_rules",
]
