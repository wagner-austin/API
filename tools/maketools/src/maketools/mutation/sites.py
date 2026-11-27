"""What a source file says about the places cosmic-ray mutates (board task fec5851c).

THREE KINDS OF PLACE ARE NOT CODE, and the step does not run a mutant there.
A type annotation is a declaration for mypy: under ``from __future__ import
annotations`` it is never evaluated, so a mutant of ``str | None`` cannot be
killed by any test, as StrykerJS never mutates a TypeScript type. The ``/``
and ``*`` that end positional and begin keyword-only parameters are not
operators, and cosmic-ray's binary-operator replacement of them is not a
program. Measured over mcp-shared-py's src on 2026-11-07: 2730 of 8038
mutants sat in annotations and 539 on those markers. ``typing``'s
``overload`` and ``final`` (or ``typing_extensions``') declare to the type
checker what a definition may be overloaded or overridden as; at run time
they only record that in an attribute and a registry no test reads, so
removing one is no program a test can tell apart (board task b9bf5690).

WHICH DEFINITION A DECORATOR'S LINE BELONGS TO (board task b9bf5690). A
decorator runs once, at import, when its definition is made, so coverage
records its line under no test, and what removing ``@property`` or
``@classmethod`` changes is seen only where the definition is used. Each
line of a decorator is mapped to the lines of the definition it decorates,
from its first decorator to its last line, whose coverage names the tests
that use it.

WHICH LINE A MUTANT'S TESTS ARE READ FROM. coverage records a statement
spread over several lines under the lines its bytecode names, which for a
continued expression is often only the first: 23 mutants of mcp-shared-py
sat on lines coverage holds no context for. A line coverage did not record
is read under the first recorded line owned by the innermost statement
holding it. For ``if (`` on its own line, the first executing line is in
the predicate. Nested body statements own their lines independently and
cannot stand for an unexecuted condition (board task ee29c9e5).
"""

from __future__ import annotations

import ast
import io
import itertools
import tokenize
from typing import NamedTuple

#: Tokens that only lay out a line, skipped when finding the token after a marker.
_LAYOUT: frozenset[int] = frozenset(
    {tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT}
)

#: The tokens a parameter marker can stand before, and an operator never can.
_AFTER_MARKER: frozenset[str] = frozenset({",", ")", ":"})

#: The modules whose decorators below are declarations for the type checker.
_TYPING_MODULES: frozenset[str] = frozenset({"typing", "typing_extensions"})

#: The decorators of those modules that change nothing a test can see.
_TYPING_DECORATORS: frozenset[str] = frozenset({"overload", "final"})


class SourceFacts(NamedTuple):
    """The places of one source file that are not code, and its statements' lines.

    Attributes:
        annotations: Each annotation's start and end, as ``(line, column)``
            with columns counted in characters from zero, as cosmic-ray
            counts them; the end is exclusive.
        markers: Where each ``/`` or ``*`` parameter marker starts.
        statement_lines: Each line of a statement, to the first line of the
            innermost statement holding it.
        typing_decorator_lines: Each line of an ``overload`` or ``final``
            decorator from :data:`_TYPING_MODULES`; a decorator's lines hold
            nothing but it.
        definitions: Each line of a decorator, to the first and last line
            of the definition it decorates, its decorators included.
    """

    annotations: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    markers: frozenset[tuple[int, int]]
    statement_lines: dict[int, int]
    typing_decorator_lines: frozenset[int]
    definitions: dict[int, tuple[int, int]]


def _annotations_of(node: ast.AST) -> list[ast.expr]:
    """The annotations one node carries.

    Args:
        node: A node of the module.

    Returns:
        A parameter's or an annotated assignment's annotation, or a
        function's return annotation; nothing for any other node.
    """
    if isinstance(node, ast.arg | ast.AnnAssign):
        return [node.annotation] if node.annotation else []
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return [node.returns] if node.returns else []
    return []


def _column(lines: list[str], line: int, byte_offset: int) -> int:
    """Turn ast's UTF-8 byte offset on a line into a character column.

    Args:
        lines: The source's lines.
        line: The line, from one.
        byte_offset: ast's offset into its UTF-8 bytes.

    Returns:
        The same place counted in characters.
    """
    return len(lines[line - 1].encode("utf-8")[:byte_offset].decode("utf-8"))


def _markers(source: str) -> frozenset[tuple[int, int]]:
    """Where each parameter marker starts.

    Args:
        source: The module's text.

    Returns:
        The ``(line, column)`` of every ``/`` or ``*`` whose next token is a
        comma, a closing parenthesis or a colon.
    """
    tokens = [
        token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in _LAYOUT
    ]
    return frozenset(
        token.start
        for token, following in itertools.pairwise(tokens)
        if token.string in {"/", "*"} and following.string in _AFTER_MARKER
    )


class _TypingNames(NamedTuple):
    """The names a module binds to typing's declaring decorators and to typing itself.

    Attributes:
        decorators: Each name ``from typing import overload`` or ``final``
            binds, under its alias when it has one.
        modules: Each name ``import typing`` binds, under its alias when it
            has one.
    """

    decorators: frozenset[str]
    modules: frozenset[str]


def _typing_names(tree: ast.Module) -> _TypingNames:
    """Read which names a module binds to :data:`_TYPING_DECORATORS` and :data:`_TYPING_MODULES`.

    Args:
        tree: The module.

    Returns:
        The names its imports bind, wherever in the module they stand.
    """
    decorators: set[str] = set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _TYPING_MODULES:
            decorators.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name in _TYPING_DECORATORS
            )
        if isinstance(node, ast.Import):
            modules.update(
                alias.asname or alias.name for alias in node.names if alias.name in _TYPING_MODULES
            )
    return _TypingNames(decorators=frozenset(decorators), modules=frozenset(modules))


def _is_typing_decorator(decorator: ast.expr, names: _TypingNames) -> bool:
    """Whether a decorator is typing's ``overload`` or ``final``.

    Args:
        decorator: One decorator's expression.
        names: What the module's imports bind (:func:`_typing_names`).

    Returns:
        True for a name bound to one, or one read off a name bound to its
        module; False for anything else, a call among them.
    """
    if isinstance(decorator, ast.Name):
        return decorator.id in names.decorators
    return (
        isinstance(decorator, ast.Attribute)
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id in names.modules
        and decorator.attr in _TYPING_DECORATORS
    )


def read_source_facts(source: str) -> SourceFacts:
    """Read what :class:`SourceFacts` holds from one module's text.

    Args:
        source: The module's text.

    Returns:
        Its annotations, parameter markers, statement lines, typing
        decorators and decorated definitions.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    names = _typing_names(tree)
    annotations: list[tuple[tuple[int, int], tuple[int, int]]] = []
    statement_lines: dict[int, int] = {}
    typing_decorator_lines: set[int] = set()
    definitions: dict[int, tuple[int, int]] = {}
    # ast.walk visits a node before the nodes inside it, so an inner
    # statement overwrites the lines its outer statement claimed first.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                held = range(decorator.lineno, (decorator.end_lineno or decorator.lineno) + 1)
                span = (node.decorator_list[0].lineno, node.end_lineno or node.lineno)
                definitions.update(dict.fromkeys(held, span))
                if _is_typing_decorator(decorator, names):
                    typing_decorator_lines.update(held)
        for annotation in _annotations_of(node):
            end_line = annotation.end_lineno or annotation.lineno
            end_offset = annotation.end_col_offset or annotation.col_offset
            annotations.append(
                (
                    (annotation.lineno, _column(lines, annotation.lineno, annotation.col_offset)),
                    (end_line, _column(lines, end_line, end_offset)),
                )
            )
        if isinstance(node, ast.stmt):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                statement_lines[line] = node.lineno
    return SourceFacts(
        annotations=tuple(annotations),
        markers=_markers(source),
        statement_lines=statement_lines,
        typing_decorator_lines=frozenset(typing_decorator_lines),
        definitions=definitions,
    )


def is_declaration(facts: SourceFacts, position: tuple[int, int]) -> bool:
    """Whether a mutant's place is an annotation, a parameter marker or a typing decorator.

    Args:
        facts: The file's facts.
        position: Where the mutant starts, ``(line, column)`` as cosmic-ray
            reports it.

    Returns:
        True when no test could tell the mutant from the original because
        the place is not code.
    """
    in_annotation = any(start <= position < end for start, end in facts.annotations)
    in_typing_decorator = position[0] in facts.typing_decorator_lines
    return in_annotation or position in facts.markers or in_typing_decorator


def coverage_line(facts: SourceFacts, line: int, recorded: frozenset[int]) -> int:
    """The line whose coverage contexts stand for a mutant's line.

    Args:
        facts: The file's facts.
        line: The mutant's line.
        recorded: The lines coverage holds contexts for in that file.

    Returns:
        The line itself when recorded, otherwise the earliest recorded
        line owned by the same innermost statement. With no such coverage,
        the statement's unrecorded first line preserves the no-test verdict.
    """
    if line in recorded:
        return line
    statement = facts.statement_lines.get(line, line)
    statements: dict[int, set[int]] = {}
    for candidate, owner in facts.statement_lines.items():
        statements.setdefault(owner, set()).add(candidate)
    return min(recorded.intersection(statements.get(statement, ())), default=statement)


__all__ = ["SourceFacts", "coverage_line", "is_declaration", "read_source_facts"]
