"""What a source file says about the places cosmic-ray mutates (board task fec5851c).

TWO KINDS OF PLACE ARE NOT CODE, and the step does not run a mutant there.
A type annotation is a declaration for mypy: under ``from __future__ import
annotations`` it is never evaluated, so a mutant of ``str | None`` cannot be
killed by any test, as StrykerJS never mutates a TypeScript type. The ``/``
and ``*`` that end positional and begin keyword-only parameters are not
operators, and cosmic-ray's binary-operator replacement of them is not a
program. Measured over mcp-shared-py's src on 2026-11-07: 2730 of 8038
mutants sat in annotations and 539 on those markers.

WHICH LINE A MUTANT'S TESTS ARE READ FROM. coverage records a statement
spread over several lines under the lines its bytecode names, which for a
continued expression is often only the first: 23 mutants of mcp-shared-py
sat on lines coverage holds no context for. A line coverage did not record
is read under the first line of the innermost statement holding it.
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


class SourceFacts(NamedTuple):
    """The places of one source file that are not code, and its statements' lines.

    Attributes:
        annotations: Each annotation's start and end, as ``(line, column)``
            with columns counted in characters from zero, as cosmic-ray
            counts them; the end is exclusive.
        markers: Where each ``/`` or ``*`` parameter marker starts.
        statement_lines: Each line of a statement, to the first line of the
            innermost statement holding it.
    """

    annotations: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    markers: frozenset[tuple[int, int]]
    statement_lines: dict[int, int]


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


def read_source_facts(source: str) -> SourceFacts:
    """Read what :class:`SourceFacts` holds from one module's text.

    Args:
        source: The module's text.

    Returns:
        Its annotations, parameter markers and statement lines.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    annotations: list[tuple[tuple[int, int], tuple[int, int]]] = []
    statement_lines: dict[int, int] = {}
    # ast.walk visits a node before the nodes inside it, so an inner
    # statement overwrites the lines its outer statement claimed first.
    for node in ast.walk(tree):
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
    )


def is_declaration(facts: SourceFacts, position: tuple[int, int]) -> bool:
    """Whether a mutant's place is an annotation or a parameter marker.

    Args:
        facts: The file's facts.
        position: Where the mutant starts, ``(line, column)`` as cosmic-ray
            reports it.

    Returns:
        True when no test could tell the mutant from the original because
        the place is not code.
    """
    in_annotation = any(start <= position < end for start, end in facts.annotations)
    return in_annotation or position in facts.markers


def coverage_line(facts: SourceFacts, line: int, recorded: frozenset[int]) -> int:
    """The line whose coverage contexts stand for a mutant's line.

    Args:
        facts: The file's facts.
        line: The mutant's line.
        recorded: The lines coverage holds contexts for in that file.

    Returns:
        The line itself when coverage recorded it, else the first line of
        the innermost statement holding it.
    """
    if line in recorded:
        return line
    return facts.statement_lines.get(line, line)


__all__ = ["SourceFacts", "coverage_line", "is_declaration", "read_source_facts"]
