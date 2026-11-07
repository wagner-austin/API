"""The verdict the step reads from its mutants, and the lines it prints (MCPs task e52e5148).

THE RULE IT ENFORCES is the coding prompt's: the source a commit changes is
mutation tested and no mutant survives; a surviving mutant is killed by a
test or, when it changes nothing observable, removed by rewriting the code;
there is no ignore list. Two outcomes fail the step: a mutant the tests ran
and passed, and one no test runs at all. The lines are the ones MCPs' steps
print (``mcp_shared_py.mutation.report`` and mcp-shared's
``src/mutation/report.ts``), so a reader of any of them reads one format:
``<words> <file>:<line>:<column> <mutator> -> <the mutated line as JSON>``,
columns counted from one.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, NamedTuple

from platform_core.json_utils import dump_json_str

#: The prefix cosmic-ray's own operators carry, dropped from a printed name.
_CORE_PREFIX: Final[str] = "core/"


class Uncaught(StrEnum):
    """Why a mutant fails the step, as the words its line opens with."""

    SURVIVED = "MUTANT SURVIVED"
    NO_TEST = "MUTANT SURVIVED, NO TEST RAN IT"


class UncaughtMutant(NamedTuple):
    """One mutant that fails the step.

    Attributes:
        path: The source file, relative to the package.
        line: Its line, from one.
        column: Its column, from zero as cosmic-ray counts it.
        operator: cosmic-ray's operator name.
        mutated: The line the mutation puts in place, or the original line
            when no test ran it and so no mutation was made.
        why: Why it fails the step.
    """

    path: str
    line: int
    column: int
    operator: str
    mutated: str
    why: Uncaught


class Tally(NamedTuple):
    """How the step's mutants were run.

    Attributes:
        files: The changed source files they were made in.
        base: The merge base those files were compared against.
        by_tests: Mutants run by the tests covering their line.
        by_suite: Mutants on lines run only at import, run by the whole suite.
        by_none: Mutants on lines no test runs.
        declarations: Places in annotations and parameter markers, not code
            and not run.
    """

    files: int
    base: str
    by_tests: int
    by_suite: int
    by_none: int
    declarations: int


def mutant_line(mutant: UncaughtMutant) -> str:
    """The line that names one uncaught mutant: its file, line and change.

    Args:
        mutant: The mutant.

    Returns:
        ``<words> <file>:<line>:<column> <operator> -> <mutated line as JSON>``,
        the column counted from one.
    """
    operator = mutant.operator.removeprefix(_CORE_PREFIX)
    place = f"{mutant.path}:{mutant.line}:{mutant.column + 1}"
    return f"{mutant.why} {place} {operator} -> {dump_json_str(mutant.mutated)}"


def _position(mutant: UncaughtMutant) -> tuple[str, int, int]:
    """Where a mutant sits, for ordering the report.

    Args:
        mutant: The mutant.

    Returns:
        Its file, line and column.
    """
    return (mutant.path, mutant.line, mutant.column)


def render_verdict(tally: Tally, uncaught: list[UncaughtMutant]) -> list[str]:
    """The lines the step prints for its run.

    Args:
        tally: How the mutants were run.
        uncaught: The mutants that fail the step.

    Returns:
        One line saying how the mutants were run, one per uncaught mutant
        in file, line and column order, then the summary: ``FAILED`` naming
        how many of how many survived, or ``passed`` naming how many were
        made.
    """
    made = tally.by_tests + tally.by_suite + tally.by_none
    ran = (
        f"mutation: {made} mutant(s): {tally.by_tests} run by the tests covering their line, "
        f"{tally.by_suite} by the whole suite (lines run only at import), {tally.by_none} by "
        f"no test; {tally.declarations} more in annotations and parameter markers are not code"
    )
    scope = (
        f"{made} mutant(s) made in {tally.files} src file(s) changed against origin/main "
        f"(merge base {tally.base}) survived"
    )
    ordered = sorted(uncaught, key=_position)
    if ordered:
        return [
            ran,
            *map(mutant_line, ordered),
            f"mutation: FAILED: {len(ordered)} of {scope}. Kill each with a test, or rewrite "
            "the code so the mutation cannot change what it does.",
        ]
    return [ran, f"mutation: passed: 0 of {scope}"]


__all__ = ["Tally", "Uncaught", "UncaughtMutant", "mutant_line", "render_verdict"]
