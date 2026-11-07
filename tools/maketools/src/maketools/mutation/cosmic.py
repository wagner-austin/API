"""cosmic-ray, as the step drives it, and what it answers (MCPs board task e52e5148).

WHY COSMIC-RAY. MCPs measured both candidates on mcp-shared-py on 2026-11-07
(board task fec5851c): mutmut 3.8.1 copies the package into ``mutants/`` and
rewrites every function into trampolines, so its own clean run failed that
suite's contract-hash test and five more; cosmic-ray 8.7.0 mutates the file
in place and runs whatever test command it is given, so a package's own
pytest configuration runs unmodified. The API step uses the same tool so a
reader of either repository reads one format.

HOW. ``cosmic-ray init`` enumerates every mutant of the changed files into a
session; ``cosmic-ray dump`` prints them as JSON lines; ``cosmic-ray
mutate-and-test`` applies one, runs the command the step chose for it,
restores the file and prints the outcome as JSON. Its ``pragma`` filter is
never run, so no comment in the source hides a mutant: there is no ignore
list. cosmic-ray ships no type information, so it is driven through its
command line and its JSON is decoded here, once, through
:mod:`platform_core.json_utils`, this repository's one JSON reader.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final, NamedTuple

from platform_core.error_codes_tooling import MutationErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_int,
    narrow_json_to_list,
    optional_str,
    require_dict,
    require_int,
    require_list,
    require_str,
)

#: The name cosmic-ray installs its command under, beside the interpreter.
COSMIC_RAY: Final[str] = "cosmic-ray"


class Site(NamedTuple):
    """One mutant cosmic-ray can make.

    Attributes:
        path: The source file, relative to the package.
        line: The line the mutation starts on, from one.
        column: Its column, counted in characters from zero.
        operator: The operator's name, e.g. ``core/ReplaceComparisonOperator_Eq_Lt``.
        occurrence: Which of that operator's places in the file it is.
    """

    path: str
    line: int
    column: int
    operator: str
    occurrence: int


class WorkerOutcome(StrEnum):
    """How cosmic-ray's worker ended."""

    NORMAL = "normal"
    EXCEPTION = "exception"
    ABNORMAL = "abnormal"
    NO_TEST = "no-test"
    SKIPPED = "skipped"


class RunOutcome(StrEnum):
    """What the tests did to a mutant."""

    SURVIVED = "survived"
    KILLED = "killed"
    INCOMPETENT = "incompetent"


class MutantResult(NamedTuple):
    """One ``mutate-and-test`` answer.

    Attributes:
        worker: How the worker ended.
        tests: What the tests did, or None when no test ran.
        diff: The mutation as a unified diff, empty when none was made.
        output: What the tests or the worker printed.
    """

    worker: WorkerOutcome
    tests: RunOutcome | None
    diff: str
    output: str


def command(python: str) -> str:
    """The cosmic-ray program installed beside an interpreter.

    Args:
        python: The interpreter of the package's environment.

    Returns:
        The path of its ``cosmic-ray`` script.
    """
    return str(Path(python).with_name(COSMIC_RAY))


def session_config(paths: list[str]) -> str:
    """The configuration ``cosmic-ray init`` enumerates mutants from.

    Only the module paths matter to ``init``; the test command and timeout
    are never used, since the step runs each mutant with its own.

    Args:
        paths: The files to mutate, relative to the package.

    Returns:
        The TOML text.
    """
    return "\n".join(
        [
            "[cosmic-ray]",
            f"module-path = {dump_json_str(paths)}",
            "timeout = 1.0",
            "excluded-modules = []",
            'test-command = "false"',
            "",
            "[cosmic-ray.distributor]",
            'name = "local"',
            "",
        ]
    )


def decode_site(line: str) -> Site:
    """Decode one line of ``cosmic-ray dump`` into the mutant it holds.

    Args:
        line: A JSON array: a work item and its result, null before the
            session ran.

    Returns:
        The mutant.

    Raises:
        AppError: ``MUTATION_COSMIC_RAY_FAILED`` when the work item holds
            other than one mutation, or the mutation carries operator
            arguments, which ``mutate-and-test`` cannot pass on.
        JSONTypeError: When the line is not the shape ``init`` writes.
    """
    item = narrow_json_to_dict(narrow_json_to_list(load_json_str(line))[0])
    mutations = require_list(item, "mutations")
    if len(mutations) != 1:
        raise AppError(
            MutationErrorCode.COSMIC_RAY_FAILED,
            f"cosmic-ray dump line {line!r} holds {len(mutations)} mutations, not one",
        )
    mutation = narrow_json_to_dict(mutations[0])
    if require_dict(mutation, "operator_args"):
        raise AppError(
            MutationErrorCode.COSMIC_RAY_FAILED,
            f"cosmic-ray dump line {line!r} carries operator arguments mutate-and-test cannot pass",
        )
    start = require_list(mutation, "start_pos")
    return Site(
        path=require_str(mutation, "module_path"),
        line=narrow_json_to_int(start[0]),
        column=narrow_json_to_int(start[1]),
        operator=require_str(mutation, "operator_name"),
        occurrence=require_int(mutation, "occurrence"),
    )


def decode_sites(dump: str) -> list[Site]:
    """Decode ``cosmic-ray dump``'s lines into the mutants they hold.

    Args:
        dump: One JSON array per line.

    Returns:
        Every mutant, in the dump's order.
    """
    return [decode_site(line) for line in dump.splitlines()]


def decode_result(text: str) -> MutantResult:
    """Decode the JSON ``cosmic-ray mutate-and-test`` prints.

    Args:
        text: Its standard output.

    Returns:
        The outcome.

    Raises:
        JSONTypeError: When it is not the shape cosmic-ray's ``WorkResult``
            serialises to.
        ValueError: When an outcome is not one cosmic-ray 8.7.0 names.
    """
    result = narrow_json_to_dict(load_json_str(text))
    tests = optional_str(result, "test_outcome")
    return MutantResult(
        worker=WorkerOutcome(require_str(result, "worker_outcome")),
        tests=None if tests is None else RunOutcome(tests),
        diff=optional_str(result, "diff") or "",
        output=optional_str(result, "output") or "",
    )


def mutated_line(diff: str) -> str:
    """The first line a mutation puts in place, from its diff.

    Args:
        diff: The unified diff cosmic-ray reports.

    Returns:
        That line without its ``+`` and surrounding blanks.
    """
    added = [
        line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
    ]
    return added[0].removeprefix("+").strip()


__all__ = [
    "COSMIC_RAY",
    "MutantResult",
    "RunOutcome",
    "Site",
    "WorkerOutcome",
    "command",
    "decode_result",
    "decode_site",
    "decode_sites",
    "mutated_line",
    "session_config",
]
