"""cosmic-ray, as the step drives it, and what it answers (board task fec5851c).

WHY COSMIC-RAY. Measured on mcp-shared-py on 2026-11-07, as the task asked:
mutmut 3.8.1 copies the package into ``mutants/`` and rewrites every function
into trampolines, so its own clean run failed the suite's contract-hash test
and five more; cosmic-ray 8.7.0 mutates the file in place and runs whatever
test command it is given, so the package's own pytest configuration runs
unmodified. It ships no type information, so it is driven through its
command line and its JSON is decoded here, once, into this step's types,
through the package's :class:`~.context.JsonCodec` (board task e52e5148:
this module is lifted into API, which reads JSON another way).

HOW. ``cosmic-ray init`` enumerates every mutant of the changed files into a
session; ``cosmic-ray dump`` prints them as JSON lines; ``cosmic-ray
mutate-and-test`` applies one, runs the command the step chose for it,
restores the file and prints the outcome as JSON. Its ``pragma`` filter is
never run, so no comment in the source hides a mutant: there is no ignore
list.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from .context import JsonCodec, JsonT

#: The name cosmic-ray installs its command under, beside the interpreter.
COSMIC_RAY = "cosmic-ray"


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


def session_config(paths: list[str], codec: JsonCodec[JsonT]) -> str:
    """The configuration ``cosmic-ray init`` enumerates mutants from.

    Only the module paths matter to ``init``; the test command and timeout
    are never used, since the step runs each mutant with its own.

    Args:
        paths: The files to mutate, relative to the package.
        codec: How JSON is written; a JSON array of strings is a TOML one.

    Returns:
        The TOML text.
    """
    return "\n".join(
        [
            "[cosmic-ray]",
            f"module-path = {codec.dumps(paths)}",
            "timeout = 1.0",
            "excluded-modules = []",
            'test-command = "false"',
            "",
            "[cosmic-ray.distributor]",
            'name = "local"',
            "",
        ]
    )


def decode_sites(dump: str, codec: JsonCodec[JsonT]) -> list[Site]:
    """Decode ``cosmic-ray dump``'s lines into the mutants they hold.

    Args:
        dump: One JSON array per line: a work item and its result, null
            before the session ran.
        codec: How JSON is read.

    Returns:
        Every mutant, in the dump's order.

    Raises:
        ValueError: When a work item holds other than one mutation, or one
            carries operator arguments, which ``mutate-and-test`` cannot
            pass on; the codec's own error when a line is not the shape
            ``init`` writes.
    """
    sites: list[Site] = []
    for number, line in enumerate(dump.splitlines(), start=1):
        field = f"cosmic-ray dump line {number}"
        item = codec.mapping(codec.sequence(codec.loads(line), field)[0], f"{field} work item")
        (raw,) = codec.sequence(item["mutations"], f"{field} mutations")
        mutation = codec.mapping(raw, f"{field} mutation")
        start = codec.sequence(mutation["start_pos"], f"{field} start_pos")
        if codec.mapping(mutation["operator_args"], f"{field} operator_args"):
            raise ValueError(f"{field} carries operator arguments mutate-and-test cannot pass")
        sites.append(
            Site(
                path=codec.text(mutation["module_path"], f"{field} module_path"),
                line=codec.integer(start[0], f"{field} start line"),
                column=codec.integer(start[1], f"{field} start column"),
                operator=codec.text(mutation["operator_name"], f"{field} operator_name"),
                occurrence=codec.integer(mutation["occurrence"], f"{field} occurrence"),
            )
        )
    return sites


def decode_result(text: str, codec: JsonCodec[JsonT]) -> MutantResult:
    """Decode the JSON ``cosmic-ray mutate-and-test`` prints.

    Args:
        text: Its standard output.
        codec: How JSON is read.

    Returns:
        The outcome.

    Raises:
        ValueError: When an outcome is not one :class:`WorkerOutcome` or
            :class:`RunOutcome` names; the codec's own error when the text
            is not the shape cosmic-ray's ``WorkResult`` serialises to.
    """
    result = codec.mapping(codec.loads(text), "mutate-and-test result")
    tests = codec.optional_text(result["test_outcome"], "test_outcome")
    return MutantResult(
        worker=WorkerOutcome(codec.text(result["worker_outcome"], "worker_outcome")),
        tests=None if tests is None else RunOutcome(tests),
        diff=codec.optional_text(result["diff"], "diff") or "",
        output=codec.optional_text(result["output"], "output") or "",
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
    "decode_sites",
    "mutated_line",
    "session_config",
]
