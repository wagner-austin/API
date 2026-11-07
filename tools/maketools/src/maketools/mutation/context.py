"""What one run of the mutation step is bound to, and how it refuses (board task fec5851c).

The refusals carry the TypeScript step's codes where the two steps refuse
the same thing (``mcp-shared/src/mutation/step.ts``, ``MutationErrorCode``),
so a reader of either log reads one vocabulary.

STANDARD LIBRARY ONLY (board task e52e5148). This module, ``changes``,
``sites``, ``cosmic``, ``report``, ``trials`` and ``step`` are the step's
core, and API's ``tools/maketools`` lifts them byte for byte (pinned in its
``lift-lock.json``), so they import nothing but the standard library and
each other. What differs between the two repositories arrives at the
boundary instead: the world the step reaches (:class:`MutationHooks`, bound
by this package's :mod:`._test_hooks` here and by maketools' hooks there)
and how JSON is read and written (:class:`JsonCodec`, bound to
:mod:`json` and :mod:`mcp_shared_py.validation` here and to
``platform_core.json_utils`` there), and where the step's time is recorded
(:attr:`StepContext.record_span`: maketools' check lock here, ``check-budget``
there). Each boundary turns a :class:`MutationStepError` into its own
refusal.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

#: The type a package's JSON reader decodes a document to.
JsonT = TypeVar("JsonT")


class MutationErrorCode(StrEnum):
    """Every way the step refuses to run or to finish."""

    BASE_UNAVAILABLE = "MUTATION_BASE_UNAVAILABLE"
    GIT_FAILED = "MUTATION_GIT_FAILED"
    TREE_DIRTY = "MUTATION_TREE_DIRTY"
    COSMIC_RAY_FAILED = "MUTATION_COSMIC_RAY_FAILED"
    COVERAGE_UNAVAILABLE = "MUTATION_COVERAGE_UNAVAILABLE"
    BASELINE_FAILED = "MUTATION_BASELINE_FAILED"
    PREPARE_FAILED = "MUTATION_PREPARE_FAILED"
    TREE_NOT_RESTORED = "MUTATION_TREE_NOT_RESTORED"
    SPAN_UNRECORDED = "MUTATION_SPAN_UNRECORDED"


class MutationStepError(Exception):
    """A refusal of the step, carrying its code as its message's first word."""

    def __init__(self, code: MutationErrorCode, detail: str) -> None:
        """Make the refusal.

        Args:
            code: Which refusal.
            detail: What was found, for the reader.
        """
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class Ran(NamedTuple):
    """How one command ended.

    Attributes:
        exit_code: Its exit status.
        stdout: What it printed, decoded as UTF-8.
        stderr: What it printed to standard error.
    """

    exit_code: int
    stdout: str
    stderr: str


class MutationHooks(NamedTuple):
    """The world the step reaches, bound to one package directory.

    Attributes:
        run: Run a command in the package directory with exactly the given
            environment, capturing its output.
        now: Seconds on a monotonic clock.
        make_run_directory: Make a fresh directory for one run's files.
        remove_tree: Remove a directory the step made, with its contents.
        write_text: Write a UTF-8 file.
        read_text: Read a UTF-8 file.
        write: Print one line of the step's output.
    """

    run: Callable[[Sequence[str], Mapping[str, str]], Ran]
    now: Callable[[], float]
    make_run_directory: Callable[[], Path]
    remove_tree: Callable[[Path], None]
    write_text: Callable[[Path, str], None]
    read_text: Callable[[Path], str]
    write: Callable[[str], None]


class JsonCodec(NamedTuple, Generic[JsonT]):
    """How the step reads cosmic-ray's and coverage's JSON and writes its own.

    Each reader raises the package's own decoding error, naming the field,
    when a value is not the shape asked for.

    Attributes:
        loads: Decode a JSON document.
        dumps: Encode a string, or a list of them, as JSON with the
            standard ``", "`` separator.
        mapping: Require an object.
        sequence: Require an array.
        text: Require a string.
        optional_text: Require a string or null.
        integer: Require an integer.
    """

    loads: Callable[[str], JsonT]
    dumps: Callable[[str | list[str]], str]
    mapping: Callable[[JsonT, str], Mapping[str, JsonT]]
    sequence: Callable[[JsonT, str], Sequence[JsonT]]
    text: Callable[[JsonT, str], str]
    optional_text: Callable[[JsonT, str], str | None]
    integer: Callable[[JsonT, str], int]


class StepContext(NamedTuple, Generic[JsonT]):
    """What one run of the step is bound to.

    Attributes:
        hooks: The world it reaches.
        codec: How it reads and writes JSON.
        package_dir: The package it mutates, where every command runs.
        python: The interpreter of the package's environment.
        prepare: The shell command that derives files from the sources
            before the tests run, or empty.
        environment: The environment the step was started with.
        run_dir: The run's scratch directory.
        record_span: Record the step's wall time, in whole seconds, as the
            check's ``mutation testing`` span, which the package's check
            holds to its own bound; called once a run has made mutants.
    """

    hooks: MutationHooks
    codec: JsonCodec[JsonT]
    package_dir: Path
    python: str
    prepare: str
    environment: Mapping[str, str]
    run_dir: Path
    record_span: Callable[[int], None]


def tail(ran: Ran) -> str:
    """How a command ended, for a refusal.

    Args:
        ran: How it ended.

    Returns:
        Its exit and the last 20 lines of its output and errors.
    """
    lines = (ran.stdout + ran.stderr).splitlines()[-20:]
    return f"exited {ran.exit_code}: " + "\n".join(lines)


def run_or_refuse(
    context: StepContext[JsonT],
    argv: list[str],
    env: Mapping[str, str],
    code: MutationErrorCode,
) -> str:
    """Run a command and require it to succeed.

    Args:
        context: The run.
        argv: The command.
        env: The whole environment it runs with.
        code: The refusal its failure is.

    Returns:
        What it printed.

    Raises:
        MutationStepError: ``code``, naming the command and its last lines.
    """
    ran = context.hooks.run(argv, env)
    if ran.exit_code:
        raise MutationStepError(code, f"{shlex.join(argv)} {tail(ran)}")
    return ran.stdout


__all__ = [
    "JsonCodec",
    "JsonT",
    "MutationErrorCode",
    "MutationHooks",
    "MutationStepError",
    "Ran",
    "StepContext",
    "run_or_refuse",
    "tail",
]
