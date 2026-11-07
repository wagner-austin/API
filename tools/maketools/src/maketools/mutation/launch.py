"""API's boundary of the lifted mutation step, run after a passing suite (MCPs task e52e5148).

The step's core (``context``, ``changes``, ``sites``, ``cosmic``, ``report``,
``trials``, ``step``) is MCPs' ``mcp_shared_py.mutation``, lifted byte for
byte and pinned in ``tools/maketools/lift-lock.json``. It imports only the
standard library, and everything that differs between the two repositories
is bound here and nowhere else:

- THE WORLD it reaches (:func:`mutation_hooks`): commands through
  :data:`maketools._test_hooks.run_answering` under the step's wall, the
  clock, files, and output through ``write_line``.
- JSON (:data:`API_CODEC`): ``platform_core.json_utils``, this repository's
  one JSON reader, its decoding errors naming the field.
- REFUSALS: the core's :class:`~maketools.mutation.context.MutationStepError`
  becomes an ``AppError`` carrying ``platform_core``'s
  :class:`~platform_core.error_codes_tooling.MutationErrorCode` of the same
  value, which ``run.py`` prints as ``CODE: message``.
- THE SPAN: a run that made mutants records its time as the check's
  ``mutation testing`` span (:mod:`maketools.bounded_spans`), which
  ``check-budget`` holds to its own bound instead of the five minutes.

``run.py test --mutation`` (:func:`maketools.test_run.run_tests`) calls
:func:`mutate_after_suite` with the package, its interpreter and the
suite's environment, which names the run's coverage file.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, TypeVar

from platform_core.error_codes_tooling import MutationErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_int,
    narrow_json_to_list,
    narrow_json_to_str,
)

from maketools import _test_hooks
from maketools.bounded_spans import (
    MUTATION_BOUND_SECONDS,
    MUTATION_SPAN,
    SPANS_ENVIRONMENT_KEY,
    record_span,
)
from maketools.mutation.context import JsonCodec, MutationHooks, MutationStepError, Ran
from maketools.mutation.step import run_in_scratch

#: The wall on any one command the step runs: its whole span's bound, so a
#: single hung command fails as the span past its bound would, and sooner.
CHILD_WALL_SECONDS: Final[int] = MUTATION_BOUND_SECONDS

NarrowedT = TypeVar("NarrowedT")


def narrowed(narrow: Callable[[JSONValue], NarrowedT], value: JSONValue, field: str) -> NarrowedT:
    """Narrow a decoded value, naming the field when it is not the shape asked for.

    Args:
        narrow: One of ``platform_core.json_utils``' narrowers.
        value: The decoded value.
        field: Where it was read, for the refusal.

    Returns:
        The narrowed value.

    Raises:
        JSONTypeError: The narrower's own, its message opened by the field.
    """
    try:
        return narrow(value)
    except JSONTypeError as refusal:
        raise JSONTypeError(f"{field}: {refusal}") from refusal


def mapping(value: JSONValue, field: str) -> Mapping[str, JSONValue]:
    """Require an object.

    Args:
        value: The decoded value.
        field: Where it was read.

    Returns:
        The object.
    """
    return narrowed(narrow_json_to_dict, value, field)


def sequence(value: JSONValue, field: str) -> Sequence[JSONValue]:
    """Require an array.

    Args:
        value: The decoded value.
        field: Where it was read.

    Returns:
        Its members.
    """
    return narrowed(narrow_json_to_list, value, field)


def text(value: JSONValue, field: str) -> str:
    """Require a string.

    Args:
        value: The decoded value.
        field: Where it was read.

    Returns:
        The string.
    """
    return narrowed(narrow_json_to_str, value, field)


def optional_text(value: JSONValue, field: str) -> str | None:
    """Require a string or null.

    Args:
        value: The decoded value.
        field: Where it was read.

    Returns:
        The string, or None for null.
    """
    return None if value is None else text(value, field)


def integer(value: JSONValue, field: str) -> int:
    """Require an integer.

    Args:
        value: The decoded value.
        field: Where it was read.

    Returns:
        The integer.
    """
    return narrowed(narrow_json_to_int, value, field)


def dump_spaced(value: str | list[str]) -> str:
    """Encode a string, or a list of them, as JSON with the standard ``", "`` separator.

    Args:
        value: What to encode.

    Returns:
        The JSON text, as MCPs' boundary writes it.
    """
    return dump_json_str(value, compact=False)


#: How API reads and writes the step's JSON.
API_CODEC: Final[JsonCodec[JSONValue]] = JsonCodec(
    loads=load_json_str,
    dumps=dump_spaced,
    mapping=mapping,
    sequence=sequence,
    text=text,
    optional_text=optional_text,
    integer=integer,
)


def answer(argv: Sequence[str], env: Mapping[str, str], project: Path) -> Ran:
    """Run one of the step's commands in the package, under the step's wall.

    Args:
        argv: The command.
        env: Its whole environment.
        project: The package, where it runs.

    Returns:
        How it ended.
    """
    result = _test_hooks.run_answering(
        argv, cwd=project, env=env, timeout_seconds=CHILD_WALL_SECONDS
    )
    return Ran(exit_code=result["returncode"], stdout=result["stdout"], stderr=result["stderr"])


def made(directory: Path) -> Path:
    """Make the step's scratch directory.

    Args:
        directory: Where, under the package's ``runs/``; it must not exist.

    Returns:
        The directory.
    """
    directory.mkdir()
    return directory


def write_text(path: Path, content: str) -> None:
    """Write a UTF-8 file.

    Args:
        path: Where.
        content: What.
    """
    path.write_text(content, encoding="utf-8")


def read_text(path: Path) -> str:
    """Read a UTF-8 file.

    Args:
        path: Which.

    Returns:
        Its text.
    """
    return path.read_text(encoding="utf-8")


def mutation_hooks(project: Path, run_dir: Path) -> MutationHooks:
    """The world the lifted step reaches, bound to maketools' hooks.

    Args:
        project: The package.
        run_dir: The scratch directory the run makes and removes.

    Returns:
        The bundle.
    """
    return MutationHooks(
        run=lambda argv, env: answer(argv, env, project),
        now=_test_hooks.now,
        make_run_directory=lambda: made(run_dir),
        remove_tree=_test_hooks.remove_tree,
        write_text=write_text,
        read_text=read_text,
        write=_test_hooks.write_line,
    )


def span_sink(environment: Mapping[str, str]) -> Callable[[int], None]:
    """Where a run that made mutants records its time.

    Args:
        environment: The suite's environment.

    Returns:
        A recorder of the run's whole seconds: into the span file
        ``check-budget`` named, or, with none named, a line saying the time
        is held to no bound.
    """

    def record(seconds: int) -> None:
        """Record one run's time.

        Args:
            seconds: Its wall time.
        """
        named = environment.get(SPANS_ENVIRONMENT_KEY, "")
        if named:
            record_span(Path(named), MUTATION_SPAN, seconds, _test_hooks.now())
            _test_hooks.write_line(f"mutation: {seconds}s, recorded as the check's {MUTATION_SPAN}")
            return
        _test_hooks.write_line(
            f"mutation: {seconds}s; no check-budget named a span file in "
            f"{SPANS_ENVIRONMENT_KEY}, so this run's time is held to no bound"
        )

    return record


def mutate_after_suite(
    project: Path, python: str, environment: Mapping[str, str], run_dir: Path
) -> int:
    """Run the lifted step over a package whose suite just passed.

    Args:
        project: The package.
        python: Its environment's interpreter.
        environment: The suite's environment, naming its coverage file.
        run_dir: A directory that does not exist yet, for the step's files.

    Returns:
        The step's exit code: 1 when a mutant survived, else 0.

    Raises:
        AppError: The step's refusal, by ``platform_core``'s code of the
            same value.
    """
    hooks = mutation_hooks(project, run_dir)
    record = span_sink(environment)
    try:
        return run_in_scratch(hooks, API_CODEC, project, python, "", environment, record)
    except MutationStepError as refusal:
        raise AppError(MutationErrorCode(refusal.code.value), refusal.detail) from refusal


__all__ = [
    "API_CODEC",
    "CHILD_WALL_SECONDS",
    "answer",
    "dump_spaced",
    "integer",
    "made",
    "mapping",
    "mutate_after_suite",
    "mutation_hooks",
    "narrowed",
    "optional_text",
    "read_text",
    "sequence",
    "span_sink",
    "text",
    "write_text",
]
