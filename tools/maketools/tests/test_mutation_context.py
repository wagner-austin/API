"""Tests for :mod:`maketools.mutation.context`, lifted from MCPs (MCPs task e52e5148).

MCPs' own cases for the module (``mcp-shared-py/tests/test_mutation_context.py``)
with the import paths made this package's.
"""

from __future__ import annotations

import pytest

from maketools.mutation.context import (
    MutationErrorCode,
    MutationStepError,
    Ran,
    run_or_refuse,
    tail,
)
from tests._mutation_world import World, failed, ok


def test_a_refusal_opens_with_its_code() -> None:
    error = MutationStepError(MutationErrorCode.TREE_DIRTY, "src/a.py differs")
    assert str(error) == "MUTATION_TREE_DIRTY: src/a.py differs"
    assert error.code == MutationErrorCode.TREE_DIRTY
    assert error.detail == "src/a.py differs"


def test_the_tail_keeps_the_last_twenty_lines_of_output_then_errors() -> None:
    stdout = "".join(f"out {n}\n" for n in range(1, 25))
    ran = Ran(exit_code=3, stdout=stdout, stderr="err 1\nerr 2\n")
    expected = [f"out {n}" for n in range(7, 25)] + ["err 1", "err 2"]
    assert tail(ran) == "exited 3: " + "\n".join(expected)


def test_a_command_that_succeeds_answers_what_it_printed_and_gets_the_given_environment() -> None:
    world = World(lambda argv: ok("printed\n"), [])
    out = run_or_refuse(
        world.context(), ["git", "status"], {"A": "1"}, MutationErrorCode.GIT_FAILED
    )
    assert out == "printed\n"
    assert world.calls == [(["git", "status"], {"A": "1"})]


def test_a_command_that_fails_is_refused_with_the_given_code_naming_it() -> None:
    world = World(lambda argv: failed(128, "fatal: not a git repository"), [])
    with pytest.raises(MutationStepError) as raised:
        run_or_refuse(world.context(), ["git", "log", "a b"], {}, MutationErrorCode.GIT_FAILED)
    assert str(raised.value) == (
        "MUTATION_GIT_FAILED: git log 'a b' exited 128: fatal: not a git repository"
    )
