"""The API launcher runs a decorator's mutant by its definition's tests (MCPs board task b9bf5690).

MCPs' real fixtures for its step (``mcp-shared-py/tests/
test_mutation_decorators_real.py``), run here through this package's
launcher with its own interpreter and pinned cosmic-ray. Coverage records a
decorator's line only under the import's empty context, so the step once ran
each decorator mutant by the whole suite; removing ``@property`` or
``@classmethod`` is seen by the tests that use the definition, and
``@final`` and ``@overload`` change nothing a test can see.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .test_mutation_real import PYPROJECT, _check, _commit, _git

DECORATED = """\
from typing import final, overload


@final
class Box:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    @classmethod
    def of(cls, value):
        return cls(value)

    @overload
    def get(self, key: int) -> int: ...
    @overload
    def get(self, key: str) -> str: ...
    def get(self, key):
        return key
"""
DECORATED_TESTS = """\
from compare import Box

def test_of():
    assert Box.of(3).value == 3

def test_get():
    assert Box(1).get("k") == "k"
"""
#: A cache no test can tell from its absence: the function returns the same either way.
CACHED = """\
import functools


@functools.cache
def double(value):
    return value * 2
"""
CACHED_TESTS = """\
from compare import double

def test_double():
    assert double(3) == 6
"""


def _mutate(tmp_path: Path, source: str, tests: str) -> subprocess.CompletedProcess[str]:
    """Commit ``source`` and ``tests`` over an empty main and run the launcher's mutation test.

    Args:
        tmp_path: Where to make the fixture repository.
        source: The module's text, committed as ``src/compare.py``.
        tests: Its tests' text.

    Returns:
        The launcher's run, once the fixture's source holds its committed bytes again.
    """
    repo = tmp_path / "fixture"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    _commit(repo, {"pyproject.toml": PYPROJECT, ".gitignore": ".venv\nruns/\n"}, "start")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _commit(repo, {"src/compare.py": source, "tests/test_compare.py": tests}, "decorated")
    spans = tmp_path / "spans"
    ran = _check(repo, spans)
    assert (repo / "src/compare.py").read_text(encoding="utf-8") == source
    assert _git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    recorded = spans.read_text(encoding="utf-8").splitlines()
    assert [line.split(" ", 2)[2] for line in recorded] == ["mutation testing"]
    return ran


def test_removed_property_and_classmethod_are_killed_by_their_definitions_tests(
    tmp_path: Path,
) -> None:
    ran = _mutate(tmp_path, DECORATED, DECORATED_TESTS)
    output = ran.stdout + ran.stderr
    assert ran.returncode == 0, output
    lines = ran.stdout.splitlines()
    # Neither baseline nor any mutant needed the whole suite.
    assert "mutation timing: baseline whole-suite runs=0 seconds=0.000" in lines, output
    assert "mutation timing: mutant whole-suite runs=0 seconds=0.000" in lines, output
    # cosmic-ray 8.7.0 dumps five RemoveDecorator mutants here, at 4:0, 9:4,
    # 13:4, 17:4 and 19:4 (columns from zero), and none of any other kind;
    # @final and both @overload are not code.
    assert (
        "mutation: 2 mutant(s): 2 run by the tests covering their line, 0 by the whole suite "
        "(lines run only at import), 0 by no test; 3 more in annotations, parameter markers "
        "and typing-only decorators are not code"
    ) in lines, output


def test_a_removed_decorator_no_test_sees_survives_with_nothing_in_its_place(
    tmp_path: Path,
) -> None:
    ran = _mutate(tmp_path, CACHED, CACHED_TESTS)
    output = ran.stdout + ran.stderr
    assert ran.returncode == 1, output
    lines = ran.stdout.splitlines()
    # cosmic-ray 8.7.0 dumps 14 mutants: 11 operators for * and two numbers
    # for 2, each giving 3 * 2 another value than 6 (5, 1, 1.5, 1, 1, 9, 12,
    # 0, 3, 2, 1; 3 and 9), and one RemoveDecorator at 4:0, run by test_double.
    assert "mutation timing: mutant whole-suite runs=0 seconds=0.000" in lines, output
    assert 'MUTANT SURVIVED src/compare.py:4:1 RemoveDecorator -> ""' in lines, output
