"""Tests for the console-script entrypoint and module-execution guards.

The CLI has two lines that only run when the process starts through them:
``entrypoint`` reading ``sys.argv``, and the ``__main__`` guard. Both are
executed here for real -- the guard via ``runpy`` with
``run_name="__main__"`` -- rather than excluded from coverage. A line
excluded because it is awkward to reach is a line nobody has ever run.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator, Sequence

import pytest

from code_corpus.cli import emit_corpus
from tests.conftest import make_repo


def _make_argv() -> Generator[list[str], None, None]:
    """Give a test control of ``sys.argv`` and restore it afterwards.

    Yields:
        The live argument list, for the test to replace in place.
    """
    original = list(sys.argv)
    yield sys.argv
    sys.argv[:] = original


# The call form resolves pytest's overloaded decorator to a concrete type;
# the bare @pytest.fixture expression carries Any under disallow_any_expr.
argv = pytest.fixture(_make_argv)


def _set(argv: list[str], args: Sequence[str]) -> None:
    """Replace the process arguments with a program name and the given args.

    Args:
        argv: The live ``sys.argv`` list.
        args: Arguments excluding the program name.
    """
    argv[:] = ["prog", *args]


def _emit_args(tmp_path: pathlib.Path) -> list[str]:
    """Build arguments for a minimal real emission.

    Args:
        tmp_path: Directory to build the repository and outputs in.

    Returns:
        Arguments excluding the program name.
    """
    repo = tmp_path / "repo"
    make_repo(repo, {"a.py": b"x = 1\n"})
    return [
        "--repo",
        f"api={repo}",
        "--out",
        str(tmp_path / "corpus.jsonl"),
        "--holdout-fraction",
        "0",
    ]


class TestEntrypoints:
    def test_emit_corpus_reads_the_process_arguments(
        self, tmp_path: pathlib.Path, argv: list[str], emitted: list[str]
    ) -> None:
        _set(argv, _emit_args(tmp_path))
        assert emit_corpus.entrypoint() == 0
        assert (tmp_path / "corpus.jsonl").exists() is True
        assert emitted[-1] == "manifest         corpus.jsonl.manifest.json"


class TestModuleExecutionGuards:
    def test_emit_corpus_runs_as_a_module(self, tmp_path: pathlib.Path, argv: list[str]) -> None:
        _set(argv, _emit_args(tmp_path))
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("code_corpus.cli.emit_corpus", run_name="__main__")
        assert excinfo.value.code == 0

    def test_the_guard_runs_as_a_module(self, tmp_path: pathlib.Path, argv: list[str]) -> None:
        """This is how the Makefile invokes it, so the path the build uses is
        the path the suite exercises."""
        _set(argv, ["--root", str(tmp_path)])
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("scripts.guard", run_name="__main__")
        assert excinfo.value.code == 0
