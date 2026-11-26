"""The API launcher detects a multiline predicate mutation (MCPs board task ee29c9e5)."""

from __future__ import annotations

import sys
from pathlib import Path

from .test_mutation_real import PYPROJECT, _check, _commit, _git

SOURCE = """\
def accepts(value):
    if (
        value
    ):
        return True
    return False
"""
TESTS = """\
from compare import accepts

def test_true():
    assert accepts(True) is True

def test_false():
    assert accepts(False) is False
"""


def test_the_launcher_runs_the_multiline_condition_negation(tmp_path: Path) -> None:
    repo = tmp_path / "fixture"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    _commit(repo, {"pyproject.toml": PYPROJECT, ".gitignore": ".venv\nruns/\n"}, "start")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _commit(repo, {"src/compare.py": SOURCE, "tests/test_compare.py": TESTS}, "condition")
    spans = tmp_path / "spans"
    ran = _check(repo, spans)
    output = ran.stdout + ran.stderr
    assert ran.returncode == 0, output
    assert "0 by the whole suite (lines run only at import), 0 by no test" in output
    assert "core/AddNot" in output
    assert (repo / "src/compare.py").read_text(encoding="utf-8") == SOURCE
    assert _git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    recorded = spans.read_text(encoding="utf-8").splitlines()
    assert [line.split(" ", 2)[2] for line in recorded] == ["mutation testing"]
