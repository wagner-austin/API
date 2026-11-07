"""The shared mutation step fails a change whose tests leave a mutant alive (MCPs task e52e5148).

Every API package that opts in runs ``run.py test --mutation``; this case
runs that same command, as a real child of this package's launcher, in a
real repository it builds, with this package's own interpreter and its
pinned cosmic-ray (the fixture's ``.venv`` is a link to this suite's
environment), so the launcher, the suite's coverage contexts, git, and
cosmic-ray's answers are all real. It is MCPs' fixture for its own step
(``packages/maketools/tests/test_mutation_step_real.py``, board task
fec5851c), so both repositories prove the same mutant with the same line.

The repository's ``origin/main`` is its first commit; its second adds
``at_least(value, floor)`` returning ``value >= floor`` with tests that
check a value above and a value below the floor but never the floor
itself. cosmic-ray's seven replacements of ``>=`` are run against the two
tests coverage names for that line, and ``>`` is the one they cannot tell
apart: the launcher must exit 1 naming it, put the source back byte for
byte, and record its time as the check's mutation span. A third commit
adds the test of the floor, and the launcher must pass.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Final

from maketools import _test_hooks
from maketools.bounded_spans import SPANS_ENVIRONMENT_KEY
from maketools.cli import repository_root

#: The launcher every Makefile recipe calls.
LAUNCHER: Final[Path] = repository_root() / "tools" / "maketools" / "scripts" / "run.py"

#: The wall on each git command and on each launcher run.
WALL_SECONDS: Final[int] = 240

#: An identity for the fixture's commits, whatever this machine's git says.
IDENTITY: Final[tuple[str, ...]] = ("-c", "user.name=t", "-c", "user.email=t@t")

PYPROJECT = """\
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]

[tool.coverage.run]
source = ["src"]
branch = true
"""

SOURCE = """\
def at_least(value: int, floor: int) -> bool:
    return value >= floor
"""

WEAK_TESTS = """\
from compare import at_least


def test_above() -> None:
    assert at_least(3, 1)


def test_below() -> None:
    assert not at_least(0, 1)
"""

FLOOR_TEST = """

def test_floor() -> None:
    assert at_least(1, 1)
"""


def _git(repo: Path, *arguments: str) -> str:
    """Run git in the fixture repository and require it to succeed.

    Args:
        repo: The repository.
        arguments: git's arguments.

    Returns:
        Its standard output.
    """
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=True,
        timeout=WALL_SECONDS,
    ).stdout


def _commit(repo: Path, files: dict[str, str], message: str) -> None:
    """Write files and commit them.

    Args:
        repo: The repository.
        files: Each path, relative to it, to its text.
        message: The commit message.
    """
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", "--", *files)
    _git(repo, *IDENTITY, "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", message)


def _check(repo: Path, spans: Path) -> subprocess.CompletedProcess[str]:
    """Run the launcher's mutation test in the fixture package.

    The environment holds nothing of this suite's own: the outer pytest-cov
    exports its coverage settings to subprocesses, and the fixture runs
    with none of them, as a package's ``make test`` does.

    Args:
        repo: The fixture package, a repository of its own.
        spans: The span file the step records into.

    Returns:
        The launcher's run.
    """
    env = {
        "PATH": _test_hooks.environ()["PATH"],
        "HOME": str(repo.parent),
        SPANS_ENVIRONMENT_KEY: str(spans),
    }
    argv = [sys.executable, str(LAUNCHER), "test", "--mutation", "--no-sweep", "--serial"]
    argv += ["--runner", "venv", "-p", "no:cacheprovider"]
    return subprocess.run(
        argv, cwd=repo, env=env, check=False, capture_output=True, text=True, timeout=WALL_SECONDS
    )


def test_a_change_whose_tests_miss_the_floor_fails_and_one_that_checks_it_passes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "fixture"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    ignored = ".venv\nruns/\n"
    _commit(repo, {"pyproject.toml": PYPROJECT, ".gitignore": ignored}, "start")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _commit(repo, {"src/compare.py": SOURCE, "tests/test_compare.py": WEAK_TESTS}, "weak")
    spans = tmp_path / "spans"

    weak = _check(repo, spans)
    output = weak.stdout + weak.stderr
    assert weak.returncode == 1, output
    lines = weak.stdout.splitlines()
    assert (
        'MUTANT SURVIVED src/compare.py:2:18 ReplaceComparisonOperator_GtE_Gt -> "return value '
        '> floor"'
    ) in lines, output
    failed = [line for line in lines if line.startswith("mutation: FAILED: ")]
    assert failed == [
        "mutation: FAILED: 1 of 7 mutant(s) made in 1 src file(s) changed against origin/main "
        f"(merge base {_git(repo, 'rev-parse', 'HEAD~1').strip()}) survived. Kill each with a "
        "test, or rewrite the code so the mutation cannot change what it does."
    ], output
    assert (repo / "src" / "compare.py").read_text(encoding="utf-8") == SOURCE
    assert _git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert list((repo / "runs").iterdir()) == []

    _commit(repo, {"tests/test_compare.py": WEAK_TESTS + FLOOR_TEST}, "floor")
    strong = _check(repo, spans)
    assert strong.returncode == 0, strong.stdout + strong.stderr
    assert any(
        line.startswith("mutation: passed: 0 of 7 mutant(s) made in 1 src file(s)")
        for line in strong.stdout.splitlines()
    ), strong.stdout
    recorded = spans.read_text(encoding="utf-8").splitlines()
    assert [line.split(" ", 2)[2] for line in recorded] == ["mutation testing"] * 2
