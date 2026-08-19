"""Shared fixtures.

Hooks are reset before and after every test so a rebinding made by one test
cannot leak into another. Leakage here would be particularly hard to
diagnose, because the symptom is a test failing only when run after a
specific other test, which ``-n auto`` reorders.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Generator

import pytest

from code_corpus.cli import _test_hooks as cli_hooks
from code_corpus.core import _test_hooks as core_hooks
from code_corpus.core.select import git_head


def make_repo(root: pathlib.Path, files: dict[str, bytes]) -> str:
    """Build a one-commit git repository holding the given files.

    The emitter pins every repository's commit behind a corpus, so a test
    repository must be a real one: the tests then exercise the production
    ``run_git`` hook rather than a stand-in. Contents are written as bytes so
    a test controls line endings and encoding exactly.

    Args:
        root: Directory to initialise.
        files: Repository-relative paths to their exact contents.

    Returns:
        The commit hash of the repository's HEAD.
    """
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    identity = ["-c", "user.email=test@example.invalid", "-c", "user.name=test"]
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(root), *identity, "commit", "-q", "-m", "fixture"],
        check=True,
    )
    return git_head(root)


@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Rebind every hook to production before and after each test."""
    core_hooks.reset_hooks()
    cli_hooks.reset_hooks()
    yield
    core_hooks.reset_hooks()
    cli_hooks.reset_hooks()


def _make_emitted() -> Generator[list[str], None, None]:
    """Capture CLI summary lines instead of writing them to stdout.

    Yields:
        The list the CLI's ``emit`` hook appends to, in emission order.
    """
    lines: list[str] = []
    cli_hooks.emit = lines.append
    yield lines
    cli_hooks.reset_hooks()


# The call form resolves pytest's overloaded decorator to a concrete type;
# the bare @pytest.fixture expression carries Any under disallow_any_expr.
emitted = pytest.fixture(_make_emitted)
