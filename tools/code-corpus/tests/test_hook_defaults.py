"""The production git runner, run for real, succeeding and failing.

Every selection test reaches git through the ``run_git`` hook, which the
autouse fixture rebinds and several tests replace with a canned fake, so
none of them is evidence of what the real runner does when git refuses. These
call the implementation itself, ``_default_run_git``, against a real
repository and against a directory that is not one (``effect-seam-twin``,
board task cc7222ca).
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from code_corpus.core import _test_hooks as core_hooks
from tests.conftest import make_repo


def test_the_real_runner_lists_a_real_repository(tmp_path: pathlib.Path) -> None:
    make_repo(tmp_path, {"a.py": b"x = 1\n", "sub/b.py": b"y = 2\n"})
    listed = core_hooks._default_run_git(tmp_path, ["ls-files", "-z"])
    assert listed.split("\0") == ["a.py", "sub/b.py", ""]


def test_a_directory_that_is_not_a_repository_raises_with_gits_status(
    tmp_path: pathlib.Path,
) -> None:
    """A corpus whose input cannot be pinned is refused, not emitted empty."""
    with pytest.raises(subprocess.CalledProcessError) as caught:
        core_hooks._default_run_git(tmp_path, ["rev-parse", "HEAD"])
    assert caught.value.returncode == 128
    argv = ["git", "-C", str(tmp_path), "rev-parse", "HEAD"]
    assert str(caught.value) == f"Command '{argv}' returned non-zero exit status 128."
