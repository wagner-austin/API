"""``lock`` and ``lock-check`` against real poetry (MCPs board task 972ea464).

The twin of :mod:`tests.test_poetry_lock`'s fakes: a package whose only
dependency is a path dependency with none of its own, so poetry resolves it
offline in about a second. Each case drives the commands through the
registry from the package's directory with the default hooks, so the poetry
that runs is the one the recipes run, and each refusal is real poetry or a
real file failing the check, never a scripted exit status.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools.cli import dispatch, main

#: A path dependency declaring no dependency of its own.
MEMBER = """[project]
name = "member"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[build-system]
requires = ["poetry-core>=2.0"]
build-backend = "poetry.core.masonry.api"
"""

#: The package: no code of its own, one path dependency.
ROOT = """[project]
name = "root"
version = "0.1.0"
requires-python = ">=3.11"
dynamic = ["dependencies"]

[tool.poetry]
package-mode = false

[tool.poetry.dependencies]
member = { path = "../member", develop = true }
"""


@pytest.fixture()
def root(tmp_path: Path) -> Generator[Path, None, None]:
    """The package beside its path dependency, as the working directory.

    Args:
        tmp_path: The test's directory.

    Yields:
        The package directory.
    """
    for name, text in (("member", MEMBER), ("root", ROOT)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "pyproject.toml").write_text(text, encoding="utf-8")
    before = Path.cwd()
    os.chdir(tmp_path / "root")
    yield tmp_path / "root"
    os.chdir(before)


def refusal(command: str) -> str:
    """The message a command refuses with.

    Args:
        command: ``lock-check`` or ``lock``.

    Returns:
        The refusal's message, after asserting its code.
    """
    with pytest.raises(AppError) as caught:
        dispatch([command])
    assert caught.value.code is MaketoolsErrorCode.LOCK_STALE
    return caught.value.message


def test_lock_writes_a_lock_and_record_that_lock_check_accepts(root: Path) -> None:
    assert dispatch(["lock"]) == 0
    assert 'url = "../member"' in (root / "poetry.lock").read_text(encoding="utf-8")
    assert '"../member"' in (root / "paths-lock.json").read_text(encoding="utf-8")
    assert dispatch(["lock-check"]) == 0


def test_a_path_dependency_moving_is_refused_although_poetry_accepts_it(root: Path) -> None:
    assert dispatch(["lock"]) == 0
    member = root.parent / "member" / "pyproject.toml"
    member.write_text(MEMBER.replace('"0.1.0"', '"0.2.0"'), encoding="utf-8")
    assert refusal("lock-check").startswith(
        "root: poetry.lock was resolved against other path dependencies than these: "
        "../member: its declared dependencies changed since the lock was written"
    )
    assert dispatch(["lock"]) == 0
    assert 'version = "0.2.0"' in (root / "poetry.lock").read_text(encoding="utf-8")
    assert dispatch(["lock-check"]) == 0


def test_the_packages_own_declarations_moving_is_refused_by_poetry(root: Path) -> None:
    assert dispatch(["lock"]) == 0
    (root.parent / "second").mkdir()
    (root.parent / "second" / "pyproject.toml").write_text(
        MEMBER.replace('"member"', '"second"'), encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        ROOT + 'second = { path = "../second", develop = true }\n', encoding="utf-8"
    )
    assert "pyproject.toml changed significantly since poetry.lock was last generated" in (
        refusal("lock-check")
    )


def test_a_package_never_locked_is_refused_by_poetry(root: Path) -> None:
    assert "poetry.lock" in refusal("lock-check")


def test_a_resolution_poetry_cannot_make_is_its_exit_status(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        ROOT.replace('path = "../member"', 'path = "../absent"'), encoding="utf-8"
    )
    assert main(["lock"]) != 0
    assert not (root / "paths-lock.json").exists()
