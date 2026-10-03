"""``ps-harness`` reads API's harness pin in Python, under every make (MCPs board task 939ec5c7).

Fleet job 2d949933 at 1f767a97f (repo API) ran tools/ps-harness on
serendipity, whose GnuWin32 make 3.81 has no ``$(file ...)``, and the recipe
handed the harness an empty ``--ref``. These pin the command that replaced
that read: the argv it builds from the pin file, the CLI routing, and one
case that executes the real chain, the shipped ``published_maketools.py``
beside a real MCPs git repository whose origin/main carries a launcher that
prints what it was handed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest
from platform_core.errors import AppError

from maketools import cli
from maketools.ps_harness import (
    HARNESS_WALL_SECONDS,
    PIN_PATH,
    PUBLISHED_MAKETOOLS,
    harness_argv,
    run_pinned_harness,
)
from tests.conftest import World, restore_defaults

#: A pin as the file holds one: forty hex characters and a line ending.
PIN: Final[str] = "3fbe8d3f00b1a1305d98afb4383e3c1d7cbfefab"

#: The launcher MCPs' origin/main carries in the executed case.
ECHOING_LAUNCHER: Final[str] = 'import sys\nprint("published", *sys.argv[1:])\n'

#: Wall for the fixture's git commands, each of which exits at once.
GIT_WALL_SECONDS: Final[int] = 120


def _repository(root: Path) -> Path:
    """A repository root holding a pin file, the way API holds it.

    Args:
        root: Where to make it.

    Returns:
        The root.
    """
    (root / PIN_PATH).parent.mkdir(parents=True)
    (root / PIN_PATH).write_text(f"{PIN}\n", encoding="utf-8")
    return root


class TestTheArgv:
    def test_it_runs_the_published_script_at_the_pin_with_mcps_beside_the_repository(
        self, world: World, tmp_path: Path
    ) -> None:
        repo = _repository(tmp_path / "API")
        world.inheriting_code = 3

        assert run_pinned_harness(repo) == 3

        (call,) = world.inheriting_calls
        assert call["argv"] == (
            sys.executable,
            str(repo / PUBLISHED_MAKETOOLS),
            "ps-harness",
            "--ref",
            PIN,
            str(tmp_path / "MCPs"),
            str(repo),
        )
        assert (call["cwd"], call["env"], call["new_session"]) == (repo, world.environment, False)
        assert call["timeout_seconds"] == HARNESS_WALL_SECONDS

    def test_a_repository_without_the_pin_file_is_an_error_not_an_empty_ref(
        self, world: World, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            run_pinned_harness(tmp_path)
        assert world.inheriting_calls == []


class TestTheCommand:
    def test_it_reads_this_repositorys_own_pin(self, world: World) -> None:
        assert cli.dispatch(["ps-harness"]) == 0

        (call,) = world.inheriting_calls
        pin = call["argv"][4]
        assert re.fullmatch(r"[0-9a-f]{40}", pin)
        assert call["argv"] == harness_argv(cli.repository_root(), pin)

    def test_it_takes_no_arguments(self, world: World) -> None:
        with pytest.raises(AppError, match="ps-harness"):
            cli.dispatch(["ps-harness", "../MCPs"])
        assert world.inheriting_calls == []


def _git(root: Path, *args: str) -> None:
    """Run git in ``root`` with a fixed identity, requiring success.

    Args:
        root: The repository.
        args: git's arguments.
    """
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=GIT_WALL_SECONDS,
    )


def test_the_real_chain_hands_the_published_command_the_pin(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """No hook is replaced: the default ``run_inheriting`` runs the shipped
    script, which extracts the launcher from MCPs' origin/main and runs it."""
    restore_defaults()
    repo = _repository(tmp_path / "API")
    shutil.copyfile(cli.repository_root() / PUBLISHED_MAKETOOLS, repo / PUBLISHED_MAKETOOLS)
    mcps = tmp_path / "MCPs"
    launcher = mcps / "packages" / "maketools" / "scripts" / "run.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(ECHOING_LAUNCHER, encoding="utf-8")
    _git(mcps, "init", "--quiet")
    _git(mcps, "add", ".")
    _git(mcps, "commit", "--quiet", "--message", "the published launcher")
    _git(mcps, "update-ref", "refs/remotes/origin/main", "HEAD")

    assert run_pinned_harness(repo) == 0

    printed = capfd.readouterr().out.strip().splitlines()[-1]
    assert printed == f"published ps-harness --ref {PIN} {mcps} {repo}"
