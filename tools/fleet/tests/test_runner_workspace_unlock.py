"""The runner job hook leaves its install's work tree deletable (MCPs board task 678de6ba).

The hook's unlock lines are executed for real under this host's bash, after
the ``install_root`` the cache lines name, over a runner install laid out
under ``tmp_path`` the way lavender's MCPs runner held it on 2026-10-28: a
sealed release payload, read-only directories, under the checkout's
``packages/maketools/.pytest-tmp``.
"""

from __future__ import annotations

import pathlib
import shlex
import stat
import subprocess

from fleet.core import runner_workspace_unlock
from tests._host_bash import host_bash

#: A read-only directory's mode, as a sealed release payload leaves one.
SEALED = 0o555


def _unlock(install: pathlib.Path, tmp_path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    """Run the hook's unlock lines once over ``install``.

    Args:
        install: The runner install whose ``_work`` is unlocked.
        tmp_path: Holds the hook's script.

    Returns:
        The finished run.
    """
    header = f"set -euo pipefail\ninstall_root={shlex.quote(install.as_posix())}\n"
    script = tmp_path / "unlock.sh"
    script.write_text(
        header + "".join(f"{line}\n" for line in runner_workspace_unlock.UNLOCK_LINES),
        encoding="utf-8",
    )
    return subprocess.run(
        [host_bash(), script.as_posix()], capture_output=True, text=True, check=False, timeout=60
    )


def _writable(path: pathlib.Path) -> bool:
    """Whether a directory's owner may write it.

    Args:
        path: The directory, read without following a link.

    Returns:
        ``True`` when its owner write bit is set.
    """
    return bool(path.lstat().st_mode & stat.S_IWUSR)


def test_a_sealed_tree_is_made_writable_and_a_link_out_of_it_is_not_followed(
    tmp_path: pathlib.Path,
) -> None:
    """Two sealed directories of seven are unlocked; the one a link names outside is not."""
    install = tmp_path / "actions-runner-mcps-1"
    work = install / "_work"
    (work / "_temp").mkdir(parents=True)
    basetemp = work / "MCPs" / "MCPs" / ".pytest-tmp"
    inner = basetemp / "sealed" / "inner"
    inner.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (basetemp / "link").symlink_to(outside, target_is_directory=True)
    for directory in (inner, inner.parent, outside):
        directory.chmod(SEALED)
    try:
        ran = _unlock(install, tmp_path)

        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == (
            f"fleet-runner-job-started: 2 of 7 directories under {work.as_posix()} were "
            "read-only and are now writable\n"
        )
        assert [_writable(path) for path in (inner, inner.parent, basetemp, outside)] == [
            True,
            True,
            True,
            False,
        ]
    finally:
        outside.chmod(0o755)


def test_a_writable_tree_is_examined_and_left_as_it_was(tmp_path: pathlib.Path) -> None:
    """A clean install reports none of its directories unlocked."""
    install = tmp_path / "actions-runner-api-1"
    (install / "_work" / "_temp").mkdir(parents=True)

    ran = _unlock(install, tmp_path)

    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == (
        f"fleet-runner-job-started: 0 of 2 directories under {(install / '_work').as_posix()} "
        "were read-only and are now writable\n"
    )
