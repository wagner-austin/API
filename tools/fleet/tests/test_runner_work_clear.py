"""The runner job-completed hook empties the job's trees (MCPs board task 621902c3).

The hook is executed for real under this host's bash, with a home and a
runner install laid out under ``tmp_path`` the way lavender's held them on
2026-10-30: an MCPs install whose job left its checkout, a venv in it, a
sealed release payload and sibling clones one level above the workspace,
beside the install's tool cache and the runner's unpacked self-update.
"""

from __future__ import annotations

import pathlib
import shlex
import subprocess
from typing import Final

import pytest

from fleet.core import runner_tool_caches, runner_work_clear
from tests._host_bash import host_bash

#: What each laid-out file holds: one MiB.
MIB: Final = 1024 * 1024

#: A read-only directory's mode, as a sealed release payload leaves one.
SEALED: Final = 0o555


def _hook(
    tmp_path: pathlib.Path, install: pathlib.Path, workspace: str
) -> subprocess.CompletedProcess[str]:
    """Run the hook once, as the runner runs it after a job's last step.

    Args:
        tmp_path: Holds the home and the hook's script.
        install: The runner install the job ran from.
        workspace: The ``GITHUB_WORKSPACE`` the runner hands the hook.

    Returns:
        The finished run.
    """
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    environment = (
        f"HOME={shlex.quote(home.as_posix())}\n"
        f"RUNNER_TEMP={shlex.quote((install / '_work' / '_temp').as_posix())}\n"
        f"GITHUB_WORKSPACE={shlex.quote(workspace)}\n"
    )
    script = tmp_path / "hook.sh"
    first, rest = runner_work_clear.JOB_COMPLETED_SCRIPT.split("\n", 1)
    script.write_text(f"{first}\n{environment}{rest}", encoding="utf-8")
    return subprocess.run(
        [host_bash(), script.as_posix()], capture_output=True, text=True, check=False, timeout=60
    )


def _mcps_install(tmp_path: pathlib.Path) -> pathlib.Path:
    """Lay out an MCPs runner install as its last job left it.

    Returns:
        The install's root.
    """
    install = tmp_path / "home" / "actions-runner-mcps-1"
    work = install / "_work"
    (work / "_temp").mkdir(parents=True)
    workspace = work / "MCPs" / "MCPs"
    venv = workspace / "packages" / "maketools" / ".venv"
    venv.mkdir(parents=True)
    (venv / "torch.so").write_bytes(bytes(3 * MIB))
    sealed = workspace / "packages" / "maketools" / ".pytest-tmp" / "sealed"
    sealed.mkdir(parents=True)
    (sealed / "payload").write_bytes(b"release")
    sealed.chmod(SEALED)
    for clone in ("wiki", "tech-wiki"):
        (work / "MCPs" / clone).mkdir()
        (work / "MCPs" / clone / "index.md").write_bytes(bytes(MIB))
    for kept in ("_tool", "_actions", "_update"):
        (work / kept).mkdir()
        (work / kept / "kept.bin").write_bytes(bytes(MIB))
    return install


@pytest.mark.host_linux
def test_the_job_s_workspace_is_emptied_and_every_clone_beside_it_removed(
    tmp_path: pathlib.Path,
) -> None:
    install = _mcps_install(tmp_path)
    work = install / "_work"
    workspace = work / "MCPs" / "MCPs"

    ran = _hook(tmp_path, install, workspace.as_posix())

    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == (
        "fleet-runner-job-completed: removed 11 entries (5 MB) under "
        f"{(work / 'MCPs').as_posix()}\n"
    )
    assert [entry.name for entry in (work / "MCPs").iterdir()] == ["MCPs"]
    assert list(workspace.iterdir()) == []
    assert sorted(entry.name for entry in work.iterdir()) == [
        "MCPs",
        "_actions",
        "_temp",
        "_tool",
        "_update",
    ]
    assert [(work / kept / "kept.bin").stat().st_size for kept in ("_tool", "_actions")] == [
        MIB,
        MIB,
    ]
    record = tmp_path / "home" / ".cache" / "fleet-jobs" / "actions-runner-mcps-1"
    instant, entries, size = (record / "work-cleared").read_text(encoding="utf-8").split()
    assert (entries, size) == ("11", "5")
    assert int(instant) > 1_790_000_000


@pytest.mark.host_linux
def test_an_emptied_tree_is_emptied_again_with_nothing_to_remove(tmp_path: pathlib.Path) -> None:
    install = _mcps_install(tmp_path)
    workspace = install / "_work" / "MCPs" / "MCPs"
    first = _hook(tmp_path, install, workspace.as_posix())

    again = _hook(tmp_path, install, workspace.as_posix())

    assert (first.returncode, again.returncode) == (0, 0), again.stderr
    assert again.stdout == (
        "fleet-runner-job-completed: removed 0 entries (0 MB) under "
        f"{(install / '_work' / 'MCPs').as_posix()}\n"
    )


@pytest.mark.host_linux
@pytest.mark.parametrize(
    "relative",
    [
        "MCPs",
        "MCPs/MCPs/packages",
        "_temp/MCPs",
        "MCPs/_actions",
        ".hidden/MCPs",
        "MCPs/.git",
    ],
)
def test_a_workspace_that_is_no_job_tree_of_the_install_fails_the_job_by_name(
    tmp_path: pathlib.Path, relative: str
) -> None:
    install = _mcps_install(tmp_path)
    workspace = f"{(install / '_work').as_posix()}/{relative}"

    ran = _hook(tmp_path, install, workspace)

    assert ran.returncode == 1
    assert ran.stderr == (
        f"FLEET_RUNNER_WORK_REFUSED: GITHUB_WORKSPACE {workspace} is no "
        f"_work/<repository>/<name> of {install.as_posix()}\n"
    )
    assert (install / "_work" / "MCPs" / "wiki" / "index.md").stat().st_size == MIB


@pytest.mark.host_linux
def test_a_workspace_of_another_install_is_refused(tmp_path: pathlib.Path) -> None:
    install = _mcps_install(tmp_path)
    other = tmp_path / "home" / "actions-runner-api-1" / "_work" / "API" / "API"
    other.mkdir(parents=True)
    (other / "kept.py").write_text("kept\n", encoding="utf-8")

    ran = _hook(tmp_path, install, other.as_posix())

    assert ran.returncode == 1
    assert ran.stderr == (
        f"FLEET_RUNNER_WORK_REFUSED: GITHUB_WORKSPACE {other.as_posix()} is no "
        f"_work/<repository>/<name> of {install.as_posix()}\n"
    )
    assert (other / "kept.py").read_text(encoding="utf-8") == "kept\n"


def test_the_hook_names_its_install_before_it_clears() -> None:
    lines = runner_work_clear.JOB_COMPLETED_SCRIPT.splitlines()
    assert runner_work_clear.JOB_COMPLETED_HOOK_PATH.endswith(".sh")
    strict = lines.index("set -euo pipefail")
    assert lines[strict + 1 :] == [
        *runner_tool_caches.INSTALL_LINES,
        *runner_work_clear.CLEAR_LINES,
    ]
