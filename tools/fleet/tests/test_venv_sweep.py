"""The fleet cache's virtualenv sweep (MCPs board task 7b07c5d2).

Which script each platform is sent, how a node is asked to run it, and, on a
linux host, the sh script executed against a cache laid out the way
lavender-wsl held it on 2026-09-29: virtualenvs whose ``.pth`` files record
only source paths under retired runs. The Windows script is executed by
``tests/pester/rendered-venv-sweep.Tests.ps1`` from its committed render.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.budget import NodeBudget
from fleet.contracts.node import NodeConfig, NodePlatform
from fleet.core import _test_hooks, venv_sweep
from fleet.core.dialect_linux import SH_INVOCATION
from tests.conftest import FakeRun, failed, ok


def _node(platform: NodePlatform, stage_root: str) -> NodeConfig:
    """Build a node for the sweep.

    Args:
        platform: Its platform.
        stage_root: Its stage root.

    Returns:
        The node.
    """
    return NodeConfig(
        host="lavender-wsl",
        platform=platform,
        stage_root=stage_root,
        logical_cores=16,
        ram_gb=25.4,
        gpu=None,
        enabled=True,
        test_database=True,
        rust=None,
        cxx="13.3.0",
        docker=None,
        stack=None,
        elevated=False,
        wsl_host="lavender",
        budget=NodeBudget(
            reserved_cores=8,
            reserved_ram_gb=6.0,
            worker_ram_gb=1.1,
            max_disk_gb=40.0,
            checks_at_once=None,
        ),
    )


def test_the_virtualenvs_live_under_the_node_cache() -> None:
    assert venv_sweep.virtualenvs_directory("/home/corvis/fleet/stage") == (
        "/home/corvis/fleet/stage/cache/pypoetry/virtualenvs"
    )


def test_each_platform_is_sent_its_own_script() -> None:
    linux = venv_sweep.script_for(NodePlatform.LINUX, stage_root="/home/corvis/fleet/stage")
    windows = venv_sweep.script_for(NodePlatform.WINDOWS, stage_root="C:/fleet/stage")

    assert linux == venv_sweep.linux_script("/home/corvis/fleet/stage/cache/pypoetry/virtualenvs")
    assert "venvs=/home/corvis/fleet/stage/cache/pypoetry/virtualenvs\n" in linux
    assert windows == venv_sweep.windows_script("C:/fleet/stage/cache/pypoetry/virtualenvs")
    assert "[string]$Venvs = 'C:/fleet/stage/cache/pypoetry/virtualenvs'" in windows


def test_a_directory_a_rendered_script_cannot_carry_is_refused() -> None:
    with pytest.raises(ValueError, match="virtualenvs directory"):
        venv_sweep.windows_script("C:/it's/virtualenvs")


@pytest.mark.parametrize(
    ("platform", "stage_root", "script_path"),
    [
        (
            NodePlatform.LINUX,
            "/home/corvis/fleet/stage",
            "/home/corvis/fleet/stage/fleet-venv-sweep.sh",
        ),
        (NodePlatform.WINDOWS, "C:/fleet/stage", "C:/fleet/stage/fleet-venv-sweep.ps1"),
    ],
)
def test_a_node_is_sent_the_sweep_under_its_stage_root_and_its_report_returned(
    platform: NodePlatform, stage_root: str, script_path: str
) -> None:
    runner = FakeRun([ok(""), ok("venv-sweep: removed 7 of 7 (24110 MB)\n")])
    _test_hooks.run = runner

    report = venv_sweep.sweep_on_node(_node(platform, stage_root))

    assert report == "venv-sweep: removed 7 of 7 (24110 MB)"
    assert script_path in " ".join(runner.calls[0])
    assert runner.stdin[0] == venv_sweep.script_for(platform, stage_root=stage_root).encode("utf-8")


def test_a_node_that_does_not_answer_is_left_to_the_next_sweep() -> None:
    """lavender-wsl at 04:05Z on 2026-10-07: the sweep after a closed row
    met a timed-out banner exchange, and raising it ended the serve (MCPs
    board task c1d48330)."""
    runner = FakeRun([failed(255, "Connection timed out during banner exchange")])
    _test_hooks.run = runner

    report = venv_sweep.sweep_on_node(_node(NodePlatform.LINUX, "/home/corvis/fleet/stage"))

    assert report == (
        "venv-sweep: lavender-wsl did not answer; the next sweep removes its orphans: ssh to "
        "lavender-wsl failed while sending /home/corvis/fleet/stage/fleet-venv-sweep.sh: "
        "Connection timed out during banner exchange"
    )
    assert len(runner.calls) == 1


def test_a_node_that_answers_with_a_failed_sweep_raises_it() -> None:
    _test_hooks.run = FakeRun([ok(""), failed(1, "rm: cannot remove: Permission denied")])

    with pytest.raises(AppError) as raised:
        venv_sweep.sweep_on_node(_node(NodePlatform.LINUX, "/home/corvis/fleet/stage"))

    assert raised.value.code is FleetErrorCode.DISPATCH_FAILED


def _virtualenv(venvs: pathlib.Path, name: str, lines: list[str] | None) -> pathlib.Path:
    """Lay out one poetry virtualenv the way a linux node holds it.

    Args:
        venvs: The virtualenvs directory.
        name: The virtualenv's folder name.
        lines: Its ``.pth`` lines, or None for one with no site-packages yet.

    Returns:
        The virtualenv's directory.
    """
    venv = venvs / name
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    if lines is not None:
        site = venv / "lib" / "python3.12" / "site-packages"
        site.mkdir(parents=True)
        (site / "weights.bin").write_bytes(bytes(2 * 1024 * 1024))
        (site / "project.pth").write_text("".join(f"{line}\n" for line in lines), "utf-8")
    return venv


@pytest.mark.host_linux
def test_the_sh_sweep_removes_only_virtualenvs_whose_sources_are_all_gone(
    tmp_path: pathlib.Path,
) -> None:
    venvs = tmp_path / "cache" / "pypoetry" / "virtualenvs"
    alive = tmp_path / "stage" / "run with space" / "src"
    alive.mkdir(parents=True)
    gone = str(tmp_path / "stage" / "MCPs-doc-extract-api-lavender-wsl-1790675845" / "src")
    orphan = _virtualenv(venvs, "doc-extract-api-1JE3TEfc-py3.12", [gone, "import _virtualenv"])
    serving = _virtualenv(venvs, "demo-alive-py3.12", [gone, str(alive)])
    unknown = _virtualenv(venvs, "demo-unknown-py3.12", ["import _virtualenv"])
    creating = _virtualenv(venvs, "demo-creating-py3.12", None)
    script = tmp_path / "sweep.sh"
    script.write_text(venv_sweep.linux_script(str(venvs)), encoding="utf-8")

    completed = subprocess.run(
        [*SH_INVOCATION, str(script)], capture_output=True, text=True, check=False, timeout=120
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "venv-sweep: removed 1 of 4 (2 MB)\n"
    assert not orphan.exists()
    assert [serving.exists(), unknown.exists(), creating.exists()] == [True, True, True]


@pytest.mark.host_linux
def test_the_sh_sweep_reports_nothing_when_the_node_has_no_virtualenvs_yet(
    tmp_path: pathlib.Path,
) -> None:
    script = tmp_path / "sweep.sh"
    script.write_text(venv_sweep.linux_script(str(tmp_path / "absent")), encoding="utf-8")

    completed = subprocess.run(
        [*SH_INVOCATION, str(script)], capture_output=True, text=True, check=False, timeout=120
    )

    assert (completed.returncode, completed.stdout) == (0, "venv-sweep: removed 0 of 0 (0 MB)\n")
