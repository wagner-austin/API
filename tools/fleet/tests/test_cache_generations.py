"""Removing the cache generations no run reads (MCPs board task 346abc45).

Which script a node is sent and how it is asked to run it, then the ``sh``
removal executed for real under this host's bash against a stage root under
``tmp_path``: a generation goes only when it is not current, no run
directory names it, and it stopped being current over an hour ago.
"""

from __future__ import annotations

import pathlib
import subprocess
import time
from typing import Final

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.node import NodeConfig
from fleet.core import _test_hooks, cache_generations, names
from fleet.core.linux_isolated_build import EXEC_HOME_LINE
from tests._host_bash import host_bash
from tests.conftest import FakeRun, failed, ok

#: One day in seconds.
DAY: Final = 86400

#: What a generation's laid-out file holds: one MiB.
MIB: Final = 1024 * 1024


#: The stage root every node declares.
STAGE_ROOT: Final = "/home/corvis/fleet/stage"


def _node(docker: str | None) -> NodeConfig:
    """A node whose cache generations are judged.

    Args:
        docker: Its execdocker daemon's version, or None.

    Returns:
        The node.
    """
    return NodeConfig(
        host="lavender-wsl",
        stage_root=STAGE_ROOT,
        logical_cores=16,
        ram_gb=25.4,
        gpu=None,
        enabled=True,
        test_database=True,
        rust=None,
        cxx=None,
        docker=docker,
        stack=None,
        budget={
            "reserved_cores": 8,
            "reserved_ram_gb": 6.0,
            "worker_ram_gb": 1.1,
            "max_disk_gb": 40.0,
            "checks_at_once": None,
            "pinned_checks_only": False,
        },
    )


def test_a_node_without_isolated_builds_judges_its_own_generations_alone() -> None:
    script = cache_generations.removal_script(STAGE_ROOT, isolated=False)

    assert f"remove_unused {STAGE_ROOT}/cache/tools {STAGE_ROOT}\n" in script
    assert "sudo" not in script


def test_a_node_running_isolated_builds_judges_execdocker_s_generations_too() -> None:
    script = cache_generations.removal_script(STAGE_ROOT, isolated=True)

    assert script.endswith(
        f"{EXEC_HOME_LINE}\nas='sudo -n'\n"
        'remove_unused "$exec_home/fleet/cache/tools" "$exec_home/fleet/stage"\n'
    )


@pytest.mark.parametrize("docker", ["29.1.3", None])
def test_a_node_is_sent_the_removal_under_its_stage_root_and_its_report_returned(
    docker: str | None,
) -> None:
    line = "cache-generations: /c: generation 1790000000 current, removed 1 of 2 (17000 MB)"
    runner = FakeRun([ok(""), ok(f"{line}\n")])
    _test_hooks.run = runner

    report = cache_generations.remove_unused_on_node(_node(docker))

    assert report == line
    assert f"{STAGE_ROOT}/fleet-cache-generations.sh" in " ".join(runner.calls[0])
    assert runner.stdin[0] == cache_generations.removal_script(
        STAGE_ROOT, isolated=docker is not None
    ).encode("utf-8")


def test_a_node_that_does_not_answer_is_left_to_the_next_removal() -> None:
    runner = FakeRun([failed(255, "Connection timed out during banner exchange")])
    _test_hooks.run = runner

    report = cache_generations.remove_unused_on_node(_node(None))

    assert report == (
        "cache-generations: lavender-wsl did not answer; the next removal judges its "
        "generations: ssh to lavender-wsl failed while sending "
        "/home/corvis/fleet/stage/fleet-cache-generations.sh: "
        "Connection timed out during banner exchange"
    )
    assert len(runner.calls) == 1


def test_a_node_that_answers_with_a_failed_removal_raises_it() -> None:
    _test_hooks.run = FakeRun([ok(""), failed(1, "rm: cannot remove: Permission denied")])

    with pytest.raises(AppError) as raised:
        cache_generations.remove_unused_on_node(_node(None))

    assert raised.value.code is FleetErrorCode.DISPATCH_FAILED


def _generation(tools: pathlib.Path, *, seconds_ago: int) -> str:
    """Lay out a generation started some time ago, holding one MiB.

    Args:
        tools: The generations directory.
        seconds_ago: How long ago it started.

    Returns:
        Its name.
    """
    name = str(int(time.time()) - seconds_ago)
    pypoetry = tools / name / "pypoetry" / "artifacts"
    pypoetry.mkdir(parents=True)
    (pypoetry / "torch.whl").write_bytes(bytes(MIB))
    return name


def _remove(stage: pathlib.Path, current: str) -> subprocess.CompletedProcess[str]:
    """Make a generation current and run the removal over the stage root.

    Args:
        stage: The stage root.
        current: The generation builds start in.

    Returns:
        The finished run.
    """
    tools = stage / names.CACHE_DIRECTORY / names.TOOL_CACHES_DIRECTORY
    (tools / names.CURRENT_GENERATION_NAME).write_text(f"{current}\n", encoding="utf-8")
    script = stage / f"{cache_generations.REMOVAL_STEM}.sh"
    script.write_text(cache_generations.removal_script(stage.as_posix(), isolated=False), "utf-8")
    return subprocess.run(
        [host_bash(), script.as_posix()], capture_output=True, text=True, check=False, timeout=60
    )


def _mark(stage: pathlib.Path, run: str, generation: str) -> None:
    """Lay out a live run's directory reading one generation.

    Args:
        stage: The stage root.
        run: The run's id.
        generation: The generation its marker names.
    """
    (stage / run).mkdir()
    (stage / run / names.GENERATION_MARKER_NAME).write_text(f"{generation}\n", encoding="utf-8")


def test_only_a_generation_no_run_reads_and_long_out_of_use_is_removed(
    tmp_path: pathlib.Path,
) -> None:
    tools = tmp_path / names.CACHE_DIRECTORY / names.TOOL_CACHES_DIRECTORY
    unread = _generation(tools, seconds_ago=30 * DAY)
    marked = _generation(tools, seconds_ago=16 * DAY)
    current = _generation(tools, seconds_ago=2 * 3600)
    racing = _generation(tools, seconds_ago=60)
    _mark(tmp_path, "MCPs-search-lavender-wsl-1792000000", marked)
    (tmp_path / names.LOGS_DIRECTORY).mkdir()

    ran = _remove(tmp_path, current)

    assert ran.returncode == 0, ran.stderr
    assert ran.stdout == (
        f"cache-generations: {tools.as_posix()}: generation {current} current, "
        "removed 1 of 4 (1 MB)\n"
    )
    assert sorted(entry.name for entry in tools.iterdir()) == sorted(
        [marked, current, racing, names.CURRENT_GENERATION_NAME]
    )
    assert not (tools / unread).exists()


def test_nothing_is_removed_within_the_hour_after_the_generation_changed(
    tmp_path: pathlib.Path,
) -> None:
    """A build that read the old name just before it changed writes its
    marker on the next line; the hour covers it."""
    tools = tmp_path / names.CACHE_DIRECTORY / names.TOOL_CACHES_DIRECTORY
    previous = _generation(tools, seconds_ago=15 * DAY)
    current = _generation(tools, seconds_ago=600)

    ran = _remove(tmp_path, current)

    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.endswith("removed 0 of 2 (0 MB)\n")
    assert (tools / previous).is_dir()


def test_a_node_whose_builds_have_not_started_a_generation_says_so(
    tmp_path: pathlib.Path,
) -> None:
    script = tmp_path / "removal.sh"
    script.write_text(
        cache_generations.removal_script(tmp_path.as_posix(), isolated=False), "utf-8"
    )

    ran = subprocess.run(
        [host_bash(), script.as_posix()], capture_output=True, text=True, check=False, timeout=60
    )

    assert (ran.returncode, ran.stdout) == (
        0,
        f"cache-generations: {tmp_path.as_posix()}/cache/tools: none yet\n",
    )


def test_an_entry_that_is_no_generation_stops_the_removal_naming_it(
    tmp_path: pathlib.Path,
) -> None:
    tools = tmp_path / names.CACHE_DIRECTORY / names.TOOL_CACHES_DIRECTORY
    current = _generation(tools, seconds_ago=2 * 3600)
    (tools / "npm").mkdir()

    ran = _remove(tmp_path, current)

    assert ran.returncode == 1
    assert ran.stderr == f"cache-generations: {tools.as_posix()}/npm is no generation\n"
