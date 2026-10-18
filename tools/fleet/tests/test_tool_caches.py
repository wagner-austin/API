"""The package managers' cache generations a build reads (MCPs board task 346abc45).

The ``sh`` lines are executed for real under this host's bash against a
cache root under ``tmp_path``: the first build on a node starts a
generation, a later one reads it while it is young, and one that finds it
:data:`~fleet.core.tool_caches.GENERATION_DAYS` days old starts the next.
The PowerShell lines run inside the build script, executed from its
committed render by ``tests/pester/rendered-dialect-build.Tests.ps1``.
"""

from __future__ import annotations

import pathlib
import subprocess
import time
from typing import Final

from fleet.core import names, tool_caches
from tests._host_bash import host_bash

#: One day in seconds.
DAY: Final = 86400


def _read(tools: pathlib.Path, marker: pathlib.Path) -> str:
    """Run the build's lines once, as a build starting now does.

    Args:
        tools: The generations directory.
        marker: The run directory's marker file.

    Returns:
        The ``$tool_cache`` the lines left.
    """
    lines = tool_caches.sh_build_lines(
        tools=f"'{tools.as_posix()}'", marker=f"'{marker.as_posix()}'", run_as=""
    )
    tools.parent.mkdir(parents=True, exist_ok=True)
    script = tools.parent / "build.sh"
    script.write_text(
        "set -eu\n" + "".join(f"{line}\n" for line in lines) + "printf '%s\\n' \"$tool_cache\"\n",
        encoding="utf-8",
    )
    ran = subprocess.run(
        [host_bash(), script.as_posix()], capture_output=True, text=True, check=False, timeout=60
    )
    assert ran.returncode == 0, ran.stderr
    return ran.stdout.strip()


def _generation(tools: pathlib.Path, *, days_ago: int) -> str:
    """Lay out a generation started some days ago, and make it current.

    Args:
        tools: The generations directory.
        days_ago: How long ago it started.

    Returns:
        Its name.
    """
    name = str(int(time.time()) - days_ago * DAY)
    (tools / name).mkdir(parents=True)
    (tools / names.CURRENT_GENERATION_NAME).write_text(f"{name}\n", encoding="utf-8")
    return name


def test_the_generations_live_under_the_cache_root_and_last_two_weeks() -> None:
    assert tool_caches.tool_caches_directory("/s/cache") == "/s/cache/tools"
    assert tool_caches.GENERATION_SECONDS == 14 * DAY


def test_every_file_operation_takes_the_prefix_execdocker_s_caches_need() -> None:
    lines = tool_caches.sh_build_lines(tools="T", marker="M", run_as="sudo -n ")

    for verb in ("mkdir -p", "test -f", "cat ", "tee ", "mv -f"):
        uses = [line for line in lines if verb in line]
        assert uses, verb
        assert all(f"sudo -n {verb}" in line for line in uses), (verb, uses)


def test_the_first_build_on_a_node_starts_a_generation_and_marks_its_run(
    tmp_path: pathlib.Path,
) -> None:
    tools = tmp_path / "cache" / "tools"
    marker = tmp_path / "run" / names.GENERATION_MARKER_NAME
    marker.parent.mkdir()
    before = int(time.time())

    cache = _read(tools, marker)

    generation = (tools / names.CURRENT_GENERATION_NAME).read_text(encoding="utf-8").strip()
    assert before <= int(generation) <= int(time.time())
    assert cache == f"{tools.as_posix()}/{generation}"
    assert marker.read_text(encoding="utf-8") == f"{generation}\n"
    assert sorted(entry.name for entry in tools.iterdir()) == sorted(
        [generation, names.CURRENT_GENERATION_NAME]
    )


def test_a_young_generation_is_read_as_it_is(tmp_path: pathlib.Path) -> None:
    tools = tmp_path / "cache" / "tools"
    young = _generation(tools, days_ago=13)
    marker = tmp_path / names.GENERATION_MARKER_NAME

    assert _read(tools, marker) == f"{tools.as_posix()}/{young}"
    assert marker.read_text(encoding="utf-8") == f"{young}\n"
    assert sorted(entry.name for entry in tools.iterdir()) == [young, "current"]


def test_a_generation_two_weeks_old_is_followed_by_a_new_one_and_kept(
    tmp_path: pathlib.Path,
) -> None:
    tools = tmp_path / "cache" / "tools"
    old = _generation(tools, days_ago=14)
    marker = tmp_path / names.GENERATION_MARKER_NAME

    cache = _read(tools, marker)

    generation = (tools / names.CURRENT_GENERATION_NAME).read_text(encoding="utf-8").strip()
    assert int(generation) > int(old)
    assert cache == f"{tools.as_posix()}/{generation}"
    assert marker.read_text(encoding="utf-8") == f"{generation}\n"
    # The old one stays for the builds that read it; the removal judges it.
    assert (tools / old).is_dir()
    assert not (tools / generation / ".pending").exists()


def test_the_powershell_lines_read_the_build_s_own_parameters() -> None:
    lines = tool_caches.powershell_build_lines()

    assert lines[0] == '$tools = "$CacheRoot/tools"'
    assert f"-ge {tool_caches.GENERATION_SECONDS}) {{" in lines[4]
    assert lines[-2] == ('[System.IO.File]::WriteAllText("$Target/cache-generation", $generation)')
    assert lines[-1] == '$toolCache = "$tools/$generation"'
