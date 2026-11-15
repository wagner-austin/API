"""The CPU bound a docker project's containers stated, gathered into the verdict.

MCPs board task 2e2df92a. The gathering lines run here under ``sh -eu``, as
the docker lane's build runs them, over a transcript holding what the
containers printed; the values are the ones lavender's lane read on
2026-11-15: ``400000 100000`` inside an osrs-lab check image run with
``--cpus 4``, and ``max 100000`` inside one run without it.
"""

from __future__ import annotations

import pathlib
import subprocess

from fleet.core.container_cpu import (
    CONTAINER_CPU_MARKER,
    CONTAINER_CPUS_MARKER,
    container_cpus_line,
    container_cpus_none_line,
    read_container_cpus,
    sh_container_cpu_lines,
)


def _gather(tmp_path: pathlib.Path, transcript: str) -> str:
    """Run the gathering lines over a transcript and return what it ends with.

    Args:
        tmp_path: The case's directory.
        transcript: The transcript's text before the lines run.

    Returns:
        The transcript after them.
    """
    log = tmp_path / "check.log"
    log.write_text(transcript, encoding="utf-8")
    subprocess.run(
        ["sh", "-eu", "-c", "\n".join(sh_container_cpu_lines(str(log)))], check=True, timeout=30
    )
    return log.read_text(encoding="utf-8")


def test_each_distinct_bound_the_containers_stated_is_gathered_once(
    tmp_path: pathlib.Path,
) -> None:
    printed = _gather(
        tmp_path,
        f"{CONTAINER_CPU_MARKER}: 400000 100000\n"
        "> Task :observer:test\n"
        f"{CONTAINER_CPU_MARKER}: max 100000\n"
        f"{CONTAINER_CPU_MARKER}: 400000 100000\n",
    )

    assert printed.endswith(f"\n{container_cpus_line('400000 100000,max 100000')}\n")
    assert read_container_cpus(printed) == "400000/100000,max/100000"


def test_a_recipe_whose_containers_stated_nothing_says_so(tmp_path: pathlib.Path) -> None:
    printed = _gather(tmp_path, "BUILD SUCCESSFUL in 41s\n")

    assert printed.endswith(f"\n{container_cpus_none_line()}\n")
    assert read_container_cpus(printed) == "none"


def test_a_tail_without_the_line_is_unread_and_the_last_line_counts() -> None:
    assert read_container_cpus("BUILD SUCCESSFUL in 41s\n") == "unread"
    two = f"{container_cpus_line('max 100000')}\n{container_cpus_line('200000 100000')}\n"
    assert read_container_cpus(two) == "200000/100000"


def test_the_markers_are_the_fleet_s_names() -> None:
    assert (CONTAINER_CPU_MARKER, CONTAINER_CPUS_MARKER) == (
        "FLEET_CONTAINER_CPU",
        "FLEET_CONTAINER_CPUS",
    )
    assert container_cpus_none_line() == "FLEET_CONTAINER_CPUS: none stated"
