"""A fleet node's package caches, read on the node (MCPs board task 346abc45, A3).

The unit suite executes the removal against laid-out trees
(:mod:`tests.test_cache_generations`); these cases read the node a fleet run
is on, through the cache root its build script names in
:data:`~fleet.core.names.CACHE_VARIABLE`. npm's and poetry's caches and
Playwright's browsers live only in generations under it, so none of them is
left at the cache root, where nothing removes it. A generation that is not
current stays while a live run's marker names it, and the removal after each
retire takes it otherwise, once it has been superseded for
:data:`~fleet.core.tool_caches.REMOVAL_GRACE_SECONDS`; so beyond the ones live
runs name, a node holds at most the one generation the next retire removes.
A second unnamed one means the removal is not running.

``tools/fleet-execution-runners`` reads the runner host's own cache, and
``tools/fleet-execution-linux`` the cache of execdocker, the account the
isolated build runs as there. Both are runs the fleet starts, the only ones
the variable is set for.
"""

from __future__ import annotations

import os
import pathlib
import time
from collections.abc import Callable
from typing import Final

import pytest

from fleet.contracts.runners import HostRunnerSpec
from fleet.core import _test_hooks, names
from tests._host import READING_PROPERTY

#: The caches that lived at the cache root before they moved into the
#: generations, and must never be there again.
RETIRED_CACHES: Final[tuple[str, ...]] = ("npm", "pypoetry", "ms-playwright")


@pytest.mark.host_runner_units
def test_the_runner_host_s_caches_are_generations_the_removal_bounds(
    runner_host: HostRunnerSpec, record_property: Callable[[str, str], None]
) -> None:
    record_property(READING_PROPERTY, f"{runner_host['name']}: {_read_generations()}")


@pytest.mark.host_linux_docker
def test_execdocker_s_caches_are_generations_the_removal_bounds(
    record_property: Callable[[str, str], None],
) -> None:
    record_property(READING_PROPERTY, f"execdocker: {_read_generations()}")


def _read_generations() -> str:
    """Hold this run's cache root to the generations' bound and say what it holds.

    Returns:
        The reading: each generation, its age, the runs naming it and its
        caches' sizes.
    """
    named = _test_hooks.env(names.CACHE_VARIABLE)
    assert named is not None, f"{names.CACHE_VARIABLE} is unset: this run is no fleet build"
    cache_root = pathlib.Path(named)
    for retired in RETIRED_CACHES:
        left = cache_root / retired
        assert not left.exists(), f"{left} is outside every generation"

    tools = cache_root / names.TOOL_CACHES_DIRECTORY
    entries = sorted(entry.name for entry in tools.iterdir())
    current = (tools / names.CURRENT_GENERATION_NAME).read_text(encoding="utf-8").strip()
    generations = [entry for entry in entries if entry != names.CURRENT_GENERATION_NAME]
    assert all(entry.isdigit() for entry in generations), f"{tools} holds {entries}"
    assert current == max(generations, key=int), f"{current} is current of {generations}"

    readers = _readers(_stage_root())
    unnamed = [name for name in generations if name != current and name not in readers]
    assert len(unnamed) <= 1, f"no live run names generations {unnamed}: no removal ran"

    now = int(time.time())
    held = []
    for generation in generations:
        caches = ", ".join(
            f"{cache.name} {_bytes(cache)} bytes"
            for cache in sorted((tools / generation).iterdir())
        )
        role = "current" if generation == current else f"named by {readers.get(generation, 0)} runs"
        days = (now - int(generation)) // 86400
        held.append(f"{generation} ({role}, {days} days old: {caches})")
    return f"{tools} holds " + "; ".join(held)


def _stage_root() -> pathlib.Path:
    """The stage root this run's directory lies in.

    The build writes the generation it read into its run's directory, which
    is the nearest directory above this suite holding that marker, so this
    holds for the node's own builds and for execdocker's copies alike.

    Returns:
        The run directory's parent.
    """
    here = pathlib.Path.cwd().resolve()
    run = next(
        directory
        for directory in (here, *here.parents)
        if (directory / names.GENERATION_MARKER_NAME).is_file()
    )
    return run.parent


def _readers(stage_root: pathlib.Path) -> dict[str, int]:
    """How many run directories name each generation.

    Args:
        stage_root: Where the run directories are.

    Returns:
        Each named generation and its count of runs.
    """
    counts: dict[str, int] = {}
    for marker in stage_root.glob(f"*/{names.GENERATION_MARKER_NAME}"):
        generation = marker.read_text(encoding="utf-8").strip()
        counts[generation] = counts.get(generation, 0) + 1
    return counts


def _bytes(tree: pathlib.Path) -> int:
    """The bytes of every file under a tree, links counted as themselves.

    Args:
        tree: The directory.

    Returns:
        The sum of its files' sizes.
    """
    return sum(
        os.lstat(os.path.join(directory, file)).st_size
        for directory, _, files in os.walk(tree)
        for file in files
    )
