"""A package that pulls CUDA wheels syncs before its check's clock (MCPs tasks 74d10d85, abb0a10e).

On 2026-11-01 four checks overran their 300 s budget with every test passing:
each was its package's first check on its node after the fleet's venvs moved
from Python 3.12 to 3.11, and each fetched its compiled wheels again inside
the clock. lavender's poetry cache took in 4168143093 bytes during
services/grandma-api's 335 s check, and libs/covenant_ml's on serendipity,
in a generation begun that evening, took in 676.5 MB and compiled
cleargbm_rs. pendragon's took in 3175258739 bytes over Wi-Fi while
services/Art-Trainer's check ran beside libs/platform_langid's, the cp311
torch, triton and cudnn wheels of both. Every rerun from the warmed cache
passed within budget, and a generation turns over every 14 days, so the
same fetch would recur.

A project's install steps run at the export root before ``make check``,
with the generation's ``POETRY_CACHE_DIR`` and in-project virtualenvs
already exported (``fleet.core.dialect.build_script``), and each is timed
as its own phase. So every package whose lock pulls a CUDA wheel (an
``nvidia-*`` library, ``torch`` or ``triton``, hundreds of MB each) runs its
recipe's own sync there first, in the phase ``deps``, the way grandma-api's
``npm ci`` already runs before the clock. The recipe's sync then finds the
venv warm, and the budget keeps measuring lint and the suite. The packages
are found from their locks, so one registered later is held to the same
rule without being named here.

NO ``poetry lock`` RUNS BEFORE THE CLOCK (MCPs board task 972ea464). A
recipe verifies its committed lock with ``run.py lock-check`` and never
re-resolves it (tools/maketools' ``makefile_poetry_lock_rule``); a deps-phase
resolution would rewrite a stale lock before that check could read it, so a
fleet check would pass a commit whose lock CI refuses.
"""

from __future__ import annotations

import pathlib
import re
from typing import Final

import pytest

from fleet.contracts.source import InstallStep
from fleet.contracts.workspace import decode_fleet_workspace
from tests._registry import shipped_document

API: Final[str] = "https://github.com/wagner-austin/API.git"

#: This repository, four hops up from ``tools/fleet/tests/test_*.py``.
REPO_ROOT: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parents[3]

#: A lock entry for a CUDA wheel.
CUDA_WHEEL: Final[re.Pattern[str]] = re.compile(
    r'^name = "(?:nvidia-[a-z0-9-]+|torch|triton)"$', re.MULTILINE
)

#: How a recipe line that runs poetry opens.
RECIPE_POETRY: Final[str] = "\tpoetry "

#: The poetry command that syncs the package's venv from its committed lock.
RECIPE_VERBS: Final[frozenset[str]] = frozenset({"sync"})

#: The packages measured overrunning, or found by hand on 2026-11-01, which the scan must find.
MEASURED: Final[frozenset[str]] = frozenset(
    {
        "clients/OrderedKernels",
        "libs/covenant_ml",
        "libs/covenant_nn",
        "libs/platform_langid",
        "libs/platform_ml",
        "services/Model-Trainer",
        "services/covenant-radar-api",
        "services/grandma-api",
        "tools/code-style-eval",
    }
)


def _api_installs() -> dict[str, tuple[InstallStep, ...]]:
    """The install steps of every registered project in this repository.

    Returns:
        Each project's steps, keyed by its path.
    """
    installs: dict[str, tuple[InstallStep, ...]] = {}
    for project in decode_fleet_workspace(shipped_document())["projects"].values():
        source = project["source"]
        if source is not None and source["remote"] == API:
            installs[source["path"]] = source["install"]
    return installs


#: Every registered API project's install steps, keyed by path.
API_INSTALLS: Final[dict[str, tuple[InstallStep, ...]]] = _api_installs()


def _pulls_cuda(path: str) -> bool:
    """Whether a package's poetry lock pulls a CUDA wheel.

    Args:
        path: The package's directory in the repository.

    Returns:
        True when its lock exists and names one.
    """
    lock = REPO_ROOT / path / "poetry.lock"
    return lock.is_file() and CUDA_WHEEL.search(lock.read_text(encoding="utf-8")) is not None


#: The registered API packages whose poetry lock pulls a CUDA wheel, sorted.
CUDA_PACKAGES: Final[tuple[str, ...]] = tuple(sorted(filter(_pulls_cuda, API_INSTALLS)))


def _recipe_deps(path: str) -> tuple[InstallStep, ...]:
    """The package recipe's own sync, run at the export root.

    Args:
        path: The package's directory in the repository.

    Returns:
        One step per distinct ``poetry sync`` line of its Makefile, in
        recipe order.
    """
    makefile = (REPO_ROOT / path / "Makefile").read_text(encoding="utf-8")
    lines: list[str] = []
    for line in makefile.splitlines():
        command = line.removeprefix(RECIPE_POETRY)
        if (
            line.startswith(RECIPE_POETRY)
            and command.split(" ")[0] in RECIPE_VERBS
            and command not in lines
        ):
            lines.append(command)
    return tuple(
        InstallStep(phase="deps", argv=("poetry", "-C", path, *line.split())) for line in lines
    )


def test_the_scan_finds_every_measured_package() -> None:
    assert frozenset(CUDA_PACKAGES) >= MEASURED, sorted(MEASURED - frozenset(CUDA_PACKAGES))


@pytest.mark.parametrize("path", CUDA_PACKAGES)
def test_the_package_syncs_before_its_check(path: str) -> None:
    expected = _recipe_deps(path)
    assert expected[-1]["argv"][3] == "sync"
    assert API_INSTALLS[path][: len(expected)] == expected


def test_no_registered_project_resolves_its_lock_before_the_clock() -> None:
    resolving = sorted(
        path
        for path, steps in API_INSTALLS.items()
        for step in steps
        if step["argv"][:1] == ("poetry",) and "lock" in step["argv"]
    )
    assert resolving == []


def test_grandma_api_still_installs_its_web_dependencies_after_them() -> None:
    assert API_INSTALLS["services/grandma-api"][-1] == InstallStep(
        phase="install", argv=("npm", "ci", "--prefix", "services/grandma-api/web")
    )
