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
recipe's own lock and sync there first, in the phase ``deps``, the way
grandma-api's ``npm ci`` already runs before the clock. The recipe's lock
and sync then find the venv warm, and the budget keeps measuring lint and
the suite. The packages are found from their locks, so one registered later
is held to the same rule without being named here.
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

#: A recipe line that locks or syncs the package's venv.
RECIPE_POETRY: Final[re.Pattern[str]] = re.compile(r"^\tpoetry (lock|sync [^\n]+)$", re.MULTILINE)

#: The packages measured overrunning, or found by hand on 2026-11-01, which the scan must find.
MEASURED: Final[frozenset[str]] = frozenset(
    {
        "clients/OrderedKernels",
        "libs/covenant_ml",
        "libs/covenant_nn",
        "libs/platform_langid",
        "libs/platform_ml",
        "services/Art-Trainer",
        "services/Model-Trainer",
        "services/covenant-radar-api",
        "services/grandma-api",
        "services/handwriting-ai",
        "tools/code-style-eval",
    }
)


def _cuda_packages() -> tuple[str, ...]:
    """The registered API packages whose poetry lock pulls a CUDA wheel.

    Returns:
        Their paths, sorted.
    """
    workspace = decode_fleet_workspace(shipped_document())
    found: list[str] = []
    for project in workspace["projects"].values():
        source = project["source"]
        if source is None or source["remote"] != API:
            continue
        lock = REPO_ROOT / source["path"] / "poetry.lock"
        if lock.is_file() and CUDA_WHEEL.search(lock.read_text(encoding="utf-8")):
            found.append(source["path"])
    return tuple(sorted(found))


CUDA_PACKAGES: Final[tuple[str, ...]] = _cuda_packages()


def _recipe_deps(path: str) -> tuple[InstallStep, ...]:
    """The package recipe's own lock and sync, run at the export root.

    Args:
        path: The package's directory in the repository.

    Returns:
        One step per distinct ``poetry lock`` or ``poetry sync`` line of its
        Makefile, in recipe order.
    """
    makefile = (REPO_ROOT / path / "Makefile").read_text(encoding="utf-8")
    lines = dict.fromkeys(match.group(1) for match in RECIPE_POETRY.finditer(makefile))
    return tuple(
        InstallStep(phase="deps", argv=("poetry", "-C", path, *line.split())) for line in lines
    )


def test_the_scan_finds_every_measured_package() -> None:
    assert MEASURED <= frozenset(CUDA_PACKAGES), sorted(MEASURED - frozenset(CUDA_PACKAGES))


@pytest.mark.parametrize("path", CUDA_PACKAGES)
def test_the_package_locks_and_syncs_before_its_check(path: str) -> None:
    project = decode_fleet_workspace(shipped_document())["projects"][path]
    source = project["source"]
    assert source is not None
    expected = _recipe_deps(path)
    assert expected
    assert source["install"][: len(expected)] == expected


def test_grandma_api_still_installs_its_web_dependencies_after_them() -> None:
    source = decode_fleet_workspace(shipped_document())["projects"]["services/grandma-api"][
        "source"
    ]
    assert source is not None
    assert source["install"][-1] == InstallStep(
        phase="install", argv=("npm", "ci", "--prefix", "services/grandma-api/web")
    )
