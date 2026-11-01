"""A heavy poetry package syncs before the check's clock (MCPs board task 74d10d85).

On 2026-11-01 four checks overran their 300 s budget with every test passing:
each was its package's first check on its node after the fleet's venvs moved
from Python 3.12 to 3.11, and each fetched its compiled wheels again inside
the clock. lavender's poetry cache took in 4168143093 bytes during
services/grandma-api's 335 s check, and libs/covenant_ml's on serendipity,
in a generation begun that evening, took in 676.5 MB and compiled
cleargbm_rs. Every rerun from the warmed cache passed within budget, and a
generation turns over every 14 days, so the same fetch would recur.

A project's install steps run at the export root before ``make check``,
with the generation's ``POETRY_CACHE_DIR`` and in-project virtualenvs
already exported (``fleet.core.dialect.build_script``), and each is timed
as its own phase. So these packages' lock and ``--no-cache`` sync run there,
in the phase ``deps``, the way grandma-api's ``npm ci`` already runs before
the clock. The recipe's own lock and sync then find the venv warm, and the
budget keeps measuring lint and the suite.
"""

from __future__ import annotations

import pytest

from fleet.contracts.source import InstallStep, ProjectSource
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document


def _deps(path: str) -> tuple[InstallStep, ...]:
    """The lock and sync of one poetry package, run at the export root.

    Args:
        path: The package's directory in the repository.

    Returns:
        The two steps, in order.
    """
    return (
        InstallStep(phase="deps", argv=("poetry", "-C", path, "lock")),
        InstallStep(
            phase="deps", argv=("poetry", "-C", path, "sync", "--no-cache", "--with", "dev")
        ),
    )


#: grandma-api's web dependencies, installed after its poetry ones.
NPM_CI_WEB = InstallStep(
    phase="install", argv=("npm", "ci", "--prefix", "services/grandma-api/web")
)


@pytest.mark.parametrize(
    ("path", "after"),
    [("libs/covenant_ml", ()), ("services/grandma-api", (NPM_CI_WEB,))],
)
def test_the_package_locks_and_syncs_before_its_check(
    path: str, after: tuple[InstallStep, ...]
) -> None:
    project = require_project(decode_fleet_workspace(shipped_document()), path)
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/API.git",
        path=path,
        install=(*_deps(path), *after),
        companions=(),
    )
