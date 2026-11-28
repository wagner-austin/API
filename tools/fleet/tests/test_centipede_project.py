"""centipede's check runs MCPs' shared mutation step from the companion beside it (MCPs board task 2359f0ac).

centipede (wagner-austin/centipede) is one game at its root, and its make
check ends in the StrykerJS mutation step MCPs keeps in
mcp-shared/src/mutation (MCPs board task ebf8750b), run from ``../MCPs`` as
``mutation lease-bounded --bound-seconds 900 --fleet-project centipede``
with no copy in centipede. So the row carries MCPs as a companion and installs
MCPs' npm workspace with MCPs' own fleet-prepare ``install`` step, the one the
personal wiki's row runs, before centipede's ``npm ci``: tsx, StrykerJS
10.1.1 and vitest are devDependencies of MCPs' mcp-shared workspace, hoisted
to MCPs' root node_modules, and the step runs from MCPs' sources without a
build. The step mutates the src lines a commit adds against origin/main, so
the row declares ``base_branch`` main and the fleet stages the commit's
history with ``refs/remotes/origin/main``; staged as one root commit, the
merge base would not resolve.

The checker's memory was measured by the step itself on ziggurat on
2026-11-28, over a centipede commit changing src/leg-pose.ts (4 mutants,
tests/browser/leg-pose.test.ts), its cap set through a scratch registry row:
300 MB and 400 MB ended in StrykerJS's OutOfMemoryError, and 500 MB finished
with 0 of 4 mutants alive, the checker peaking at 744220 KB RSS (0.71 GB).
By stream-client's convention the row declares 1.25 times the least cap that
finished, 625 MB, and that plus 0.25 GB, 0.86 GB rounded up to 0.87.
"""

from __future__ import annotations

from fleet.contracts.mutation_checker import MutationChecker
from fleet.contracts.source import InstallStep, ProjectCompanion, ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document

#: The measurement the row cites, word for word.
MEASURED = (
    "ziggurat 2026-11-28, StrykerJS 10.1.1: one TypeScript checker finished src/leg-pose.ts "
    "(4 mutants) at a 500 MB heap cap, the least in 100 MB steps, peaking at 0.71 GB RSS; "
    "heap_mb is 1.25 times that cap and ram_gb heap_mb plus 0.25 GB "
    "(MCPs wiki typescript-mutation-step)"
)


def test_the_check_installs_the_mcps_companion_and_stages_main_s_history() -> None:
    project = require_project(decode_fleet_workspace(shipped_document()), "centipede")

    assert project["required_tags"] == (NodeTag.GPU,)
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/centipede.git",
        path="",
        install=(
            InstallStep(
                phase="mcps-install",
                argv=("make", "-s", "--directory=../MCPs/scripts/fleet-prepare", "install"),
            ),
            InstallStep(phase="install", argv=("npm", "ci")),
            InstallStep(phase="install", argv=("npx", "playwright", "install", "chromium")),
        ),
        companions=(
            ProjectCompanion(
                remote="https://github.com/wagner-austin/MCPs.git", ref="main", directory="MCPs"
            ),
        ),
        base_branch="main",
    )


def test_the_mutation_checker_is_the_measured_cap_with_its_margin() -> None:
    project = require_project(decode_fleet_workspace(shipped_document()), "centipede")

    assert project["mutation_checker"] == MutationChecker(
        heap_mb=625, ram_gb=0.87, measurement=MEASURED
    )
