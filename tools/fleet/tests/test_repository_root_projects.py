"""Each room's repository has its whole make check on the fleet (MCPs board task a8ee9b21).

The review pass gathers a PASSED check for every repository a closure
touches, and a repository whose check can run only on a workstation shows as
FAILED there. hardware-wiki's, tech-wiki's (MCPs board task 4558ee91),
creator-wiki's (5a83f413), metabolomics-dashboard's, chat's and
corvis-stick's checks run MCPs'
published maketools, which reads ``../MCPs``
at ``origin/main``, so each
carries MCPs as a companion at ``main``: the fleet clones it beside the export
with ``origin/main`` at its real commit (fleet.core.stage_repository,
companion_repository_commands). idle's check reads nothing beside it, and its
browser project draws with WebGL in Chromium, so it takes the gpu tag; so
does centipede's (MCPs board task 2a06915d), whose lint checks its lifted
files against lift-lock.json offline and whose browser project draws the game
with WebGL in Chromium.
tree-bot's and Dashboards/rabbit's checks read nothing beside them and
install through their own make check (MCPs board task ddd25310), and like
the Dashboards root's and metabolomics-dashboard's they run under sh on any
node, so none of them takes a tag. LSTM's check is a poetry install inside make check with no
repository beside it, since platform_core arrives as a git dependency
(board task bc12f18b). It takes the gpu tag, because its resume test
restores CUDA RNG states on the card, the only place the bug it guards
reproduces; on serendipity, with no card, that left train.py at 99.67% and
the check under its 100% floor (fleet job 1b48e90e). The Dashboards root's
check ends in a provenance gate that runs the deliverable-write validator
against the wikis its articles cite, all found beside the repository as a
workstation keeps them in ``~/PROJECTS``, so the root carries those five
repositories as companions under the directory names they have there. The
personal wiki's check (MCPs board task 96def18f) is TypeScript tooling whose
package.json links corvis-db, corvis-wiki-check and corvis-wiki-cli by
``file:../MCPs/packages/...``, so it carries MCPs as a companion, installs
MCPs' workspace and builds those three packages there with MCPs' own
fleet-prepare before its own ``npm ci``, and takes the pdftotext and cxx tags
MCPs' wiki-check project takes. One of its cases drives the production vault
read against a live Postgres with the vault's schema, so it also takes the
testdb tag and the node's ``corvis-fleet-testdb`` container, and MCPs'
``test-database`` step migrates corvis_test and writes the ``../MCPs/.env``
whose DATABASE_TEST_URL the wiki's ``make test`` lifts (fleet job 601bba96
failed that case with nothing to connect to).
"""

from __future__ import annotations

import pytest

from fleet.contracts.source import InstallStep, ProjectCompanion, ProjectSource
from fleet.contracts.tags import NodeTag
from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document

#: The install step of every npm repository's root check.
NPM_CI = InstallStep(phase="install", argv=("npm", "ci"))

#: MCPs, staged beside a project whose check runs MCPs' published maketools.
MCPS_COMPANION = ProjectCompanion(
    remote="https://github.com/wagner-austin/MCPs.git", ref="main", directory="MCPs"
)


@pytest.mark.parametrize(
    ("name", "tags", "install", "companions"),
    [
        (
            "idle",
            (NodeTag.GPU,),
            (
                NPM_CI,
                InstallStep(phase="install", argv=("npx", "playwright", "install", "chromium")),
            ),
            (),
        ),
        (
            "centipede",
            (NodeTag.GPU,),
            (
                NPM_CI,
                InstallStep(phase="install", argv=("npx", "playwright", "install", "chromium")),
            ),
            (),
        ),
        ("chat", (), (NPM_CI,), (MCPS_COMPANION,)),
        ("hardware-wiki", (), (), (MCPS_COMPANION,)),
        ("tech-wiki", (), (), (MCPS_COMPANION,)),
        ("creator-wiki", (), (), (MCPS_COMPANION,)),
        (
            "wiki",
            (NodeTag.TESTDB, NodeTag.CXX, NodeTag.PDFTOTEXT),
            (
                InstallStep(
                    phase="mcps-install",
                    argv=("make", "-s", "--directory=../MCPs/scripts/fleet-prepare", "install"),
                ),
                InstallStep(
                    phase="mcps-build",
                    argv=(
                        "make",
                        "-s",
                        "--directory=../MCPs/scripts/fleet-prepare",
                        "workspace-build",
                        "ROOTS=packages/db:packages/wiki-check:packages/wiki-cli",
                    ),
                ),
                InstallStep(
                    phase="test-database",
                    argv=(
                        "make",
                        "-s",
                        "--directory=../MCPs/scripts/fleet-prepare",
                        "test-database",
                        "CONTAINER=corvis-fleet-testdb",
                    ),
                ),
                NPM_CI,
            ),
            (MCPS_COMPANION,),
        ),
        ("metabolomics-dashboard", (), (), (MCPS_COMPANION,)),
        ("corvis-stick", (NodeTag.LINUX,), (NPM_CI,), (MCPS_COMPANION,)),
        ("tree-bot", (), (), ()),
        ("LSTM", (NodeTag.GPU,), (), ()),
    ],
)
def test_each_repository_s_root_check_is_declared_with_what_it_reads(
    name: str,
    tags: tuple[NodeTag, ...],
    install: tuple[InstallStep, ...],
    companions: tuple[ProjectCompanion, ...],
) -> None:
    workspace = decode_fleet_workspace(shipped_document())
    project = require_project(workspace, name)
    assert project["required_tags"] == tags
    assert project["source"] == ProjectSource(
        remote=f"https://github.com/wagner-austin/{name}.git",
        path="",
        install=install,
        companions=companions,
    )


def test_the_wiki_holds_the_node_s_test_database_for_its_whole_run() -> None:
    workspace = decode_fleet_workspace(shipped_document())
    project = require_project(workspace, "wiki")
    assert project["exclusive_resources"] == ("corvis-fleet-testdb",)


def test_rabbit_is_checked_from_its_directory_in_dashboards() -> None:
    workspace = decode_fleet_workspace(shipped_document())
    project = require_project(workspace, "Dashboards/rabbit")
    assert project["required_tags"] == ()
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/Dashboards.git",
        path="rabbit",
        install=(),
        companions=(),
    )


def test_extraction_eval_is_checked_from_its_directory_in_the_wiki() -> None:
    """The wiki's one Python package, a poetry project whose check reads
    nothing beside it; its recipes run under either platform's shell since
    MCPs board task 22cb1ba7, whose review found it on no node."""
    workspace = decode_fleet_workspace(shipped_document())
    project = require_project(workspace, "wiki/tools/extraction-eval")
    assert project["required_tags"] == ()
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/wiki.git",
        path="tools/extraction-eval",
        install=(),
        companions=(),
    )


def test_the_dashboards_root_carries_the_validator_and_the_wikis_it_reads() -> None:
    workspace = decode_fleet_workspace(shipped_document())
    project = require_project(workspace, "Dashboards")
    assert project["required_tags"] == ()
    assert project["source"] == ProjectSource(
        remote="https://github.com/wagner-austin/Dashboards.git",
        path="",
        install=(),
        companions=(
            ProjectCompanion(
                remote="https://github.com/wagner-austin/deliverable-write.git",
                ref="master",
                directory="deliverable-write",
            ),
            ProjectCompanion(
                remote="https://github.com/wagner-austin/wiki.git", ref="main", directory="wiki"
            ),
            ProjectCompanion(
                remote="https://github.com/wagner-austin/me-wiki.git",
                ref="main",
                directory="me-wiki",
            ),
            MCPS_COMPANION,
            ProjectCompanion(
                remote="https://github.com/wagner-austin/API.git", ref="main", directory="API"
            ),
        ),
    )
