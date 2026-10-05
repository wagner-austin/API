"""``hpc3-register`` end to end, against a copy of the real registry and index.

THE ACCEPTANCE TEST THE TASK SET (board task cf5f54c0): register a new project
and show the files landing green without a red check in between. The happy
path below does that on a copy of the tree: the person writes the section,
the command writes the document and the table, and then every gate the
registry carries -- the filename rule, no project declared twice, every
project with a section, the table equal to what the registry renders -- is
evaluated on what was left, with no edit made in between.

The refusals matter as much: a missing section and a missing flag are
reported before the cluster is touched, and an image that does not hold the
declared pins leaves nothing written.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.cli import _test_hooks as cli_hooks
from hpc3.cli.register import FLAGS, main
from hpc3.core.index_sections import projects_without_section
from hpc3.core.registry import declared_projects, workspace_filename
from hpc3.core.research_index import extract_projects_block, render_projects_block
from tests._registry_tree import (
    NEWCOMER,
    NEWCOMER_DIGEST,
    NEWCOMER_IMAGE,
    copy_tree,
    index_of,
    repo_of,
    runs_of,
    write_section,
)
from tests.conftest import IMAGE_INTERPRETER, FakeRun


def _args(root: pathlib.Path, **overrides: str) -> list[str]:
    """Build a full command line registering the newcomer.

    Args:
        root: The copied tree's root, which holds the newcomer's repo.
        **overrides: Flag values to replace, keyed by flag without dashes,
            hyphens as underscores.

    Returns:
        The argument list.
    """
    values = {
        "project": NEWCOMER,
        "partition": "free",
        "gpu": "none",
        "cpus": "2",
        "mem_gb": "4",
        "minutes": "60",
        "requeue": "yes",
        "resumes": "no",
        "deterministic": "yes",
        "certified_inputs": "no",
        "image": NEWCOMER_IMAGE,
        "env_path": "/opt/env",
        "pins": "newcomer==0.1.0",
        "gpu_hours": "0",
        "billing": "free",
        "repo": str(repo_of(root)),
    }
    values.update(overrides)
    return [
        token for key, value in values.items() for token in ("--" + key.replace("_", "-"), value)
    ]


def _tree(tmp_path: pathlib.Path, *, section: bool) -> pathlib.Path:
    """Copy the tree, point the command at it, and optionally write the section.

    Args:
        tmp_path: Directory to hold the copy.
        section: Whether the person has written the newcomer's section yet.

    Returns:
        The copy's root.
    """
    root = copy_tree(tmp_path)
    if section:
        write_section(root, NEWCOMER)
    cli_hooks.monorepo_root = lambda: root
    return root


def _script_image(fake_run: FakeRun, *, installed: str) -> None:
    """Script the built image: its digest, and what its environment reports.

    Args:
        fake_run: The scripted cluster.
        installed: The ``name==version`` lines the environment answers with.
    """
    fake_run.add("sha256sum", stdout=f"{NEWCOMER_DIGEST}  {NEWCOMER_IMAGE}\n")
    fake_run.add("/opt/env/bin/python", stdout=IMAGE_INTERPRETER + installed)


class TestRegisteringANewProject:
    def test_the_files_land_and_every_gate_holds_on_what_is_left(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        """Args:
        tmp_path: Directory to hold the copy.
        fake_run: The scripted cluster.
        emitted: The command's report lines.
        """
        root = _tree(tmp_path, section=True)
        _script_image(fake_run, installed="newcomer==0.1.0\n")
        before = set(declared_projects(runs_of(root)))

        assert main(_args(root)) == 0

        runs = runs_of(root)
        document = runs / workspace_filename(NEWCOMER)
        registry = declared_projects(runs)
        assert set(registry) == before | {NEWCOMER}
        assert emitted == [
            f"registered {NEWCOMER} in {document}",
            f"  image {NEWCOMER_IMAGE} sha256 {NEWCOMER_DIGEST}",
            "  /opt/env verified inside it, 1 pin(s) held",
            f"  table in {index_of(root)} regenerated from the registry:"
            f" {len(before) + 1} projects",
            f"next: hpc3-preflight --config {document} --run <run document>",
        ]
        declared = registry[NEWCOMER]
        assert declared["image"] == {
            "path": NEWCOMER_IMAGE,
            "sha256": NEWCOMER_DIGEST,
            "binds": ["/pub/wagnera3"],
        }
        assert declared["repo"] == str(runs / "../../../clients/Newcomer")
        assert declared["pinned_packages"] == {"newcomer": "0.1.0"}
        text = index_of(root).read_text(encoding="utf-8")
        assert extract_projects_block(text) == render_projects_block(registry)
        assert projects_without_section(text, registry) == []
        assert document.is_file()


class TestRefusalsBeforeTheCluster:
    def test_a_bare_invocation_names_every_flag(self, tmp_path: pathlib.Path) -> None:
        """Args:
        tmp_path: Directory to hold the copy.
        """
        _ = _tree(tmp_path, section=True)

        with pytest.raises(ValueError, match=f"missing {len(FLAGS)}: "):
            _ = main([])

    def test_a_project_without_a_section_never_reaches_the_cluster(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """Args:
        tmp_path: Directory to hold the copy.
        fake_run: The scripted cluster, which must not be asked anything.
        """
        root = _tree(tmp_path, section=False)
        before = index_of(root).read_bytes()

        with pytest.raises(AppError) as refused:
            _ = main(_args(root))

        assert refused.value.code is Hpc3ErrorCode.REGISTRATION_INCOMPLETE
        assert "has no section headed '### `newcomer`'" in refused.value.message
        assert fake_run.calls == []
        assert index_of(root).read_bytes() == before


class TestRefusalsAtTheImage:
    def test_an_image_that_lacks_a_declared_pin_leaves_nothing_written(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """The declaration is proven inside the image before it is written.

        Args:
            tmp_path: Directory to hold the copy.
            fake_run: The scripted cluster.
        """
        root = _tree(tmp_path, section=True)
        before = index_of(root).read_bytes()
        _script_image(fake_run, installed="newcomer==0.0.9\n")

        with pytest.raises(AppError) as refused:
            _ = main(_args(root))

        assert refused.value.code is Hpc3ErrorCode.ENV_PACKAGE_MISMATCH
        assert not (runs_of(root) / workspace_filename(NEWCOMER)).exists()
        assert index_of(root).read_bytes() == before
