"""The generated half of the research index, and that it agrees with the registry."""

from __future__ import annotations

import pathlib

import pytest

from hpc3.cli.research_index import (
    declared_projects,
    index_path,
    main,
    runs_directory,
)
from hpc3.contracts.cluster import GpuRequest
from hpc3.contracts.workspace import ProjectConfig
from hpc3.core.research_index import (
    BLOCK_END,
    BLOCK_START,
    extract_projects_block,
    render_project_row,
    render_projects_block,
    replace_projects_block,
)


def _project(
    *,
    gpu: GpuRequest | None = None,
    image_sha: str | None = None,
    cpus: int = 4,
    minutes: int = 60,
) -> ProjectConfig:
    """Build a project configuration.

    Args:
        gpu: GPU request, or None for CPU-only work.
        image_sha: Image digest, or None when the project declares no image.
        cpus: Cores per job.
        minutes: Wall clock per job.

    Returns:
        The configuration.
    """
    return ProjectConfig(
        partition="free",
        gpu=gpu,
        cpus=cpus,
        mem_gb=16,
        minutes=minutes,
        requeue=True,
        checkpoint_steps=0,
        image=None
        if image_sha is None
        else {"path": "/pub/x.sif", "sha256": image_sha, "binds": ["/pub"]},
        env_path="/opt/env",
        pinned_packages={},
        deterministic=True,
        budget={
            "self_imposed_gpu_hours": 0.0,
            "max_service_units": 0.0,
            "charge_account": "",
        },
        repo="../../..",
    )


class TestRenderingARow:
    """Each cell states a declared fact, or says it is absent."""

    def test_a_project_with_no_image_says_none(self) -> None:
        """A blank cell reads as a formatting fault, not as an absence."""
        row = render_project_row("cleargbm", _project())

        assert "| none |" in row

    def test_an_image_is_shown_by_its_digest(self) -> None:
        """The digest is the thing that differs between two images."""
        row = render_project_row("mi", _project(image_sha="a" * 64))

        assert "`aaaaaaaaaaaa`" in row

    def test_a_cpu_project_says_cpu(self) -> None:
        """Rendered as a word for the same reason as the image cell."""
        assert "| cpu |" in render_project_row("cleargbm", _project())

    def test_a_gpu_is_rendered_field_by_field(self) -> None:
        """Formatting the mapping would put Python dict syntax in a document.

        The first version of this renderer did exactly that and produced
        ``{'model': 'A100', 'count': 1}`` in the committed index.
        """
        row = render_project_row("mi", _project(gpu=GpuRequest(model="A100", count=1)))

        assert "`A100` x1" in row
        assert "{" not in row


class TestRenderingTheBlock:
    """The block is what replaces hand-maintained numbers."""

    def test_projects_are_sorted_so_two_renderings_match(self) -> None:
        """An unstable order would make every regeneration a diff."""
        projects = {"zeta": _project(), "alpha": _project()}

        block = render_projects_block(projects)

        assert block.index("`alpha`") < block.index("`zeta`")

    def test_the_block_carries_its_own_markers(self) -> None:
        """Without them nothing can find the block to replace it."""
        block = render_projects_block({"alpha": _project()})

        assert block.startswith(BLOCK_START)
        assert block.endswith(BLOCK_END)

    def test_the_block_names_how_to_regenerate_it(self) -> None:
        """A reader who finds it stale should not have to guess."""
        assert "hpc3-research-index --write" in render_projects_block({"a": _project()})


class TestSubstitutingTheBlock:
    """Replacement is surgical, and refuses rather than guessing."""

    def test_the_surrounding_prose_is_untouched(self) -> None:
        """Everything outside the markers is a human's to write."""
        document = f"before\n{BLOCK_START}\nold\n{BLOCK_END}\nafter\n"

        result = replace_projects_block(document, render_projects_block({"a": _project()}))

        assert result.startswith("before\n")
        assert result.endswith("\nafter\n")
        assert "old" not in result

    def test_a_document_without_markers_is_refused(self) -> None:
        """Appending would leave the stale table above the fresh one."""
        with pytest.raises(ValueError, match="carries no generated block"):
            _ = replace_projects_block("no markers here", "block")

    def test_markers_out_of_order_are_refused(self) -> None:
        """A closing marker before its opener describes no region."""
        with pytest.raises(ValueError, match="out of order"):
            _ = replace_projects_block(f"{BLOCK_END}\n{BLOCK_START}", "block")

    def test_extracting_a_document_without_markers_is_refused(self) -> None:
        """The reading half refuses on the same terms as the writing half."""
        with pytest.raises(ValueError, match="carries no generated block"):
            _ = extract_projects_block("no markers here")


class TestTheCommittedIndexAgreesWithTheRegistry:
    """The check this whole module exists for.

    ``test_committed_runs.py`` asserts that every registered project APPEARS
    in the index. It never asserted that what the index SAYS about one is
    true, and three entries were wrong at once because of it: rusted's cores
    and wall clock, cleargbm's record shape, code-style's fingerprint. This
    closes the half that is derivable.
    """

    def test_the_generated_block_matches_what_the_registry_declares(self) -> None:
        """Fails the build when the registry moves and the document does not."""
        expected = render_projects_block(declared_projects(runs_directory()))

        assert extract_projects_block(index_path().read_text(encoding="utf-8")) == expected

    def test_every_registered_project_has_a_row(self) -> None:
        """A project missing from the table would drift unnoticed."""
        block = extract_projects_block(index_path().read_text(encoding="utf-8"))
        missing = sorted(
            name for name in declared_projects(runs_directory()) if f"`{name}`" not in block
        )

        assert missing == []

    def test_checking_returns_zero_when_the_document_is_current(self) -> None:
        """The command a build step would run."""
        assert main(["--check"]) == 0


class TestTheCommandLine:
    """Checking is the default; writing is asked for."""

    def test_an_unknown_flag_is_refused(self) -> None:
        """A typo must not be read as the checking form and silently pass."""
        with pytest.raises(ValueError, match="unknown argument"):
            _ = main(["--rewrite"])

    def test_a_bare_invocation_is_refused(self) -> None:
        """Every command in this package refuses one, and the shape test checks it.

        Guessing an action for a caller who named none is how a tracked
        document gets rewritten by somebody who meant to check it.
        """
        with pytest.raises(ValueError, match="exactly one"):
            _ = main([])

    def test_naming_both_actions_is_refused(self) -> None:
        """Check and write are different intentions, not a sequence."""
        with pytest.raises(ValueError, match="exactly one"):
            _ = main(["--check", "--write"])

    def test_the_index_path_resolves(self) -> None:
        """A path computed from __file__ is one nobody re-checks by hand."""
        assert index_path().is_file()

    def test_the_runs_directory_resolves(self) -> None:
        """Same reasoning as the index path."""
        directory: pathlib.Path = runs_directory()

        assert directory.is_dir()
