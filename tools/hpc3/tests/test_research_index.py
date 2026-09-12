"""The generated half of the research index, and that it agrees with the registry."""

from __future__ import annotations

import pathlib

import pytest

from hpc3.cli.research_index import (
    CLAIM_GUIDANCE,
    declared_projects,
    index_path,
    main,
    runs_directory,
)
from hpc3.contracts.cluster import GpuRequest
from hpc3.contracts.project import ProjectConfig
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.research_index import (
    BLOCK_END,
    BLOCK_START,
    LEDGER_ROW_UNIT,
    REGENERATE_HINT,
    SCALE_FIELD,
    extract_projects_block,
    image_digest_claims,
    ledger_state_claims,
    render_project_row,
    render_projects_block,
    replace_projects_block,
)


def _project(
    *,
    gpu: GpuRequest | None = None,
    image_sha: str = "b" * 64,
    cpus: int = 4,
    minutes: int = 60,
) -> ProjectConfig:
    """Build a project configuration.

    Args:
        gpu: GPU request, or None for CPU-only work.
        image_sha: Image digest. Not optional, because the configuration it
            builds is not: every project declares an image.
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
        resumes_from_checkpoint=False,
        image={"path": "/pub/x.sif", "sha256": image_sha, "binds": ["/pub"]},
        env_path="/opt/env",
        pinned_packages={},
        deterministic=True,
        certified_inputs=False,
        budget={
            "self_imposed_gpu_hours": 0.0,
            "max_service_units": 0.0,
            "charge_account": "",
        },
        repo="../../..",
    )


class TestRenderingARow:
    """Each cell states a declared fact, or says it is absent.

    There is no "no image" case to render. Every project declares one, so the
    cell that used to say ``none`` is unreachable and the test asserting it is
    gone rather than kept passing against a state the decoder refuses.
    """

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

    def test_extracting_markers_out_of_order_is_refused(self) -> None:
        """Both halves refuse this, and only one was covered.

        Slicing from the opener to a closer that precedes it yields an empty
        string, which compares unequal to the rendered block and would report
        the document as stale -- a true verdict reached for a false reason,
        and the misleading kind of pass.
        """
        with pytest.raises(ValueError, match="out of order"):
            _ = extract_projects_block(f"{BLOCK_END}\n{BLOCK_START}")


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


class TestTheWritingAndStaleBranches:
    """The two outcomes a build step acts on, and the refusals around them.

    Exercised through the package's own file hooks rather than against
    docs/RESEARCH.md, because a test that rewrites a tracked document to prove
    it can rewrite a tracked document is a test nobody can run twice.
    """

    def test_writing_replaces_the_block_and_reports_where(
        self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--write is the half that mutates, so it says what it touched.

        Args:
            tmp_path: Temporary directory.
            capsys: Captured process output.
        """
        written: dict[pathlib.Path, str] = {}
        current = index_path().read_text(encoding="utf-8")
        stale = current.replace(BLOCK_END, "stale row\n" + BLOCK_END, 1)
        core_hooks.read_bytes = lambda path: (
            stale.encode("utf-8") if path == index_path() else path.read_bytes()
        )
        core_hooks.write_text = lambda path, text: written.__setitem__(path, text)

        assert main(["--write"]) == 0

        assert written[index_path()] == current
        assert capsys.readouterr().out == f"wrote the project table into {index_path()}\n"

    def test_a_stale_document_reports_and_returns_one(
        self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Exit 1 is what makes this usable as a build step.

        Args:
            tmp_path: Temporary directory.
            capsys: Captured process output.
        """
        stale = (
            index_path()
            .read_text(encoding="utf-8")
            .replace(BLOCK_END, "stale row\n" + BLOCK_END, 1)
        )
        core_hooks.read_bytes = lambda path: (
            stale.encode("utf-8") if path == index_path() else path.read_bytes()
        )

        assert main(["--check"]) == 1

        block = render_projects_block(declared_projects(runs_directory()))
        assert capsys.readouterr().out == (
            f"the project table in {index_path()} is stale; run `{REGENERATE_HINT}`\n\n{block}\n"
        )


class TestRefusingAssertedRunState:
    """The half that cannot be generated, so it is refused instead.

    A ledger row count restates a file that is machine-local and untracked, so
    unlike every other number in the index it cannot be rendered from a source
    and cannot be checked by a reader elsewhere. Both entries carrying one were
    wrong when this was written.
    """

    def test_a_count_before_the_unit_is_a_claim(self) -> None:
        """The spelling both live instances used."""
        claims = ledger_state_claims(f"- **Runs:** 131 {LEDGER_ROW_UNIT}s, the largest.\n")

        assert claims == (f"an asserted ledger row count: 131 {LEDGER_ROW_UNIT}s",)

    def test_a_grouped_count_reads_as_one_number(self) -> None:
        """``13,008`` is one count, and a digit scan that stops at the comma
        would report ``008`` and read as a different, smaller claim."""
        claims = ledger_state_claims(f"13,008 {LEDGER_ROW_UNIT}s")

        assert claims == (f"an asserted ledger row count: 13,008 {LEDGER_ROW_UNIT}s",)

    def test_the_scale_field_is_refused_on_its_own(self) -> None:
        """The field is banned as an affordance, whatever it is filled with."""
        claims = ledger_state_claims(f"{SCALE_FIELD} big.\n")

        assert claims == (f"the `Scale` field is back: {SCALE_FIELD} big.",)

    def test_an_indented_scale_field_is_refused_too(self) -> None:
        """Nesting the bullet must not smuggle the field back in."""
        assert len(ledger_state_claims(f"    {SCALE_FIELD} 12\n")) == 1

    def test_a_job_id_after_the_unit_is_not_a_claim(self) -> None:
        """The index legitimately cites job ids beside this noun.

        ``tankpit``'s entry says "the ledger row for `55715577` carries", and
        a rule that convicted it would be one an author learns to skip.
        """
        assert ledger_state_claims(f"the {LEDGER_ROW_UNIT} for `55715577` carries") == ()

    def test_the_unit_without_a_count_is_not_a_claim(self) -> None:
        """Describing the ledger is the behaviour being asked for."""
        assert ledger_state_claims(f"a {LEDGER_ROW_UNIT} is written for you") == ()

    def test_a_document_asserting_nothing_yields_nothing(self) -> None:
        """The passing case, stated so the rule cannot fire vacuously."""
        assert ledger_state_claims("- **Runs:** `hpc3-submit`\n") == ()

    def test_every_claim_is_reported_not_just_the_first(self) -> None:
        """Two entries carried one at once; reporting one would hide the other."""
        text = f"{SCALE_FIELD} x\nand 108 {LEDGER_ROW_UNIT}s\n"

        assert len(ledger_state_claims(text)) == 2

    def test_the_committed_index_asserts_no_run_state(self) -> None:
        """The assertion this rule exists for, against the real document."""
        assert ledger_state_claims(index_path().read_text(encoding="utf-8")) == ()


class TestRefusingARestatedImageDigest:
    """The generated block renders every declared digest, and one was retyped.

    ``rusted``'s entry named ``images/v4/rusted.sif`` and ``b1eaaa2e`` while
    the registry declared v5 and ``97a80bdeb16d``, with the rendered table
    carrying the right answer two screens above it. Sitting beside the
    generated value is not what makes a restatement safe.
    """

    def _index(self, body: str) -> str:
        """Wrap a fragment in the heading the section reader needs.

        Args:
            body: The entry's text.

        Returns:
            A document with one ``rusted`` section.
        """
        return f"### `rusted` — a title\n\n{body}\n"

    def test_a_digest_the_registry_contradicts_is_a_claim(self) -> None:
        """The live instance, reduced to its shape."""
        text = self._index("- declares `/pub/x/images/v4/rusted.sif` pinned by sha256 `b1eaaa2e`")

        claims = image_digest_claims(text, {"rusted": _project(image_sha="9" * 64)})

        assert claims == (
            "`rusted` restates an image digest the registry contradicts: "
            "b1eaaa2e against 999999999999",
        )

    def test_a_digest_that_agrees_is_not_a_claim(self) -> None:
        """``tankpit``'s restatement was CORRECT when this was written.

        A rule that fired on it would be one an author learns to skip, and the
        point is to catch disagreement rather than to ban the sentence.
        """
        text = self._index("- ships `/pub/x/images/v2/t.sif`, sha256 `aaaaaaaaaaaa…`, 127 MB")

        assert image_digest_claims(text, {"rusted": _project(image_sha="a" * 64)}) == ()

    def test_an_elided_digest_is_read_up_to_the_ellipsis(self) -> None:
        """``0cfdd5592a1a…`` must compare as twelve hex characters, not as
        twelve plus a character that is not in any digest."""
        text = self._index("- ships `/pub/x/images/v2/t.sif`, sha256 `bbbbbbbbbbbb…`")

        assert image_digest_claims(text, {"rusted": _project(image_sha="c" * 64)}) != ()

    def test_a_digest_far_from_the_path_is_not_attributed_to_it(self) -> None:
        """A later bullet's digest belongs to that bullet, not to this image."""
        text = self._index("- ships `/pub/x/images/v2/t.sif`" + " padding" * 40 + " sha256 `dddd`")

        assert image_digest_claims(text, {"rusted": _project(image_sha="e" * 64)}) == ()

    def test_a_path_with_no_digest_is_not_a_claim(self) -> None:
        """Naming the image without retyping its digest is the fixed shape."""
        text = self._index("- declares `/pub/x/images/v5/rusted.sif`, binding `/pub`")

        assert image_digest_claims(text, {"rusted": _project()}) == ()

    def test_a_digest_outside_any_registered_section_is_left_alone(self) -> None:
        """The preamble and the unregistered entries are not registry claims."""
        text = "prose `/pub/x/images/v1/s.sif` sha256 `abcabcabcabc` with no heading above it\n"

        assert image_digest_claims(text, {"rusted": _project()}) == ()

    def test_an_unregistered_project_section_is_left_alone(self) -> None:
        """``sirius`` is described at length and declares nothing."""
        text = "### `sirius` — never run\n\n`/pub/x/s.sif` sha256 `abcabcabcabc`\n"

        assert image_digest_claims(text, {"rusted": _project()}) == ()

    def test_the_committed_index_restates_no_contradicted_digest(self) -> None:
        """The assertion this rule exists for, against the real document."""
        text = index_path().read_text(encoding="utf-8")

        assert image_digest_claims(text, declared_projects(runs_directory())) == ()


class TestReportingAssertedRunStateFromTheCommandLine:
    """A claim fails both forms, and writing cannot clear it.

    Driven through the file hooks rather than against ``docs/RESEARCH.md``,
    for the reason ``TestTheWritingAndStaleBranches`` gives: a test that
    rewrites a tracked document to prove it can is one nobody runs twice.
    """

    def _with_claim(self) -> str:
        """Build a document carrying one asserted count.

        Returns:
            The committed index with a claim appended, so the generated block
            is current and the claim is the only thing wrong with it.
        """
        return index_path().read_text(encoding="utf-8") + f"\n- 131 {LEDGER_ROW_UNIT}s\n"

    def test_checking_fails_and_says_to_delete_rather_than_update(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ "Update it" is the wrong instruction: the next count rots too.

        Args:
            capsys: Captured process output.
        """
        document = self._with_claim()
        core_hooks.read_bytes = lambda path: (
            document.encode("utf-8") if path == index_path() else path.read_bytes()
        )

        assert main(["--check"]) == 1

        assert capsys.readouterr().out == (
            f"{index_path()}: an asserted ledger row count: 131 {LEDGER_ROW_UNIT}s\n"
            f"{CLAIM_GUIDANCE}"
        )

    def test_writing_still_fails_because_the_claim_is_prose(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--write fixes the block and cannot fix a sentence.

        The written document still carries the claim, which is the point: a
        caller who reads exit 0 from the writing form as "clean" would be
        reading it from the half no generator owns.

        Args:
            capsys: Captured process output.
        """
        document = self._with_claim()
        written: dict[pathlib.Path, str] = {}
        core_hooks.read_bytes = lambda path: (
            document.encode("utf-8") if path == index_path() else path.read_bytes()
        )
        core_hooks.write_text = lambda path, text: written.__setitem__(path, text)

        assert main(["--write"]) == 1

        assert written[index_path()] == document
        assert capsys.readouterr().out == (
            f"{index_path()}: an asserted ledger row count: 131 {LEDGER_ROW_UNIT}s\n"
            f"{CLAIM_GUIDANCE}"
            f"wrote the project table into {index_path()}\n"
        )


class TestARegistryThatContradictsItself:
    """Two workspaces declaring one project leaves no answer to which governs."""

    def test_a_project_declared_twice_is_refused(self, tmp_path: pathlib.Path) -> None:
        """The name of the second file is in the message, because that is the fix.

        Args:
            tmp_path: Temporary directory holding two workspace documents.
        """
        document = (runs_directory() / "hpc3-rusted.json").read_text(encoding="utf-8")
        for name in ("hpc3-a.json", "hpc3-b.json"):
            (tmp_path / name).write_text(document, encoding="utf-8")

        with pytest.raises(ValueError, match="declared twice"):
            _ = declared_projects(tmp_path)
