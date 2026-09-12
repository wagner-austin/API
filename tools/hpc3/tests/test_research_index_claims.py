"""The three rules that refuse a claim the index cannot support.

Split from ``test_research_index`` when that module crossed the size ceiling.
The division is by role rather than by size: that module is about RENDERING the
generated block and substituting it, and this one is about the prose around it,
where nothing can be generated and the only available move is to refuse.

The three are siblings and fail for different reasons, which is why they are
here together:

* ``ledger_state_claims`` -- a count restating a machine-local, untracked file,
  so no reader elsewhere can check it.
* ``image_digest_claims`` -- a digest restating a value the block already
  renders, which went stale beside the thing it copied.
* ``stale_review_claims`` -- neither of those. A sentence that was TRUE when
  written, whose evidence moved afterwards and which nobody read again.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import pytest

from hpc3.cli.research_index import (
    CLAIM_GUIDANCE,
    declared_projects,
    index_path,
    main,
    runs_directory,
    tracked_counts,
)
from hpc3.core import _test_hooks as core_hooks
from hpc3.core._test_hooks import CommandResult
from hpc3.core.research_index import (
    LEDGER_ROW_UNIT,
    REVIEW_MARKER_PREFIX,
    REVIEW_MARKER_SEPARATOR,
    REVIEW_MARKER_SUFFIX,
    SCALE_FIELD,
    image_digest_claims,
    ledger_state_claims,
    parse_review_markers,
    stale_review_claims,
)
from tests._project_config import project_config


def _two_tracked_files(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
    """Stand in for git listing two files, with a trailing blank line.

    Args:
        argv: The command, ignored: this fake answers one question.
        stdin_bytes: Unused, and present because the protocol declares it.

    Returns:
        Two paths and the empty line git's output ends on.
    """
    return CommandResult(returncode=0, stdout="a.json\nb.json\n\n", stderr="")


def _a_refused_pathspec(argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
    """Stand in for git rejecting a pathspec it cannot parse.

    Args:
        argv: The command, ignored.
        stdin_bytes: Unused, and present because the protocol declares it.

    Returns:
        A non-zero status with the reason on stderr, which is where git puts
        it and what the caller has to surface.
    """
    return CommandResult(returncode=128, stdout="", stderr="bad pathspec")


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

    ``rusted``'s entry named the v4 image and gave its digest as ``b1eaaa2e``
    while the registry declared v5, with the rendered table carrying the right
    answer two screens above it. Sitting beside the generated value is not what
    makes a restatement safe.
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

        claims = image_digest_claims(text, {"rusted": project_config(image_sha="9" * 64)})

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

        assert image_digest_claims(text, {"rusted": project_config(image_sha="a" * 64)}) == ()

    def test_an_elided_digest_is_read_up_to_the_ellipsis(self) -> None:
        """``0cfdd5592a1a…`` must compare as twelve hex characters, not as
        twelve plus a character that is not in any digest."""
        text = self._index("- ships `/pub/x/images/v2/t.sif`, sha256 `bbbbbbbbbbbb…`")

        assert image_digest_claims(text, {"rusted": project_config(image_sha="c" * 64)}) != ()

    def test_a_digest_far_from_the_path_is_not_attributed_to_it(self) -> None:
        """A later bullet's digest belongs to that bullet, not to this image."""
        text = self._index("- ships `/pub/x/images/v2/t.sif`" + " padding" * 40 + " sha256 `dddd`")

        assert image_digest_claims(text, {"rusted": project_config(image_sha="e" * 64)}) == ()

    def test_a_path_with_no_digest_is_not_a_claim(self) -> None:
        """Naming the image without retyping its digest is the fixed shape."""
        text = self._index("- declares `/pub/x/images/v5/rusted.sif`, binding `/pub`")

        assert image_digest_claims(text, {"rusted": project_config()}) == ()

    def test_a_digest_outside_any_registered_section_is_left_alone(self) -> None:
        """The preamble and the unregistered entries are not registry claims."""
        text = "prose `/pub/x/images/v1/s.sif` sha256 `abcabcabcabc` with no heading above it\n"

        assert image_digest_claims(text, {"rusted": project_config()}) == ()

    def test_an_unregistered_project_section_is_left_alone(self) -> None:
        """``sirius`` is described at length and declares nothing."""
        text = "### `sirius` — never run\n\n`/pub/x/s.sif` sha256 `abcabcabcabc`\n"

        assert image_digest_claims(text, {"rusted": project_config()}) == ()

    def test_the_committed_index_restates_no_contradicted_digest(self) -> None:
        """The assertion this rule exists for, against the real document."""
        text = index_path().read_text(encoding="utf-8")

        assert image_digest_claims(text, declared_projects(runs_directory())) == ()


class TestTheReviewMarker:
    """The rule for a sentence that was true when written and never re-read.

    Both live misses had this shape: ``code-style`` named the width that would
    settle its result and the run at that width landed the next day, and
    ``floor`` described seven jobs while ninety-six run documents accumulated.
    Nothing was wrong with any sentence in isolation, which is why the two
    rules above are blind to it.
    """

    def test_a_marker_is_read_as_its_glob_and_count(self) -> None:
        """The pairing is the whole declaration."""
        text = f"{REVIEW_MARKER_PREFIX}runs/*.json{REVIEW_MARKER_SEPARATOR}7{REVIEW_MARKER_SUFFIX}"

        assert parse_review_markers(text) == (("runs/*.json", 7),)

    def test_several_markers_are_read_in_order(self) -> None:
        """One entry can rest on more than one class of evidence."""
        one = f"{REVIEW_MARKER_PREFIX}a/*{REVIEW_MARKER_SEPARATOR}1{REVIEW_MARKER_SUFFIX}"
        two = f"{REVIEW_MARKER_PREFIX}b/*{REVIEW_MARKER_SEPARATOR}2{REVIEW_MARKER_SUFFIX}"

        assert parse_review_markers(f"{one}\ntext\n{two}") == (("a/*", 1), ("b/*", 2))

    def test_a_document_with_no_markers_yields_none(self) -> None:
        """Carrying one is optional, and absence is not an error."""
        assert parse_review_markers("no markers here") == ()

    def test_an_unterminated_marker_is_refused(self) -> None:
        """Skipping it would disable the check on the entry being edited."""
        with pytest.raises(ValueError, match="unterminated review marker"):
            _ = parse_review_markers(f"{REVIEW_MARKER_PREFIX}runs/*.json = 7")

    def test_a_marker_without_the_separator_is_refused(self) -> None:
        """A glob with no count declares nothing to check against."""
        with pytest.raises(ValueError, match="omits"):
            _ = parse_review_markers(f"{REVIEW_MARKER_PREFIX}runs/*.json{REVIEW_MARKER_SUFFIX}")

    def test_a_non_numeric_count_is_refused(self) -> None:
        """Caught on the real file: prose describing the syntax parsed as one."""
        text = f"{REVIEW_MARKER_PREFIX}glob{REVIEW_MARKER_SEPARATOR}N{REVIEW_MARKER_SUFFIX}"

        with pytest.raises(ValueError, match="non-numeric count"):
            _ = parse_review_markers(text)

    def test_a_glob_containing_the_separator_keeps_its_count(self) -> None:
        """Split from the RIGHT, so a path with " = " in it still resolves."""
        text = (
            f"{REVIEW_MARKER_PREFIX}odd = name/*.json"
            f"{REVIEW_MARKER_SEPARATOR}3{REVIEW_MARKER_SUFFIX}"
        )

        assert parse_review_markers(text) == (("odd = name/*.json", 3),)

    def test_a_count_that_moved_is_a_claim(self) -> None:
        """The code-style miss, reduced: one more file than was read."""
        claims = stale_review_claims((("runs/*.json", 5),), {"runs/*.json": 6})

        assert claims == (
            "`runs/*.json` now matches 6 tracked file(s) and this entry was read "
            "against 5; re-read the entry and bump its marker",
        )

    def test_evidence_that_disappeared_is_also_a_claim(self) -> None:
        """Fewer files is equally a reason to re-read, not only more."""
        assert len(stale_review_claims((("runs/*.json", 5),), {"runs/*.json": 0})) == 1

    def test_a_count_that_agrees_is_not_a_claim(self) -> None:
        """The passing case, so the rule cannot pass vacuously."""
        assert stale_review_claims((("runs/*.json", 6),), {"runs/*.json": 6}) == ()

    def test_the_committed_index_is_current_against_its_markers(self) -> None:
        """The assertion this rule exists for, against the real document."""
        markers = parse_review_markers(index_path().read_text(encoding="utf-8"))

        assert markers != ()
        assert stale_review_claims(markers, tracked_counts(tuple(g for g, _ in markers))) == ()


class TestCountingTrackedFiles:
    """Counted from git, never from the disk.

    A count over what happens to be present locally would be the same
    uncheckable claim ``ledger_state_claims`` refuses. A count over what is
    committed is one any clone reproduces.
    """

    def test_a_glob_is_counted_from_what_git_tracks(self) -> None:
        """Blank lines in git's output must not inflate the count."""
        core_hooks.run = _two_tracked_files

        assert tracked_counts(("x/*.json",)) == {"x/*.json": 2}

    def test_a_pathspec_git_refuses_is_not_counted_as_zero(self) -> None:
        """Zero would read as "the evidence was deleted", a louder and
        different claim than "this marker has a typo"."""
        core_hooks.run = _a_refused_pathspec

        with pytest.raises(RuntimeError, match="refused the review marker pathspec"):
            _ = tracked_counts(("[",))


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
