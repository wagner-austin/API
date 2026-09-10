"""The subscription record: what a push hook writes and the bridge reads.

The validation tests carry most of the weight here, and they are about ONE
thing: a bad row enrolled now is a post the board refuses on every later
cycle, which wedges every other session's announcement behind it. Every
refusal below converts that permanent outage into one failed push, told to
the only person who can fix it.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.error_codes_tooling import CiWakeErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import InvalidJsonError, JSONTypeError, dump_json_str

from ci_wake import _test_hooks
from ci_wake.enrolment import (
    PushAttempt,
    append_attempt,
    attempt_key,
    decode_push_attempt,
    encode_push_attempt,
    latest_attempts,
    read_attempts,
    require_agent,
    require_repository,
    require_sha,
)
from tests.conftest import AGENT, OTHER_SHA, REPO, SHA


def _attempt(
    *, sha: str = SHA, agent: str = AGENT, repo: str = REPO, at: int = 1788700000
) -> PushAttempt:
    """Build one enrolment row.

    Args:
        sha: The commit sha.
        agent: The pushing session's label, or the empty string.
        repo: The repository.
        at: When the hook ran.

    Returns:
        The row.
    """
    return PushAttempt(repo=repo, sha=sha, ref="refs/heads/main", agent=agent, attempted_unix=at)


class TestAttemptKey:
    def test_it_names_the_repository_as_well_as_the_sha(self) -> None:
        """A fork and its upstream can hold the same commit, and announcing
        one repository's verdict under the other's name would be worse than
        announcing nothing."""
        assert attempt_key(REPO, SHA) == f"{REPO}@{SHA}"
        assert attempt_key("wagner-austin/API", SHA) != attempt_key(REPO, SHA)


class TestRequireRepository:
    def test_an_owner_name_pair_is_accepted(self) -> None:
        assert require_repository(REPO) == REPO

    @pytest.mark.parametrize("value", ["MCPs", "", "a/b/c", "owner/na me", "owner/"])
    def test_anything_that_addresses_no_repository_refuses(self, value: str) -> None:
        with pytest.raises(AppError) as caught:
            require_repository(value)

        assert caught.value.code is CiWakeErrorCode.ENROLMENT_FIELD_MALFORMED
        assert "--repo" in caught.value.message


class TestRequireSha:
    def test_a_full_lowercase_sha_is_accepted(self) -> None:
        assert require_sha(SHA) == SHA

    def test_an_abbreviation_refuses(self) -> None:
        """``head_sha`` in the Actions API is always the full 40, so a short
        one matches no run and the bridge would report "no run ever
        appeared" for a push that had one."""
        with pytest.raises(AppError) as caught:
            require_sha(SHA[:7])

        assert caught.value.code is CiWakeErrorCode.ENROLMENT_FIELD_MALFORMED

    @pytest.mark.parametrize("value", ["", SHA.upper(), "z" * 40, SHA + "a"])
    def test_anything_else_refuses(self, value: str) -> None:
        with pytest.raises(AppError):
            require_sha(value)


class TestRequireAgent:
    def test_the_empty_string_is_accepted_and_means_unaddressed(self) -> None:
        """A human pushing from a terminal has no board label, and their
        push is announced board-level rather than refused."""
        assert require_agent("") == ""

    def test_a_kebab_case_label_is_accepted(self) -> None:
        assert require_agent(AGENT) == AGENT

    @pytest.mark.parametrize(
        "value",
        ["Opus-CI-0909", "opus_ci_0909", "-opus-ci", "opus--ci", "ab", "a" * 65],
    )
    def test_a_label_the_board_would_refuse_is_refused_here_instead(self, value: str) -> None:
        """At the push, not three minutes later. See the module docstring."""
        with pytest.raises(AppError) as caught:
            require_agent(value)

        assert caught.value.code is CiWakeErrorCode.ENROLMENT_FIELD_MALFORMED
        assert "BOARD_AGENT_LABEL" in caught.value.message


class TestCodec:
    def test_a_row_round_trips(self) -> None:
        record = _attempt()

        assert decode_push_attempt(encode_push_attempt(record)) == record

    def test_a_non_object_refuses(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_push_attempt(["not", "an", "object"])

    def test_a_missing_field_refuses(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_push_attempt({"repo": REPO, "sha": SHA, "ref": "refs/heads/main"})

    def test_a_well_typed_but_unaddressable_field_still_refuses(self) -> None:
        """Re-validated on the way IN as well as out. This file is written by
        a git hook and hand-edited by whoever is debugging one, so a row that
        only the writer validated is a row nobody validated."""
        encoded = encode_push_attempt(_attempt())
        encoded["sha"] = "short"

        with pytest.raises(AppError) as caught:
            decode_push_attempt(encoded)

        assert caught.value.code is CiWakeErrorCode.ENROLMENT_FIELD_MALFORMED


class TestReadAttempts:
    def test_an_absent_record_reads_as_empty(self, tmp_path: pathlib.Path) -> None:
        """A machine that has not pushed since the bridge was installed has
        enrolled nothing; refusing the first cycle would make the bridge
        impossible to start."""
        assert read_attempts(tmp_path / "pushes.jsonl") == ()

    def test_rows_are_returned_in_the_order_the_hooks_wrote_them(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "pushes.jsonl"
        append_attempt(path, _attempt(sha=SHA))
        append_attempt(path, _attempt(sha=OTHER_SHA))

        rows = read_attempts(path)

        assert [row["sha"] for row in rows] == [SHA, OTHER_SHA]

    def test_blank_lines_are_skipped(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "pushes.jsonl"
        path.write_text(
            dump_json_str(encode_push_attempt(_attempt())) + "\n\n   \n", encoding="utf-8"
        )

        assert len(read_attempts(path)) == 1

    def test_a_non_object_line_is_fatal_and_names_its_line_number(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Not skipped. A line read as absent is a push whose author is never
        told what their CI did."""
        path = tmp_path / "pushes.jsonl"
        path.write_text(
            dump_json_str(encode_push_attempt(_attempt())) + "\n[1, 2]\n", encoding="utf-8"
        )

        with pytest.raises(JSONTypeError, match="line 2"):
            read_attempts(path)

    def test_a_line_that_is_not_json_at_all_is_fatal(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "pushes.jsonl"
        path.write_text("{not json\n", encoding="utf-8")

        with pytest.raises(InvalidJsonError):
            read_attempts(path)

    def test_it_reads_through_the_seam_rather_than_the_filesystem(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The seam is what a cycle test uses to give the two records
        different behaviour, so a reader that bypassed it would be untestable
        in the case that decides whether an announcement repeats or is lost.
        """
        seen: list[pathlib.Path] = []

        def _exists(path: pathlib.Path) -> bool:
            seen.append(path)
            return False

        _test_hooks.file_exists = _exists

        assert read_attempts(tmp_path / "pushes.jsonl") == ()
        assert seen == [tmp_path / "pushes.jsonl"]


class TestLatestAttempts:
    def test_a_re_pushed_sha_collapses_to_its_most_recent_row(self) -> None:
        """The session waiting on the verdict is the one that pushed last."""
        rows = latest_attempts(
            [_attempt(agent="opus-first-0909", at=10), _attempt(agent="opus-second-0909", at=20)]
        )

        assert len(rows) == 1
        assert rows[0]["agent"] == "opus-second-0909"
        assert rows[0]["attempted_unix"] == 20

    def test_distinct_shas_are_all_kept(self) -> None:
        rows = latest_attempts([_attempt(sha=SHA), _attempt(sha=OTHER_SHA)])

        assert [row["sha"] for row in rows] == [SHA, OTHER_SHA]

    def test_the_same_sha_in_two_repositories_is_two_pushes(self) -> None:
        rows = latest_attempts([_attempt(repo=REPO), _attempt(repo="wagner-austin/API")])

        assert len(rows) == 2

    def test_order_is_first_appearance_so_a_re_push_does_not_reorder_a_cycle(self) -> None:
        """Last-wins on CONTENT, first-appearance on ORDER. Reordering would
        make two cycles over the same work post different things."""
        rows = latest_attempts(
            [_attempt(sha=SHA, at=10), _attempt(sha=OTHER_SHA, at=20), _attempt(sha=SHA, at=30)]
        )

        assert [row["sha"] for row in rows] == [SHA, OTHER_SHA]
        assert rows[0]["attempted_unix"] == 30

    def test_no_rows_collapse_to_no_rows(self) -> None:
        assert latest_attempts([]) == ()


class TestAppendAttempt:
    def test_it_writes_one_decodable_line_per_call(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "runs" / "pushes.jsonl"

        append_attempt(path, _attempt(sha=SHA))
        append_attempt(path, _attempt(sha=OTHER_SHA))

        assert len(path.read_text(encoding="utf-8").splitlines()) == 2
        assert read_attempts(path)[1]["sha"] == OTHER_SHA
