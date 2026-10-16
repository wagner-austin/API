"""Whether a run-less push ever reached its branch, read from GitHub's two answers.

The replies are the shapes ``gh`` printed on 2026-10-16 for MCPs main (MCPs
board task be748588): ``{"object":null}`` for d357385da52f, which pre-push
refused and GitHub never received; ``{"object":{"__typename":"Commit"}}``
and compare ``diverged`` for 1091c36a200b, which the remote refused after
storing it; compare ``behind`` for main against 29182688d, which landed.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError

from ci_wake import _test_hooks
from ci_wake.coverage import compare_argv
from ci_wake.landing import (
    ABSENT,
    OBJECT_QUERY,
    UnlandedPush,
    decode_held,
    holds,
    landing_of,
    object_argv,
)
from tests.conftest import (
    REPO,
    SHA,
    FakeCompleted,
    FakeGh,
    compare_reply,
    make_attempt,
    object_reply,
)


def test_the_object_question_is_one_graphql_call_wrapped_for_jq() -> None:
    """Wrapped, because gh's --jq prints nothing for a bare null."""
    assert object_argv(REPO, SHA) == (
        "gh",
        "api",
        "graphql",
        "-f",
        f"query={OBJECT_QUERY}",
        "-f",
        "owner=wagner-austin",
        "-f",
        "name=MCPs",
        "-f",
        f"sha={SHA}",
        "--jq",
        "{object: .data.repository.object}",
    )
    assert "object(expression:$sha)" in OBJECT_QUERY


class TestDecodeHeld:
    def test_a_null_object_is_a_sha_github_never_received(self) -> None:
        assert decode_held({"object": None}) is False

    def test_an_object_is_held(self) -> None:
        assert decode_held({"object": {"__typename": "Commit"}}) is True

    def test_a_reply_without_the_key_is_refused_never_read_as_absent(self) -> None:
        with pytest.raises(JSONTypeError, match="carries no 'object' key"):
            decode_held({"data": None})

    def test_an_object_that_is_not_one_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_held({"object": "Commit"})

    def test_a_reply_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_held(None)


class TestLandingOf:
    def test_a_sha_github_never_received_is_absent_and_no_compare_is_made(self) -> None:
        gh = FakeGh({object_argv(REPO, SHA): object_reply(held=False)})
        _test_hooks.run_process = gh
        assert landing_of(REPO, "main", SHA) == ABSENT
        assert gh.calls == [object_argv(REPO, SHA)]

    @pytest.mark.parametrize("status", ["ahead", "identical", "behind", "diverged"])
    def test_a_held_sha_is_compared_with_its_branch(self, status: str) -> None:
        _test_hooks.run_process = FakeGh(
            {
                object_argv(REPO, SHA): object_reply(held=True),
                compare_argv(REPO, SHA, "main"): compare_reply(status),
            }
        )
        assert landing_of(REPO, "main", SHA) == status

    def test_a_gh_that_cannot_answer_propagates(self) -> None:
        _test_hooks.run_process = FakeGh(
            {object_argv(REPO, SHA): FakeCompleted(returncode=1, stderr="gh: Bad credentials")}
        )
        with pytest.raises(AppError, match="Bad credentials"):
            landing_of(REPO, "main", SHA)


@pytest.mark.parametrize(
    ("landing", "held"),
    [("ahead", True), ("identical", True), ("behind", False), ("diverged", False), (ABSENT, False)],
)
def test_only_a_branch_descending_from_the_sha_holds_it(landing: str, held: bool) -> None:
    assert holds(landing) is held


class TestTheLine:
    def test_a_sha_github_never_received_says_so(self) -> None:
        push = UnlandedPush(attempt=make_attempt(), branch="main", landing=ABSENT)
        assert push.line() == (
            f"unlanded {REPO}@{SHA[:12]}: refs/heads/main never held it (GitHub never "
            "received it), so no run was due; closed unannounced"
        )

    def test_a_refused_sha_github_holds_names_the_compare(self) -> None:
        push = UnlandedPush(attempt=make_attempt(), branch="main", landing="diverged")
        assert push.line() == (
            f"unlanded {REPO}@{SHA[:12]}: refs/heads/main never held it (compare of main "
            "against it reads diverged), so no run was due; closed unannounced"
        )
