"""Which later run checked a superseded push, over GitHub's recorded answers.

THE CASE IS A MEASURED ONE (MCPs board task 5af9e749). On 2026-10-15
``wagner-austin/MCPs`` 82fa981f2 was pushed to main at 07:05:30Z; its own
Check run 38696451632 was cancelled with 0 jobs. Of the branch's completed
runs that followed, run 38699479621 (head c91c08f97, created 07:32:49Z) and
run 38716940740 (head e7cb362b9, created 09:58:56Z) completed, every run
between them was cancelled, and GitHub's compare read 82fa981f2...c91c08f97
as ``ahead`` by 9 and 82fa981f2...e7cb362b9 as ``ahead`` by 33. So the
covering run is 38699479621, a failure, though a later green also contained
the push: the first verdict is the one its pusher must hear. The ids, shas
and instants below are copied from those ``gh api`` answers.
"""

from __future__ import annotations

import datetime

import pytest
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str

from ci_wake import _test_hooks
from ci_wake.coverage import (
    BranchRun,
    branch_of,
    branch_runs,
    branch_runs_argv,
    candidates,
    compare_argv,
    decode_branch_runs,
    decode_compare_status,
    find_cover,
)
from tests.conftest import FakeCompleted, FakeGh

REPO = "wagner-austin/MCPs"
PUSHED = "82fa981f26a6db820fcd2c4e4e65dcb25be5e4ce"
RED = "c91c08f97ffa847a1bbcc33b3fdb24a88300e1dc"
GREEN = "e7cb362b93312932e7a8bf4787c89a4ba2e8b211"
EVICTED = "8fb508fb115a9820366179a8cf931aea8e7e7bcf"


def _unix(instant: str) -> int:
    """Whole epoch seconds of an ISO instant, as GitHub writes them."""
    return int(datetime.datetime.fromisoformat(instant).timestamp())


PUSHED_AT = _unix("2026-10-15T07:05:30Z")


def _entry(
    run_id: int, head: str, created: str, conclusion: str, *, name: str = "Check"
) -> JSONObject:
    """One ``workflow_runs`` element as the branch listing returns it."""
    return {
        "id": run_id,
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
        "head_sha": head,
        "created_at": created,
    }


RECORDED: tuple[JSONObject, ...] = (
    _entry(38716940740, GREEN, "2026-10-15T09:58:56Z", "success"),
    _entry(38708367436, EVICTED, "2026-10-15T08:48:59Z", "cancelled"),
    _entry(38699479621, RED, "2026-10-15T07:32:49Z", "failure"),
)


def _listing(entries: tuple[JSONObject, ...], total: int) -> FakeCompleted:
    """Script one page of ``actions/runs``."""
    payload: JSONObject = {"total_count": total, "workflow_runs": list(entries)}
    return FakeCompleted(stdout=dump_json_str(payload))


def _compared(status: str) -> FakeCompleted:
    """Script the ``--jq`` reply of a compare."""
    payload: JSONObject = {"status": status}
    return FakeCompleted(stdout=dump_json_str(payload))


def _decoded() -> tuple[BranchRun, ...]:
    payload: JSONValue = {"total_count": 3, "workflow_runs": list(RECORDED)}
    return decode_branch_runs(payload)[1]


class TestTheBranch:
    def test_a_branch_ref_names_its_branch(self) -> None:
        assert branch_of("refs/heads/main") == "main"
        assert branch_of("refs/heads/feature/x") == "feature/x"

    def test_a_tag_has_no_branch_for_a_later_run_to_share(self) -> None:
        assert branch_of("refs/tags/v1.0") == ""


class TestTheQueries:
    def test_the_listing_asks_for_completed_runs_since_the_push_one_page_at_a_time(self) -> None:
        assert branch_runs_argv(REPO, "main", PUSHED_AT, 2) == (
            "gh",
            "api",
            f"repos/{REPO}/actions/runs?branch=main&status=completed"
            "&created=%3E%3D2026-10-15T07%3A05%3A30Z&per_page=100&page=2",
        )

    def test_the_compare_keeps_only_the_status(self) -> None:
        assert compare_argv(REPO, PUSHED, RED) == (
            "gh",
            "api",
            f"repos/{REPO}/compare/{PUSHED}...{RED}",
            "--jq",
            "{status: .status}",
        )
        payload: JSONValue = {"status": "ahead"}
        assert decode_compare_status(payload) == "ahead"

    def test_a_listing_decodes_with_its_heads_and_creation_times(self) -> None:
        payload: JSONValue = {"total_count": 3, "workflow_runs": list(RECORDED)}
        total, runs = decode_branch_runs(payload)
        assert total == 3
        assert [(run["run"]["run_id"], run["head_sha"]) for run in runs] == [
            (38716940740, GREEN),
            (38708367436, EVICTED),
            (38699479621, RED),
        ]
        assert runs[2]["created_unix"] == _unix("2026-10-15T07:32:49Z")
        assert runs[2]["run"]["conclusion"] == "failure"


class TestReadingEveryPage:
    def test_pages_are_read_until_the_count_is_reached(self) -> None:
        gh = FakeGh(
            {
                branch_runs_argv(REPO, "main", PUSHED_AT, 1): _listing(RECORDED[:2], 3),
                branch_runs_argv(REPO, "main", PUSHED_AT, 2): _listing(RECORDED[2:], 3),
            }
        )
        _test_hooks.run_process = gh
        runs = branch_runs(REPO, "main", PUSHED_AT)
        assert [run["run"]["run_id"] for run in runs] == [38716940740, 38708367436, 38699479621]
        assert len(gh.calls) == 2

    def test_an_empty_page_ends_the_reading_even_short_of_the_count(self) -> None:
        """A count that grew while paging must not page forever."""
        gh = FakeGh(
            {
                branch_runs_argv(REPO, "main", PUSHED_AT, 1): _listing(RECORDED, 5),
                branch_runs_argv(REPO, "main", PUSHED_AT, 2): _listing((), 5),
            }
        )
        _test_hooks.run_process = gh
        assert len(branch_runs(REPO, "main", PUSHED_AT)) == 3
        assert len(gh.calls) == 2


class TestTheCandidates:
    def test_only_the_workflows_completed_verdicts_since_the_push_earliest_first(self) -> None:
        runs = _decoded()
        before = decode_branch_runs(
            {
                "total_count": 4,
                "workflow_runs": [
                    _entry(38690000000, RED, "2026-10-15T06:00:00Z", "success"),
                    _entry(38716940741, GREEN, "2026-10-15T09:58:57Z", "success", name="Live"),
                    {**_entry(38716940742, GREEN, "2026-10-15T09:59:00Z", ""), "status": "queued"},
                ],
            }
        )[1]
        ordered = candidates(runs + before, "Check", PUSHED_AT)
        assert [run["run"]["run_id"] for run in ordered] == [38699479621, 38716940740]


class TestFindingTheCover:
    def test_the_first_run_containing_the_push_covers_it_though_a_later_one_was_green(
        self,
    ) -> None:
        gh = FakeGh({compare_argv(REPO, PUSHED, RED): _compared("ahead")})
        _test_hooks.run_process = gh
        runs = _decoded()
        assert runs[2]["run"]["run_id"] == 38699479621
        assert find_cover(REPO, PUSHED, "Check", PUSHED_AT, runs) == runs[2]
        assert gh.calls == [compare_argv(REPO, PUSHED, RED)]

    @pytest.mark.parametrize("status", ["behind", "diverged"])
    def test_a_head_not_descending_from_the_push_is_passed_over(self, status: str) -> None:
        _test_hooks.run_process = FakeGh(
            {
                compare_argv(REPO, PUSHED, RED): _compared(status),
                compare_argv(REPO, PUSHED, GREEN): _compared("identical"),
            }
        )
        runs = _decoded()
        assert runs[0]["head_sha"] == GREEN
        assert find_cover(REPO, PUSHED, "Check", PUSHED_AT, runs) == runs[0]

    def test_no_run_containing_the_push_yet_is_none(self) -> None:
        _test_hooks.run_process = FakeGh(
            {
                compare_argv(REPO, PUSHED, RED): _compared("behind"),
                compare_argv(REPO, PUSHED, GREEN): _compared("behind"),
            }
        )
        assert find_cover(REPO, PUSHED, "Check", PUSHED_AT, _decoded()) is None
