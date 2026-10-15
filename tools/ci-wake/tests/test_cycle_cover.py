"""A superseded push, end to end: held for the run that checked it, then announced with it.

MCPs board task 5af9e749. The shape is the one ``wagner-austin/MCPs`` gives
every push of a burst but the newest: the push's own run is cancelled before
any job, and a later run whose head contains it gives the verdict. The
records are real files written by this package's own writers, and every
``gh`` call the cycle makes is scripted, so a call the change did not mean to
add fails here rather than passing quietly.
"""

from __future__ import annotations

import datetime
import pathlib

from platform_core.json_utils import JSONObject, JSONValue, dump_json_str
from platform_core.mcp_testing import announcing_poster, notes_sent

from ci_wake import _test_hooks
from ci_wake.coverage import branch_runs_argv, compare_argv
from ci_wake.cycle import run_cycle
from ci_wake.enrolment import PushAttempt, append_attempt
from ci_wake.position import position_path
from ci_wake.runs import jobs_argv, runs_argv
from ci_wake.verdicts import STALL_SECONDS
from tests.conftest import (
    AGENT,
    CONFIGURED_ENV,
    FROZEN_NOW,
    OTHER_SHA,
    REPO,
    SHA,
    FakeCompleted,
    FakeGh,
    pin_env,
)

OWN_RUN = 38696451632
OTHER_OWN_RUN = 38696451633
RED_RUN = 38699479621
GREEN_RUN = 38716940740
RED = "c91c08f97ffa847a1bbcc33b3fdb24a88300e1dc"
GREEN = "e7cb362b93312932e7a8bf4787c89a4ba2e8b211"


def _iso(unix: int) -> str:
    """An epoch instant as GitHub writes ``created_at``."""
    return datetime.datetime.fromtimestamp(unix, tz=datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _enrol(
    path: pathlib.Path, *, sha: str = SHA, ago: int = 3600, ref: str = "refs/heads/main"
) -> PushAttempt:
    """Write one enrolment row through the package's own writer."""
    record = PushAttempt(repo=REPO, sha=sha, ref=ref, agent=AGENT, attempted_unix=FROZEN_NOW - ago)
    append_attempt(path, record)
    return record


def _reply(payload: JSONObject) -> FakeCompleted:
    return FakeCompleted(stdout=dump_json_str(payload))


def _own_run(run_id: int = OWN_RUN) -> FakeCompleted:
    """The push's own run: cancelled, as a superseded push's is."""
    return _reply(
        {
            "total_count": 1,
            "workflow_runs": [
                {
                    "id": run_id,
                    "name": "Check",
                    "status": "completed",
                    "conclusion": "cancelled",
                    "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
                }
            ],
        }
    )


def _jobs(total: int, failed: tuple[str, ...] = ()) -> FakeCompleted:
    jobs: list[JSONValue] = [
        {
            "name": name,
            "conclusion": "failure",
            "steps": [{"name": "Set up job", "conclusion": "failure"}],
        }
        for name in failed
    ]
    jobs.extend(
        {"name": f"pkg{index}", "conclusion": "success", "steps": []}
        for index in range(total - len(failed))
    )
    return _reply({"total_count": total, "jobs": jobs})


def _branch(*runs: tuple[int, str, int, str]) -> FakeCompleted:
    """The branch's completed runs: (id, head, seconds after the push, conclusion)."""
    return _reply(
        {
            "total_count": len(runs),
            "workflow_runs": [
                {
                    "id": run_id,
                    "name": "Check",
                    "status": "completed",
                    "conclusion": conclusion,
                    "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
                    "head_sha": head,
                    "created_at": _iso(FROZEN_NOW - 3600 + after),
                }
                for run_id, head, after, conclusion in runs
            ],
        }
    )


def _status(value: str) -> FakeCompleted:
    return _reply({"status": value})


class TestASupersededPushIsAnnouncedWithItsCover:
    def test_the_first_run_containing_it_is_its_verdict(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment)
        _test_hooks.run_process = FakeGh(
            {
                runs_argv(REPO, SHA): _own_run(),
                jobs_argv(REPO, OWN_RUN): _jobs(0),
                branch_runs_argv(REPO, "main", FROZEN_NOW - 3600, 1): _branch(
                    (GREEN_RUN, GREEN, 3200, "success"), (RED_RUN, RED, 1600, "failure")
                ),
                compare_argv(REPO, SHA, RED): _status("ahead"),
                jobs_argv(REPO, RED_RUN): _jobs(61, ("coloring-book",)),
            }
        )
        poster = announcing_poster()
        _test_hooks.http_post = poster

        run_cycle(enrolment)

        (note,) = notes_sent(poster)
        body = str(note["body"])
        assert body.splitlines()[0] == (
            f"CI-TERMINAL {REPO}: 1 push(es), 1 run(s) (failure by a later run x1)"
        )
        assert (
            f"  Check: its own run was cancelled before any job; checked by run {RED_RUN} at "
            "c91c08f, failure -- 61 jobs -- 1 failed: coloring-book (step: Set up job)"
        ) in body
        assert "NOT RUN" not in body
        assert body.endswith(f"@{AGENT} your push has its CI verdict")
        assert '"state":"covered"' in position_path(enrolment).read_text(encoding="utf-8")

    def test_two_pushes_to_one_branch_read_its_runs_once(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment)
        _enrol(enrolment, sha=OTHER_SHA, ago=3000)
        branch = branch_runs_argv(REPO, "main", FROZEN_NOW - 3600, 1)
        gh = FakeGh(
            {
                runs_argv(REPO, SHA): _own_run(),
                runs_argv(REPO, OTHER_SHA): _own_run(OTHER_OWN_RUN),
                jobs_argv(REPO, OWN_RUN): _jobs(0),
                jobs_argv(REPO, OTHER_OWN_RUN): _jobs(0),
                branch: _branch((GREEN_RUN, GREEN, 3200, "success")),
                compare_argv(REPO, SHA, GREEN): _status("ahead"),
                compare_argv(REPO, OTHER_SHA, GREEN): _status("ahead"),
                jobs_argv(REPO, GREEN_RUN): _jobs(5),
            }
        )
        _test_hooks.run_process = gh
        poster = announcing_poster()
        _test_hooks.http_post = poster

        run_cycle(enrolment)

        assert gh.calls.count(branch) == 1
        (note,) = notes_sent(poster)
        assert str(note["body"]).count(f"checked by run {GREEN_RUN} at e7cb362, success") == 2


class TestASupersededPushWithNoCoverYet:
    def test_inside_the_stall_horizon_nothing_is_posted_or_recorded(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment)
        _test_hooks.run_process = FakeGh(
            {
                runs_argv(REPO, SHA): _own_run(),
                jobs_argv(REPO, OWN_RUN): _jobs(0),
                branch_runs_argv(REPO, "main", FROZEN_NOW - 3600, 1): _branch(
                    (GREEN_RUN, GREEN, 3200, "success")
                ),
                compare_argv(REPO, SHA, GREEN): _status("behind"),
            }
        )
        poster = announcing_poster()
        _test_hooks.http_post = poster

        run_cycle(enrolment)

        assert notes_sent(poster) == []
        assert not position_path(enrolment).exists()
        assert emitted[-1] == "1 push(es) outstanding, none decided yet"

    def test_a_tag_has_no_branch_so_it_waits_without_asking_for_one(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment, ref="refs/tags/v1.0")
        gh = FakeGh({runs_argv(REPO, SHA): _own_run(), jobs_argv(REPO, OWN_RUN): _jobs(0)})
        _test_hooks.run_process = gh
        _test_hooks.http_post = announcing_poster()

        run_cycle(enrolment)

        assert len(gh.calls) == 2
        assert emitted[-1] == "1 push(es) outstanding, none decided yet"

    def test_past_the_stall_horizon_it_is_announced_not_run(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        ago = STALL_SECONDS + 60
        _enrol(enrolment, ago=ago)
        _test_hooks.run_process = FakeGh(
            {
                runs_argv(REPO, SHA): _own_run(),
                jobs_argv(REPO, OWN_RUN): _jobs(0),
                branch_runs_argv(REPO, "main", FROZEN_NOW - ago, 1): _reply(
                    {"total_count": 0, "workflow_runs": []}
                ),
            }
        )
        poster = announcing_poster()
        _test_hooks.http_post = poster

        run_cycle(enrolment)

        (note,) = notes_sent(poster)
        body = str(note["body"])
        assert "cancelled -- no job ever ran, NOT RUN" in body
        assert f"NOT RUN: CI cancelled work for {SHA}" in body
        assert '"state":"ripe"' in position_path(enrolment).read_text(encoding="utf-8")
