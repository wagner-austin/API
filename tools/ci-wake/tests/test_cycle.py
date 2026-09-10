"""One poll, end to end, against the real writers and the real gh boundary.

THE RECORDS ARE REAL FILES AND THE ROWS ARE WRITTEN BY THIS PACKAGE'S OWN
WRITERS. Nothing here hand-crafts a line of either record, so a reader that
skipped ``latest_attempts`` or spelled a key differently from
``attempt_key`` fails these tests rather than passing them against a fixture
built on the same mistake.

THE ORDERING TEST IS THE ONE THAT MATTERS. Post-then-write is the delivery
guarantee, and the only way to see it is to make the WRITE fail after the
POST succeeded and then check that the position record did not move -- which
is what ``test_a_failed_position_write_leaves_the_push_to_be_announced_again``
does. A suite that only asserted the happy path would pass just as
comfortably against a bridge that recorded first and announced second, which
is the version that loses announcements.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.error_codes_tooling import BoardWatchErrorCode, CiWakeErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments

from ci_wake import _test_hooks
from ci_wake.cycle import run_cycle
from ci_wake.enrolment import PushAttempt, append_attempt, attempt_key
from ci_wake.identity import IDENTITY
from ci_wake.position import position_path, read_announced
from ci_wake.runs import jobs_argv, runs_argv
from ci_wake.verdicts import NO_RUN_SECONDS
from tests.conftest import (
    AGENT,
    CONFIGURED_ENV,
    FROZEN_NOW,
    OTHER_SHA,
    REPO,
    SHA,
    TASK_ID,
    FakeCompleted,
    FakeGh,
    pin_env,
)

RUN_ID = 34397357156
OTHER_RUN_ID = 34397042137


def _enrol(
    path: pathlib.Path,
    *,
    sha: str = SHA,
    agent: str = AGENT,
    repo: str = REPO,
    ago: int = 3600,
) -> PushAttempt:
    """Write one enrolment row through the package's own writer.

    Args:
        path: The enrolment record.
        sha: The commit sha.
        agent: The pushing session's label, or the empty string.
        repo: The repository.
        ago: How long before the frozen clock the push happened.

    Returns:
        The row that was written.
    """
    record = PushAttempt(
        repo=repo,
        sha=sha,
        ref="refs/heads/main",
        agent=agent,
        attempted_unix=FROZEN_NOW - ago,
    )
    append_attempt(path, record)
    return record


def _runs_reply(*runs: JSONObject) -> FakeCompleted:
    """Script an ``actions/runs`` answer.

    Args:
        runs: The ``workflow_runs`` elements.

    Returns:
        The finished process.
    """
    listing: JSONObject = {"total_count": len(runs), "workflow_runs": list(runs)}
    return FakeCompleted(stdout=dump_json_str(listing))


def _run_entry(
    *,
    run_id: int = RUN_ID,
    name: str = "Check",
    status: str = "completed",
    conclusion: JSONValue = "success",
) -> JSONObject:
    """Build one ``workflow_runs`` element.

    Args:
        run_id: The run's numeric id.
        name: The workflow's display name.
        status: The run's status.
        conclusion: Its conclusion, or None while in flight.

    Returns:
        The element.
    """
    return {
        "id": run_id,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
    }


def _jobs_reply(*, total: int = 52, failed: tuple[str, ...] = ()) -> FakeCompleted:
    """Script a ``runs/{id}/jobs`` answer.

    Args:
        total: The payload's ``total_count``.
        failed: Names of jobs that did not succeed.

    Returns:
        The finished process.
    """
    jobs: list[JSONValue] = [{"name": name, "conclusion": "failure"} for name in failed]
    jobs.extend(
        {"name": f"check (pkg{index})", "conclusion": "success"}
        for index in range(total - len(failed))
    )
    listing: JSONObject = {"total_count": total, "jobs": jobs}
    return FakeCompleted(stdout=dump_json_str(listing))


class TestNothingToDo:
    def test_an_empty_enrolment_record_says_so_and_asks_github_nothing(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        gh = FakeGh({})
        _test_hooks.run_process = gh
        _test_hooks.http_post = FakeHttpPost([])

        run_cycle(tmp_path / "pushes.jsonl")

        assert emitted == ["enrolment record is empty; nothing has been pushed from this machine"]
        assert gh.calls == []

    def test_a_fully_announced_record_asks_github_nothing(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """The cost of a cycle tracks what is IN FLIGHT rather than what has
        ever been pushed, which is what keeps a quiet week cheap."""
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment)
        _test_hooks.run_process = FakeGh(
            {
                runs_argv(REPO, SHA): _runs_reply(_run_entry()),
                jobs_argv(REPO, RUN_ID): _jobs_reply(),
            }
        )
        _test_hooks.http_post = FakeHttpPost([posted_ok()])
        run_cycle(enrolment)

        gh = FakeGh({})
        _test_hooks.run_process = gh
        _test_hooks.http_post = FakeHttpPost([])
        emitted.clear()

        run_cycle(enrolment)

        assert emitted == ["1 push(es) enrolled, all already announced"]
        assert gh.calls == []

    def test_a_push_still_in_flight_is_asked_about_but_not_posted(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment)
        _test_hooks.run_process = FakeGh(
            {runs_argv(REPO, SHA): _runs_reply(_run_entry(status="in_progress", conclusion=None))}
        )
        poster = FakeHttpPost([])
        _test_hooks.http_post = poster

        run_cycle(enrolment)

        assert emitted == ["1 push(es) outstanding, none decided yet"]
        assert poster.bodies == []
        assert read_announced(position_path(enrolment)) == frozenset()


class TestAVerdictIsAnnounced:
    def test_it_posts_once_and_records_the_position(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment)
        _test_hooks.run_process = FakeGh(
            {
                runs_argv(REPO, SHA): _runs_reply(_run_entry(conclusion="failure")),
                jobs_argv(REPO, RUN_ID): _jobs_reply(total=52, failed=("audit",)),
            }
        )
        poster = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = poster

        run_cycle(enrolment)

        assert len(poster.bodies) == 1
        assert read_announced(position_path(enrolment)) == {attempt_key(REPO, SHA)}
        assert emitted[-1] == ("cycle: 1 enrolled, 1 outstanding, 1 announced, positions recorded")

    def test_the_post_carries_the_bridge_identity_and_the_standing_task(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """The board binds session to label on first write, so the arguments
        are asserted here rather than assumed -- a bridge posting under a
        fresh identity is refused for the rest of its life."""
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment)
        _test_hooks.run_process = FakeGh(
            {
                runs_argv(REPO, SHA): _runs_reply(_run_entry()),
                jobs_argv(REPO, RUN_ID): _jobs_reply(),
            }
        )
        poster = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = poster

        run_cycle(enrolment)

        arguments = sent_arguments(poster.bodies[0])
        assert arguments["taskId"] == TASK_ID
        assert arguments["agent"] == IDENTITY["agent"]
        assert arguments["sessionId"] == IDENTITY["session_id"]
        assert arguments["cwd"] == IDENTITY["cwd"]
        assert arguments["kind"] == "note"

    def test_the_jobs_call_is_made_only_for_a_push_about_to_be_announced(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """One runs query per outstanding push; the jobs query only for the
        ones with something to say. A cycle that fetched jobs for every push
        would pay a second call per waiting push forever."""
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment, sha=SHA)
        _enrol(enrolment, sha=OTHER_SHA)
        gh = FakeGh(
            {
                runs_argv(REPO, SHA): _runs_reply(_run_entry()),
                jobs_argv(REPO, RUN_ID): _jobs_reply(),
                runs_argv(REPO, OTHER_SHA): _runs_reply(
                    _run_entry(run_id=OTHER_RUN_ID, status="queued", conclusion=None)
                ),
            }
        )
        _test_hooks.run_process = gh
        _test_hooks.http_post = FakeHttpPost([posted_ok()])

        run_cycle(enrolment)

        assert jobs_argv(REPO, OTHER_RUN_ID) not in gh.calls
        assert gh.calls.count(jobs_argv(REPO, RUN_ID)) == 1

    def test_two_pushes_by_one_session_land_in_one_post(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment, sha=SHA)
        _enrol(enrolment, sha=OTHER_SHA)
        _test_hooks.run_process = FakeGh(
            {
                runs_argv(REPO, SHA): _runs_reply(_run_entry()),
                jobs_argv(REPO, RUN_ID): _jobs_reply(),
                runs_argv(REPO, OTHER_SHA): _runs_reply(_run_entry(run_id=OTHER_RUN_ID)),
                jobs_argv(REPO, OTHER_RUN_ID): _jobs_reply(),
            }
        )
        poster = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = poster

        run_cycle(enrolment)

        assert len(poster.bodies) == 1
        assert read_announced(position_path(enrolment)) == {
            attempt_key(REPO, SHA),
            attempt_key(REPO, OTHER_SHA),
        }

    def test_an_unaddressed_push_says_so_in_the_report_line(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment, agent="")
        _test_hooks.run_process = FakeGh(
            {
                runs_argv(REPO, SHA): _runs_reply(_run_entry()),
                jobs_argv(REPO, RUN_ID): _jobs_reply(),
            }
        )
        _test_hooks.http_post = FakeHttpPost([posted_ok()])

        run_cycle(enrolment)

        assert f"posted {REPO}: unaddressed, no BOARD_AGENT_LABEL" in emitted

    def test_an_addressed_push_names_its_label_in_the_report_line(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment)
        _test_hooks.run_process = FakeGh(
            {
                runs_argv(REPO, SHA): _runs_reply(_run_entry()),
                jobs_argv(REPO, RUN_ID): _jobs_reply(),
            }
        )
        _test_hooks.http_post = FakeHttpPost([posted_ok()])

        run_cycle(enrolment)

        assert f"posted {REPO}: tagged @{AGENT}" in emitted


class TestSilenceIsAnnounced:
    def test_a_push_that_never_produced_a_run_is_announced_and_closed(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """The state that makes this a monitor rather than a nicety. No jobs
        call is made, because there is no run to ask about."""
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment, ago=NO_RUN_SECONDS + 1)
        gh = FakeGh({runs_argv(REPO, SHA): _runs_reply()})
        _test_hooks.run_process = gh
        poster = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = poster

        run_cycle(enrolment)

        assert "NO RUN EVER APPEARED" in str(sent_arguments(poster.bodies[0])["body"])
        assert read_announced(position_path(enrolment)) == {attempt_key(REPO, SHA)}
        assert gh.calls == [runs_argv(REPO, SHA)]


class TestNothingIsCaught:
    def test_a_gh_that_cannot_answer_ends_the_cycle_and_records_no_position(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """A bridge that swallowed this would write its position anyway and
        never announce that work again."""
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment)
        _test_hooks.run_process = FakeGh(
            {runs_argv(REPO, SHA): FakeCompleted(returncode=1, stderr="gh: command not found")}
        )
        _test_hooks.http_post = FakeHttpPost([])

        with pytest.raises(AppError) as caught:
            run_cycle(enrolment)

        assert caught.value.code is CiWakeErrorCode.GH_COMMAND_FAILED
        assert read_announced(position_path(enrolment)) == frozenset()

    def test_missing_credentials_refuse_before_anything_is_asked(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env({})
        gh = FakeGh({})
        _test_hooks.run_process = gh

        with pytest.raises(AppError) as caught:
            run_cycle(tmp_path / "pushes.jsonl")

        assert caught.value.code is BoardWatchErrorCode.API_KEY_MISSING
        assert gh.calls == []

    def test_a_failed_position_write_leaves_the_push_to_be_announced_again(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """POST BEFORE WRITE, and this is the test that can see it. The post
        landed; the mark did not; the next cycle repeats the announcement
        rather than losing it. At-least-once, with the position file as the
        mark."""
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment)
        _test_hooks.run_process = FakeGh(
            {
                runs_argv(REPO, SHA): _runs_reply(_run_entry()),
                jobs_argv(REPO, RUN_ID): _jobs_reply(),
            }
        )
        poster = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = poster

        def _unwritable(path: pathlib.Path, line: str) -> None:
            raise OSError("the position record is on a full disk")

        _test_hooks.append_text = _unwritable

        with pytest.raises(OSError, match="full disk"):
            run_cycle(enrolment)

        assert len(poster.bodies) == 1
        _test_hooks.reset_hooks()
        assert read_announced(position_path(enrolment)) == frozenset()


class TestReEnrolment:
    def test_a_re_pushed_sha_is_asked_about_once_under_its_newest_label(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """The session waiting on the verdict is the one that pushed last,
        and the sha is queried once however many times it was enrolled."""
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment, agent="opus-first-0909", ago=7200)
        _enrol(enrolment, agent="opus-second-0909", ago=3600)
        gh = FakeGh(
            {
                runs_argv(REPO, SHA): _runs_reply(_run_entry()),
                jobs_argv(REPO, RUN_ID): _jobs_reply(),
            }
        )
        _test_hooks.run_process = gh
        poster = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = poster

        run_cycle(enrolment)

        assert gh.calls.count(runs_argv(REPO, SHA)) == 1
        assert "@opus-second-0909" in str(sent_arguments(poster.bodies[0])["body"])
        assert emitted[-1] == ("cycle: 1 enrolled, 1 outstanding, 1 announced, positions recorded")
