"""A push that never landed is closed without a post; a landed one with no run is not.

MCPs board task be748588. On 2026-10-16 the bridge posted NO RUN EVER
APPEARED for d357385da52f, 1091c36a200b and 868ae9417 on MCPs main, each a
worktree-publish attempt refused before or at the remote while its retry
landed. These run whole cycles against the real record writers: the
position file is read back as the package's own reader decodes it.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import JSONObject, dump_json_str, load_json_str
from platform_core.mcp_testing import FakeHttpPost, announcing_poster, notes_sent

from ci_wake import _test_hooks
from ci_wake.coverage import compare_argv
from ci_wake.cycle import run_cycle
from ci_wake.enrolment import PushAttempt, append_attempt, attempt_key
from ci_wake.landing import object_argv
from ci_wake.position import decode_announced_push, position_path
from ci_wake.runs import runs_argv
from ci_wake.verdicts import NO_RUN_SECONDS
from tests.conftest import (
    AGENT,
    CONFIGURED_ENV,
    FROZEN_NOW,
    OTHER_SHA,
    REPO,
    SHA,
    FakeCompleted,
    FakeGh,
    compare_reply,
    object_reply,
    pin_env,
)


def _enrol(path: pathlib.Path, *, sha: str = SHA, ref: str = "refs/heads/main") -> None:
    """Write one enrolment row, old enough to be abandoned, through the package's writer."""
    append_attempt(
        path,
        PushAttempt(
            repo=REPO, sha=sha, ref=ref, agent=AGENT, attempted_unix=FROZEN_NOW - NO_RUN_SECONDS - 1
        ),
    )


def _no_runs() -> FakeCompleted:
    """An ``actions/runs`` answer listing nothing."""
    listing: JSONObject = {"total_count": 0, "workflow_runs": []}
    return FakeCompleted(stdout=dump_json_str(listing))


def _closed_states(enrolment: pathlib.Path) -> dict[str, str]:
    """Every closed row's key and state, as the position record holds them."""
    text = position_path(enrolment).read_text(encoding="utf-8")
    rows = [decode_announced_push(load_json_str(line)) for line in text.splitlines()]
    return {row["key"]: row["state"] for row in rows}


class TestAPushThatNeverLandedIsClosedUnannounced:
    def test_a_sha_github_never_received_posts_nothing_and_registers_nothing(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """d357385da52f's shape: pre-push refused it, so GitHub never got it.
        A poster with no replies raises on any call, checkin included."""
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment)
        _test_hooks.run_process = FakeGh(
            {runs_argv(REPO, SHA): _no_runs(), object_argv(REPO, SHA): object_reply(held=False)}
        )
        _test_hooks.http_post = FakeHttpPost([])

        run_cycle(enrolment)

        assert _closed_states(enrolment) == {attempt_key(REPO, SHA): "unlanded"}
        assert emitted == [
            f"unlanded {REPO}@{SHA[:12]}: refs/heads/main never held it (GitHub never received "
            "it), so no run was due; closed unannounced",
            "cycle: 1 enrolled, 1 outstanding, 0 announced, 1 closed unannounced as never "
            "landed, positions recorded",
        ]

    def test_a_sha_the_remote_refused_after_storing_it_is_closed_the_same_way(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """1091c36a200b's shape: GitHub holds the object, main diverged from it."""
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment)
        _test_hooks.run_process = FakeGh(
            {
                runs_argv(REPO, SHA): _no_runs(),
                object_argv(REPO, SHA): object_reply(held=True),
                compare_argv(REPO, SHA, "main"): compare_reply("diverged"),
            }
        )
        _test_hooks.http_post = FakeHttpPost([])

        run_cycle(enrolment)

        assert _closed_states(enrolment) == {attempt_key(REPO, SHA): "unlanded"}
        assert "compare of main against it reads diverged" in emitted[0]


class TestOnlyTheLandedPushIsAnnounced:
    def test_a_landed_silent_push_posts_beside_an_unlanded_one_that_does_not(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment, sha=SHA)
        _enrol(enrolment, sha=OTHER_SHA)
        _test_hooks.run_process = FakeGh(
            {
                runs_argv(REPO, SHA): _no_runs(),
                object_argv(REPO, SHA): object_reply(held=True),
                compare_argv(REPO, SHA, "main"): compare_reply("identical"),
                runs_argv(REPO, OTHER_SHA): _no_runs(),
                object_argv(REPO, OTHER_SHA): object_reply(held=False),
            }
        )
        poster = announcing_poster()
        _test_hooks.http_post = poster

        run_cycle(enrolment)

        notes = notes_sent(poster)
        assert len(notes) == 1
        assert "NO RUN EVER APPEARED" in str(notes[0]["body"])
        assert OTHER_SHA[:7] not in str(notes[0]["body"])
        assert _closed_states(enrolment) == {
            attempt_key(REPO, SHA): "abandoned",
            attempt_key(REPO, OTHER_SHA): "unlanded",
        }
        assert emitted[-1] == (
            "cycle: 2 enrolled, 2 outstanding, 1 announced, 1 closed unannounced as never "
            "landed, positions recorded"
        )

    def test_a_tag_has_no_branch_to_ask_and_keeps_its_announcement(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        enrolment = tmp_path / "pushes.jsonl"
        _enrol(enrolment, ref="refs/tags/v1")
        gh = FakeGh({runs_argv(REPO, SHA): _no_runs()})
        _test_hooks.run_process = gh
        poster = announcing_poster()
        _test_hooks.http_post = poster

        run_cycle(enrolment)

        assert "NO RUN EVER APPEARED" in str(notes_sent(poster)[0]["body"])
        assert gh.calls == [runs_argv(REPO, SHA)]
        assert _closed_states(enrolment) == {attempt_key(REPO, SHA): "abandoned"}
