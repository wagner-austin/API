"""The two commands: one enrols a push, one announces verdicts.

``ci-wake-enrol`` IS THE HALF THAT RUNS INSIDE ``pre-push``, so its refusals
are a push failing. That is deliberate and it is tested here rather than
described: a bad row enrolled now is a post the board refuses on every later
cycle, wedging every other session's announcement behind it.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.error_codes_tooling import (
    CiWakeErrorCode,
    McpClientErrorCode,
    SessionLabelErrorCode,
)
from platform_core.errors import AppError
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, sent_arguments, tool_text_body
from platform_core.session_label import (
    API_KEY_NAME,
    LABEL_VARIABLE,
    SESSION_ID_VARIABLE,
    STACK_ENV_PATH,
    TASKBOARD_URL,
    TENANT_ID_NAME,
)

from ci_wake import _test_hooks
from ci_wake.cli import enrol as enrol_cli
from ci_wake.cli import wake as wake_cli
from ci_wake.enrolment import read_attempts
from tests.conftest import AGENT, CONFIGURED_ENV, FROZEN_NOW, REPO, SHA, FakeGh, pin_env

#: The session of the 20:38Z case (MCPs board task 3843d29f).
SESSION = "4f8a2c1e-9b3d-4e7f-8a6b-1c2d3e4f5a6b"
BOUND = "opus-rebuild-deadlock-0910"
WRONG = "opus-dashboard-0911"


def _whereis(label: str) -> McpHttpResponse:
    """``task_whereis`` answering for :data:`SESSION`, bound to ``label``
    or, for the empty string, never written to the board."""
    board = (
        f"board     {label} — 2 posts, last post 6m ago, C:\\x\n"
        if label != ""
        else "board     never written to this board (no label)\n"
    )
    text = f"WHEREIS session {SESSION}\n\nsession   {SESSION}\n{board}[showing all 1 sessions]"
    return McpHttpResponse(
        status=200, body=tool_text_body(text), content_type=EVENT_STREAM_MEDIA_TYPE
    )


def _in_session(label: str, exported: str | None) -> tuple[FakeHttpPost, list[pathlib.Path]]:
    """Put the enrolment inside a session: the harness's id in the
    environment, the stack's ``.env`` on disk, and a board answering for
    the session. Returns the poster and the files the enrolment read."""
    pin_env(
        {SESSION_ID_VARIABLE: SESSION} | ({} if exported is None else {LABEL_VARIABLE: exported})
    )
    read: list[pathlib.Path] = []
    # The same seam reads the enrolment record back (``read_attempts``),
    # so only the stack's ``.env`` is answered here and every other path
    # goes to the real file the enrolment wrote under ``tmp_path``.
    real_read_text = _test_hooks.read_text

    def _read_text(path: pathlib.Path) -> str:
        if path != STACK_ENV_PATH:
            return real_read_text(path)
        read.append(path)
        return f"{API_KEY_NAME}=internal-key\n{TENANT_ID_NAME}=2e137b5f-0000-4000-8000-00000000aa\n"

    _test_hooks.read_text = _read_text
    post = FakeHttpPost([_whereis(label)])
    _test_hooks.http_post = post
    return post, read


def _enrol_argv(path: pathlib.Path) -> list[str]:
    """Build a well-formed enrolment command line.

    Args:
        path: The enrolment record to append to.

    Returns:
        The tokens, excluding the program name.
    """
    return [
        enrol_cli.ENROLMENT_FLAG,
        str(path),
        enrol_cli.REPO_FLAG,
        REPO,
        enrol_cli.SHA_FLAG,
        SHA,
        enrol_cli.REF_FLAG,
        "refs/heads/main",
    ]


class TestEnrolMain:
    def test_outside_a_session_it_records_the_push_under_the_exported_label(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """No session id: nothing to check the export against, and the
        board is never asked (an unscripted POST would raise)."""
        pin_env({LABEL_VARIABLE: AGENT})
        _test_hooks.http_post = FakeHttpPost([])
        path = tmp_path / "runs" / "pushes.jsonl"

        assert enrol_cli.main(_enrol_argv(path)) == 0

        rows = read_attempts(path)
        assert len(rows) == 1
        assert rows[0]["repo"] == REPO
        assert rows[0]["sha"] == SHA
        assert rows[0]["ref"] == "refs/heads/main"
        assert rows[0]["agent"] == AGENT
        assert rows[0]["attempted_unix"] == FROZEN_NOW
        assert emitted == [f"ci-wake: enrolled {SHA[:7]} in {REPO} for @{AGENT}"]

    def test_a_push_with_no_label_is_recorded_unaddressed_rather_than_refused(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """A human pushing from a terminal has no board label. Refusing their
        push would make the hook a thing people disable."""
        pin_env({})
        _test_hooks.http_post = FakeHttpPost([])
        path = tmp_path / "pushes.jsonl"

        assert enrol_cli.main(_enrol_argv(path)) == 0

        assert read_attempts(path)[0]["agent"] == ""
        assert emitted == [
            f"ci-wake: enrolled {SHA[:7]} in {REPO} for nobody "
            f"(no ${LABEL_VARIABLE} and no board binding)"
        ]

    def test_inside_a_session_an_unset_label_is_filled_from_the_boards_binding(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """Acceptance 2: the push is addressed without any export at all."""
        post, read = _in_session(BOUND, None)
        path = tmp_path / "pushes.jsonl"

        assert enrol_cli.main(_enrol_argv(path)) == 0

        assert read_attempts(path)[0]["agent"] == BOUND
        assert emitted == [f"ci-wake: enrolled {SHA[:7]} in {REPO} for @{BOUND}"]
        assert read == [STACK_ENV_PATH]
        assert post.urls == [TASKBOARD_URL]
        assert post.headers[0]["x-api-key"] == "internal-key"
        assert sent_arguments(post.bodies[0]) == {"session": SESSION}

    def test_inside_a_session_a_label_other_than_the_boards_binding_fails_the_push(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """Acceptance 1, the 20:38Z case at this writer: refused naming both,
        and nothing is enrolled."""
        _in_session(BOUND, WRONG)
        path = tmp_path / "pushes.jsonl"

        with pytest.raises(AppError) as caught:
            enrol_cli.main(_enrol_argv(path))

        assert caught.value.code is SessionLabelErrorCode.LABEL_MISMATCH
        assert BOUND in caught.value.message and WRONG in caught.value.message
        assert read_attempts(path) == ()
        assert emitted == []

    def test_inside_a_session_an_export_that_agrees_with_the_binding_is_recorded(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        _in_session(BOUND, BOUND)
        path = tmp_path / "pushes.jsonl"

        assert enrol_cli.main(_enrol_argv(path)) == 0

        assert read_attempts(path)[0]["agent"] == BOUND

    def test_a_session_the_board_never_saw_keeps_its_exported_label(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """No binding to disagree with: the export is what the board would
        bind on the session's first write."""
        _in_session("", AGENT)
        path = tmp_path / "pushes.jsonl"

        assert enrol_cli.main(_enrol_argv(path)) == 0

        assert read_attempts(path)[0]["agent"] == AGENT

    def test_a_board_that_does_not_answer_fails_the_push_rather_than_enrolling_blind(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env({SESSION_ID_VARIABLE: SESSION, LABEL_VARIABLE: BOUND})

        def _read_text(path: pathlib.Path) -> str:
            return f"{API_KEY_NAME}=k\n{TENANT_ID_NAME}=t\n"

        _test_hooks.read_text = _read_text
        _test_hooks.http_post = FakeHttpPost(
            [McpHttpResponse(status=401, body="unauthorized", content_type="text/plain")]
        )
        path = tmp_path / "pushes.jsonl"

        with pytest.raises(AppError) as caught:
            enrol_cli.main(_enrol_argv(path))

        assert caught.value.code is McpClientErrorCode.HTTP_STATUS
        assert read_attempts(path) == ()

    def test_a_malformed_label_fails_the_push(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env({LABEL_VARIABLE: "Opus_CI_0909"})

        with pytest.raises(AppError) as caught:
            enrol_cli.main(_enrol_argv(tmp_path / "pushes.jsonl"))

        assert caught.value.code is CiWakeErrorCode.ENROLMENT_FIELD_MALFORMED

    def test_an_abbreviated_sha_fails_the_push(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env({})
        argv = _enrol_argv(tmp_path / "pushes.jsonl")
        argv[argv.index(enrol_cli.SHA_FLAG) + 1] = SHA[:7]

        with pytest.raises(AppError) as caught:
            enrol_cli.main(argv)

        assert caught.value.code is CiWakeErrorCode.ENROLMENT_FIELD_MALFORMED

    def test_a_local_remote_alias_instead_of_owner_name_fails_the_push(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """Two machines can name one remote differently and the Actions API
        cannot, so the hook must resolve the address rather than pass its
        own nickname for it."""
        pin_env({})
        argv = _enrol_argv(tmp_path / "pushes.jsonl")
        argv[argv.index(enrol_cli.REPO_FLAG) + 1] = "origin"

        with pytest.raises(AppError):
            enrol_cli.main(argv)

    def test_a_missing_flag_refuses(self, tmp_path: pathlib.Path) -> None:
        pin_env({})

        with pytest.raises(ValueError, match="--ref"):
            enrol_cli.main(
                [
                    enrol_cli.ENROLMENT_FLAG,
                    str(tmp_path / "pushes.jsonl"),
                    enrol_cli.REPO_FLAG,
                    REPO,
                    enrol_cli.SHA_FLAG,
                    SHA,
                ]
            )

    def test_an_unknown_flag_refuses_rather_than_being_ignored(self) -> None:
        with pytest.raises(ValueError, match="unknown argument"):
            enrol_cli.main(["--branch", "main"])


class TestEnrolEntrypoint:
    def test_it_exits_with_mains_status(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env({})
        original = sys.argv
        sys.argv = ["ci-wake-enrol", *_enrol_argv(tmp_path / "pushes.jsonl")]

        try:
            with pytest.raises(SystemExit) as caught:
                enrol_cli.entrypoint()
        finally:
            sys.argv = original

        assert caught.value.code == 0

    def test_running_as_a_module_actually_enrols(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """Without the ``__main__`` guard, ``python -m ci_wake.cli.enrol``
        imports the module, runs nothing and exits 0 -- which the hook would
        read as a successful enrolment."""
        pin_env({})
        path = tmp_path / "pushes.jsonl"
        original = sys.argv
        sys.argv = ["ci-wake-enrol", *_enrol_argv(path)]

        try:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("ci_wake.cli.enrol", run_name="__main__")
        finally:
            sys.argv = original

        assert caught.value.code == 0
        assert len(read_attempts(path)) == 1


class TestWakeMain:
    def test_a_cycle_runs_against_the_named_record(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.run_process = FakeGh({})
        _test_hooks.http_post = FakeHttpPost([])

        assert wake_cli.main([wake_cli.ENROLMENT_FLAG, str(tmp_path / "pushes.jsonl")]) == 0

        assert emitted == ["enrolment record is empty; nothing has been pushed from this machine"]

    def test_a_missing_enrolment_flag_refuses(self) -> None:
        """There is no default record. A bridge that guessed one would
        announce nothing while every push was enrolled correctly, which
        reads exactly like a quiet week."""
        with pytest.raises(ValueError, match="--enrolment"):
            wake_cli.main([])

    def test_an_unknown_flag_refuses_rather_than_being_ignored(self) -> None:
        with pytest.raises(ValueError, match="unknown argument"):
            wake_cli.main(["--follow"])


class TestWakeEntrypoint:
    def test_it_exits_with_mains_status(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.run_process = FakeGh({})
        _test_hooks.http_post = FakeHttpPost([])
        original = sys.argv
        sys.argv = ["ci-wake", wake_cli.ENROLMENT_FLAG, str(tmp_path / "pushes.jsonl")]

        try:
            with pytest.raises(SystemExit) as caught:
                wake_cli.entrypoint()
        finally:
            sys.argv = original

        assert caught.value.code == 0

    def test_running_as_a_module_actually_runs(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.run_process = FakeGh({})
        _test_hooks.http_post = FakeHttpPost([])
        original = sys.argv
        sys.argv = ["ci-wake", wake_cli.ENROLMENT_FLAG, str(tmp_path / "pushes.jsonl")]

        try:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("ci_wake.cli.wake", run_name="__main__")
        finally:
            sys.argv = original

        assert caught.value.code == 0
        assert emitted == ["enrolment record is empty; nothing has been pushed from this machine"]
