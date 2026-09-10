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
from platform_core.error_codes_tooling import CiWakeErrorCode
from platform_core.errors import AppError
from platform_core.mcp_testing import FakeHttpPost

from ci_wake import _test_hooks
from ci_wake.cli import enrol as enrol_cli
from ci_wake.cli import wake as wake_cli
from ci_wake.enrolment import AGENT_VARIABLE, read_attempts
from tests.conftest import AGENT, CONFIGURED_ENV, FROZEN_NOW, REPO, SHA, FakeGh, pin_env


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
    def test_it_records_the_push_under_the_exported_label(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env({AGENT_VARIABLE: AGENT})
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
        path = tmp_path / "pushes.jsonl"

        assert enrol_cli.main(_enrol_argv(path)) == 0

        assert read_attempts(path)[0]["agent"] == ""
        assert emitted == [
            f"ci-wake: enrolled {SHA[:7]} in {REPO} for nobody (no ${AGENT_VARIABLE})"
        ]

    def test_a_malformed_label_fails_the_push(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env({AGENT_VARIABLE: "Opus_CI_0909"})

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
