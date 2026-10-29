"""Session-host input uses the existing delivery contract and real SSH failures."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError, require_str
from platform_core.mcp_client import McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, announcing_poster, notes_sent, posted_ok

from ci_wake import _test_hooks
from ci_wake.cli.wake import main
from ci_wake.enrolment import attempt_key
from ci_wake.position import position_path, read_announced
from ci_wake.remote import enrolment_argv, read_remote_attempts
from ci_wake.runs import jobs_argv, runs_argv
from tests.conftest import (
    AGENT,
    CONFIGURED_ENV,
    OTHER_SHA,
    REPO,
    SHA,
    FakeCompleted,
    FakeGh,
    pin_env,
)
from tests.test_cycle import RUN_ID, _enrol, _jobs_reply, _run_entry, _runs_reply


@pytest.mark.parametrize("host", ["", "-oProxyCommand=false", "host;echo", "user@host"])
def test_host_cannot_supply_options_or_shell_commands(host: str, tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="CI_WAKE_ENROLMENT_SOURCE_INVALID"):
        enrolment_argv(host, tmp_path / "pushes.jsonl")


def test_remote_path_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="CI_WAKE_ENROLMENT_SOURCE_INVALID"):
        enrolment_argv("ziggurat", pathlib.Path("runs/pushes.jsonl"))


def test_shell_quotes_the_path_as_one_literal_argument() -> None:
    assert enrolment_argv("ziggurat", pathlib.Path("/tmp/a b/$(echo bad).jsonl")) == (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "ziggurat",
        "cat -- '/tmp/a b/$(echo bad).jsonl'",
    )


@pytest.mark.parametrize("code,detail", [(1, "No such file"), (255, "Connection refused")])
def test_missing_source_or_failed_transport_never_uses_local_file(
    code: int, detail: str, tmp_path: pathlib.Path, frozen_clock: int
) -> None:
    pin_env(CONFIGURED_ENV)
    path = tmp_path / "pushes.jsonl"
    _enrol(path)
    _test_hooks.run_process = FakeGh(
        {enrolment_argv("ziggurat", path): FakeCompleted(returncode=code, stderr=detail)}
    )
    poster = FakeHttpPost([])
    _test_hooks.http_post = poster

    with pytest.raises(ValueError, match=f"CI_WAKE_ENROLMENT_UNREADABLE:.*{detail}"):
        main(["--source-host", "ziggurat", "--enrolment", str(path)])

    assert read_announced(position_path(path)) == frozenset()
    assert poster.bodies == []


def test_malformed_remote_row_names_its_source(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "pushes.jsonl"
    _test_hooks.run_process = FakeGh(
        {enrolment_argv("ziggurat", path): FakeCompleted(stdout="[]\n")}
    )
    with pytest.raises(JSONTypeError, match="ziggurat:.*line 1"):
        read_remote_attempts("ziggurat", path)


def test_real_ssh_failure_propagates_without_delivery_marks(tmp_path: pathlib.Path) -> None:
    """The reserved invalid domain cannot serve a journal, regardless of local sshd."""
    pin_env(CONFIGURED_ENV)
    path = tmp_path / "pushes.jsonl"
    _test_hooks.run_process = _test_hooks._default_run_process
    poster = FakeHttpPost([])
    _test_hooks.http_post = poster

    with pytest.raises(ValueError, match="CI_WAKE_ENROLMENT_UNREADABLE:.*exit 255"):
        main(["--source-host", "ci-wake-unreachable.invalid", "--enrolment", str(path)])

    assert read_announced(position_path(path)) == frozenset()
    assert poster.bodies == []


def test_remote_delivery_retries_refused_post_then_acknowledges_only_once(
    tmp_path: pathlib.Path, frozen_clock: int, emitted: list[str]
) -> None:
    """The stale local row must never replace the new remote owner's enrollment."""
    pin_env(CONFIGURED_ENV)
    local = tmp_path / "hub" / "pushes.jsonl"
    remote = tmp_path / "session-host" / "pushes.jsonl"
    _enrol(local, sha=OTHER_SHA, agent="opus-stale-1022")
    expected = _enrol(remote)
    remote_text = remote.read_text(encoding="utf-8")
    argv = ["--source-host", "ziggurat", "--enrolment", str(local)]
    ssh = enrolment_argv("ziggurat", local)
    _test_hooks.run_process = FakeGh(
        {
            ssh: FakeCompleted(stdout=remote_text),
            runs_argv(REPO, SHA): _runs_reply(_run_entry()),
            jobs_argv(REPO, RUN_ID): _jobs_reply(),
        }
    )
    _test_hooks.http_post = FakeHttpPost(
        [
            posted_ok(),
            McpHttpResponse(status=503, body="board unavailable", content_type="text/plain"),
        ]
    )

    with pytest.raises(AppError):
        main(argv)

    assert read_announced(position_path(local)) == frozenset()
    poster = announcing_poster()
    _test_hooks.http_post = poster
    assert main(argv) == 0
    notes = notes_sent(poster)
    assert len(notes) == 1
    assert f"@{AGENT}" in require_str(notes[0], "body")
    assert read_announced(position_path(local)) == frozenset({attempt_key(REPO, SHA)})
    assert read_remote_attempts("ziggurat", local) == (expected,)
    assert remote.read_text(encoding="utf-8") == remote_text
    assert position_path(remote).exists() is False

    gh = FakeGh({ssh: FakeCompleted(stdout=remote_text)})
    _test_hooks.run_process = gh
    _test_hooks.http_post = FakeHttpPost([])
    assert main(argv) == 0
    assert gh.calls == [ssh]
    assert len(notes_sent(poster)) == 1


__all__: list[str] = []
