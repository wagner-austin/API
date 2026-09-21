"""Resolving the acting session's label against the board: the credentials
read, the ``task_whereis`` call and its parse, and the rule that fills an
unset label and refuses a wrong one (MCPs board task 3843d29f)."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest

from platform_core.error_codes_tooling import McpClientErrorCode, SessionLabelErrorCode
from platform_core.errors import AppError
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, sent_arguments, tool_text_body
from platform_core.session_label import (
    API_KEY_NAME,
    LABEL_VARIABLE,
    STACK_ENV_PATH,
    TASKBOARD_URL,
    TENANT_ID_NAME,
    WHEREIS_TOOL,
    SessionIdentity,
    bound_label,
    disagrees,
    mismatch,
    read_identity,
    require_session_id,
    resolve_label,
    resolved_label,
    stack_credentials,
)

SESSION = "4f8a2c1e-9b3d-4e7f-8a6b-1c2d3e4f5a6b"
BOUND = "opus-rebuild-deadlock-0910"
WRONG = "opus-dashboard-0911"
ENV_TEXT = (
    "# the stack\n"
    "POSTGRES_PASSWORD='pg'\n"
    f'{API_KEY_NAME}="internal-key"\n'
    "not a pair\n"
    f"{TENANT_ID_NAME}=2e137b5f-0000-4000-8000-000000000000\n"
)

#: ``task_whereis`` for a session the board has written under a label, as
#: taskboard-mcp renders it (the em dash and the double-space column).
WHEREIS_BOUND = (
    f"WHEREIS session {SESSION}\n\n"
    f"session   {SESSION}\n"
    f"board     {BOUND} — 2 posts, last post 6m ago, C:\\Users\\Test\\PROJECTS\\MCPs\n"
    f"name      {BOUND} (user; informational — NOT the address)\n"
    "[showing all 1 sessions]"
)

#: The same session observed by the ledger but never written under a label.
WHEREIS_UNLABELLED = (
    f"WHEREIS session {SESSION}\n\n"
    f"session   {SESSION}\n"
    "board     never written to this board (no label)\n"
    "[showing all 1 sessions]"
)

#: A session nothing knows.
WHEREIS_MISS = f"WHEREIS: nothing on this board or in the ledger knows session {SESSION}."


def reply(text: str) -> McpHttpResponse:
    return McpHttpResponse(
        status=200, body=tool_text_body(text), content_type=EVENT_STREAM_MEDIA_TYPE
    )


def env_reader(text: str = ENV_TEXT) -> tuple[list[pathlib.Path], Callable[[pathlib.Path], str]]:
    """A file seam answering ``text`` and recording what was asked for."""
    asked: list[pathlib.Path] = []

    def read_text(path: pathlib.Path) -> str:
        asked.append(path)
        return text

    return asked, read_text


def identity(session_id: str, exported: str, bound: str) -> SessionIdentity:
    return SessionIdentity(session_id=session_id, exported=exported, bound=bound)


# ---------------------------------------------------------------------------
# require_session_id
# ---------------------------------------------------------------------------


def test_a_session_id_as_the_harness_writes_it_passes() -> None:
    assert require_session_id(SESSION) == SESSION


@pytest.mark.parametrize("value", ["not-a-uuid", SESSION.upper(), SESSION[:-1], " " + SESSION])
def test_a_session_id_the_harness_could_not_have_written_is_refused(value: str) -> None:
    with pytest.raises(AppError) as caught:
        require_session_id(value)
    assert caught.value.code is SessionLabelErrorCode.SESSION_ID_MALFORMED
    assert repr(value) in caught.value.message


# ---------------------------------------------------------------------------
# stack_credentials
# ---------------------------------------------------------------------------


def test_the_credentials_are_the_two_keys_unquoted_bound_to_the_loopback_url() -> None:
    credentials = stack_credentials(ENV_TEXT)
    assert credentials["url"] == TASKBOARD_URL
    assert credentials["api_key"] == "internal-key"
    assert credentials["tenant_id"] == "2e137b5f-0000-4000-8000-000000000000"


@pytest.mark.parametrize(
    ("text", "missing"),
    [
        (f"{TENANT_ID_NAME}=t\n", API_KEY_NAME),
        (f"{API_KEY_NAME}=k\n{TENANT_ID_NAME}=\n", TENANT_ID_NAME),
        (f"{API_KEY_NAME}=''\n{TENANT_ID_NAME}=t\n", API_KEY_NAME),
        ("", API_KEY_NAME),
    ],
)
def test_a_missing_or_empty_key_is_refused_naming_it_and_the_file(text: str, missing: str) -> None:
    with pytest.raises(AppError) as caught:
        stack_credentials(text)
    assert caught.value.code is SessionLabelErrorCode.CREDENTIALS_MISSING
    assert f"carries no {missing}" in caught.value.message
    assert str(STACK_ENV_PATH) in caught.value.message


# ---------------------------------------------------------------------------
# bound_label
# ---------------------------------------------------------------------------


def test_the_board_line_names_the_binding() -> None:
    assert bound_label(WHEREIS_BOUND) == BOUND


@pytest.mark.parametrize("text", [WHEREIS_UNLABELLED, WHEREIS_MISS, ""])
def test_an_unwritten_or_unknown_session_has_no_binding(text: str) -> None:
    assert bound_label(text) == ""


# ---------------------------------------------------------------------------
# read_identity
# ---------------------------------------------------------------------------


def test_outside_a_session_nothing_is_read_and_the_export_is_kept() -> None:
    asked, read_text = env_reader()
    post = FakeHttpPost([])
    assert read_identity(
        session_id=None, exported="fleet-agent-0917", read_text=read_text, post=post
    ) == identity("", "fleet-agent-0917", "")
    assert read_identity(session_id=None, exported=None, read_text=read_text, post=post) == (
        identity("", "", "")
    )
    assert asked == []
    assert post.urls == []


def test_inside_a_session_the_board_is_asked_with_the_stacks_credentials() -> None:
    asked, read_text = env_reader()
    post = FakeHttpPost([reply(WHEREIS_BOUND)])
    assert read_identity(
        session_id=SESSION, exported=None, read_text=read_text, post=post
    ) == identity(SESSION, "", BOUND)
    assert asked == [STACK_ENV_PATH]
    assert post.urls == [TASKBOARD_URL]
    assert post.headers[0]["x-api-key"] == "internal-key"
    assert post.headers[0]["X-Tenant-Id"] == "2e137b5f-0000-4000-8000-000000000000"
    assert sent_arguments(post.bodies[0]) == {"session": SESSION}
    envelope = post.bodies[0].decode("utf-8")
    assert f'"name":"{WHEREIS_TOOL}"' in envelope


def test_a_session_the_board_never_saw_reads_as_unbound() -> None:
    _, read_text = env_reader()
    post = FakeHttpPost([reply(WHEREIS_MISS)])
    assert read_identity(
        session_id=SESSION, exported=WRONG, read_text=read_text, post=post
    ) == identity(SESSION, WRONG, "")


def test_a_board_that_cannot_be_asked_raises_rather_than_reading_as_unbound() -> None:
    _, read_text = env_reader()
    post = FakeHttpPost(
        [McpHttpResponse(status=401, body="unauthorized", content_type="text/plain")]
    )
    with pytest.raises(AppError) as caught:
        read_identity(session_id=SESSION, exported=None, read_text=read_text, post=post)
    assert caught.value.code is McpClientErrorCode.HTTP_STATUS


def test_a_malformed_session_id_is_refused_before_the_credentials_are_read() -> None:
    asked, read_text = env_reader()
    post = FakeHttpPost([])
    with pytest.raises(AppError) as caught:
        read_identity(session_id="nope", exported=None, read_text=read_text, post=post)
    assert caught.value.code is SessionLabelErrorCode.SESSION_ID_MALFORMED
    assert asked == []


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (identity(SESSION, "", BOUND), BOUND),
        (identity(SESSION, BOUND, BOUND), BOUND),
        (identity(SESSION, WRONG, ""), WRONG),
        (identity("", "fleet-agent-0917", ""), "fleet-agent-0917"),
        (identity(SESSION, "", ""), ""),
        (identity("", "", ""), ""),
    ],
)
def test_resolved_label_prefers_the_binding_then_the_export_then_nothing(
    given: SessionIdentity, expected: str
) -> None:
    assert not disagrees(given)
    assert resolved_label(given) == expected


def test_only_a_binding_beside_a_different_export_disagrees() -> None:
    assert disagrees(identity(SESSION, WRONG, BOUND))
    assert not disagrees(identity(SESSION, "", BOUND))
    assert not disagrees(identity(SESSION, WRONG, ""))


def test_the_mismatch_names_both_labels_and_the_session() -> None:
    error = mismatch(identity(SESSION, WRONG, BOUND))
    assert error.code == SessionLabelErrorCode.LABEL_MISMATCH
    assert f"session {SESSION} writes to the board as '{BOUND}'" in error.message
    assert f"cannot act as '{WRONG}'" in error.message
    assert LABEL_VARIABLE in error.message


# ---------------------------------------------------------------------------
# resolve_label, the one call an enrolling site makes
# ---------------------------------------------------------------------------


def test_the_20_38z_case_is_refused_at_resolution() -> None:
    _, read_text = env_reader()
    post = FakeHttpPost([reply(WHEREIS_BOUND)])
    with pytest.raises(AppError) as caught:
        resolve_label(session_id=SESSION, exported=WRONG, read_text=read_text, post=post)
    assert caught.value.code is SessionLabelErrorCode.LABEL_MISMATCH
    assert BOUND in caught.value.message and WRONG in caught.value.message


def test_an_unset_label_is_filled_from_the_binding() -> None:
    _, read_text = env_reader()
    post = FakeHttpPost([reply(WHEREIS_BOUND)])
    assert resolve_label(session_id=SESSION, exported=None, read_text=read_text, post=post) == BOUND


def test_an_export_that_agrees_passes_and_a_terminal_keeps_its_own() -> None:
    _, read_text = env_reader()
    post = FakeHttpPost([reply(WHEREIS_BOUND)])
    agreed = resolve_label(session_id=SESSION, exported=BOUND, read_text=read_text, post=post)
    assert agreed == BOUND
    assert resolve_label(session_id=None, exported=None, read_text=read_text, post=post) == ""
