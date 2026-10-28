"""The WRDS inventories' own board reports and the identity that writes them."""

from __future__ import annotations

import pytest
from platform_core.board import mint_service_session_id
from platform_core.errors import AppError
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments

from hpc3.cli import _test_hooks as cli_hooks
from hpc3.cli._wrds_report import HARNESS, post_load_report, report_identity
from tests.wrds_board import ENV_TEXT, TASK, TENANT, accepting_board, bind_stack_files


def test_identity_is_derived_from_the_task_and_stable() -> None:
    identity = report_identity(TASK)
    assert identity == {
        "agent": "load-report-c4522545",
        "session_id": mint_service_session_id("corvis:hpc3:wrds-inventory:load-report-c4522545"),
        "cwd": "service://hpc3-wrds-inventory",
    }
    assert report_identity(TASK) == identity
    other = report_identity("0b80f22f-177b-4d03-8a90-0c5c6e6b189c")
    assert other["agent"] == "load-report-0b80f22f"
    assert other["session_id"] != identity["session_id"]


@pytest.mark.parametrize("task", ["c4522545", TASK.upper(), f"task:{TASK}", f"{TASK}0", ""])
def test_identity_refuses_anything_but_a_full_task_id(task: str) -> None:
    with pytest.raises(ValueError, match="WRDS_REPORT_TASK"):
        report_identity(task)


def test_report_registers_the_identity_then_posts_on_the_task() -> None:
    post = accepting_board(2)
    post_load_report(TASK, "WRDS SQL LOAD-REPORT body")
    assert [headers["X-Tenant-Id"] for headers in post.headers] == [TENANT, TENANT]
    assert [headers["x-api-key"] for headers in post.headers] == ["internal-key"] * 2
    checkin, note = (sent_arguments(body) for body in post.bodies)
    identity = report_identity(TASK)
    assert checkin == {
        "room": "main",
        "kind": "checkin",
        "harness": HARNESS,
        "body": (
            "load-report-c4522545 checking in: posts the WRDS inventories' measured "
            "counts and experiment attempts' measured outcomes on their task"
        ),
        "agent": identity["agent"],
        "sessionId": identity["session_id"],
        "cwd": identity["cwd"],
    }
    assert note == {
        "room": "main",
        "taskId": TASK,
        "kind": "note",
        "body": "WRDS SQL LOAD-REPORT body",
        "agent": identity["agent"],
        "sessionId": identity["session_id"],
        "cwd": identity["cwd"],
    }


def test_a_refused_post_fails_the_report() -> None:
    bind_stack_files(ENV_TEXT)
    refusal = McpHttpResponse(status=401, body="bad key", content_type=EVENT_STREAM_MEDIA_TYPE)
    post = FakeHttpPost([posted_ok(), refusal])
    cli_hooks.http_post = post
    with pytest.raises(AppError, match="HTTP 401"):
        post_load_report(TASK, "body")
    assert len(post.bodies) == 2


def test_missing_board_credentials_fail_before_any_post() -> None:
    bind_stack_files("OTHER=1\n")
    post = FakeHttpPost([])
    cli_hooks.http_post = post
    with pytest.raises(AppError, match="MCP_INTERNAL_KEY"):
        post_load_report(TASK, "body")
    assert post.bodies == []
