"""A measuring program's own board report and the identity that writes it."""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Literal

import pytest

from platform_core.board import mint_service_session_id
from platform_core.errors import AppError
from platform_core.load_report import post_load_report, report_identity
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments, stack_endpoint_files
from platform_core.session_label import API_KEY_NAME, STACK_ENV_PATH, TENANT_ID_NAME

TASK = "c4522545-5bfc-449d-8863-87ed1d87e5b1"
TENANT = "2e137b5f-0000-4000-8000-0000aa"
ENV_TEXT = f"{API_KEY_NAME}=internal-key\n{TENANT_ID_NAME}={TENANT}\n"


def _stack(env_text: str) -> Callable[[pathlib.Path], str]:
    """Answer the stack's ``.env`` and the files the taskboard's url is formed from."""
    files = stack_endpoint_files()
    files[STACK_ENV_PATH] = env_text
    return files.__getitem__


def test_identity_is_derived_from_the_task_and_stable() -> None:
    identity = report_identity(TASK, "wrds")
    assert identity == {
        "agent": "load-report-c4522545",
        "session_id": mint_service_session_id("corvis:hpc3:wrds-inventory:load-report-c4522545"),
        "cwd": "service://hpc3-wrds-inventory",
    }
    assert report_identity(TASK, "wrds") == identity
    other = report_identity("0b80f22f-177b-4d03-8a90-0c5c6e6b189c", "wrds")
    assert other["agent"] == "load-report-0b80f22f"
    assert other["session_id"] != identity["session_id"]


def test_the_wrds_identity_is_the_one_that_already_posted_on_the_board() -> None:
    identity = report_identity("1f60a20d-d860-43b3-9616-68f192bcedd9", "wrds")
    assert identity["agent"] == "load-report-1f60a20d"
    assert identity["session_id"] == mint_service_session_id(
        "corvis:hpc3:wrds-inventory:load-report-1f60a20d"
    )


def test_the_hpc_identity_is_a_separate_writer() -> None:
    assert report_identity(TASK, "hpc") == {
        "agent": "hpc-report-c4522545",
        "session_id": mint_service_session_id("corvis:hpc3:hpc-evidence:hpc-report-c4522545"),
        "cwd": "service://hpc3-hpc-evidence",
    }


@pytest.mark.parametrize("task", ["c4522545", TASK.upper(), f"task:{TASK}", f"{TASK}0", ""])
def test_identity_refuses_anything_but_a_full_task_id(task: str) -> None:
    with pytest.raises(ValueError, match="WRDS_REPORT_TASK"):
        report_identity(task, "wrds")


@pytest.mark.parametrize(
    ("family", "harness", "purpose"),
    [
        (
            "wrds",
            "hpc3-wrds-inventory",
            "posts the WRDS inventories' measured counts, experiment attempts' measured "
            "outcomes and evaluator verifications on their task",
        ),
        (
            "hpc",
            "hpc3-hpc-evidence",
            "posts native HPC3 accounting, verified artifacts and replay comparisons",
        ),
    ],
)
def test_report_registers_the_identity_then_posts_on_the_task(
    family: Literal["wrds", "hpc"], harness: str, purpose: str
) -> None:
    post = FakeHttpPost([posted_ok(), posted_ok()])
    post_load_report(
        post=post, read_text=_stack(ENV_TEXT), task_id=TASK, body="LOAD-REPORT body", family=family
    )
    assert [headers["X-Tenant-Id"] for headers in post.headers] == [TENANT, TENANT]
    assert [headers["x-api-key"] for headers in post.headers] == ["internal-key"] * 2
    checkin, note = (sent_arguments(body) for body in post.bodies)
    identity = report_identity(TASK, family)
    assert checkin == {
        "room": "main",
        "kind": "checkin",
        "harness": harness,
        "body": f"{identity['agent']} checking in: {purpose}",
        "agent": identity["agent"],
        "sessionId": identity["session_id"],
        "cwd": identity["cwd"],
    }
    assert note == {
        "room": "main",
        "taskId": TASK,
        "kind": "note",
        "body": "LOAD-REPORT body",
        "agent": identity["agent"],
        "sessionId": identity["session_id"],
        "cwd": identity["cwd"],
    }


def test_a_refused_post_fails_the_report() -> None:
    refusal = McpHttpResponse(status=401, body="bad key", content_type=EVENT_STREAM_MEDIA_TYPE)
    post = FakeHttpPost([posted_ok(), refusal])
    with pytest.raises(AppError, match="HTTP 401"):
        post_load_report(
            post=post, read_text=_stack(ENV_TEXT), task_id=TASK, body="body", family="wrds"
        )
    assert len(post.bodies) == 2


def test_missing_board_credentials_fail_before_any_post() -> None:
    post = FakeHttpPost([])
    with pytest.raises(AppError, match="MCP_INTERNAL_KEY"):
        post_load_report(
            post=post, read_text=_stack("OTHER=1\n"), task_id=TASK, body="body", family="wrds"
        )
    assert post.bodies == []


def test_a_malformed_task_fails_before_any_post() -> None:
    post = FakeHttpPost([])
    with pytest.raises(ValueError, match="WRDS_REPORT_TASK"):
        post_load_report(
            post=post, read_text=_stack(ENV_TEXT), task_id="c4522545", body="b", family="wrds"
        )
    assert post.bodies == []
