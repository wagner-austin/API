"""The one board write: what is sent, as whom, and how refusal surfaces."""

from __future__ import annotations

import pytest
from platform_core.error_codes import McpClientErrorCode
from platform_core.errors import AppError
from platform_core.mcp_client import McpCredentials, McpHttpResponse

from hpc_wake import _test_hooks
from hpc_wake.announce import Announcement
from hpc_wake.board import post_announcement
from hpc_wake.identity import BRIDGE_AGENT, BRIDGE_SESSION_ID
from tests.conftest import TASK_ID, FakeHttpPost, posted_ok, sent_arguments

CREDENTIALS = McpCredentials(
    url="http://127.0.0.1:8033/mcp",
    api_key="test-key",
    tenant_id="2e137b5f-0000-4000-8000-000000000000",
)

ANNOUNCEMENT = Announcement(
    submitter="label-a-0906",
    project="mi",
    body="JOB-TERMINAL mi: 1 job(s) ended (COMPLETED x1)\n@label-a-0906",
)


class TestPostAnnouncement:
    def test_the_post_carries_the_body_the_thread_and_the_bridge_identity(self) -> None:
        fake = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = fake

        post_announcement(CREDENTIALS, TASK_ID, ANNOUNCEMENT)

        assert fake.urls == ["http://127.0.0.1:8033/mcp"]
        assert fake.headers[0]["x-api-key"] == "test-key"
        assert fake.headers[0]["X-Tenant-Id"] == "2e137b5f-0000-4000-8000-000000000000"
        arguments = sent_arguments(fake.bodies[0])
        assert arguments == {
            "room": "main",
            "taskId": TASK_ID,
            "kind": "note",
            "body": ANNOUNCEMENT["body"],
            "agent": BRIDGE_AGENT,
            "sessionId": BRIDGE_SESSION_ID,
            "cwd": "service://hpc-wake",
        }

    def test_a_refused_post_raises_with_the_status_in_it(self) -> None:
        """A rotated key is the ordinary failure; it must end the cycle
        loudly, not read as a quiet board."""
        fake = FakeHttpPost(
            [McpHttpResponse(status=401, body="unauthorized", content_type="text/plain")]
        )
        _test_hooks.http_post = fake

        with pytest.raises(AppError) as caught:
            post_announcement(CREDENTIALS, TASK_ID, ANNOUNCEMENT)
        assert caught.value.code is McpClientErrorCode.HTTP_STATUS
        assert "401" in caught.value.message
