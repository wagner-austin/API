"""One MCP call: what goes on the wire, and every way the answer can fail."""

from __future__ import annotations

import pytest
from platform_core.error_codes import BoardWatchErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict

from board_watch import _test_hooks
from board_watch.client import call_tool
from tests.conftest import (
    TEST_CREDENTIALS,
    FakeHttpPost,
    ok,
    refused,
    rpc_body,
    sent_arguments,
    tool_text,
)


def test_posts_a_jsonrpc_tool_call_with_both_headers() -> None:
    """The request shape is the contract the endpoint accepts.

    Both headers matter and neither is optional: without ``x-api-key`` the
    endpoint answers 401, and without ``X-Tenant-Id`` it answers a tenant
    error from inside the tool. Both were found by making the call.
    """
    poster = FakeHttpPost([ok(tool_text("rendered"))])
    _test_hooks.http_post = poster
    assert call_tool(TEST_CREDENTIALS, "task_events", {"limit": 5}) == "rendered"

    assert poster.urls == [TEST_CREDENTIALS["url"]]
    headers = poster.headers[0]
    assert headers["x-api-key"] == TEST_CREDENTIALS["api_key"]
    assert headers["X-Tenant-Id"] == TEST_CREDENTIALS["tenant_id"]
    assert "text/event-stream" in headers["Accept"]

    envelope = narrow_json_to_dict(load_json_str(poster.bodies[0].decode("utf-8")))
    assert envelope["method"] == "tools/call"
    assert narrow_json_to_dict(envelope["params"])["name"] == "task_events"
    assert sent_arguments(poster.bodies[0]) == {"limit": 5}


def test_joins_every_text_block() -> None:
    """A result may carry more than one block and all of it is the answer."""
    payload: JSONObject = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"text": "first "}, {"text": "second"}]},
    }
    _test_hooks.http_post = FakeHttpPost([ok(rpc_body(payload))])
    assert call_tool(TEST_CREDENTIALS, "task_events", {}) == "first second"


def test_a_refused_status_raises_with_the_status_in_the_message() -> None:
    """A rotated key is a 401 and is the ordinary way this fails."""
    _test_hooks.http_post = FakeHttpPost([refused(401, '{"error":"Unauthorized"}')])
    with pytest.raises(AppError) as raised:
        call_tool(TEST_CREDENTIALS, "task_events", {})
    assert raised.value.code is BoardWatchErrorCode.HTTP_STATUS
    assert "401" in raised.value.message
    assert "Unauthorized" in raised.value.message


def test_a_body_with_no_data_line_raises_its_own_code() -> None:
    """An endpoint answering in a shape this client cannot read is distinct
    from one answering an error, and the two send a reader different ways."""
    _test_hooks.http_post = FakeHttpPost([ok("event: message\n\n")])
    with pytest.raises(AppError) as raised:
        call_tool(TEST_CREDENTIALS, "task_events", {})
    assert raised.value.code is BoardWatchErrorCode.RESPONSE_NOT_EVENT_STREAM


def test_a_jsonrpc_error_raises_its_own_code() -> None:
    """The transport succeeded and the tool failed; that is not an HTTP fault."""
    payload: JSONObject = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32602, "message": "bad cursor"},
    }
    _test_hooks.http_post = FakeHttpPost([ok(rpc_body(payload))])
    with pytest.raises(AppError) as raised:
        call_tool(TEST_CREDENTIALS, "task_events", {})
    assert raised.value.code is BoardWatchErrorCode.RPC_ERROR
    assert "bad cursor" in raised.value.message


def test_the_timeout_reaches_the_transport() -> None:
    """A caller-set timeout has to arrive at the socket to mean anything."""
    poster = FakeHttpPost([ok(tool_text("x")), ok(tool_text("x"))])
    _test_hooks.http_post = poster
    call_tool(TEST_CREDENTIALS, "task_events", {}, timeout_seconds=3)
    call_tool(TEST_CREDENTIALS, "task_events", {})
    assert poster.timeouts == [3, _test_hooks.DEFAULT_TIMEOUT_SECONDS]


__all__ = [
    "test_a_body_with_no_data_line_raises_its_own_code",
    "test_a_jsonrpc_error_raises_its_own_code",
    "test_a_refused_status_raises_with_the_status_in_the_message",
    "test_joins_every_text_block",
    "test_posts_a_jsonrpc_tool_call_with_both_headers",
    "test_the_timeout_reaches_the_transport",
]
