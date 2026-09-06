"""The shared MCP fakes, tested as the load-bearing utilities they are.

THESE ARE SHIPPED IN ``src``, NOT IN A TEST DIRECTORY, because three packages
import them. That makes them ordinary code with an ordinary obligation to be
correct: a fake whose recording is wrong makes every assertion built on it
wrong in the same direction, and silently.

THE EXHAUSTION GUARD IS THE ONE THAT MATTERS. Several suites assert that a
code path makes NO request, or exactly one, by scripting exactly that many
replies and letting an extra call raise. If :class:`FakeHttpPost` answered an
unscripted call instead -- by repeating the last reply, or returning a default
-- every one of those tests would keep passing while the behaviour they pin
had changed. That is the assertion this file exists to protect.
"""

from __future__ import annotations

import pytest

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.mcp_client import EVENT_STREAM_MEDIA_TYPE, McpHttpResponse, rpc_envelope
from platform_core.mcp_testing import (
    FakeHttpPost,
    posted_ok,
    sent_arguments,
    sse_body,
    tool_text_body,
)

REPLY = McpHttpResponse(status=200, body="first", content_type="text/plain")
SECOND = McpHttpResponse(status=200, body="second", content_type="text/plain")


class TestFakeHttpPostRecordsWhatItWasAsked:
    def test_every_part_of_the_request_is_kept_in_order(self) -> None:
        fake = FakeHttpPost([REPLY, SECOND])

        first = fake("http://a", headers={"k": "1"}, body=b"one", timeout_seconds=7)
        second = fake("http://b", headers={"k": "2"}, body=b"two", timeout_seconds=9)

        assert (first, second) == (REPLY, SECOND)
        assert fake.urls == ["http://a", "http://b"]
        assert fake.headers == [{"k": "1"}, {"k": "2"}]
        assert fake.bodies == [b"one", b"two"]
        assert fake.timeouts == [7, 9]

    def test_the_recorded_headers_are_a_copy_not_the_callers_mapping(self) -> None:
        """A caller that reuses and mutates one header dict across calls would
        otherwise leave every recorded entry showing the LAST call's headers,
        and an assertion about the first request would silently be about the
        second."""
        fake = FakeHttpPost([REPLY, SECOND])
        headers = {"k": "1"}

        fake("http://a", headers=headers, body=b"", timeout_seconds=1)
        headers["k"] = "2"
        fake("http://b", headers=headers, body=b"", timeout_seconds=1)

        assert fake.headers == [{"k": "1"}, {"k": "2"}]


class TestAnUnscriptedCallRaises:
    def test_a_call_past_the_script_names_the_url(self) -> None:
        """THE GUARD EVERY "no request was made" ASSERTION RESTS ON. Answering
        it instead would keep those tests green through exactly the change
        they exist to catch."""
        fake = FakeHttpPost([REPLY])
        fake("http://a", headers={}, body=b"", timeout_seconds=1)

        with pytest.raises(AssertionError, match=r"unscripted POST to http://b"):
            fake("http://b", headers={}, body=b"", timeout_seconds=1)

    def test_an_empty_script_refuses_the_very_first_call(self) -> None:
        """How a suite asserts that a path touches the network not at all."""
        fake = FakeHttpPost([])

        with pytest.raises(AssertionError, match=r"unscripted POST"):
            fake("http://a", headers={}, body=b"", timeout_seconds=1)

    def test_the_refused_call_is_still_recorded(self) -> None:
        """So a failing test can show WHAT the unexpected call was, rather
        than only that there was one."""
        fake = FakeHttpPost([])

        with pytest.raises(AssertionError):
            fake("http://a", headers={"k": "1"}, body=b"body", timeout_seconds=3)

        assert fake.urls == ["http://a"]
        assert fake.bodies == [b"body"]


class TestResponseBuilders:
    def test_the_framing_is_what_the_server_actually_sends(self) -> None:
        assert sse_body('{"a":1}') == 'event: message\ndata: {"a":1}\n\n'

    def test_a_tool_result_body_carries_the_text_block(self) -> None:
        body = tool_text_body("rendered")
        payload = narrow_json_to_dict(
            load_json_str(body.removeprefix("event: message\ndata: ").strip())
        )

        result = narrow_json_to_dict(payload["result"])
        assert result == {"content": [{"type": "text", "text": "rendered"}]}

    def test_posted_ok_declares_the_event_stream_content_type(self) -> None:
        """The client checks it; a fake that claimed ``application/json``
        would exercise a branch the real server never takes."""
        assert posted_ok()["content_type"] == EVENT_STREAM_MEDIA_TYPE
        assert posted_ok()["status"] == 200


class TestSentArguments:
    def test_it_reads_back_exactly_what_the_envelope_carried(self) -> None:
        """Round-tripped through the REAL envelope builder rather than a
        hand-written shape, so this cannot pass against an envelope the client
        does not actually produce."""
        arguments: JSONObject = {"limit": 5, "cursor": "abc"}

        assert sent_arguments(rpc_envelope("task_events", arguments)) == arguments

    def test_a_body_that_is_not_an_envelope_raises(self) -> None:
        with pytest.raises(Exception, match=r"params"):
            sent_arguments(dump_json_str({"jsonrpc": "2.0"}).encode("utf-8"))
