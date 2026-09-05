"""Shared fakes and the hook reset that keeps tests independent.

Everything here is a FAKE, not a mock. Each implements the same Protocol the
production implementation does and records what it was asked for, so an
assertion is about the request this package builds rather than about a
patching library's call-recording API. Nothing patches anything: the hooks in
:mod:`board_watch._test_hooks` are module-level names and a test rebinds them.

HOOKS ARE RESET BEFORE AND AFTER EVERY TEST. A rebinding that leaked would
produce a test that fails only when it runs after a specific other one, and
``-n auto`` reorders freely, so the symptom would be an intermittent failure
whose cause is invisible in the failing test.

THE BOARD FIXTURES ARE REAL CAPTURED BYTES. :data:`LIVE_MENTION_LINE` and
:data:`LIVE_CHECKIN_LINE` were taken verbatim from ``task_events`` against
the live board on 2026-09-05. A fixture written by hand would agree with
whatever the parser happened to do, which is the failure mode this package
was built after: a watcher whose cursor regex matched nothing, forever,
while every test it had passed.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator, Sequence
from typing import Final

import pytest
from platform_core.config import config_test_hooks
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from board_watch import _test_hooks
from board_watch.config import API_KEY_VARIABLE, TENANT_ID_VARIABLE, BoardCredentials
from board_watch.contracts import EMPTY_FEED_SENTINEL

#: A real ``task_events`` row carrying mentions and a truncated body.
LIVE_MENTION_LINE: Final = (
    "2026-09-04T22:57:47.747Z [note] opus-lavender-gpu-0824 "
    "mentions:@all-sessions,@opus-nclex-licensure-0904,@opus-artifact-sweep-0902: "
    "@opus-nclex-licensure-0904 URGENT, READ BEFORE YOUR `make deploy` [+3149 more chars]"
)

#: A real row of a different kind, with no mentions and no truncation.
LIVE_CHECKIN_LINE: Final = (
    "2026-09-05T00:03:06.343Z [checkin] opus-fleet-mcp-0904: LANDING fleet NOW"
)

#: A real row scoped to a task thread.
LIVE_TASK_LINE: Final = (
    "2026-09-04T23:47:26.390Z [status_change] fable-brain-audit-0903 "
    "task:8793517e-5c6b-4edd-a127-0234b40404d4 mentions:@opus-nclex-licensure-0904: "
    "claimed -> done"
)

#: A real row whose summary spans several physical lines.
#:
#: Captured 2026-09-05. This shape is why the first live run failed: every
#: other fixture here happens to be single-line, so a decoder that split on
#: newlines passed the whole suite and broke on the first real page.
LIVE_MULTILINE_ROW: Final = (
    "2026-08-16T01:57:20.613Z [note] opus-portaclaude-0815 mentions:@100: "
    "** HOSTNAME CHANGE: desktop-jah\n"
    "call — they wanted it renamed to fit the existing colour/place scheme\n"
    "\n"
    "Done this session:"
)

#: The credentials every client test posts with.
TEST_CREDENTIALS: Final = BoardCredentials(
    url="http://127.0.0.1:8033/mcp",
    api_key="test-key",
    tenant_id="2e137b5f-0000-4000-8000-000000000000",
)


def sse(payload: str) -> str:
    """Wrap a JSON payload the way the endpoint's event stream does.

    Args:
        payload: The JSON-RPC body.

    Returns:
        The response body, with the ``event:`` and ``data:`` framing.
    """
    return f"event: message\ndata: {payload}\n\n"


def tool_text(text: str) -> str:
    """Build a successful ``tools/call`` response body carrying one text block.

    Args:
        text: The rendered text the tool returns.

    Returns:
        The whole response body.
    """
    payload: JSONObject = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": text}]},
    }
    return sse(dump_json_str(payload))


def rpc_body(payload: JSONObject) -> str:
    """Wrap an arbitrary JSON-RPC payload in the endpoint's event stream.

    Args:
        payload: The payload object.

    Returns:
        The whole response body.
    """
    return sse(dump_json_str(payload))


def sent_arguments(body: bytes) -> JSONObject:
    """Read back the tool arguments from a recorded request body.

    Args:
        body: The bytes the fake poster recorded.

    Returns:
        The ``params.arguments`` object that was sent.
    """
    envelope = narrow_json_to_dict(load_json_str(body.decode("utf-8")))
    return narrow_json_to_dict(narrow_json_to_dict(envelope["params"])["arguments"])


def page_text(lines: Sequence[str], next_cursor: str | None) -> str:
    """Render a whole ``task_events`` response the way the board does.

    Args:
        lines: The event rows.
        next_cursor: The cursor the footer should offer, or None for a short
            page where the caller is caught up.

    Returns:
        The rendered text.
    """
    footer = (
        f"[showing {len(lines)} events]"
        if next_cursor is None
        else f"[showing {len(lines)} events; next cursor: {next_cursor}]"
    )
    body = "\n".join(lines) if len(lines) > 0 else EMPTY_FEED_SENTINEL
    return f"{body}\n\n{footer}"


class FakeHttpPost:
    """An HTTP poster that answers from a script and records the calls.

    Satisfies :class:`~board_watch._test_hooks.HttpPostProtocol`.

    Attributes:
        urls: Every URL it was given, in order.
        headers: Every header mapping it was given, in order.
        bodies: Every request body it was given, in order.
    """

    urls: list[str]
    headers: list[dict[str, str]]
    bodies: list[bytes]
    timeouts: list[int]
    _replies: list[_test_hooks.HttpResponse]

    def __init__(self, replies: Sequence[_test_hooks.HttpResponse]) -> None:
        """Build a poster that will answer with these responses in order.

        Args:
            replies: One response per expected call. Running out is an error
                rather than a default: a test that made more calls than it
                declared has changed behaviour it did not mean to assert on.
        """
        self.urls = []
        self.headers = []
        self.bodies = []
        self.timeouts = []
        self._replies = list(replies)

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: int,
    ) -> _test_hooks.HttpResponse:
        """Record a call and answer with the next scripted response.

        Args:
            url: The URL posted to.
            headers: The request headers.
            body: The request body.
            timeout_seconds: The timeout the caller chose.

        Returns:
            The next scripted response.

        Raises:
            AssertionError: If more calls are made than responses were given.
        """
        self.urls.append(url)
        self.headers.append(dict(headers))
        self.bodies.append(body)
        self.timeouts.append(timeout_seconds)
        assert self._replies, f"unscripted POST to {url}"
        return self._replies.pop(0)


class FakeEnv:
    """An environment backed by a dictionary.

    Satisfies :class:`~board_watch._test_hooks.EnvProtocol`.

    Attributes:
        values: The variables that are set.
    """

    values: dict[str, str]

    def __init__(self, values: dict[str, str]) -> None:
        """Build the environment.

        Args:
            values: The variables that are set.
        """
        self.values = dict(values)

    def __call__(self, name: str) -> str | None:
        """Read a variable, normalising blank to unset.

        The normalisation is part of :class:`~board_watch._test_hooks.EnvProtocol`,
        not a convenience. A fake that returned ``""`` where the real reader
        returns None would share the blind spot with the code under test, and
        the blank-credential case would pass here and fail against the board.

        Args:
            name: The variable name.

        Returns:
            Its trimmed value, or None when unset or blank.
        """
        raw = self.values.get(name)
        if raw is None or raw.strip() == "":
            return None
        return raw.strip()


class FakeFiles:
    """An in-memory filesystem for cursor documents.

    Satisfies the three filesystem Protocols in
    :mod:`board_watch._test_hooks`. In memory rather than ``tmp_path``
    because these tests are about WHICH path is written and what goes in it,
    and a real directory adds an operating system to that question without
    adding an assertion.

    Attributes:
        contents: Every file written, keyed by path.
    """

    contents: dict[pathlib.Path, str]

    def __init__(self, contents: dict[pathlib.Path, str] | None = None) -> None:
        """Build the filesystem.

        Args:
            contents: Files that already exist, keyed by path.
        """
        self.contents = {} if contents is None else dict(contents)

    def read_text(self, path: pathlib.Path) -> str:
        """Read a file.

        Args:
            path: The file.

        Returns:
            Its contents.

        Raises:
            AssertionError: If the file was never written, which in
                production would be a caller that skipped ``file_exists``.
        """
        assert path in self.contents, f"read of a file that does not exist: {path}"
        return self.contents[path]

    def write_text(self, path: pathlib.Path, content: str) -> None:
        """Write a file.

        Args:
            path: The file.
            content: What to write.
        """
        self.contents[path] = content

    def file_exists(self, path: pathlib.Path) -> bool:
        """Report whether a file was written.

        Args:
            path: The path.

        Returns:
            True when it exists.
        """
        return path in self.contents


class FakeEmit:
    """An output sink that records the lines it was given.

    Satisfies :class:`~board_watch._test_hooks.EmitProtocol`.

    Attributes:
        lines: Every line emitted, in order.
    """

    lines: list[str]

    def __init__(self) -> None:
        """Build an empty sink."""
        self.lines = []

    def __call__(self, line: str) -> None:
        """Record a line.

        Args:
            line: The line.
        """
        self.lines.append(line)


def ok(body: str) -> _test_hooks.HttpResponse:
    """Build a successful HTTP response.

    Args:
        body: The response body.

    Returns:
        The response, status 200 and an event-stream content type.
    """
    return _test_hooks.HttpResponse(status=200, content_type="text/event-stream", body=body)


def refused(status: int, body: str) -> _test_hooks.HttpResponse:
    """Build a refusing HTTP response.

    Args:
        status: The status code.
        body: The response body.

    Returns:
        The response, with a JSON content type.
    """
    return _test_hooks.HttpResponse(
        status=status, content_type="application/json; charset=utf-8", body=body
    )


def set_environment() -> FakeEnv:
    """Bind a complete environment and return it.

    Returns:
        The bound environment, so a test can mutate it further.
    """
    environment = FakeEnv(
        {
            API_KEY_VARIABLE: TEST_CREDENTIALS["api_key"],
            TENANT_ID_VARIABLE: TEST_CREDENTIALS["tenant_id"],
        }
    )
    _test_hooks.env = environment
    return environment


@pytest.fixture(name="files")
def _files() -> FakeFiles:
    """Bind an empty in-memory filesystem.

    Returns:
        The bound filesystem.
    """
    fake = FakeFiles()
    _test_hooks.read_text = fake.read_text
    _test_hooks.write_text = fake.write_text
    _test_hooks.file_exists = fake.file_exists
    return fake


@pytest.fixture(name="emitted")
def _emitted() -> FakeEmit:
    """Bind a recording output sink.

    Returns:
        The bound sink.
    """
    sink = FakeEmit()
    _test_hooks.emit = sink
    return sink


@pytest.fixture(name="reset_hooks", autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Put every hook back to its real implementation around each test."""
    _restore()
    yield None
    _restore()


def _restore() -> None:
    """Rebind every hook to the implementation it starts life with.

    Includes ``platform_core``'s environment hook, because
    :func:`board_watch._test_hooks._default_env` delegates to it and a test
    that rebinds it would otherwise leak that binding into whichever test
    ``-n auto`` happened to schedule next.
    """
    _test_hooks.http_post = _test_hooks._default_http_post
    _test_hooks.env = _test_hooks._default_env
    _test_hooks.read_text = _test_hooks._default_read_text
    _test_hooks.write_text = _test_hooks._default_write_text
    _test_hooks.file_exists = _test_hooks._default_file_exists
    _test_hooks.emit = _test_hooks._default_emit
    config_test_hooks.get_env = config_test_hooks._default_get_env
