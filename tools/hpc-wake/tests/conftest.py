"""Shared fakes and the hook reset that keeps tests independent.

Everything here is a FAKE implementing the production Protocol, never a
mock, matching ``board-watch``'s conventions. The seams rebound are this
package's own (``hpc_wake._test_hooks``), hpc3's command runner
(``hpc3.core._test_hooks.run`` -- the same seam hpc3's suite uses), and the
one sanctioned environment reader
(``platform_core.config.config_test_hooks.get_env`` -- which also feeds
``board_watch.config.load_credentials``, so pinning it configures the whole
credential chain from one place).

Ledger and closure files are REAL files under ``tmp_path``: the production
hpc3 defaults do the I/O, exactly as hpc3's own suite exercises them.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Final

import pytest
from hpc3.core import _test_hooks as hpc3_hooks
from hpc3.core._test_hooks import CommandResult
from platform_core.config import config_test_hooks
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.mcp_client import McpHttpResponse

from hpc_wake import _test_hooks
from hpc_wake.identity import TASK_ID_VARIABLE

FROZEN_NOW: Final = "2026-09-06T07:00:00+00:00"

#: The standing task id every configured test posts into.
TASK_ID: Final = "50e693d6-c3aa-4464-b43b-adbc07149a67"

#: The environment the configured tests run in, in full.
CONFIGURED_ENV: Final[dict[str, str]] = {
    "TASKBOARD_MCP_API_KEY": "test-key",
    "CORVIS_TENANT_ID": "2e137b5f-0000-4000-8000-000000000000",
    TASK_ID_VARIABLE: TASK_ID,
}


@pytest.fixture(autouse=True)
def _reset_hooks() -> Generator[None, None, None]:
    """Rebind every touched seam to production before and after each test."""
    _test_hooks.reset_hooks()
    hpc3_hooks.reset_hooks()
    original_env = config_test_hooks.get_env
    yield
    _test_hooks.reset_hooks()
    hpc3_hooks.reset_hooks()
    config_test_hooks.get_env = original_env


def pin_env(values: dict[str, str]) -> None:
    """Answer environment reads from a dictionary and nothing else.

    Args:
        values: The variables that are set. Every other variable reads as
            unset, so a test's environment is this call, not the developer's
            shell.
    """

    def _env(name: str) -> str | None:
        return values.get(name)

    config_test_hooks.get_env = _env


def _make_frozen_clock() -> Generator[str, None, None]:
    """Pin the bridge clock so closure timestamps are assertable.

    Yields:
        The timestamp every closure will record.
    """

    def _now() -> str:
        return FROZEN_NOW

    _test_hooks.now_iso = _now
    yield FROZEN_NOW
    _test_hooks.reset_hooks()


def _make_emitted() -> Generator[list[str], None, None]:
    """Capture report lines instead of writing them to stdout.

    Yields:
        The list the ``emit`` hook appends to, in emission order.
    """
    lines: list[str] = []

    def _emit(line: str) -> None:
        lines.append(line)

    _test_hooks.emit = _emit
    yield lines
    _test_hooks.reset_hooks()


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


def posted_ok() -> McpHttpResponse:
    """A 200 whose payload is a successful ``task_post``.

    Returns:
        The response.
    """
    return McpHttpResponse(status=200, body=tool_text("posted"), content_type="text/event-stream")


def sent_arguments(body: bytes) -> JSONObject:
    """Read back the tool arguments from a recorded request body.

    Args:
        body: The bytes the fake poster recorded.

    Returns:
        The ``params.arguments`` object that was sent.
    """
    envelope = narrow_json_to_dict(load_json_str(body.decode("utf-8")))
    return narrow_json_to_dict(narrow_json_to_dict(envelope["params"])["arguments"])


class FakeHttpPost:
    """An HTTP poster that answers from a script and records the calls.

    Satisfies :class:`~platform_core.mcp_client.McpPostProtocol`.

    Attributes:
        urls: Every URL it was given, in order.
        headers: Every header mapping it was given, in order.
        bodies: Every request body it was given, in order.
    """

    urls: list[str]
    headers: list[dict[str, str]]
    bodies: list[bytes]

    def __init__(self, replies: Sequence[McpHttpResponse]) -> None:
        """Build a poster that will answer with these responses in order.

        Args:
            replies: One response per expected call. Running out is an
                error rather than a default: a test that made more calls
                than it declared has changed behaviour it did not mean to
                assert on.
        """
        self.urls = []
        self.headers = []
        self.bodies = []
        self._replies = list(replies)

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: int,
    ) -> McpHttpResponse:
        """Record a call and answer with the next scripted response.

        Args:
            url: The URL posted to.
            headers: The request headers.
            body: The request body.
            timeout_seconds: The timeout the caller chose.

        Returns:
            The next scripted response.

        Raises:
            AssertionError: If more calls are made than responses were
                scripted.
        """
        self.urls.append(url)
        self.headers.append(dict(headers))
        self.bodies.append(body)
        assert self._replies, f"unscripted POST to {url}"
        return self._replies.pop(0)


class FakeRun:
    """A scripted stand-in for hpc3's remote command runner.

    Satisfies the ``run`` seam in :mod:`hpc3.core._test_hooks`: responses
    match by substring against the remote command (the final argv element),
    first match wins, and an unmatched command returns empty success.

    Attributes:
        commands: Every remote command received, in order.
    """

    commands: list[str]

    def __init__(self) -> None:
        """Start with no rules and no recorded calls."""
        self.commands = []
        self._rules: list[tuple[str, CommandResult]] = []

    def add(self, contains: str, *, stdout: str = "") -> None:
        """Script a response for commands containing a substring.

        Args:
            contains: Substring matched against the remote command.
            stdout: Standard output to return.
        """
        self._rules.append((contains, CommandResult(returncode=0, stdout=stdout, stderr="")))

    def __call__(self, argv: Sequence[str], *, stdin_bytes: bytes | None = None) -> CommandResult:
        """Record the invocation and return its scripted response.

        Args:
            argv: Executable and arguments.
            stdin_bytes: Bytes offered on standard input, or None.

        Returns:
            The first matching scripted response, or empty success.
        """
        command = argv[-1]
        self.commands.append(command)
        for contains, result in self._rules:
            if contains in command:
                return result
        return CommandResult(returncode=0, stdout="", stderr="")


def _make_fake_run() -> Generator[FakeRun, None, None]:
    """Install the fake command runner for the duration of a test.

    Yields:
        The runner, for scripting responses and asserting on calls.
    """
    fake = FakeRun()
    hpc3_hooks.run = fake
    yield fake
    hpc3_hooks.reset_hooks()


emitted = pytest.fixture(_make_emitted)
fake_run = pytest.fixture(_make_fake_run)
frozen_clock = pytest.fixture(_make_frozen_clock)
