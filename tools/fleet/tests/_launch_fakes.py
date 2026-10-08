"""The node and the queue answering a claim and the launch beside it from separate scripts.

Support module, not a test module. Lifted out of ``test_node_launch.py``
(MCPs board task 8993c306) when ``test_node_launch_host.py`` (MCPs board
task a85ef09e) needed the same routing for two runners of one host: a
runner's claims run on the calling thread and its launches on the
launcher's pool (:class:`fleet.cli.node_launch.Launcher`), so each answers
from the script of the thread asking, and the launch's first command waits
for the case's go-ahead, holding the launch under way for as long as the
case needs.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

from platform_core.mcp_client import McpHttpResponse

from fleet.core import _test_hooks
from tests._queue_fakes import FakeQueue
from tests._thread_fakes import await_event
from tests.conftest import FakeRun

#: The prefix of a launch thread's name (:class:`fleet.cli.node_launch.Launcher`).
LAUNCH_THREAD = "fleet-launch-"


def on_a_launch_thread() -> bool:
    """Whether the caller runs on one of a launcher's threads.

    Returns:
        True on a launch thread.
    """
    return threading.current_thread().name.startswith(LAUNCH_THREAD)


class ThreadRoutedRun:
    """The node and the hub's git, answering the claim and the launch from scripts of their own.

    Satisfies :class:`~fleet.core._test_hooks.RunProtocol`. The launch's first
    command waits for the case's go-ahead, so the launch is under way for as
    long as the case holds it.

    Attributes:
        claims: The claim's answers, the probes.
        launches: The launch's answers.
    """

    claims: FakeRun
    launches: FakeRun

    def __init__(
        self,
        *,
        claims: FakeRun,
        launches: FakeRun,
        release: threading.Event,
    ) -> None:
        """Bind the scripts and the go-ahead.

        Args:
            claims: The claim's answers.
            launches: The launch's answers.
            release: Set when the launch may go on.
        """
        self.claims = claims
        self.launches = launches
        self._release = release

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        """Answer from the caller's thread's script.

        Args:
            argv: The command.
            timeout_seconds: The deadline the caller chose.
            stdin_bytes: Its standard input, or None.
            unset_env: The variables the caller withheld from the child.
            set_env: The variables the caller set in the child.

        Returns:
            The next scripted result for that thread.
        """
        script = self.launches if on_a_launch_thread() else self.claims
        if script is self.launches and not script.calls:
            await_event(self._release, what="the case's go-ahead to launch")
        return script(
            argv,
            timeout_seconds=timeout_seconds,
            stdin_bytes=stdin_bytes,
            unset_env=unset_env,
            set_env=set_env,
        )


class ThreadRoutedQueue:
    """The queue, answering the claim and the launch from scripts of their own.

    Satisfies :class:`~platform_core.mcp_client.McpPostProtocol`.
    """

    def __init__(self, *, claims: FakeQueue, launches: FakeQueue) -> None:
        """Bind the scripts.

        Args:
            claims: The claim's answers.
            launches: The launch's answers.
        """
        self._claims = claims
        self._launches = launches

    def __call__(
        self, url: str, *, headers: dict[str, str], body: bytes, timeout_seconds: int
    ) -> McpHttpResponse:
        """Answer from the caller's thread's script.

        Args:
            url: Absolute URL posted to.
            headers: Every request header.
            body: The encoded JSON-RPC body.
            timeout_seconds: The caller's timeout.

        Returns:
            The next scripted answer for that thread.
        """
        script = self._launches if on_a_launch_thread() else self._claims
        return script(url, headers=headers, body=body, timeout_seconds=timeout_seconds)


__all__ = ["LAUNCH_THREAD", "ThreadRoutedQueue", "ThreadRoutedRun", "on_a_launch_thread"]
