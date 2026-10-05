"""Fakes that answer a node runner's two threads apart (MCPs board task c1d48330).

A tick's watch polls and settles on a thread of its own while the passes run
in the main thread (:mod:`fleet.cli.node_watch`), so one scripted list of
answers would be consumed in whatever order the two threads happened to
reach it. These route each call by the thread that made it to a fake of its
own, :class:`~tests.conftest.FakeRun` or :class:`~tests._queue_fakes.FakeQueue`,
so each thread's calls are answered and recorded in their own order, and
the runner holds a call until an event a test names has happened, so a test
fixes the one interleaving it asserts on: the fill pass's first probe held
until the watch has closed a run, say, which shows the run closed while
that pass was going.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Final

from platform_core.json_utils import load_json_str, narrow_json_to_dict, narrow_json_to_str
from platform_core.mcp_client import McpHttpResponse

from fleet.cli.node_watch import THREAD_PREFIX
from fleet.core import _test_hooks
from tests._queue_fakes import FakeQueue
from tests.conftest import FakeRun

#: How long a held call waits for its event before the test fails, rather
#: than hanging the suite on an interleaving that never came.
WAIT_SECONDS: Final = 30


def on_watch_thread() -> bool:
    """Whether the calling thread is a tick's watch.

    Returns:
        True on the thread :func:`fleet.cli.node_watch.run_tick` starts.
    """
    return threading.current_thread().name.startswith(THREAD_PREFIX)


def await_event(event: threading.Event, *, what: str) -> None:
    """Wait for an event a held call depends on.

    Args:
        event: The event.
        what: What it marks, for the failure message.

    Raises:
        AssertionError: When it has not happened within :data:`WAIT_SECONDS`.
    """
    assert event.wait(WAIT_SECONDS), f"waited {WAIT_SECONDS}s for {what}"


class RunByThread:
    """A command runner answering the watch thread and the main thread apart.

    Satisfies :class:`~fleet.core._test_hooks.RunProtocol`.

    Attributes:
        main: Answers every call made outside the watch thread.
        watch: Answers every call made on it.
    """

    main: FakeRun
    watch: FakeRun

    def __init__(
        self,
        main: FakeRun,
        watch: FakeRun,
        *,
        watch_after: threading.Event,
        main_waits: Mapping[int, threading.Event],
    ) -> None:
        """Bind the two runners.

        Args:
            main: The main thread's runner.
            watch: The watch thread's runner.
            watch_after: Every watch call waits for it, so the watch reads
                nothing before the point a test chose.
            main_waits: For the index of a main-thread call, counted from
                zero, the event it waits for before it is answered.
        """
        self.main = main
        self.watch = watch
        self._watch_after = watch_after
        self._main_waits = main_waits

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        stdin_bytes: bytes | None = None,
        unset_env: Sequence[str] = (),
        set_env: Sequence[tuple[str, str]] = (),
    ) -> _test_hooks.CommandResult:
        """Answer from the calling thread's runner.

        Args:
            argv: The command.
            timeout_seconds: The deadline the caller chose.
            stdin_bytes: Its standard input, or None.
            unset_env: The variables the caller withheld from the child.
            set_env: The variables the caller set in the child.

        Returns:
            That runner's next scripted result.
        """
        if not on_watch_thread():
            index = len(self.main.calls)
            if index in self._main_waits:
                await_event(self._main_waits[index], what=f"the point main may make call {index}")
            return self.main(
                argv,
                timeout_seconds=timeout_seconds,
                stdin_bytes=stdin_bytes,
                unset_env=unset_env,
                set_env=set_env,
            )
        await_event(self._watch_after, what="the point the watch may read the node")
        return self.watch(
            argv,
            timeout_seconds=timeout_seconds,
            stdin_bytes=stdin_bytes,
            unset_env=unset_env,
            set_env=set_env,
        )


def tool_of(body: bytes) -> str:
    """The tool a JSON-RPC request body calls.

    Args:
        body: The encoded request.

    Returns:
        The tool's name.
    """
    envelope = narrow_json_to_dict(load_json_str(body.decode("utf-8")))
    return narrow_json_to_str(narrow_json_to_dict(envelope["params"])["name"])


class QueueByThread:
    """A queue endpoint answering the watch thread and the main thread apart.

    Satisfies :class:`~platform_core.mcp_client.McpPostProtocol`.

    Attributes:
        main: Answers every call made outside the watch thread.
        watch: Answers every call made on it.
        order: ``main:<tool>`` or ``watch:<tool>`` for every call, across
            both threads, in the order they were answered.
    """

    main: FakeQueue
    watch: FakeQueue
    order: list[str]

    def __init__(
        self,
        main: FakeQueue,
        watch: FakeQueue,
        *,
        signals: Mapping[str, threading.Event],
    ) -> None:
        """Bind the two endpoints and the events their answers set.

        Args:
            main: The main thread's endpoint.
            watch: The watch thread's endpoint.
            signals: For ``main:<tool>`` or ``watch:<tool>``, the event set
                once that call has been answered.
        """
        self.main = main
        self.watch = watch
        self.order = []
        self._signals = signals
        self._lock = threading.Lock()

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: int,
    ) -> McpHttpResponse:
        """Answer from the calling thread's endpoint.

        Args:
            url: Absolute URL posted to.
            headers: Every request header.
            body: The encoded JSON-RPC body.
            timeout_seconds: The caller's timeout.

        Returns:
            That endpoint's next scripted answer.
        """
        tool = tool_of(body)
        watching = on_watch_thread()
        endpoint = self.watch if watching else self.main
        answer = endpoint(url, headers=headers, body=body, timeout_seconds=timeout_seconds)
        key = f"{'watch' if watching else 'main'}:{tool}"
        with self._lock:
            self.order.append(key)
        if key in self._signals:
            self._signals[key].set()
        return answer


__all__ = [
    "WAIT_SECONDS",
    "QueueByThread",
    "RunByThread",
    "await_event",
    "on_watch_thread",
    "tool_of",
]
