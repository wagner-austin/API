"""Waiting on another thread in a test (MCPs board tasks c1d48330 and 8993c306).

A serving node runner's watch polls and settles on a thread of its own
(:mod:`fleet.cli.node_watch`), so a case that fixes one interleaving, a
handover asked for while a settle is under way, say, waits for an event the
other thread sets, and fails rather than hanging the suite when it never
comes.

Its claimed jobs launch on a pool too (:mod:`fleet.cli.node_launch`), and a
case that scripts the node's and the queue's answers in one order needs
each launch over before the claim goes on: :class:`InOrderExecutor` runs
every launch on a real thread and returns only once it has finished, so the
order is the one a runner had before its launches ran beside its claims.
The cases about launching beside the claim bind the real pool instead.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait
from typing import Final, ParamSpec, TypeVar

#: How long a case waits for an event before it fails, rather than hanging
#: the suite on an interleaving that never came.
WAIT_SECONDS: Final = 30

_P = ParamSpec("_P")
_T = TypeVar("_T")


def await_event(event: threading.Event, *, what: str) -> None:
    """Wait for an event another thread sets.

    Args:
        event: The event.
        what: What it marks, for the failure message.

    Raises:
        AssertionError: When it has not happened within :data:`WAIT_SECONDS`.
    """
    assert event.wait(WAIT_SECONDS), f"waited {WAIT_SECONDS}s for {what}"


class InOrderExecutor(Executor):
    """A pool of one real thread whose submit returns once the call has finished."""

    def __init__(self, *, name: str) -> None:
        """Open the thread's pool.

        Args:
            name: The prefix of its thread's name.
        """
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)

    def submit(self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_T]:
        """Run a call on the thread and wait for it.

        Args:
            fn: The call.
            *args: Its positional arguments.
            **kwargs: Its keyword arguments.

        Returns:
            Its future, already finished, its result or its error held.
        """
        call = self._pool.submit(fn, *args, **kwargs)
        wait([call])
        return call

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Close the thread's pool.

        Args:
            wait: Whether to wait for the call under way, of which there is
                none once a submit has returned.
            cancel_futures: Whether to cancel calls not yet started.
        """
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)


def in_order_executor(*, workers: int, name: str) -> Executor:
    """Make an :class:`InOrderExecutor`, as the launch pool's seam asks.

    Satisfies :class:`~fleet.core._test_hooks.ExecutorProtocol`.

    Args:
        workers: Ignored: one launch at a time is the point.
        name: The prefix of its thread's name.

    Returns:
        The executor.
    """
    return InOrderExecutor(name=name)


__all__ = ["WAIT_SECONDS", "InOrderExecutor", "await_event", "in_order_executor"]
