from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Final, Protocol, TypedDict


class Runnable(Protocol):
    async def run(self, *, limit: int | None = None) -> None: ...


class Closable(Protocol):
    async def close(self) -> None: ...


class _BuildResult(TypedDict):
    runnable: Runnable
    closable: Closable


BuildFunc = Callable[[], _BuildResult]
OnError = Callable[[BaseException], None]


class TaskRunner:
    """Encapsulate start/stop/done-callback lifecycle for long-running tasks.

    This wrapper standardizes the pattern used by Discord bot notifiers that
    subscribe to event streams (e.g., Redis Pub/Sub) and run until cancelled.

    It avoids Any by using minimal Protocols for `Runnable` and `Closable`.
    """

    __slots__ = ("_build", "_closable", "_name", "_on_error", "_runnable", "_task")

    def __init__(
        self,
        *,
        build: BuildFunc,
        name: str,
        on_error: OnError | None = None,
    ) -> None:
        self._build = build
        self._name: Final[str] = name
        self._on_error = on_error
        self._runnable: Runnable | None = None
        self._closable: Closable | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            return
        built = self._build()
        runnable = built["runnable"]
        closable = built["closable"]

        async def _run() -> None:
            try:
                await runnable.run()
            finally:
                await closable.close()

        task = asyncio.create_task(_run(), name=self._name)
        task.add_done_callback(self._on_done)
        self._runnable = runnable
        self._closable = closable
        self._task = task

    async def stop(self) -> None:
        task = self._task
        # Reset first to prevent reentrancy issues during awaiting
        self._task = None
        if task is None:
            # No running task; close any existing closable
            closable = self._closable
            if closable is not None:
                await closable.close()
                self._closable = None
            self._runnable = None
            return
        task.cancel()
        done, _ = await asyncio.wait({task})
        finished = next(iter(done))
        if finished.cancelled():
            closable = self._closable
            if closable is not None:
                await closable.close()
                self._closable = None
            self._runnable = None
            return
        exc = finished.exception()
        closable = self._closable
        if closable is not None:
            await closable.close()
            self._closable = None
        self._runnable = None
        if exc is not None:
            raise exc

    def _on_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        cb = self._on_error
        if cb is not None:
            cb(exc)

    async def run_once(self) -> None:
        """Build and run the runnable once (limit=1) for testing.

        If already started, reuses the existing runnable and does not rebuild.
        """
        runnable = self._runnable
        if runnable is None:
            built = self._build()
            runnable = built["runnable"]
            closable = built["closable"]
            try:
                await runnable.run(limit=1)
            finally:
                await closable.close()
            return
        await runnable.run(limit=1)


__all__ = ["BuildFunc", "Closable", "OnError", "Runnable", "TaskRunner"]
