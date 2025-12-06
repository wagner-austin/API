from __future__ import annotations

import asyncio
from typing import TypedDict

import pytest

from platform_discord.task_runner import BuildFunc, Closable, Runnable, TaskRunner


class _FakeClosable:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeRunnable(Runnable):
    def __init__(
        self,
        *,
        sleep_forever: bool = False,
        should_raise: bool = False,
    ) -> None:
        self.calls: list[int | None] = []
        self.sleep_forever = sleep_forever
        self.should_raise = should_raise

    async def run(self, *, limit: int | None = None) -> None:
        self.calls.append(limit)
        if self.should_raise:
            raise RuntimeError("boom")
        if self.sleep_forever:
            await asyncio.sleep(10)
        else:
            await asyncio.sleep(0)


class _BR(TypedDict):
    runnable: Runnable
    closable: Closable


def _build_pair(r: Runnable, c: Closable) -> BuildFunc:
    def _builder() -> _BR:
        return {"runnable": r, "closable": c}

    return _builder


@pytest.mark.asyncio
async def test_start_then_stop_cancels_and_closes() -> None:
    r = _FakeRunnable(sleep_forever=True)
    c = _FakeClosable()
    runner = TaskRunner(build=_build_pair(r, c), name="t")
    runner.start()
    await asyncio.sleep(0)
    await runner.stop()
    assert c.closed is True


@pytest.mark.asyncio
async def test_start_idempotent_then_stop() -> None:
    r = _FakeRunnable(sleep_forever=True)
    c = _FakeClosable()
    runner = TaskRunner(build=_build_pair(r, c), name="t2")
    runner.start()
    runner.start()  # no-op
    await asyncio.sleep(0)
    await runner.stop()
    assert c.closed is True


@pytest.mark.asyncio
async def test_run_once_builds_and_closes() -> None:
    r = _FakeRunnable()
    c = _FakeClosable()
    runner = TaskRunner(build=_build_pair(r, c), name="t3")
    await runner.run_once()
    assert r.calls == [1]
    assert c.closed is True


@pytest.mark.asyncio
async def test_stop_raises_when_task_failed_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    r = _FakeRunnable(should_raise=True)
    c = _FakeClosable()
    seen: list[str] = []

    def _on_err(exc: BaseException) -> None:
        seen.append(str(exc))

    runner = TaskRunner(build=_build_pair(r, c), name="t4", on_error=_on_err)
    runner.start()
    await asyncio.sleep(0)  # let run execute and fail
    with pytest.raises(RuntimeError):
        await runner.stop()
    assert c.closed is True
    assert seen and "boom" in seen[0]


@pytest.mark.asyncio
async def test_run_once_when_started_uses_existing_runnable() -> None:
    r = _FakeRunnable(sleep_forever=True)
    c = _FakeClosable()
    runner = TaskRunner(build=_build_pair(r, c), name="t5")
    runner.start()
    await asyncio.sleep(0)
    # Even if started, run_once should call with limit=1 on the same runnable
    # (and not close the closable, which is owned by the background task)
    # It will not complete until we stop the background; so call with a quick await
    task = asyncio.create_task(runner.run_once())
    await asyncio.sleep(0)
    await runner.stop()
    await task
    # The first call from background has limit=None, then run_once used 1
    assert r.calls and r.calls[-1] == 1


@pytest.mark.asyncio
async def test_stop_without_task_closes_existing_closable() -> None:
    r = _FakeRunnable()
    c = _FakeClosable()
    runner = TaskRunner(build=_build_pair(r, c), name="t6")
    # Simulate an externally set closable without an active task
    object.__setattr__(runner, "_closable", c)
    await runner.stop()
    assert c.closed is True


@pytest.mark.asyncio
async def test_stop_after_task_success_closes_without_raise() -> None:
    r = _FakeRunnable(sleep_forever=False, should_raise=False)
    c = _FakeClosable()
    runner = TaskRunner(build=_build_pair(r, c), name="t7")
    runner.start()
    # Allow runnable to complete successfully
    await asyncio.sleep(0)
    await runner.stop()
    assert c.closed is True


@pytest.mark.asyncio
async def test_on_done_paths() -> None:
    errors: list[str] = []

    def _on_err(exc: BaseException) -> None:
        errors.append(str(exc))

    # Runner for invoking callbacks; build is unused here
    runner = TaskRunner(
        build=_build_pair(_FakeRunnable(), _FakeClosable()),
        name="t8",
        on_error=_on_err,
    )

    # Completed task without exception -> returns
    t_ok = asyncio.create_task(asyncio.sleep(0))
    await t_ok
    runner._on_done(t_ok)

    # Completed task with exception -> on_error called
    async def _boom() -> None:
        raise RuntimeError("E")

    t_bad = asyncio.create_task(_boom())
    await asyncio.sleep(0)
    runner._on_done(t_bad)
    assert errors and "E" in errors[0]


@pytest.mark.asyncio
async def test_stop_no_task_no_closable_noop() -> None:
    r = _FakeRunnable()
    c = _FakeClosable()
    runner = TaskRunner(build=_build_pair(r, c), name="t9")
    # No task, no closable set
    await runner.stop()


@pytest.mark.asyncio
async def test_stop_cancelled_without_closable_path() -> None:
    r = _FakeRunnable(sleep_forever=True)
    c = _FakeClosable()
    runner = TaskRunner(build=_build_pair(r, c), name="t10")
    runner.start()
    await asyncio.sleep(0)
    # clear closable to exercise the false branch of the conditional
    object.__setattr__(runner, "_closable", None)
    await runner.stop()


@pytest.mark.asyncio
async def test_stop_finished_no_closable_no_exception() -> None:
    r = _FakeRunnable(sleep_forever=False)
    c = _FakeClosable()
    runner = TaskRunner(build=_build_pair(r, c), name="t11")
    runner.start()
    await asyncio.sleep(0)  # let it finish successfully
    object.__setattr__(runner, "_closable", None)
    await runner.stop()


@pytest.mark.asyncio
async def test_on_done_error_no_callback() -> None:
    runner = TaskRunner(build=_build_pair(_FakeRunnable(), _FakeClosable()), name="t12")

    async def _boom() -> None:
        raise RuntimeError("X")

    t = asyncio.create_task(_boom())
    await asyncio.sleep(0)
    runner._on_done(t)


@pytest.mark.asyncio
async def test_stop_with_finished_task_no_exception_and_no_closable() -> None:
    runner = TaskRunner(build=_build_pair(_FakeRunnable(), _FakeClosable()), name="t13")

    async def _ok() -> None:
        await asyncio.sleep(0)

    t = asyncio.create_task(_ok())
    await t
    object.__setattr__(runner, "_task", t)
    object.__setattr__(runner, "_closable", None)
    await runner.stop()
