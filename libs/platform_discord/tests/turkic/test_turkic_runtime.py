from __future__ import annotations

from collections.abc import Generator

import pytest

from platform_discord.testing import fake_load_discord_module, hooks
from platform_discord.turkic.runtime import (
    RequestAction,
    TurkicRuntime,
    new_runtime,
    on_completed,
    on_failed,
    on_progress,
    on_started,
)


def _rt() -> TurkicRuntime:
    return new_runtime()


@pytest.fixture(autouse=True)
def _use_fake_discord() -> Generator[None, None, None]:
    """Set up fake discord module via hooks."""
    hooks.load_discord_module = fake_load_discord_module
    yield


def test_turkic_runtime_flow_with_user() -> None:
    rt = _rt()
    a1: RequestAction = on_started(rt, user_id=1, job_id="j", queue="turkic")
    if a1["embed"] is None:
        pytest.fail("expected embed in a1")
    assert a1["user_id"] == 1
    a2 = on_progress(rt, user_id=1, job_id="j", progress=10, message="ok")
    if a2["embed"] is None:
        pytest.fail("expected embed in a2")
    a3 = on_completed(rt, user_id=1, job_id="j", result_id="fid", result_bytes=1024)
    if a3["embed"] is None:
        pytest.fail("expected embed in a3")


def test_turkic_runtime_skips_when_no_user() -> None:
    rt = _rt()
    a1 = on_started(rt, user_id=None, job_id="x", queue="turkic")
    assert a1["embed"] is None and a1["user_id"] == 0
    a2 = on_failed(
        rt, user_id=None, job_id="x", error_kind="system", message="boom", status="failed"
    )
    assert a2["embed"] is None


def test_turkic_completed_without_user_skips() -> None:
    rt = _rt()
    a = on_completed(rt, user_id=None, job_id="j5", result_id="fid", result_bytes=10)
    assert a["embed"] is None


def test_turkic_progress_without_message_with_user_and_failed_with_user() -> None:
    rt = _rt()
    on_started(rt, user_id=7, job_id="j6", queue="turkic")
    a1 = on_progress(rt, user_id=7, job_id="j6", progress=25, message=None)
    assert a1["embed"] is not None and a1["user_id"] == 7
    a2 = on_failed(rt, user_id=7, job_id="j6", error_kind="system", message="x", status="failed")
    assert a2["embed"] is not None and a2["user_id"] == 7


def test_turkic_progress_without_user_skips() -> None:
    rt = _rt()
    a = on_progress(rt, user_id=None, job_id="j7", progress=1, message=None)
    assert a["embed"] is None
