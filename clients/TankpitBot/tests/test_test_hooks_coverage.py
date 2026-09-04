"""Coverage tests for _test_hooks/runtime.py real implementations."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import psutil

from tankpit_bot._test_hooks.runtime import (
    _git_head_ref,
    _kill_browser_engines,
    _real_kill_browser_processes,
    _real_resolve_build_ref,
    _real_start_watchdog,
)


def test_real_start_watchdog_fires_callback() -> None:
    """_real_start_watchdog arms a daemon timer that calls the callback."""
    fired = threading.Event()

    def on_fire() -> None:
        fired.set()

    _real_start_watchdog(0.05, on_fire)

    assert fired.wait(timeout=2.0), "watchdog callback did not fire within 2s"


class _FakeEngineProcess:
    """A killable process double matching the psutil slice the loop reads."""

    def __init__(
        self,
        pid: int,
        name: str,
        *,
        name_gone: bool = False,
        kill_gone: bool = False,
    ) -> None:
        self.pid = pid
        self._name = name
        self._name_gone = name_gone
        self._kill_gone = kill_gone
        self.kill_calls = 0

    def name(self) -> str:
        if self._name_gone:
            raise psutil.NoSuchProcess(self.pid)
        return self._name

    def kill(self) -> None:
        self.kill_calls += 1
        if self._kill_gone:
            raise psutil.NoSuchProcess(self.pid)


class TestKillBrowserEngines:
    def test_kills_engines_and_spares_the_driver(self) -> None:
        """Chromium names die (case-insensitively); node.exe survives.

        Sparing the driver is the rung's whole design: it must stay
        alive to observe the engine's death and resolve the pending
        ``browser.close()``.
        """
        engine = _FakeEngineProcess(101, "Chrome.exe")
        headless = _FakeEngineProcess(102, "headless_shell")
        driver = _FakeEngineProcess(103, "node.exe")

        killed = _kill_browser_engines([engine, headless, driver])

        assert killed == [101, 102]
        assert engine.kill_calls == 1
        assert headless.kill_calls == 1
        assert driver.kill_calls == 0

    def test_a_process_gone_before_the_name_read_is_skipped(self) -> None:
        """The OS boundary: exit between enumeration and name() is legal."""
        gone = _FakeEngineProcess(201, "chrome.exe", name_gone=True)

        assert _kill_browser_engines([gone]) == []
        assert gone.kill_calls == 0

    def test_a_process_gone_before_the_kill_is_not_reported_killed(self) -> None:
        """Exit between name() and kill() removes it without our credit."""
        racing = _FakeEngineProcess(301, "chromium", kill_gone=True)

        assert _kill_browser_engines([racing]) == []
        assert racing.kill_calls == 1


def test_git_head_ref_answers_from_a_real_repository() -> None:
    """Inside the checkout the ref IS the repository's HEAD."""
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    assert _git_head_ref(".") == expected
    assert len(expected) == 40


def test_git_head_ref_answers_empty_outside_any_repository(tmp_path: Path) -> None:
    """A directory with no repository answers "" — a fact, not an error.

    The release tree is a ``git archive`` and HAS no repository; the
    stamp records that nothing identified the build rather than
    inventing one.
    """
    assert _git_head_ref(str(tmp_path)) == ""


def test_resolve_build_ref_prefers_the_stamped_environment() -> None:
    """The container class: TANKPIT_BUILD_REF answers without asking git."""
    from tankpit_bot._test_hooks import env as env_hooks
    from tests.conftest import FakeEnv

    original = env_hooks.get_env
    fake = FakeEnv()
    fake.set("TANKPIT_BUILD_REF", "v0.1.0-abc1234")
    env_hooks.get_env = fake
    try:
        assert _real_resolve_build_ref() == "v0.1.0-abc1234"
    finally:
        env_hooks.get_env = original


def test_resolve_build_ref_falls_to_git_in_a_checkout() -> None:
    """The checkout class: no stamp in the environment, git answers."""
    from tankpit_bot._test_hooks import env as env_hooks
    from tests.conftest import FakeEnv

    original = env_hooks.get_env
    env_hooks.get_env = FakeEnv()
    try:
        ref = _real_resolve_build_ref()
    finally:
        env_hooks.get_env = original

    assert len(ref) == 40


def test_real_kill_browser_processes_spares_non_engine_children() -> None:
    """The real enumeration finds this process's children and the name
    filter passes over a python child untouched.

    A real sleeper child stands in the descendant tree; it is not a
    browser engine, so the sweep returns nothing and the child stays
    alive to be reaped by the test.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    try:
        killed = _real_kill_browser_processes()
        assert child.pid not in killed
        assert child.poll() is None, "the sleeper must survive the sweep"
    finally:
        child.kill()
        child.wait(timeout=10)
