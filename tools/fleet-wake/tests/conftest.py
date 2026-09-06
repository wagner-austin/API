"""Shared fakes and the hook reset that keeps tests independent.

Everything here is a FAKE implementing the production Protocol, never a mock,
matching ``hpc-wake``'s and ``board-watch``'s conventions. The seams rebound
are this package's own (``fleet_wake._test_hooks``) and the one sanctioned
environment reader (``platform_core.config.config_test_hooks.get_env``, which
also feeds ``board_watch.config.load_credentials``, so pinning it configures
the whole credential chain from one place).

THE FLEET LEDGER IS A REAL FILE UNDER ``tmp_path``, WRITTEN BY FLEET'S OWN
WRITER. ``records.append_ledger`` does the I/O exactly as ``fleet-run`` does,
so these tests read rows that were produced the way production produces them
rather than rows a fixture invented. That is what makes the
append-only/current-row distinction testable at all: a fixture that wrote one
row per dispatch could never catch a bridge reading raw rows instead of
current ones.

The MCP poster fake is :class:`platform_core.mcp_testing.FakeHttpPost`, shared
with ``platform_core``'s and ``hpc-wake``'s suites rather than copied a third
time.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Final

import pytest
from platform_core.config import config_test_hooks

from fleet_wake import _test_hooks
from fleet_wake.identity import TASK_ID_VARIABLE

#: The clock every test runs against, so position timestamps are assertable.
FROZEN_NOW: Final = 1788700000

#: The standing task id every configured test posts into.
TASK_ID: Final = "df6f1dc8-cd6b-4314-b28a-eb3625390ae0"

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
    original_env = config_test_hooks.get_env
    yield
    _test_hooks.reset_hooks()
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


def _make_frozen_clock() -> Generator[int, None, None]:
    """Pin the bridge clock so position timestamps are assertable.

    Yields:
        The timestamp every position row will record.
    """

    def _now() -> int:
        return FROZEN_NOW

    _test_hooks.now = _now
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


emitted = pytest.fixture(_make_emitted)
frozen_clock = pytest.fixture(_make_frozen_clock)
