"""Shared fakes and the hook reset that keeps tests independent.

Everything here is a FAKE implementing the production Protocol, never a
mock, matching the three sibling bridges' conventions. The seams rebound
are this package's own (``lock_wake._test_hooks``) and the one sanctioned
environment reader (``platform_core.config.config_test_hooks.get_env``,
which also feeds ``board_watch.config.load_credentials``, so pinning it
configures the whole credential chain from one place).

THE JOURNAL AND THE POSITION ARE REAL FILES UNDER ``tmp_path``. The
journal fixture writes lines byte-for-byte in the lock wrapper's own form
(compact JSON, ``\\n``-terminated, seven-digit fractional timestamps), so
these tests read rows shaped the way production shapes them rather than
rows a fixture invented -- including the torn tail and the pre-``agent``
history rows, which are the cases that decide whether the cursor is
correct.

The MCP poster fake is :class:`platform_core.mcp_testing.FakeHttpPost`,
shared with every sibling's suite rather than copied a fifth time.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator
from typing import Final

import pytest
from platform_core.config import config_test_hooks

from lock_wake import _test_hooks
from lock_wake.identity import TASK_ID_VARIABLE

#: The standing task id every configured test posts into.
TASK_ID: Final = "0b892f1e-0000-4000-8000-00000000c0de"

#: The environment the configured tests run in, in full.
CONFIGURED_ENV: Final[dict[str, str]] = {
    "TASKBOARD_MCP_API_KEY": "test-key",
    "CORVIS_TENANT_ID": "2e137b5f-0000-4000-8000-000000000000",
    TASK_ID_VARIABLE: TASK_ID,
}


def journal_line(
    *,
    ts: str = "2026-09-09T19:28:00.0608673Z",
    kind: str = "acquired",
    holder_pid: int = 2688,
    label: str = "up-transcriber",
    op: str = "service-up",
    only: str = "all",
    detail: str = "",
    agent: str | None = "opus-mosh-reboot-0909",
) -> str:
    """One journal line in the lock wrapper's own byte form.

    Args:
        ts: The transition's timestamp, wrapper-form (seven fractional
            digits, trailing ``Z``).
        kind: The transition kind.
        holder_pid: The lock-taking process id (the journal's ``pid``).
        label: The make target's label.
        op: The wrapper operation.
        only: The language scope.
        detail: Kind-specific text.
        agent: The session label, or None for a pre-2026-09-09 history row
            that lacks the key entirely.

    Returns:
        The line, ``\\n``-terminated.
    """
    agent_part = "" if agent is None else f',"agent":"{agent}"'
    return (
        f'{{"ts":"{ts}","kind":"{kind}","pid":{holder_pid},"label":"{label}",'
        f'"op":"{op}","only":"{only}","detail":"{detail}"{agent_part}}}\n'
    )


def stage_journal(tmp_path: pathlib.Path, content: bytes) -> pathlib.Path:
    """Write a journal file with exact bytes.

    Args:
        tmp_path: The test's temporary directory.
        content: The journal's full contents, torn tails included.

    Returns:
        The journal's path.
    """
    journal = tmp_path / ".fleet-events.jsonl"
    journal.write_bytes(content)
    return journal


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
            unset, so a test's environment is this call, not the
            developer's shell.
    """

    def _env(name: str) -> str | None:
        return values.get(name)

    config_test_hooks.get_env = _env


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
