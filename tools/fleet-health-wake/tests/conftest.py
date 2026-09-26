"""Shared fakes and the hook reset that keeps tests independent.

Everything here is a FAKE implementing the production Protocol, never a
mock, as in lock-wake's suite. The seams rebound are this package's own
(``fleet_health_wake._test_hooks``) and the one sanctioned environment
reader (``platform_core.config.config_test_hooks.get_env``, which also
feeds ``board_watch.config.load_credentials``).

THE JOURNAL AND THE POSITION ARE REAL FILES UNDER ``tmp_path``, and the
journal lines are the fleet audit's own bytes: :data:`TRANSITIONS_LINE` and
:data:`REFUSED_LINE` are what fleet-mcp's ``encodeHealthEvent`` writes
(compact JSON, keys in its order, ``\\n``-terminated, a UTF-8 em dash in
the body), so a decoder that drifted from the writer fails here.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator
from typing import Final

import pytest
from platform_core.config import config_test_hooks
from platform_core.journal_cursor import (
    cursor_path,
    file_is_present,
    read_file_bytes,
    read_offset,
    write_file_text,
    write_offset,
)
from platform_core.mcp_testing import DECLARED_TASKBOARD_URL

from fleet_health_wake import _test_hooks
from fleet_health_wake.identity import CURSOR_READER, TASK_ID_VARIABLE

#: The standing task id every configured test posts into.
TASK_ID: Final = "88b20894-0000-4000-8000-00000000c0de"

#: The environment the configured tests run in, in full.
CONFIGURED_ENV: Final[dict[str, str]] = {
    "BOARD_WATCH_URL": DECLARED_TASKBOARD_URL,
    "TASKBOARD_MCP_API_KEY": "test-key",
    "CORVIS_TENANT_ID": "2e137b5f-0000-4000-8000-000000000000",
    TASK_ID_VARIABLE: TASK_ID,
}

#: The body of the 06:14Z run's line, as fleet-mcp renders it.
TRANSITIONS_BODY: Final = (
    "FLEET HEALTH at 2026-09-26T06:14:13.504Z (audit on AustinPC, 2 node(s) compared "
    "with the run before): 1 newly failing, 1 recovered.\n"
    "FAILING  austinpc boot.up.task:\\API-FleetNode-sedona-3min — Ready, S4U, last run "
    "2026-09-25T23:12:01-07:00 result 0x1 (runs if 0x0 / 0x41303)\n"
    "RECOVERED  sedona boot.up.service:cloudflared — Running, starts Auto"
)

#: That line, byte for byte as ``encodeHealthEvent`` writes it.
TRANSITIONS_LINE: Final = (
    '{"at":"2026-09-26T06:14:13.504Z","kind":"transitions","key":"2026-09-26T06:14:13.504Z",'
    '"body":"FLEET HEALTH at 2026-09-26T06:14:13.504Z (audit on AustinPC, 2 node(s) compared '
    "with the run before): 1 newly failing, 1 recovered.\\nFAILING  austinpc "
    "boot.up.task:\\\\API-FleetNode-sedona-3min — Ready, S4U, last run "
    "2026-09-25T23:12:01-07:00 result 0x1 (runs if 0x0 / 0x41303)\\nRECOVERED  sedona "
    'boot.up.service:cloudflared — Running, starts Auto"}\n'
)

#: The body of a refused run's line.
REFUSED_BODY: Final = (
    "FLEET AUDIT REFUSED TO RUN at 2026-09-26T06:20:00.000Z: its dist/ does not match its "
    "src/ (hash-mismatch), so it probed nothing and wrote no snapshot; fleet_status keeps "
    "serving the last one, aging. Remedy on the hub: npm run build in fleet-mcp. This is "
    "said once per stale build."
)

#: That line, as ``encodeHealthEvent`` writes it.
REFUSED_LINE: Final = (
    '{"at":"2026-09-26T06:20:00.000Z","kind":"refused","key":"hash-mismatch:'
    + "a" * 64
    + '","body":"'
    + REFUSED_BODY
    + '"}\n'
)


def stage_journal(tmp_path: pathlib.Path, content: bytes) -> pathlib.Path:
    """Write the health journal with exact bytes.

    Args:
        tmp_path: The test's temporary directory.
        content: The journal's full contents, torn tails included.

    Returns:
        The journal's path.
    """
    journal = tmp_path / "health-events.jsonl"
    journal.write_bytes(content)
    return journal


def offset_of(journal: pathlib.Path) -> int:
    """This bridge's recorded position in the journal, as the cycle reads it.

    Args:
        journal: The journal's path.

    Returns:
        The offset in its cursor file, 0 when the file was never written.
    """
    return read_offset(file_is_present, read_file_bytes, cursor_path(journal, CURSOR_READER))


def set_offset(journal: pathlib.Path, offset: int) -> None:
    """Record this bridge's position in the journal, as the cycle writes it.

    Args:
        journal: The journal's path.
        offset: The offset to record.
    """
    write_offset(write_file_text, cursor_path(journal, CURSOR_READER), offset)


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
        values: The variables that are set; every other reads as unset.
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
