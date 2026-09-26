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

import pathlib
from collections.abc import Generator
from typing import Final

import pytest
from fleet.contracts.budget import NodeBudget
from fleet.contracts.node import NodeConfig, encode_node_config
from fleet.contracts.project import ProjectConfig, encode_project_config
from platform_core.config import config_test_hooks
from platform_core.json_utils import JSONObject, dump_json_str
from platform_core.mcp_testing import DECLARED_TASKBOARD_URL

from fleet_wake import _test_hooks
from fleet_wake.identity import TASK_ID_VARIABLE

#: The clock every test runs against, so position timestamps are assertable.
FROZEN_NOW: Final = 1788700000

#: The standing task id every configured test posts into.
TASK_ID: Final = "df6f1dc8-cd6b-4314-b28a-eb3625390ae0"

#: The environment the configured tests run in, in full.
CONFIGURED_ENV: Final[dict[str, str]] = {
    # The override, so no test reads the MCPs checkout's endpoint declaration
    # (board-watch's own suite tests that default).
    "BOARD_WATCH_URL": DECLARED_TASKBOARD_URL,
    "TASKBOARD_MCP_API_KEY": "test-key",
    "CORVIS_TENANT_ID": "2e137b5f-0000-4000-8000-000000000000",
    TASK_ID_VARIABLE: TASK_ID,
}


def write_fleet_workspace(tmp_path: pathlib.Path, *, project: str) -> pathlib.Path:
    """Write a one-node, one-project workspace THROUGH FLEET'S OWN ENCODERS.

    THE DRIFT THIS REMOVES, and it cost fleet-wake eleven hours of red CI
    on 2026-09-21. Both suites hand-built this document as a literal, and
    both spelled the project entry as three keys. `339c66e30` then made
    `source` a required field of `ProjectConfig` (board task fd5cabfa, A2),
    the literals did not follow, and all fourteen cycle and CLI cases died
    inside `decode_project_config` before reaching a line of this package.
    The suite that proves fleet-wake reads a workspace cannot be the one
    place that invents what a workspace looks like.

    `encode_node_config` and `encode_project_config` are the writers the
    registry itself round-trips through, and `NodeConfig`/`ProjectConfig`
    are total TypedDicts, so the NEXT required field fails mypy here rather
    than fourteen tests at runtime. The same reasoning as the ledger rows
    this module's header describes: produced the way production produces
    them, never invented beside them.

    Args:
        tmp_path: Directory the workspace's records resolve into.
        project: The project key, which the tests assert dispatch rows and
            board posts against.

    Returns:
        Path to the written document.
    """
    document: JSONObject = {
        "nodes": {
            "lavender": encode_node_config(
                NodeConfig(
                    host="lavender",
                    platform="windows",
                    stage_root="C:/fleet/stage",
                    logical_cores=16,
                    ram_gb=32.0,
                    gpu=None,
                    enabled=True,
                    test_database=False,
                    budget=NodeBudget(
                        reserved_cores=2,
                        reserved_ram_gb=4.0,
                        worker_ram_gb=1.1,
                        max_concurrent_runs=2,
                        max_disk_gb=20.0,
                    ),
                )
            )
        },
        "not_dispatchable": {},
        # Required by the workspace contract since board task 140e7042. Empty
        # here because nothing this package does reads it: the field scopes a
        # dispatch's ARCHIVE, and fleet-wake reads the workspace to find nodes
        # and projects, never to export one.
        "data_paths": {},
        "projects": {
            project: encode_project_config(
                ProjectConfig(
                    worker_ram_gb=1.1,
                    minimum_workers=2,
                    expected_minutes=5,
                    exclusive_resources=(),
                    external_paths=(),
                    required_tags=(),
                    # NULL, SPELLED OUT: this bridge announces dispatch
                    # outcomes and never checks anything out, so the
                    # fixture states the contract's "no remote" rather
                    # than implying a remote nothing here would fetch.
                    source=None,
                )
            )
        },
        "ledger": "runs/ledger.jsonl",
        "feed": "runs/feed.jsonl",
        "leases": "runs/leases.json",
    }
    path = tmp_path / "fleet.json"
    path.write_text(dump_json_str(document), encoding="utf-8")
    return path


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
