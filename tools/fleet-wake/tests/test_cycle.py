"""One poll, end to end: fleet's ledger to the board to the position record.

INTEGRATION, NOT UNIT. The ledger is a real file written by
``fleet.core.records.append_ledger`` -- the same writer ``fleet-run`` and
``fleet-collect`` use -- and the position file is a real file written by this
package's production hooks. Only the HTTP poster and the clock are fakes,
because those are the two things a test cannot have.

THAT MATTERS FOR ONE CASE IN PARTICULAR. The fleet ledger is APPEND-ONLY, so a
finished dispatch has BOTH a ``running`` row and a terminal row in the file. A
fixture that wrote one row per dispatch would make a bridge reading raw rows
look identical to one reading current rows, and the defect -- announcing every
dispatch twice, once from each row -- would ship. These tests write both rows,
the way production does.
"""

from __future__ import annotations

import pathlib

import pytest
from fleet.contracts.ledger import LedgerEntry, decode_ledger_entry
from fleet.core import records
from platform_core.error_codes_tooling import BoardBridgeErrorCode, McpClientErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject, dump_json_str, require_str
from platform_core.mcp_client import McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments

from fleet_wake import _test_hooks
from fleet_wake.announce import MARKER
from fleet_wake.cycle import load_workspace, run_cycle
from fleet_wake.position import position_path, read_announced
from tests.conftest import CONFIGURED_ENV, TASK_ID, pin_env

DEMO_PROJECT = "tools/fleet"


def _row(
    run_id: str,
    *,
    outcome: str,
    project: str = DEMO_PROJECT,
    agent: str = "opus-fleet-0906",
    exit_code: int = 0,
) -> LedgerEntry:
    """Build one ledger row through the production decoder.

    Args:
        run_id: The dispatch's id.
        outcome: How it ended, or ``running``.
        project: Repo-relative project path.
        agent: Board label of the dispatching session.
        exit_code: The recipe's exit status.

    Returns:
        The decoded row.
    """
    document: JSONObject = {
        "run_id": run_id,
        "node": "lavender",
        "host": "lavender",
        "project": project,
        "agent": agent,
        "session_id": "acc774c0-3bc3-4cce-9dda-c7a12fb99519",
        "started_unix": 1788633781,
        "ended_unix": 1788633884,
        "outcome": outcome,
        "exit_code": exit_code,
        "workers": 12,
        "detail": "",
    }
    return decode_ledger_entry(document)


def _workspace(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a minimal but real fleet workspace document.

    Decoded by fleet's own decoder in the cycle, so a change to that contract
    fails here rather than at the first live run.

    Args:
        tmp_path: Directory the records resolve into.

    Returns:
        Path to the written document.
    """
    document: JSONObject = {
        "nodes": {
            "lavender": {
                "host": "lavender",
                "stage_root": "C:/fleet/stage",
                "logical_cores": 16,
                "ram_gb": 32.0,
                "gpu": None,
                "enabled": True,
                "budget": {
                    "reserved_cores": 2,
                    "reserved_ram_gb": 4.0,
                    "worker_ram_gb": 1.1,
                    "max_concurrent_runs": 2,
                    "max_disk_gb": 20.0,
                },
            }
        },
        "not_dispatchable": {},
        "projects": {
            DEMO_PROJECT: {
                "worker_ram_gb": 1.1,
                "minimum_workers": 2,
                "expected_minutes": 5,
            }
        },
        "ledger": "runs/ledger.jsonl",
        "feed": "runs/feed.jsonl",
        "leases": "runs/leases.json",
    }
    path = tmp_path / "fleet.json"
    path.write_text(dump_json_str(document), encoding="utf-8")
    return path


def _dispatch(
    ledger: pathlib.Path,
    run_id: str,
    *,
    outcome: str,
    project: str = DEMO_PROJECT,
    agent: str = "opus-fleet-0906",
    exit_code: int = 0,
) -> None:
    """Record a dispatch the way production does: running row, then outcome.

    THE POINT OF THIS HELPER. Writing only the terminal row would hide a
    bridge that read every row instead of the current one -- the defect that
    announces every dispatch twice.

    Spelled out rather than taking ``**fields``, because a kwargs passthrough
    types every forwarded value the same and this helper forwards both strings
    and an int.

    Args:
        ledger: The ledger path.
        run_id: The dispatch's id.
        outcome: The terminal outcome to close it with.
        project: Repo-relative project path.
        agent: Board label of the dispatching session.
        exit_code: The recipe's exit status on the terminal row.
    """
    records.append_ledger(ledger, _row(run_id, outcome="running", project=project, agent=agent))
    records.append_ledger(
        ledger,
        _row(run_id, outcome=outcome, project=project, agent=agent, exit_code=exit_code),
    )


class TestQuietCycles:
    def test_an_empty_ledger_reports_and_posts_nothing(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([])

        run_cycle(load_workspace(_workspace(tmp_path)))

        assert emitted == ["ledger is empty; nothing has been dispatched from this machine"]

    def test_a_running_dispatch_posts_nothing_and_records_nothing(
        self, tmp_path: pathlib.Path, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([])
        config = _workspace(tmp_path)
        ledger = tmp_path / "runs" / "ledger.jsonl"
        records.append_ledger(ledger, _row("a", outcome="running"))

        run_cycle(load_workspace(config))

        assert emitted == ["1 dispatch(es) recorded, none newly terminal"]
        assert not position_path(ledger).exists()

    def test_a_second_cycle_over_the_same_ledger_posts_nothing(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        """THE WHOLE REASON THE POSITION FILE EXISTS. A scheduler runs this
        every few minutes over a ledger that mostly does not change."""
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([posted_ok()])
        config = _workspace(tmp_path)
        ledger = tmp_path / "runs" / "ledger.jsonl"
        _dispatch(ledger, "a", outcome="passed")
        run_cycle(load_workspace(config))

        _test_hooks.http_post = FakeHttpPost([])
        run_cycle(load_workspace(config))

        assert emitted[-1] == "1 dispatch(es) recorded, none newly terminal"


class TestAnnouncingCycles:
    def test_a_finished_dispatch_is_posted_then_recorded(
        self, tmp_path: pathlib.Path, emitted: list[str], frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        fake = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = fake
        config = _workspace(tmp_path)
        ledger = tmp_path / "runs" / "ledger.jsonl"
        _dispatch(ledger, "tools-fleet-1788633781", outcome="passed")

        run_cycle(load_workspace(config))

        arguments = sent_arguments(fake.bodies[0])
        assert arguments["taskId"] == TASK_ID
        assert arguments["agent"] == "bridge-fleet-wake-0906"
        body = require_str(arguments, "body")
        assert body.startswith(f"{MARKER} tools/fleet: 1 dispatch(es) ended (passed x1)")
        assert "tools-fleet-1788633781 lavender passed exit 0 103s" in body
        assert "@opus-fleet-0906" in body
        assert read_announced(position_path(ledger)) == frozenset({"tools-fleet-1788633781"})
        assert emitted == [
            "posted tools/fleet: tagged @opus-fleet-0906",
            "cycle: 1 recorded, 1 newly terminal, positions recorded",
        ]

    def test_a_dispatch_with_both_rows_is_announced_exactly_once(
        self, tmp_path: pathlib.Path, frozen_clock: int
    ) -> None:
        """THE APPEND-ONLY TRAP. The ledger holds a ``running`` row AND a
        ``passed`` row for this dispatch. A bridge reading raw rows rather
        than current ones would post twice -- and the scripted poster, holding
        exactly one reply, raises rather than letting that pass."""
        pin_env(CONFIGURED_ENV)
        fake = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = fake
        config = _workspace(tmp_path)
        ledger = tmp_path / "runs" / "ledger.jsonl"
        _dispatch(ledger, "a", outcome="passed")

        run_cycle(load_workspace(config))

        assert len(fake.bodies) == 1
        body = require_str(sent_arguments(fake.bodies[0]), "body")
        assert "1 dispatch(es) ended" in body

    def test_two_sessions_get_one_post_each(
        self, tmp_path: pathlib.Path, frozen_clock: int
    ) -> None:
        """Each mention must land on the session that dispatched, not on
        whichever one happened to be first in the ledger."""
        pin_env(CONFIGURED_ENV)
        fake = FakeHttpPost([posted_ok(), posted_ok()])
        _test_hooks.http_post = fake
        config = _workspace(tmp_path)
        ledger = tmp_path / "runs" / "ledger.jsonl"
        _dispatch(ledger, "a", outcome="passed", agent="opus-one-0906")
        _dispatch(ledger, "b", outcome="failed", agent="opus-two-0906")

        run_cycle(load_workspace(config))

        bodies = [require_str(sent_arguments(sent), "body") for sent in fake.bodies]
        assert len(bodies) == 2
        assert "@opus-one-0906" in bodies[0]
        assert "@opus-two-0906" in bodies[1]

    def test_a_failure_is_announced_as_a_failure(
        self, tmp_path: pathlib.Path, frozen_clock: int
    ) -> None:
        pin_env(CONFIGURED_ENV)
        fake = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = fake
        config = _workspace(tmp_path)
        ledger = tmp_path / "runs" / "ledger.jsonl"
        _dispatch(ledger, "a", outcome="failed", exit_code=1)

        run_cycle(load_workspace(config))

        body = require_str(sent_arguments(fake.bodies[0]), "body")
        assert "(failed x1)" in body
        assert "a lavender failed exit 1" in body


class TestFailuresEndTheCycle:
    def test_a_refused_post_leaves_the_position_unwritten(
        self, tmp_path: pathlib.Path, frozen_clock: int
    ) -> None:
        """POST-THEN-WRITE IS THE DELIVERY GUARANTEE. The next cycle must
        retry an announcement the board never accepted; recording it anyway
        would mean that dispatch is never announced at all, which is the
        silence this bridge exists to remove."""
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost(
            [McpHttpResponse(status=401, body="unauthorized", content_type="text/plain")]
        )
        config = _workspace(tmp_path)
        ledger = tmp_path / "runs" / "ledger.jsonl"
        _dispatch(ledger, "a", outcome="passed")

        with pytest.raises(AppError) as caught:
            run_cycle(load_workspace(config))

        assert caught.value.code is McpClientErrorCode.HTTP_STATUS
        assert not position_path(ledger).exists()

    def test_the_retry_actually_posts_on_the_next_cycle(
        self, tmp_path: pathlib.Path, frozen_clock: int
    ) -> None:
        """The other half of the guarantee, asserted rather than assumed: a
        failed cycle must leave the work announceable, not merely unrecorded."""
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost(
            [McpHttpResponse(status=401, body="unauthorized", content_type="text/plain")]
        )
        config = _workspace(tmp_path)
        ledger = tmp_path / "runs" / "ledger.jsonl"
        _dispatch(ledger, "a", outcome="passed")
        with pytest.raises(AppError):
            run_cycle(load_workspace(config))

        retry = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = retry
        run_cycle(load_workspace(config))

        assert len(retry.bodies) == 1
        assert read_announced(position_path(ledger)) == frozenset({"a"})

    def test_a_second_group_is_not_posted_after_the_first_is_refused(
        self, tmp_path: pathlib.Path, frozen_clock: int
    ) -> None:
        """The cycle stops on the failure rather than carrying on, so the
        position file cannot end up holding a group whose post never landed."""
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost(
            [McpHttpResponse(status=401, body="unauthorized", content_type="text/plain")]
        )
        config = _workspace(tmp_path)
        ledger = tmp_path / "runs" / "ledger.jsonl"
        _dispatch(ledger, "a", outcome="passed", agent="opus-one-0906")
        _dispatch(ledger, "b", outcome="passed", agent="opus-two-0906")

        with pytest.raises(AppError):
            run_cycle(load_workspace(config))

        assert not position_path(ledger).exists()

    def test_a_missing_task_id_refuses_before_reading_anything(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Configuration is checked before work: a bridge that read a ledger
        and then discovered it had nowhere to post would do the reading on
        every scheduled cycle forever."""
        pin_env(
            {
                "TASKBOARD_MCP_API_KEY": "test-key",
                "CORVIS_TENANT_ID": "2e137b5f-0000-4000-8000-000000000000",
            }
        )
        _test_hooks.http_post = FakeHttpPost([])

        with pytest.raises(AppError) as caught:
            run_cycle(load_workspace(_workspace(tmp_path)))

        assert caught.value.code is BoardBridgeErrorCode.TASK_ID_MISSING
