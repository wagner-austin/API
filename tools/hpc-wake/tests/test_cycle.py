"""One poll, end to end: ledger to accounting to board to closures."""

from __future__ import annotations

import pathlib

import pytest
from hpc3.clusters.hpc3 import HPC3
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.workspace import WorkspaceConnection, decode_workspace_connection
from hpc3.core import ledger
from platform_core.error_codes import McpClientErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue, require_str
from platform_core.mcp_client import McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments

from hpc_wake import _test_hooks
from hpc_wake.announce import MARKER
from hpc_wake.cycle import run_cycle
from tests.conftest import CONFIGURED_ENV, FakeRun, pin_env


def _connection(tmp_path: pathlib.Path) -> WorkspaceConnection:
    """Decode a minimal workspace connection rooted at ``tmp_path``.

    Args:
        tmp_path: Directory the ledger resolves into.

    Returns:
        The connection, with a real ledger path under ``tmp_path``.
    """
    document: dict[str, JSONValue] = {
        "cluster": "hpc3",
        "host": "hpc3",
        "root": "/pub/w",
        "ledger": "ledger.jsonl",
        "quiet_seconds": 1800,
    }
    return decode_workspace_connection(document, config_dir=tmp_path)


def _entry(job_id: str, *, submitter: str | None = "label-a-0906") -> LedgerEntry:
    """Build a ledger entry accounting will be asked about.

    Args:
        job_id: The job's id.
        submitter: The recorded board label, ``""``, or None.

    Returns:
        The entry.
    """
    return LedgerEntry(
        job_id=job_id,
        project="abl",
        name=f"abl.job-{job_id}",
        host="hpc3",
        partition="free-gpu",
        submitted_at="2026-09-06T05:00:00+00:00",
        log_dir="/pub/w/logs",
        deterministic=False,
        experiment={"arm": "x"},
        image_digest="",
        submitter=submitter,
        artifact=None,
    )


def _sacct_row(job_id: str, state: str, *, elapsed: int = 4688) -> str:
    """Render one accounting row the way ``sacct -P`` does.

    Args:
        job_id: The row's id, task or aggregate.
        state: The reported state, suffix and all.
        elapsed: ``ElapsedRaw`` seconds.

    Returns:
        The pipe-delimited row.
    """
    tres = "billing=8,cpu=8,gres/gpu=1"
    return f"{job_id}|abl.job-{job_id}|free-gpu|{state}|{elapsed}|{tres}|hpc3-gpu-18-02"


def _write_ledger(tmp_path: pathlib.Path, entries: list[LedgerEntry]) -> pathlib.Path:
    """Record submissions the way the production writer does.

    Args:
        tmp_path: Directory holding the ledger.
        entries: What was submitted.

    Returns:
        The ledger path.
    """
    path = tmp_path / "ledger.jsonl"
    for entry in entries:
        ledger.append(path, entry)
    return path


class TestQuietCycles:
    def test_an_empty_ledger_reports_and_asks_nothing(
        self, tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([])

        run_cycle(_connection(tmp_path), HPC3)

        assert emitted == ["ledger is empty; nothing has been submitted from this machine"]
        assert fake_run.commands == []

    def test_a_fully_closed_ledger_reports_and_asks_nothing(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([])
        path = _write_ledger(tmp_path, [_entry("101")])
        ledger.append_closure(
            ledger.closure_path(path),
            {
                "job_id": "101",
                "state": "COMPLETED",
                "closed_at": frozen_clock,
                "elapsed_seconds": 4688,
            },
        )

        run_cycle(_connection(tmp_path), HPC3)

        assert emitted == ["1 recorded, all closed; nothing to announce"]
        assert fake_run.commands == []

    def test_a_still_running_job_posts_nothing_and_closes_nothing(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
    ) -> None:
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost([])
        path = _write_ledger(tmp_path, [_entry("101")])
        fake_run.add("sacct", stdout=_sacct_row("101", "RUNNING") + "\n")

        run_cycle(_connection(tmp_path), HPC3)

        assert emitted == ["1 open job(s), none newly terminal"]
        assert not ledger.closure_path(path).exists()


class TestAnnouncingCycles:
    def test_a_completed_job_is_posted_with_its_tag_then_closed(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
    ) -> None:
        pin_env(CONFIGURED_ENV)
        fake_http = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = fake_http
        path = _write_ledger(tmp_path, [_entry("101")])
        fake_run.add("sacct", stdout=_sacct_row("101", "COMPLETED") + "\n")

        run_cycle(_connection(tmp_path), HPC3)

        body = require_str(sent_arguments(fake_http.bodies[0]), "body")
        assert body.startswith(f"{MARKER} abl: 1 job(s) ended (COMPLETED x1)")
        assert "101 abl.job-101 COMPLETED 4688s" in body
        assert "@label-a-0906" in body

        closed = ledger.read_closures(ledger.closure_path(path))
        assert closed["101"]["state"] == "COMPLETED"
        assert closed["101"]["closed_at"] == frozen_clock
        assert closed["101"]["elapsed_seconds"] == 4688
        assert emitted == [
            "posted abl: tagged @label-a-0906",
            "cycle: 1 open, 1 newly terminal, closures recorded",
        ]

    def test_a_refused_post_leaves_the_closure_unwritten(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
    ) -> None:
        """Post-then-close is the delivery guarantee: the next cycle must
        retry an announcement the board never accepted."""
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost(
            [McpHttpResponse(status=401, body="unauthorized", content_type="text/plain")]
        )
        path = _write_ledger(tmp_path, [_entry("101")])
        fake_run.add("sacct", stdout=_sacct_row("101", "COMPLETED") + "\n")

        with pytest.raises(AppError) as caught:
            run_cycle(_connection(tmp_path), HPC3)

        assert caught.value.code is McpClientErrorCode.HTTP_STATUS
        assert not ledger.closure_path(path).exists()

    def test_an_aggregate_row_announces_only_the_tasks_not_already_closed(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
    ) -> None:
        """``closures_for`` expands a cancelled pending aggregate to every
        task it names; re-announcing the already-closed ones would repeat
        old news on every later cycle that sees the aggregate."""
        pin_env(CONFIGURED_ENV)
        fake_http = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = fake_http
        path = _write_ledger(tmp_path, [_entry("555_2"), _entry("555_3")])
        ledger.append_closure(
            ledger.closure_path(path),
            {
                "job_id": "555_2",
                "state": "CANCELLED",
                "closed_at": "2026-09-06T06:00:00+00:00",
                "elapsed_seconds": 0,
            },
        )
        fake_run.add("sacct", stdout=_sacct_row("555_[2-3]", "CANCELLED by 99", elapsed=0) + "\n")

        run_cycle(_connection(tmp_path), HPC3)

        body = require_str(sent_arguments(fake_http.bodies[0]), "body")
        assert "555_3" in body
        assert "555_2 " not in body
        closed = ledger.read_closures(ledger.closure_path(path))
        assert set(closed) == {"555_2", "555_3"}
        assert closed["555_3"]["closed_at"] == frozen_clock

    def test_a_job_with_no_recorded_label_is_announced_untagged(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
    ) -> None:
        pin_env(CONFIGURED_ENV)
        fake_http = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = fake_http
        _write_ledger(tmp_path, [_entry("101", submitter=None)])
        fake_run.add("sacct", stdout=_sacct_row("101", "COMPLETED") + "\n")

        run_cycle(_connection(tmp_path), HPC3)

        body = require_str(sent_arguments(fake_http.bodies[0]), "body")
        assert "@" not in body
        assert emitted[0] == "posted abl: no submitter label on record"
