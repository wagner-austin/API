"""One poll, end to end: ledger to accounting to board to closures."""

from __future__ import annotations

import pathlib

import pytest
from hpc3.clusters.hpc3 import HPC3
from hpc3.contracts.ledger import LedgerEntry
from hpc3.contracts.workspace import WorkspaceConnection, decode_workspace_connection
from hpc3.core import _test_hooks as hpc3_hooks
from hpc3.core import ledger
from platform_core.error_codes_tooling import McpClientErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue, require_str
from platform_core.mcp_client import McpHttpResponse
from platform_core.mcp_testing import FakeHttpPost, posted_ok, sent_arguments

from hpc_wake import _test_hooks
from hpc_wake.announce import MARKER
from hpc_wake.cycle import run_cycle
from hpc_wake.pending import PendingClosure, pending_path, read_pending, write_pending
from hpc_wake.settling import MAX_HOLD_SECONDS, SETTLE_SECONDS
from tests.conftest import (
    CONFIGURED_ENV,
    FROZEN_EPOCH,
    TASK_ID,
    FakeRun,
    MovingClock,
    pin_env,
)


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


def _install_accounting(ended_ids: list[str]) -> None:
    """Point the cluster seam at a fresh fake reporting these jobs as ended.

    Rebuilt rather than extended because :class:`FakeRun` matches by
    substring and the FIRST matching rule wins: adding a second ``sacct``
    rule to an existing fake is dead code, and a multi-cycle test built that
    way silently replays its first response forever.

    Args:
        ended_ids: Every job accounting should now report COMPLETED. All of
            them, not only the newest -- that is what ``sacct`` does.
    """
    fake = FakeRun()
    rows = "".join(_sacct_row(job_id, "COMPLETED") + "\n" for job_id in ended_ids)
    fake.add("sacct", stdout=rows)
    hpc3_hooks.run = fake


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

        assert emitted == ["1 open job(s), none newly terminal and none waiting"]
        assert not ledger.closure_path(path).exists()


class TestAnnouncingCycles:
    def test_a_completed_job_is_posted_with_its_tag_then_closed(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
        moving_clock: MovingClock,
    ) -> None:
        """The full path, across the two cycles it now genuinely takes.

        The first cycle OBSERVES the ending and posts nothing -- it cannot
        know yet whether more members of the same sweep are seconds behind.
        The second, one settle window later, finds the group quiet and
        announces it. Asserting this over two cycles rather than pinning a
        clock far enough ahead to collapse them is deliberate: the two-cycle
        shape IS the behaviour, and a test that hid it would pass equally
        well against the per-job posting this replaced.
        """
        pin_env(CONFIGURED_ENV)
        fake_http = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = fake_http
        path = _write_ledger(tmp_path, [_entry("101")])
        fake_run.add("sacct", stdout=_sacct_row("101", "COMPLETED") + "\n")

        run_cycle(_connection(tmp_path), HPC3)

        assert fake_http.bodies == []
        assert not ledger.closure_path(path).exists()
        assert emitted == ["1 ending(s) waiting, none settled; 1 arrived this cycle"]

        moving_clock.advance(SETTLE_SECONDS)
        run_cycle(_connection(tmp_path), HPC3)

        arguments = sent_arguments(fake_http.bodies[0])
        # THE IDENTITY BINDING, asserted here since board.py was deleted.
        # That module existed to bind these two constants into the call and
        # was a wrapper doing nothing else; the binding itself is still worth
        # pinning, because posting under the wrong label is refused by the
        # board permanently and posting under the wrong cwd silently rewrites
        # this bridge's audit trail.
        assert arguments["agent"] == "bridge-hpc-wake-0906"
        assert arguments["sessionId"] == "b6048b2e-2e32-5247-a488-7b4ccc35f2cc"
        assert arguments["cwd"] == "service://hpc-wake"
        assert arguments["taskId"] == TASK_ID
        assert arguments["kind"] == "note"

        body = require_str(arguments, "body")
        assert body.startswith(f"{MARKER} abl: 1 job(s) ended (COMPLETED x1)")
        assert "101 abl.job-101 COMPLETED 4688s" in body
        assert "@label-a-0906" in body

        closed = ledger.read_closures(ledger.closure_path(path))
        assert closed["101"]["state"] == "COMPLETED"
        assert closed["101"]["closed_at"] == frozen_clock
        assert closed["101"]["elapsed_seconds"] == 4688
        assert emitted == [
            "1 ending(s) waiting, none settled; 1 arrived this cycle",
            "posted abl: tagged @label-a-0906",
            "cycle: 1 open, 0 newly terminal, 1 announced, 0 still settling",
        ]
        assert not pending_path(path).read_text(encoding="utf-8")

    def test_a_refused_post_leaves_the_closure_unwritten(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
        moving_clock: MovingClock,
    ) -> None:
        """Post-then-close is the delivery guarantee: the next cycle must
        retry an announcement the board never accepted.

        THE PENDING RECORD MUST SURVIVE THE REFUSAL TOO, which is the half
        settling added. The ending is durable before the post is attempted,
        so a board that refuses leaves the record in place and the retry
        needs nothing from the cluster -- if the refusal had cleared it, the
        ending would be closed by nobody and announced by nobody.
        """
        pin_env(CONFIGURED_ENV)
        _test_hooks.http_post = FakeHttpPost(
            [McpHttpResponse(status=401, body="unauthorized", content_type="text/plain")]
        )
        path = _write_ledger(tmp_path, [_entry("101")])
        fake_run.add("sacct", stdout=_sacct_row("101", "COMPLETED") + "\n")

        run_cycle(_connection(tmp_path), HPC3)
        moving_clock.advance(SETTLE_SECONDS)

        with pytest.raises(AppError) as caught:
            run_cycle(_connection(tmp_path), HPC3)

        assert caught.value.code is McpClientErrorCode.HTTP_STATUS
        assert not ledger.closure_path(path).exists()
        still_waiting = read_pending(pending_path(path))
        assert [r["closure"]["job_id"] for r in still_waiting] == ["101"]

    def test_an_aggregate_row_announces_only_the_tasks_not_already_closed(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
        moving_clock: MovingClock,
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
        moving_clock.advance(SETTLE_SECONDS)
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
        moving_clock: MovingClock,
    ) -> None:
        pin_env(CONFIGURED_ENV)
        fake_http = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = fake_http
        _write_ledger(tmp_path, [_entry("101", submitter=None)])
        fake_run.add("sacct", stdout=_sacct_row("101", "COMPLETED") + "\n")

        run_cycle(_connection(tmp_path), HPC3)
        moving_clock.advance(SETTLE_SECONDS)
        run_cycle(_connection(tmp_path), HPC3)

        body = require_str(sent_arguments(fake_http.bodies[0]), "body")
        assert "@" not in body
        assert emitted[1] == "posted abl: no submitter label on record"


class TestSettlingAcrossCycles:
    """The measured defect and its bounds, end to end through run_cycle."""

    def test_a_trickling_array_produces_one_post_not_one_per_job(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
        moving_clock: MovingClock,
    ) -> None:
        """THE DEFECT THIS PACKAGE WAS CHANGED FOR, as an end-to-end test.

        Six members of one array finish 180 seconds apart -- the median gap
        measured on the live board in the burst that produced 116 posts in
        24 hours. Each is observed by its own cycle, exactly as the real
        poller would observe it.

        Under the previous behaviour this was six posts. It must now be one,
        carrying all six, once the array stops and the group goes quiet.
        """
        pin_env(CONFIGURED_ENV)
        fake_http = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = fake_http
        ids = [f"777_{index}" for index in range(6)]
        path = _write_ledger(tmp_path, [_entry(job_id) for job_id in ids])

        for count in range(1, len(ids) + 1):
            # Accounting reports EVERY ended job on every poll, not just the
            # newest, so the fake is rebuilt each cycle with the cumulative
            # set. Appending rules to one fake instead would have matched the
            # first rule every time -- first match wins -- and the array
            # would never have trickled at all. It did not, at first, and the
            # test passed a post it should have caught.
            _install_accounting(ids[:count])
            run_cycle(_connection(tmp_path), HPC3)
            assert fake_http.bodies == [], f"posted after {count} of {len(ids)} endings"
            moving_clock.advance(180)
        _install_accounting(ids)

        # The array has stopped. One settle window of quiet, then one post.
        moving_clock.advance(SETTLE_SECONDS)
        run_cycle(_connection(tmp_path), HPC3)

        assert len(fake_http.bodies) == 1
        body = require_str(sent_arguments(fake_http.bodies[0]), "body")
        assert body.startswith(f"{MARKER} abl: 6 job(s) ended (COMPLETED x6)")
        for job_id in ids:
            assert job_id in body
        assert set(ledger.read_closures(ledger.closure_path(path))) == set(ids)
        assert read_pending(pending_path(path)) == []

    def test_a_group_that_never_goes_quiet_is_still_announced(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
        moving_clock: MovingClock,
    ) -> None:
        """The latency ceiling, proved against a group that stays busy.

        A member arrives every 180 seconds and never stops, so the quiet
        rule can never fire. Without MAX_HOLD_SECONDS this array would be
        announced never; the operator would have traded 116 posts for
        silence, which is the same bug pointing the other way.
        """
        pin_env(CONFIGURED_ENV)
        fake_http = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = fake_http
        ids = [f"888_{index}" for index in range(MAX_HOLD_SECONDS // 180 + 2)]
        _write_ledger(tmp_path, [_entry(job_id) for job_id in ids])

        for count in range(1, len(ids) + 1):
            _install_accounting(ids[:count])
            run_cycle(_connection(tmp_path), HPC3)
            if fake_http.bodies != []:
                break
            moving_clock.advance(180)

        assert len(fake_http.bodies) == 1, "the hold ceiling never fired"
        held_for = moving_clock.epoch - FROZEN_EPOCH
        assert held_for >= MAX_HOLD_SECONDS

    def test_an_announced_ending_is_never_announced_twice(
        self,
        tmp_path: pathlib.Path,
        fake_run: FakeRun,
        emitted: list[str],
        frozen_clock: str,
        moving_clock: MovingClock,
    ) -> None:
        """Idempotence across the crash window the ordering leaves open.

        A cycle that dies after writing closures but before rewriting the
        pending file leaves an ending in BOTH records. The next cycle must
        drop it on the way in -- otherwise it is re-announced on every
        subsequent cycle forever, which is the original defect made
        permanent rather than fixed.

        The state is constructed directly rather than by killing a process,
        because that on-disk state is precisely what a crash leaves and it
        is the state the code must survive.
        """
        pin_env(CONFIGURED_ENV)
        fake_http = FakeHttpPost([posted_ok()])
        _test_hooks.http_post = fake_http
        path = _write_ledger(tmp_path, [_entry("101")])
        fake_run.add("sacct", stdout=_sacct_row("101", "COMPLETED") + "\n")

        run_cycle(_connection(tmp_path), HPC3)
        moving_clock.advance(SETTLE_SECONDS)
        run_cycle(_connection(tmp_path), HPC3)
        assert len(fake_http.bodies) == 1

        # The crash: closures written, pending never cleared.
        stranded = PendingClosure(
            closure=ledger.read_closures(ledger.closure_path(path))["101"],
            observed_epoch=moving_clock.epoch,
        )
        write_pending(pending_path(path), [*read_pending(pending_path(path)), stranded])
        moving_clock.advance(SETTLE_SECONDS)
        run_cycle(_connection(tmp_path), HPC3)

        assert len(fake_http.bodies) == 1, "the ending was announced a second time"
        assert read_pending(pending_path(path)) == []
