"""Shared queue lifecycle, fair polling and native receipt recovery."""

from __future__ import annotations

import pathlib
import socket

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str
from platform_core.mcp_client import McpCredentials

from fleet.cli import research_queue
from fleet.contracts.dispatch import DispatchStatus, decode_job
from fleet.contracts.lease import Lease
from fleet.core import _test_hooks, leases
from tests._queue_fakes import QUEUE_CREDENTIALS, RUNNER_IDENTITY, FakeQueue
from tests._research_fixtures import research_job, worker
from tests.conftest import FakeRun, failed, ok


@pytest.mark.parametrize("state", ["running", "complete", "failed", "unreachable", "malformed"])
def test_native_tick_records_only_verified_outcomes_and_releases_resources(
    tmp_path: pathlib.Path, state: str
) -> None:
    row = research_job()
    running = research_job(status="running", runId="preserved-receipt")
    endpoint = FakeQueue([dump_json_str({"job": running}), dump_json_str({"job": running})])
    _test_hooks.http_post = endpoint
    result: JSONObject = {
        "state": state,
        "native_job_id": "123",
        "record_sha256": "a" * 64 if state == "complete" else None,
    }
    outcome = (
        failed(255, "private transport details")
        if state == "unreachable"
        else ok(dump_json_str(result))
    )
    runner = FakeRun([ok(""), outcome])
    _test_hooks.run = runner
    path = tmp_path / "leases.json"
    if state == "malformed":
        with pytest.raises(JSONTypeError, match="Invalid state"):
            research_queue.reconcile(
                QUEUE_CREDENTIALS, RUNNER_IDENTITY, worker(), path, decode_job(row, answer="")
            )
        assert len(endpoint.tools) == 1
    else:
        changed = research_queue.reconcile(
            QUEUE_CREDENTIALS, RUNNER_IDENTITY, worker(), path, decode_job(row, answer="")
        )
        assert changed["status"] is DispatchStatus.RUNNING
        reported = endpoint.arguments[-1]
        assert reported["action"] == ("close" if state in {"complete", "failed"} else "progress")
        if state in {"complete", "failed"}:
            assert reported["status"] == ("passed" if state == "complete" else "failed")
            assert reported["exitCode"] == (0 if state == "complete" else 1)
        else:
            assert "exitCode" not in reported
        assert "private transport details" not in dump_json_str(reported)
    assert leases.read_leases(path) == ()
    assert runner.timeouts == [120, research_queue.CONTROL_TIMEOUT_SECONDS]
    assert runner.stdin[0] is not None and b"fleet.cli.research_native" in runner.stdin[0]
    assert b"--attempt pilot" in runner.stdin[0]


def test_running_receipt_survives_repeated_poll_without_start(tmp_path: pathlib.Path) -> None:
    row = research_job(status="running", runId="native-receipt")
    endpoint = FakeQueue([dump_json_str({"job": row}), dump_json_str({"job": row})])
    _test_hooks.http_post = endpoint
    _test_hooks.run = FakeRun(
        [ok(""), ok('{"state":"running","native_job_id":"123","record_sha256":null}')] * 2
    )
    job = decode_job(row, answer="")
    for _ in range(2):
        job = research_queue.reconcile(
            QUEUE_CREDENTIALS, RUNNER_IDENTITY, worker(), tmp_path / "leases", job
        )
        assert job["run_id"] == "native-receipt"
    assert [row["action"] for row in endpoint.arguments] == ["progress", "progress"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("command", "check"),
        ("requestedNode", "other"),
        ("taskId", None),
        ("project", "research/hpc3"),
    ],
)
def test_wrong_routing_never_touches_native_controller(
    tmp_path: pathlib.Path, field: str, value: str | None
) -> None:
    row = research_job()
    row[field] = value
    with pytest.raises(ValueError, match="RESEARCH_QUEUE_ROUTING"):
        research_queue.reconcile(
            QUEUE_CREDENTIALS,
            RUNNER_IDENTITY,
            worker(),
            tmp_path / "leases",
            decode_job(row, answer=""),
        )
    assert not (tmp_path / "leases").exists()


def test_tenant_mismatch_and_existing_controller_lease_prevent_claims(
    tmp_path: pathlib.Path,
) -> None:
    place = worker()
    with pytest.raises(ValueError, match="RESEARCH_QUEUE_TENANT"):
        research_queue.tick(
            {**QUEUE_CREDENTIALS, "tenant_id": "other"}, RUNNER_IDENTITY, place, tmp_path / "leases"
        )
    now = _test_hooks.now()
    held = Lease(
        node=place["node"],
        project="research/controller",
        run_id="prior",
        agent="prior",
        session_id="prior",
        acquired_unix=now,
        expires_unix=now + 600,
        resources=(),
    )
    leases.acquire(tmp_path / "leases", held, now_unix=now)
    endpoint = FakeQueue([])
    _test_hooks.http_post = endpoint
    assert (
        research_queue.tick(QUEUE_CREDENTIALS, RUNNER_IDENTITY, place, tmp_path / "leases") is None
    )
    assert endpoint.tools == []
    assert leases.read_leases(tmp_path / "leases") == (held,)


@pytest.mark.parametrize("claimed", [True, False])
def test_empty_queue_claims_only_research_and_handles_idle(
    tmp_path: pathlib.Path, claimed: bool
) -> None:
    row = research_job()
    replies = [dump_json_str({"jobs": []}), dump_json_str({"claimed": row if claimed else None})]
    if claimed:
        replies.extend([dump_json_str({"job": research_job(status="running")})] * 2)
    endpoint = FakeQueue(replies)
    _test_hooks.http_post = endpoint
    _test_hooks.run = FakeRun(
        [ok(""), ok('{"state":"running","native_job_id":"123","record_sha256":null}')]
    )
    result = research_queue.tick(QUEUE_CREDENTIALS, RUNNER_IDENTITY, worker(), tmp_path / "leases")
    assert (result is not None) == claimed
    assert endpoint.arguments[1]["lane"] == "research"
    assert endpoint.arguments[1]["projects"] == ["research/hpc3", "research/wrds"]


def test_full_lane_polls_oldest_renewal_without_claiming_more(tmp_path: pathlib.Path) -> None:
    rows = [
        research_job(
            id=f"job-{i}",
            status="running",
            runId=f"receipt-{i}",
            leaseExpiresAt=f"2026-10-28T00:0{i}:00.000Z",
        )
        for i in range(4)
    ]
    endpoint = FakeQueue(
        [dump_json_str({"jobs": list(reversed(rows))}), dump_json_str({"job": rows[0]})]
    )
    _test_hooks.http_post = endpoint
    _test_hooks.run = FakeRun(
        [ok(""), ok('{"state":"running","native_job_id":"123","record_sha256":null}')]
    )
    result = research_queue.tick(QUEUE_CREDENTIALS, RUNNER_IDENTITY, worker(), tmp_path / "leases")
    assert result == decode_job(rows[0], answer="")
    assert endpoint.tools == ["dispatch_list", "dispatch_report"]


def test_reclaimed_row_is_polled_once_with_its_existing_run_identity(
    tmp_path: pathlib.Path,
) -> None:
    row = research_job(status="running", runId="saved")
    endpoint = FakeQueue(
        [
            dump_json_str({"jobs": [row]}),
            dump_json_str({"claimed": row}),
            dump_json_str({"job": row}),
        ]
    )
    _test_hooks.http_post = endpoint
    _test_hooks.run = FakeRun(
        [ok(""), ok('{"state":"running","native_job_id":"123","record_sha256":null}')]
    )
    result = research_queue.tick(QUEUE_CREDENTIALS, RUNNER_IDENTITY, worker(), tmp_path / "leases")
    assert result == decode_job(row, answer="")
    assert endpoint.tools == ["dispatch_list", "dispatch_claim", "dispatch_report"]


def test_real_queue_connection_failure_does_not_acquire_a_resource(tmp_path: pathlib.Path) -> None:
    """The real HTTP twin reaches a closed listener and fails before native effects."""
    listener = socket.create_server(("127.0.0.1", 0))
    port: int = listener.getsockname()[1]
    listener.close()
    credentials: McpCredentials = {**QUEUE_CREDENTIALS, "url": f"http://127.0.0.1:{port}/mcp"}
    with pytest.raises(AppError) as raised:
        research_queue.tick(credentials, RUNNER_IDENTITY, worker(), tmp_path / "leases")
    assert raised.value.code is FleetErrorCode.QUEUE_UNANSWERED
    assert not (tmp_path / "leases").exists()
