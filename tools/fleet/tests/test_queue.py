"""Talking to the dispatch queue: what goes on the wire, and what comes back.

The endpoint is a FAKE that speaks the real JSON-RPC-over-SSE shape, so what
is asserted is the request this package actually builds and the answer it can
actually read -- not a stubbed return value.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import dump_json_str

from fleet.core import _test_hooks, queue
from tests._queue_fakes import (
    QUEUE_CREDENTIALS,
    RUNNER_IDENTITY,
    FakeEnv,
    FakeQueue,
    FakeRefusingQueue,
    queue_job,
)


class TestCredentials:
    def test_reads_both_secrets_and_defaults_the_endpoint(self) -> None:
        _test_hooks.env = FakeEnv(
            {
                queue.API_KEY_VARIABLE: "k",
                queue.TENANT_ID_VARIABLE: "t",
            }
        )

        credentials = queue.load_credentials()

        assert credentials["api_key"] == "k"
        assert credentials["tenant_id"] == "t"
        assert credentials["url"] == queue.DEFAULT_URL

    def test_an_explicit_endpoint_overrides_the_default(self) -> None:
        _test_hooks.env = FakeEnv(
            {
                queue.API_KEY_VARIABLE: "k",
                queue.TENANT_ID_VARIABLE: "t",
                queue.URL_VARIABLE: "http://elsewhere:9000/mcp",
            }
        )

        assert queue.load_credentials()["url"] == "http://elsewhere:9000/mcp"

    def test_a_missing_api_key_names_where_its_value_comes_from(self) -> None:
        """A refusal that says only "unset" costs the operator a search."""
        _test_hooks.env = FakeEnv({queue.TENANT_ID_VARIABLE: "t"})

        with pytest.raises(AppError) as raised:
            queue.load_credentials()

        assert raised.value.code is FleetErrorCode.QUEUE_CREDENTIALS_MISSING
        assert "mcp-fleet container's" in raised.value.message

    def test_a_missing_tenant_id_names_where_its_value_comes_from(self) -> None:
        _test_hooks.env = FakeEnv({queue.API_KEY_VARIABLE: "k"})

        with pytest.raises(AppError) as raised:
            queue.load_credentials()

        assert raised.value.code is FleetErrorCode.QUEUE_CREDENTIALS_MISSING
        assert "tenants row" in raised.value.message


class TestClaim:
    def test_an_empty_queue_answers_none(self) -> None:
        _test_hooks.http_post = FakeQueue([dump_json_str({"claimed": None})])

        assert (
            queue.claim_next(
                QUEUE_CREDENTIALS,
                node=None,
                lease_seconds=3600,
                identity=RUNNER_IDENTITY,
            )
            is None
        )

    def test_sends_the_lease_and_the_identity_and_omits_an_absent_node(self) -> None:
        """Omitting ``node`` is what a runner serving the whole fleet does;
        sending it as null would be a different request, and the tool's
        schema is strict about unknown and mistyped keys."""
        endpoint = FakeQueue([dump_json_str({"claimed": queue_job(status="claimed")})])
        _test_hooks.http_post = endpoint

        job = queue.claim_next(
            QUEUE_CREDENTIALS, node=None, lease_seconds=900, identity=RUNNER_IDENTITY
        )

        assert job == {
            "job_id": "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
            "project": "libs/demo",
            "command": "check",
            "status": "claimed",
            "requested_node": None,
            "node": None,
            "run_id": "",
            "claimed_by": None,
            "submitted_by": "opus-dispatch-0905",
            "session_id": "11111111-aaaa-4aaa-8aaa-111111111111",
        }
        assert endpoint.tools == ["dispatch_claim"]
        assert endpoint.arguments[0] == {
            "leaseSeconds": 900,
            "agent": "fleet-runner-austinpc",
            "sessionId": "33333333-cccc-4ccc-8ccc-333333333333",
            "cwd": "C:/fleet",
        }

    def test_a_node_scoped_runner_sends_its_node(self) -> None:
        endpoint = FakeQueue([dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        queue.claim_next(
            QUEUE_CREDENTIALS,
            node="lavender",
            lease_seconds=900,
            identity=RUNNER_IDENTITY,
        )

        assert endpoint.arguments[0]["node"] == "lavender"


class TestReport:
    def test_start_names_the_node_and_the_ledger_run_id(self) -> None:
        """The run id is the JOIN between the queue row and this machine's own
        records. A start that omitted it would leave the two unlinkable."""
        endpoint = FakeQueue([dump_json_str({"job": queue_job(status="running", node="lavender")})])
        _test_hooks.http_post = endpoint

        job = queue.report_start(
            QUEUE_CREDENTIALS,
            job_id="aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
            node="lavender",
            run_id="libs-demo-1757000000",
            lease_seconds=3600,
            identity=RUNNER_IDENTITY,
        )

        assert job["status"] == "running"
        assert endpoint.tools == ["dispatch_report"]
        assert endpoint.arguments[0]["action"] == "start"
        assert endpoint.arguments[0]["runId"] == "libs-demo-1757000000"
        assert endpoint.arguments[0]["node"] == "lavender"

    def test_close_sends_an_exit_code_when_there_is_one(self) -> None:
        endpoint = FakeQueue([dump_json_str({"job": queue_job(status="failed")})])
        _test_hooks.http_post = endpoint

        queue.report_close(
            QUEUE_CREDENTIALS,
            job_id="aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
            status="failed",
            exit_code=1,
            detail="make check exited 1",
            identity=RUNNER_IDENTITY,
        )

        assert endpoint.arguments[0]["exitCode"] == 1
        assert endpoint.arguments[0]["status"] == "failed"

    def test_close_omits_the_exit_code_for_a_refusal(self) -> None:
        """``refused`` means the command never ran, so there is nothing to
        have exited -- and the queue REFUSES the pair if one is sent. Sending
        a zero here would turn every capacity refusal into a rejected report."""
        endpoint = FakeQueue([dump_json_str({"job": queue_job(status="refused")})])
        _test_hooks.http_post = endpoint

        queue.report_close(
            QUEUE_CREDENTIALS,
            job_id="aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
            status="refused",
            exit_code=None,
            detail="no node had capacity",
            identity=RUNNER_IDENTITY,
        )

        assert "exitCode" not in endpoint.arguments[0]

    def test_a_tool_refusal_surfaces_with_its_own_message(self) -> None:
        """A runner reporting on a job it no longer holds must hear WHY. The
        SDK answers a thrown tool with ``isError`` and prose, so a client
        reading only the JSON-RPC error member would report a JSON fault."""
        endpoint = FakeRefusingQueue(
            "dispatch_report failed: DISPATCH_NOT_CLAIMANT: job aaaa is held by someone else"
        )
        _test_hooks.http_post = endpoint

        with pytest.raises(AppError) as raised:
            queue.report_close(
                QUEUE_CREDENTIALS,
                job_id="aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa",
                status="passed",
                exit_code=0,
                detail="green",
                identity=RUNNER_IDENTITY,
            )

        assert "DISPATCH_NOT_CLAIMANT" in raised.value.message
        assert endpoint.tools == ["dispatch_report"]


class TestHeldBy:
    def test_asks_the_queue_which_live_jobs_this_runner_holds(self) -> None:
        """Asked of the QUEUE rather than remembered locally: a runner that
        crashed between launching and writing a note to itself would strand a
        job on a node with nothing left that knows to collect it."""
        endpoint = FakeQueue(
            [dump_json_str({"jobs": [queue_job(status="running")], "pagination": {}})]
        )
        _test_hooks.http_post = endpoint

        jobs = queue.held_by(QUEUE_CREDENTIALS, agent="fleet-runner-austinpc")

        assert len(jobs) == 1
        assert endpoint.tools == ["dispatch_list"]
        assert endpoint.arguments[0] == {
            "claimedBy": "fleet-runner-austinpc",
            "status": "live",
        }

    def test_holding_nothing_is_an_empty_tuple(self) -> None:
        _test_hooks.http_post = FakeQueue([dump_json_str({"jobs": []})])

        assert queue.held_by(QUEUE_CREDENTIALS, agent="fleet-runner-austinpc") == ()
