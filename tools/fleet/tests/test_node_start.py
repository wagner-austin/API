"""A job cancelled between its launch and its start report (MCPs board task 88b8fe61).

Measured 2026-09-29: lavender-wsl claimed MCPs/packages/db b4c6c447 at
18:57:14Z and launched it, its submitter cancelled it at 18:57:25Z, the start
report was answered ``DISPATCH_BAD_TRANSITION``, and the tick ended with exit
1 and the suite still running until the next tick stopped it. Each test here
drives one whole claim tick through ``node_agent.main`` against the fake queue
and the fake ssh runner, with the start report refused, and asserts what
reached the node, the queue and the ledger.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str

from fleet.cli import _config, node_agent
from fleet.core import _test_hooks, leases, records, staging
from tests._node_agent_fixtures import (
    _credentials_in_env,
    _sourced_config,
    claim_replies,
    node_argv,
    prebuilt_export,
)
from tests._queue_fakes import DEFAULT_JOB_ID, FakeQueue, ToolRefusal, queue_job
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeRun, ok, retire_replies

__all__ = ["_credentials_in_env", "_sourced_config"]

#: The start report's refusal as the queue worded it for b4c6c447.
BAD_TRANSITION = (
    "dispatch_report failed: DISPATCH_BAD_TRANSITION: a job that is 'cancelled' cannot "
    "become 'running' — legal next states are none, because it is already terminal"
)


def _tick(config_path: pathlib.Path, *, read_back: str) -> tuple[FakeQueue, FakeRun]:
    """Arm one claim tick whose start report is refused, and the job's read-back.

    Args:
        config_path: The workspace document.
        read_back: The ``dispatch_get`` answer the refusal is followed by.

    Returns:
        The queue and the ssh runner, to assert on after the tick.
    """
    payload = prebuilt_export(config_path)
    stop = [ok(""), ok("stopped"), *retire_replies()]
    runner = FakeRun([*claim_replies(staging.digest(payload), commit_present=True), *stop])
    _test_hooks.run = runner
    endpoint = FakeQueue(
        [
            dump_json_str({"jobs": []}),
            dump_json_str({"claimed": queue_job(status="claimed")}),
            ToolRefusal(BAD_TRANSITION),
            read_back,
        ]
    )
    _test_hooks.http_post = endpoint
    return endpoint, runner


def test_a_job_cancelled_while_it_launched_is_stopped_and_the_tick_succeeds(
    sourced_config: pathlib.Path,
) -> None:
    cancelled = queue_job(status="cancelled", claimedBy="fleet-node-lavender")
    read_back = dump_json_str({"job": cancelled, "trail": []})
    endpoint, runner = _tick(sourced_config, read_back=read_back)

    assert node_agent.main(node_argv(sourced_config)) == 0

    assert endpoint.tools == ["dispatch_list", "dispatch_claim", "dispatch_report", "dispatch_get"]
    assert endpoint.arguments[3] == {"jobId": DEFAULT_JOB_ID}
    # Every scripted reply was consumed: the launch, then the stop and the
    # retire of the run it launched.
    assert len(runner.calls) == len(claim_replies("x", commit_present=True)) + 4
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
    last = records.read_ledger(loaded.ledger)[-1]
    assert (last["run_id"], last["outcome"]) == (DEMO_RUN_ID, "cancelled")
    assert last["detail"] == (
        f"queue job {DEFAULT_JOB_ID} was cancelled while it ran; stopped by "
        "fleet-node-lavender; was dispatched by opus-dispatch-0905"
    )
    assert leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW) is None


def test_a_refused_start_for_a_job_not_cancelled_propagates_and_stops_nothing(
    sourced_config: pathlib.Path,
) -> None:
    claimed = queue_job(status="claimed", claimedBy="fleet-node-lavender")
    endpoint, runner = _tick(sourced_config, read_back=dump_json_str({"job": claimed, "trail": []}))

    with pytest.raises(AppError) as excinfo:
        node_agent.main(node_argv(sourced_config))

    assert excinfo.value.message == f"MCP tool reported a failure: {BAD_TRANSITION}"
    assert endpoint.tools[-1] == "dispatch_get"
    assert len(runner.calls) == len(claim_replies("x", commit_present=True))
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
    assert records.read_ledger(loaded.ledger)[-1]["outcome"] == "running"
