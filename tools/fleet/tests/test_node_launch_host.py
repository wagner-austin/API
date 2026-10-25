"""Two processes claiming for one host claim in turn, against its limit (MCPs board task a85ef09e).

serendipity had 8 cores, 4 left to its owner, and two runner identities,
ordinary and elevated, two processes on the hub. Each charged its launches
under way from its own memory, so on 2026-10-07 the elevated runner claimed
MCPs/scripts/ps-harness at 22:57:23 PDT, the ordinary runner
MCPs/sms-gateway at 22:59:57 and the elevated runner
MCPs/execution-elevated at 23:01:34, and fleet job c858b46b's sms-gateway
check ran 321 s of its 300 s budget. Earlier the same night two checks the
pool admitted, 2 + 2 workers inside its 4 cores, took sms-gateway from 162 s
to 306 s (fleet job 25d07c7c), so serendipity's budget declared
``checks_at_once: 1``.

The elevated runner left with the Windows nodes (board task e705f08c), and
the host's claims and turn hold for any two processes claiming for it. The
case binds the real launch pool and two launchers of lavender's one runner,
declared to carry one check at once. The first claims and holds its launch;
the second, a launcher of its own that never saw that claim, reads it from
the host and claims nothing while the launch is under way, nor while its
run is live on the ledger, and asks the queue once the run has ended.
"""

from __future__ import annotations

import pathlib
import threading

import pytest
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.cli import _config, node_agent
from fleet.cli.node_launch import Launcher
from fleet.contracts.ledger import LedgerEntry, LedgerOutcome
from fleet.core import _test_hooks, host_claims, leases, queue, records, staging
from fleet.core.workspace_files import read_workspace_document
from tests._launch_fakes import ThreadRoutedQueue, ThreadRoutedRun
from tests._node_agent_fixtures import (
    PROBED,
    VERDICT_TASK,
    _credentials_in_env,
    _sourced_config,
    launch_steps,
    prebuilt_export,
)
from tests._queue_fakes import DEFAULT_JOB_ID, FakeQueue, queue_job
from tests._registry import write_registry
from tests._roster_fixtures import dispatch_of
from tests.conftest import DEMO_RUN_ID, FakeRun

__all__ = ["_credentials_in_env", "_sourced_config"]

#: What the second process's gate says while the host holds its one check.
ONE_AT_ONCE = (
    "has room for nothing; claiming nothing: NODE_OWNER_RESERVED: lavender's CPUs carry "
    "1 budgeted check(s) at once across its runners, and 1 are live or launching on it now"
)


def _one_check_at_once(config_path: pathlib.Path) -> None:
    """Declare lavender's CPUs measured to carry one check at once.

    Args:
        config_path: The sourced workspace document, rewritten in place.
    """
    document = read_workspace_document(config_path)
    assert isinstance(document, dict)
    budget = dispatch_of(document, "lavender")["budget"]
    assert isinstance(budget, dict)
    budget["checks_at_once"] = 1
    write_registry(config_path, document)


def _process(
    loaded: _config.LoadedWorkspace, *, held: list[frozenset[str]]
) -> tuple[JSONObject, Launcher]:
    """One process of lavender's runner: its identity and its own launcher.

    Args:
        loaded: The workspace.
        held: Receives each run its launcher hands to the watch.

    Returns:
        Its identity arguments and its launcher.
    """
    agent, session_id = node_agent.node_identity("lavender")
    identity = queue.identity_arguments(agent, session_id, str(loaded.directory))
    launcher = Launcher(
        loaded,
        queue.load_credentials(),
        identity,
        alias="lavender",
        node=loaded.workspace["nodes"]["lavender"],
        agent=agent,
        hold=held.append,
    )
    return identity, launcher


def _claim(loaded: _config.LoadedWorkspace, identity: JSONObject, launcher: Launcher) -> str | None:
    """One claim pass of one of lavender's processes.

    Args:
        loaded: The workspace.
        identity: The runner's identity arguments.
        launcher: The process's launcher.

    Returns:
        The job it claimed, or None.
    """
    return node_agent.claim_pass(
        loaded,
        queue.load_credentials(),
        identity,
        alias="lavender",
        node=loaded.workspace["nodes"]["lavender"],
        launcher=launcher,
    )


def _ended(row: LedgerEntry) -> LedgerEntry:
    """The row a collect pass appends once a run has passed, beside giving back its lease.

    Args:
        row: The run's running row.

    Returns:
        The same run, passed.
    """
    return LedgerEntry(
        run_id=row["run_id"],
        node=row["node"],
        host=row["host"],
        project=row["project"],
        agent=row["agent"],
        session_id=row["session_id"],
        started_unix=row["started_unix"],
        ended_unix=row["started_unix"] + 162,
        outcome=LedgerOutcome.PASSED,
        exit_code=0,
        workers=row["workers"],
        detail="make check exited 0",
    )


class TestTwoProcessesOfOneHost:
    def test_claim_in_turn_against_the_hosts_one_check(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _one_check_at_once(sourced_config)
        digest = staging.digest(prebuilt_export(sourced_config))
        _test_hooks.executor = _test_hooks._default_executor
        release = threading.Event()
        _test_hooks.run = ThreadRoutedRun(
            claims=FakeRun([*PROBED, *PROBED, *PROBED, *PROBED]),
            launches=FakeRun(launch_steps(digest, registry=sourced_config, commit_present=True)),
            release=release,
        )
        claims = FakeQueue(
            [
                dump_json_str({"claimed": queue_job(status="claimed", taskId=VERDICT_TASK)}),
                dump_json_str({"claimed": None}),
            ]
        )
        _test_hooks.http_post = ThreadRoutedQueue(
            claims=claims,
            launches=FakeQueue([dump_json_str({"job": queue_job(status="running")})]),
        )
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced_config)})
        held: list[frozenset[str]] = []
        first_identity, first = _process(loaded, held=held)
        second_identity, second = _process(loaded, held=[])

        with caplog.at_level("INFO"), second:
            with first:
                claimed = _claim(loaded, first_identity, first)
                launching = host_claims.live_claims(loaded.host_claims, alias="lavender")
                while_launching = _claim(loaded, second_identity, second)
                release.set()
            row = records.read_ledger(loaded.ledger)[-1]
            while_live = _claim(loaded, second_identity, second)
            records.append_ledger(loaded.ledger, _ended(row))
            leases.release(loaded.leases, run_id=row["run_id"], now_unix=_test_hooks.now())
            once_ended = _claim(loaded, second_identity, second)

        assert claimed == DEFAULT_JOB_ID
        assert [(claim["runner"], claim["job_id"]) for claim in launching] == [
            ("fleet-node-lavender", DEFAULT_JOB_ID)
        ]
        assert row["run_id"] == DEMO_RUN_ID
        assert held == [frozenset({DEMO_RUN_ID})]
        assert (while_launching, while_live, once_ended) == (None, None, None)
        second_ticks = claims.ticks[1:]
        verdicts = [tick["verdict"] for tick in second_ticks]
        assert len(verdicts) == 3
        assert all(isinstance(verdict, str) for verdict in verdicts)
        assert str(verdicts[0]).startswith(ONE_AT_ONCE)
        assert str(verdicts[1]).startswith(ONE_AT_ONCE)
        assert verdicts[2] == "asked for 1 fitting project(s); nothing in the node lane matched"
        assert [tick["claiming"] for tick in second_ticks] == [False, False, True]
        assert host_claims.live_claims(loaded.host_claims, alias="lavender") == ()
        assert not host_claims.lock_path(loaded.host_claims, alias="lavender").exists()
