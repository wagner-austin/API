"""A held run launched under another stage root is given back, and the pass goes on.

MCPs board task f02b144e.

Measured 2026-10-30: colossus's tick took the roll of API cf5b3308d, which
moved its stage root from /home/user/fleet/stage to /home/corvis/fleet/stage,
at 07:11:58Z, two seconds after jobs f492e270 and 4c3824c0 had passed under
the old root. Every read after it looked under the new root, found no
result, and the runs were stopped ``LEASE_NOT_HELD`` at 07:32Z. Each case
runs the real collect pass against the fake queue and the fake ssh runner,
on the demo run lavender launched, with a workspace whose lavender stage
root a roll has moved; the pass holds a second running job after it, which
it must still reach.
"""

from __future__ import annotations

import logging
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from platform_core.json_utils import dump_json_str

from fleet.cli import _config, node_collect, node_watch
from fleet.cli.node_collected import Collected, CollectOutcome
from fleet.contracts.ledger import NO_EXIT_CODE
from fleet.contracts.node import NodeConfig
from fleet.contracts.workspace import FleetWorkspace
from fleet.core import _test_hooks, leases, queue, records
from tests._holder_fakes import RecordingHolder
from tests._node_agent_fixtures import _credentials_in_env, _sourced_config, launch
from tests._queue_fakes import FakeQueue, queue_job
from tests._thread_fakes import await_event
from tests._watch_fixtures import collected, lavender_watch, poll_every_second
from tests.conftest import DEMO_NOW, DEMO_RUN_ID, FakeRun, ok, retire_replies
from tests.test_node_retired import LAVENDER, _held, _reached_the_second

__all__ = ["_credentials_in_env", "_sourced_config"]

#: The stage root the roll gave lavender, where the demo run was never launched.
MOVED_ROOT = "/home/corvis/fleet/stage-after-the-roll"

#: The node's answers to one read of a directory no launch wrote into: the
#: read script's send, then its run.
MOVED = (ok(""), ok("unlaunched\n"))


def _moved(config_path: pathlib.Path) -> _config.LoadedWorkspace:
    """The workspace as a roll that moved lavender's stage root left it.

    Args:
        config_path: The workspace document.

    Returns:
        The loaded workspace, lavender declaring :data:`MOVED_ROOT`.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    lavender: NodeConfig = {**loaded.workspace["nodes"]["lavender"], "stage_root": MOVED_ROOT}
    workspace: FleetWorkspace = {
        **loaded.workspace,
        "nodes": {**loaded.workspace["nodes"], "lavender": lavender},
    }
    return _config.LoadedWorkspace(workspace, loaded.registry, loaded.directory)


class TestARunTheRollMoved:
    def test_is_given_back_then_stopped_and_closed_lost_and_the_pass_goes_on(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        launch(sourced_config)
        loaded = _moved(sourced_config)
        runner = FakeRun([*MOVED, ok(""), ok("stopped"), *retire_replies()])
        _test_hooks.run = runner
        endpoint = FakeQueue([_held(), dump_json_str({"job": queue_job(status="queued")})])
        _test_hooks.http_post = endpoint

        with caplog.at_level(logging.INFO):
            node_collect.collect_pass(
                loaded,
                queue.load_credentials(),
                queue.load_credentials(),
                {},
                agent=LAVENDER,
                alias="lavender",
                holder=RecordingHolder(),
                launching=frozenset(),
            )

        detail = (
            f"DISPATCH_NOT_LAUNCHED: {MOVED_ROOT}/{DEMO_RUN_ID} on lavender holds no unit-end "
            f"script, which the launch writes before it starts the unit, so {DEMO_RUN_ID} was "
            f"launched under another stage root than {MOVED_ROOT}, the one the rolled workspace "
            "declares now; its job was given back and the runner ended its unit"
        )
        # The read went to the moved root, the only one the rolled workspace names.
        assert f"{MOVED_ROOT}/{DEMO_RUN_ID}/collect.sh" in " ".join(runner.calls[0])
        # Given back before the stop: no exit status is invented for it.
        assert endpoint.tools == ["dispatch_list", "dispatch_report"]
        report = endpoint.arguments[1]
        assert report["action"] == "lost"
        assert "exitCode" not in report
        assert report["detail"] == detail
        assert "stop-" in " ".join(runner.calls[2])
        last = records.read_ledger(loaded.ledger)[-1]
        assert last["run_id"] == DEMO_RUN_ID
        assert last["outcome"] == "lost"
        assert last["exit_code"] == NO_EXIT_CODE
        assert last["detail"] == detail
        assert leases.find_by_run(loaded.leases, run_id=DEMO_RUN_ID, now_unix=DEMO_NOW) is None
        assert any(
            record.getMessage().endswith(f": {detail}; the queue left it queued")
            for record in caplog.records
        )
        assert _reached_the_second(caplog)

    def test_the_serve_watch_reads_it_as_moved_and_settles_it_at_its_first_poll(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Without this the watch read it as running, as colossus's did for
        20 minutes, and only the lease's end reached it."""
        launch(sourced_config)
        poll_every_second(sourced_config)
        loaded = _moved(sourced_config)
        node = FakeRun([*MOVED, *MOVED])
        _test_hooks.run = node
        settling = threading.Event()
        settled: list[str] = []

        def settle(*, run_id: str) -> Collected:
            settled.append(run_id)
            settling.set()
            return collected(run_id, CollectOutcome.SETTLED)

        watch = lavender_watch(loaded, settle)
        read = watch.read(DEMO_RUN_ID)
        with caplog.at_level(logging.INFO), ThreadPoolExecutor(max_workers=1) as pool:
            watching = pool.submit(watch.watch)
            with watch:
                watch.hold(frozenset({DEMO_RUN_ID}))
                await_event(settling, what="the moved run's settle")
            watching.result()

        assert read is node_watch.RunRead.MOVED
        assert settled == [DEMO_RUN_ID]
        assert watch.polls == 1
        assert len(node.calls) == 2 * len(MOVED)
        assert any(
            "was launched under another stage root" in record.getMessage()
            for record in caplog.records
        )
