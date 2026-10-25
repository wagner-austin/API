"""A run that never settles is retired by a later collect pass (MCPs board task 3d9eec0d).

Measured 2026-10-18: 12 run directories on six nodes were left by runs the
hub's feed ends at ``refused`` (``NODE_UNREACHABLE`` while staging,
``STAGE_DIGEST_MISMATCH`` of a companion) or at ``leased`` or ``staged`` when
their runner ended, and the retire record held no line for any of them, so
nothing retired them. These cases run the real node runner
(:func:`fleet.cli.node_agent.main`): a run refused while staging is retired
by the next start's collect pass, a launch whose runner ended is left alone
until its lease lapses and retired then. A ``LEASE_HELD`` refusal, which
took no lease and staged nothing, owes nothing
(``tests/test_node_agent_staging_refusal.py``).
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import dump_json_str

from fleet.cli import _config, node_agent
from fleet.contracts.retire_record import RetireState
from fleet.contracts.workspace import require_project
from fleet.core import _test_hooks, records, retire_owed, run_lease
from fleet.core.queue_transport import answering
from tests._node_agent_fixtures import (
    PROBED,
    _credentials_in_env,
    _sourced_config,
    node_argv,
    prebuilt_export,
)
from tests._queue_fakes import FakeQueue, queue_job
from tests.conftest import (
    DEMO_NOW,
    DEMO_PROJECT,
    DEMO_RUN_ID,
    FakeRun,
    ok,
    pin_clock,
    retire_replies,
    timed_out,
)

__all__ = ["_credentials_in_env", "_sourced_config"]

#: Where the runner's retire script goes on the demo node.
RETIRE_PATH = f"/home/corvis/fleet/stage/retire-{DEMO_RUN_ID}.sh"


def _loaded(config_path: pathlib.Path) -> _config.LoadedWorkspace:
    """The workspace and its record paths.

    Args:
        config_path: The workspace document.

    Returns:
        The loaded workspace.
    """
    return _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})


def _states(config_path: pathlib.Path) -> list[RetireState]:
    """Every line of the runner's retire record for the demo run, as its state.

    Args:
        config_path: The workspace document.

    Returns:
        One state per line, in order.
    """
    return [
        record["state"]
        for record in records.read_retires(_loaded(config_path).retires)
        if record["run_id"] == DEMO_RUN_ID
    ]


def _idle_start(config_path: pathlib.Path, run: FakeRun) -> FakeQueue:
    """Start the runner holding nothing and with nothing to claim.

    Args:
        config_path: The workspace document.
        run: The node's fake.

    Returns:
        The queue's fake, for its tools.
    """
    _test_hooks.run = run
    queue = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
    _test_hooks.http_post = answering(queue)
    assert node_agent.main(node_argv(config_path)) == 0
    return queue


def _sent_retire(run: FakeRun) -> int:
    """How many calls carried the demo run's retire script.

    Args:
        run: The node's fake.

    Returns:
        The count, two per retire sent (the script written, then run).
    """
    return sum(RETIRE_PATH in " ".join(call) for call in run.calls)


class TestARunRefusedWhileStaging:
    def test_is_retired_by_the_next_start_s_collect_pass(
        self, sourced_config: pathlib.Path
    ) -> None:
        prebuilt_export(sourced_config)
        _test_hooks.run = FakeRun(
            [
                *PROBED,
                ok(""),  # git init --bare (the mirror is new)
                ok(""),  # git cat-file -e: the sha is present
                ok(""),  # git archive -o
                ok(""),  # stage: send mkdir script
                ok(""),  # stage: run mkdir, so the run's directory is on the node
                timed_out(120),  # stage: send the payload, ended at its deadline
            ]
        )
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="refused")}),
            ]
        )

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert _states(sourced_config) == [RetireState.OWED]

        answered = FakeRun([*retire_replies(), *PROBED])
        queue = _idle_start(sourced_config, answered)

        assert queue.tools == ["dispatch_list", "dispatch_claim"]
        assert _sent_retire(answered) == 2
        assert any("fleet-cache-generations" in " ".join(call) for call in answered.calls)
        assert _states(sourced_config) == [RetireState.OWED, RetireState.RETIRED]
        assert retire_owed.owed_on(_loaded(sourced_config).retires, alias="lavender") == ()


class TestALaunchWhoseRunnerEnded:
    def test_is_left_alone_until_its_lease_lapses_and_retired_then(
        self, sourced_config: pathlib.Path
    ) -> None:
        """Its lease taken and nothing after: the feed ends at ``leased``."""
        loaded = _loaded(sourced_config)
        clock = pin_clock(DEMO_NOW)
        lease = run_lease.take(
            loaded.leases,
            loaded.feed,
            loaded.retires,
            node_name="lavender",
            project=DEMO_PROJECT,
            plan=require_project(loaded.workspace, DEMO_PROJECT),
            workers=1,
            agent="opus-fleet-0904",
            session_id="acc774c0-3bc3-4cce-9dda-c7a12fb99519",
            node_local=loaded.workspace["node_local_resources"],
        )
        assert lease["run_id"] == DEMO_RUN_ID

        held = FakeRun(list(PROBED))
        _idle_start(sourced_config, held)

        assert _sent_retire(held) == 0
        assert _states(sourced_config) == [RetireState.OWED]

        clock.seconds = lease["expires_unix"] + 1
        lapsed = FakeRun([*retire_replies(), *PROBED])
        _idle_start(sourced_config, lapsed)

        assert _sent_retire(lapsed) == 2
        assert _states(sourced_config) == [RetireState.OWED, RetireState.RETIRED]
