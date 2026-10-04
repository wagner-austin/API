"""A node runner fills its node's room in one tick (MCPs board task 48842bfd).

Measured 2026-10-03 06:36-06:45Z with 14 checks queued: every node claimed
exactly one job per three-minute tick, each finishing within about a tick,
so each ran about one job at a time while lavender-wsl's probe showed
18.6 GB free. The runner now claims again after every job it launches,
re-probing between claims, until the node has no room or the lane nothing
it fits, and stops at the first job it refuses. These drive whole ticks of
lavender's runner against two registered projects, each half the node's
spare cores wide (:func:`fleet.core.capacity.job_ceiling`).
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import dump_json_str, narrow_json_to_str

from fleet.cli import _config, node_agent
from fleet.contracts.source import ProjectSource, encode_project_source
from fleet.core import _test_hooks, records, staging
from tests._node_agent_fixtures import (
    NOTHING_LAUNCHED,
    NPM_CI,
    PROBED,
    REMOTE,
    _credentials_in_env,
    _sourced_config,
    claim_replies,
    launch_steps,
    node_argv,
    prebuilt_export,
    sourced_document,
)
from tests._queue_fakes import FakeQueue, queue_job, tick_body
from tests.conftest import DEMO_NOW, DEMO_PROJECT, DEMO_RUN_ID, FakeRun

__all__ = ["_credentials_in_env", "_sourced_config"]

#: The second registered project, as wide as the demo one.
OTHER_PROJECT = "libs/other"
OTHER_RUN_ID = f"libs-other-lavender-{DEMO_NOW}"
OTHER_JOB_ID = "bbbbbbbb-2222-4222-8222-bbbbbbbbbbbb"


@pytest.fixture(name="two_projects")
def _two_projects(config_path: pathlib.Path) -> pathlib.Path:
    """Rewrite the workspace with ``libs/other`` sourced beside the demo.

    Args:
        config_path: The shared workspace document, clock pinned.

    Returns:
        The same path, rewritten.
    """
    document = sourced_document((NPM_CI,))
    projects = document["projects"]
    assert isinstance(projects, dict)
    demo = projects[DEMO_PROJECT]
    assert isinstance(demo, dict)
    projects[OTHER_PROJECT] = {
        **demo,
        "source": encode_project_source(
            ProjectSource(remote=REMOTE, path=OTHER_PROJECT, install=(NPM_CI,), companions=())
        ),
    }
    config_path.write_text(dump_json_str(document), encoding="utf-8")
    return config_path


def _other_export(config_path: pathlib.Path) -> None:
    """Write the second run's archive where its ``git archive`` would have.

    Args:
        config_path: The workspace document.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    source = loaded.archives / f"{DEMO_RUN_ID}.tgz"
    (loaded.archives / f"{OTHER_RUN_ID}.tgz").write_bytes(source.read_bytes())


def _tick(
    config_path: pathlib.Path, runs: list[_test_hooks.CommandResult], replies: list[str]
) -> tuple[FakeQueue, FakeRun]:
    """Run one tick of lavender's runner.

    Args:
        config_path: The workspace document.
        runs: What each ssh and git call returns, in order.
        replies: What the queue answers, ``dispatch_tick`` aside.

    Returns:
        The queue and the runner, every call recorded.
    """
    runner = FakeRun(runs)
    _test_hooks.run = runner
    endpoint = FakeQueue(replies)
    _test_hooks.http_post = endpoint
    assert node_agent.main(node_argv(config_path)) == 0
    return endpoint, runner


class TestARoomyNode:
    def test_launches_two_jobs_in_one_tick_and_stops_when_the_room_is_gone(
        self, two_projects: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        digest = staging.digest(prebuilt_export(two_projects))
        _other_export(two_projects)
        runs = [
            *PROBED,
            *launch_steps(digest, commit_present=True),
            *PROBED,
            *launch_steps(digest, commit_present=True),
            *PROBED,
        ]
        other = {"id": OTHER_JOB_ID, "project": OTHER_PROJECT}
        replies = [
            dump_json_str({"jobs": []}),
            dump_json_str({"claimed": queue_job(status="claimed")}),
            dump_json_str({"job": queue_job(status="running", node="lavender")}),
            dump_json_str({"claimed": queue_job(status="claimed", **other)}),
            dump_json_str({"job": queue_job(status="running", node="lavender", **other)}),
        ]

        with caplog.at_level("INFO"):
            endpoint, runner = _tick(two_projects, runs, replies)

        assert endpoint.tools == [
            "dispatch_list",
            "dispatch_claim",
            "dispatch_report",
            "dispatch_claim",
            "dispatch_report",
        ]
        # The second claim leaves out the project the first launched.
        assert endpoint.arguments[1]["projects"] == [DEMO_PROJECT, OTHER_PROJECT]
        assert endpoint.arguments[3]["projects"] == [OTHER_PROJECT]
        assert len(runner.calls) == len(runs)
        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(two_projects)})
        rows = records.read_ledger(loaded.ledger)
        assert [(row["run_id"], row["outcome"], row["workers"]) for row in rows] == [
            (DEMO_RUN_ID, "running", 7),
            (OTHER_RUN_ID, "running", 7),
        ]
        # One tick record per pass; the third re-probe found no room.
        ticks = [tick_body(tick) for tick in endpoint.ticks]
        assert [tick["claiming"] for tick in ticks] == [True, True, False]
        assert [tick["load"] for tick in ticks] == [
            {"runs": 0, "workers": 0, "freeRamGb": 27.0},
            {"runs": 1, "workers": 7, "freeRamGb": 27.0},
            {"runs": 2, "workers": 14, "freeRamGb": 27.0},
        ]
        assert narrow_json_to_str(ticks[2]["verdict"]).startswith(
            "has room for nothing; claiming nothing: NODE_OWNER_RESERVED: "
        )
        messages = [record.getMessage() for record in caplog.records]
        assert messages[-1] == "lavender launched 2 job(s) this tick"


class TestARefusedJobEndsTheTick:
    def test_a_refusal_claims_no_second_job_though_the_node_has_room(
        self, two_projects: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A node that stopped answering mid-staging would refuse the next
        job and the next; stopping at the first leaves the lane for the next
        tick or another node."""
        with caplog.at_level("INFO"):
            endpoint, runner = _tick(
                two_projects,
                list(PROBED),
                [
                    dump_json_str({"jobs": []}),
                    dump_json_str({"claimed": queue_job(status="claimed", requiredTags=["gpu"])}),
                    dump_json_str({"job": queue_job(status="refused")}),
                ],
            )

        assert endpoint.tools == ["dispatch_list", "dispatch_claim", "dispatch_report"]
        assert endpoint.arguments[2]["status"] == "refused"
        assert len(runner.calls) == len(PROBED)
        assert len(endpoint.ticks) == 1
        assert [record.getMessage() for record in caplog.records][-1] == NOTHING_LAUNCHED


class TestAHeldProjectIsNotClaimed:
    def test_the_pass_after_a_launch_asks_for_nothing_when_its_one_project_is_held(
        self, sourced_config: pathlib.Path
    ) -> None:
        """The re-run gate leaves out what a lease on the node holds
        (:func:`fleet.core.run_lease.held_on_node`, MCPs board task
        939ec5c7), so that pass never claims a second job of the project just
        launched and refuses it ``LEASE_HELD``, which closes the job for good."""
        payload = prebuilt_export(sourced_config)

        endpoint, _ = _tick(
            sourced_config,
            claim_replies(staging.digest(payload), commit_present=True),
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ],
        )

        assert endpoint.tools == ["dispatch_list", "dispatch_claim", "dispatch_report"]
        last = tick_body(endpoint.ticks[-1])
        assert (last["claiming"], last["fits"]) == (False, [])
        assert last["verdict"] == (
            "has room, but every project it fits is held by a lease on it; claiming nothing"
        )
