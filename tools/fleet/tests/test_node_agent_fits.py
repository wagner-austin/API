"""A node runner claims only a project its node can run now (MCPs 939ec5c7).

On 2026-10-02 diphtheria had room for a small suite, claimed the oldest job
its tags matched, MCPs/execution-deploy with its 4.5 GB worker, and refused
it on capacity, which closed the job for good: 8ccd89db and 86208e6b. The
claim now names the projects the node fits this tick
(:func:`fleet.core.capacity.fitting_projects`), so the queue offers nothing
else, and a node that fits none asks for nothing.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, load_json_str, narrow_json_to_dict

from fleet.cli import _config, node_agent
from fleet.contracts.lease import Lease
from fleet.contracts.resources import encode_names
from fleet.core import _test_hooks, leases
from tests._node_agent_fixtures import (
    NO_WATCH,
    NOTHING_LAUNCHED,
    PROBED,
    _credentials_in_env,
    _sourced_config,
    node_argv,
)
from tests._queue_fakes import FakeQueue, tick_body
from tests._toolchain_fixtures import LAVENDER_2026_09_23
from tests.conftest import DEMO_NOW, DEMO_PROJECT, FakeRun, ok

__all__ = ["_credentials_in_env", "_sourced_config"]

#: The container each testdb node runs for itself, as fleet.json declares it.
TESTDB = "corvis-fleet-testdb"

#: A second registered project, declaring nothing exclusive.
FREE_PROJECT = "libs/free"


def _declare(config_path: pathlib.Path, *, demo_exclusive: tuple[str, ...]) -> None:
    """Add :data:`FREE_PROJECT`, and make the testdb container node-local when declared.

    Args:
        config_path: The sourced workspace document, rewritten in place.
        demo_exclusive: What the demo project declares exclusive.
    """
    document: JSONObject = narrow_json_to_dict(load_json_str(config_path.read_text("utf-8")))
    projects = narrow_json_to_dict(document["projects"])
    demo = narrow_json_to_dict(projects[DEMO_PROJECT])
    demo["exclusive_resources"] = encode_names(demo_exclusive)
    projects[FREE_PROJECT] = {**demo, "exclusive_resources": [], "source": None}
    # The workspace refuses a node-local name no project declares.
    document["node_local_resources"] = encode_names(
        tuple(name for name in demo_exclusive if name == TESTDB)
    )
    config_path.write_text(dump_json_str(document), encoding="utf-8")


def _hold(
    config_path: pathlib.Path, *, node: str, project: str, resources: tuple[str, ...]
) -> None:
    """Record a live lease in the workspace's lease file, as a running dispatch holds it.

    Args:
        config_path: The workspace document.
        node: The node the lease is on.
        project: Its project.
        resources: What it holds, scoped as a lease records them.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    leases.acquire(
        loaded.leases,
        Lease(
            node=node,
            project=project,
            run_id=f"{project.replace('/', '-')}-{node}-{DEMO_NOW - 60}",
            agent="opus-fleet-w4-1003",
            session_id="s",
            acquired_unix=DEMO_NOW - 60,
            expires_unix=DEMO_NOW + 600,
            resources=resources,
        ),
        now_unix=DEMO_NOW,
    )


def test_the_claim_names_the_projects_the_node_fits(sourced_config: pathlib.Path) -> None:
    _test_hooks.run = FakeRun(PROBED)
    endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
    _test_hooks.http_post = endpoint

    assert node_agent.main(node_argv(sourced_config)) == 0

    assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
    assert endpoint.arguments[1]["projects"] == [DEMO_PROJECT]


def test_a_node_with_room_for_one_worker_but_no_project_s_minimum_claims_nothing(
    sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """6.0 GB free is 2.0 GB past the owner's 4.0 GB reservation: one 1.1 GB
    worker, which passes the room gate, but the one registered project needs
    two. It used to claim that project and refuse it; now it asks the queue
    for nothing."""
    _test_hooks.run = FakeRun(
        [ok(""), ok("free_ram_gb=6.0\nfree_disk_gb=860.0\n"), ok(""), ok(LAVENDER_2026_09_23)]
    )
    endpoint = FakeQueue([dump_json_str({"jobs": []})])
    _test_hooks.http_post = endpoint

    with caplog.at_level("INFO"):
        assert node_agent.main(node_argv(sourced_config)) == 0

    assert endpoint.tools == ["dispatch_list"]
    assert [record.getMessage() for record in caplog.records][-3:] == [
        "lavender has room for one worker but not for any project's minimum; claiming nothing",
        NOTHING_LAUNCHED,
        NO_WATCH,
    ]


class TestWhatALeaseOnTheNodeHolds:
    """MCPs board task 939ec5c7: a node running several jobs at once fitted,
    by room, a project its own lease file would refuse, and a claimed job
    refused ``LEASE_HELD`` or ``RESOURCE_HELD`` is closed for good. The case
    that made it: every testdb project declares the node's own
    ``corvis-fleet-testdb``, so diphtheria running MCPs/packages/db would
    take MCPs/fleet-mcp and refuse it. The runner now leaves such a project
    out of its claim, and the job stays queued."""

    def test_a_project_whose_node_local_container_a_running_suite_holds_is_not_asked_for(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _declare(sourced_config, demo_exclusive=(TESTDB,))
        _hold(sourced_config, node="lavender", project="libs/db", resources=(f"{TESTDB}@lavender",))
        _test_hooks.run = FakeRun(PROBED)
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
        assert endpoint.arguments[1]["projects"] == [FREE_PROJECT]
        assert (
            "lavender leaves out what a lease on it holds now, which the queue keeps for "
            f"another node or a later tick: {DEMO_PROJECT}"
        ) in [record.getMessage() for record in caplog.records]
        assert tick_body(endpoint.ticks[0])["fits"] == [FREE_PROJECT]

    def test_another_node_s_copy_of_the_container_holds_nothing_here(
        self, sourced_config: pathlib.Path
    ) -> None:
        _declare(sourced_config, demo_exclusive=(TESTDB,))
        _hold(
            sourced_config,
            node="diphtheria",
            project="libs/db",
            resources=(f"{TESTDB}@diphtheria",),
        )
        _test_hooks.run = FakeRun(PROBED)
        endpoint = FakeQueue([dump_json_str({"jobs": []}), dump_json_str({"claimed": None})])
        _test_hooks.http_post = endpoint

        assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.arguments[1]["projects"] == [DEMO_PROJECT, FREE_PROJECT]

    def test_a_node_whose_every_fitting_project_is_held_asks_the_queue_for_nothing(
        self, sourced_config: pathlib.Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Both projects are held: the demo by its own run (``LEASE_HELD``),
        the other by a fleet-wide resource a run on another node holds
        (``RESOURCE_HELD``). The node has room and says why it claims
        nothing anyway."""
        _declare(sourced_config, demo_exclusive=())
        document = narrow_json_to_dict(load_json_str(sourced_config.read_text("utf-8")))
        projects = narrow_json_to_dict(document["projects"])
        narrow_json_to_dict(projects[FREE_PROJECT])["exclusive_resources"] = ["corvis_test"]
        sourced_config.write_text(dump_json_str(document), encoding="utf-8")
        _hold(sourced_config, node="lavender", project=DEMO_PROJECT, resources=())
        _hold(sourced_config, node="diphtheria", project="libs/db", resources=("corvis_test",))
        _test_hooks.run = FakeRun(PROBED)
        endpoint = FakeQueue([dump_json_str({"jobs": []})])
        _test_hooks.http_post = endpoint

        with caplog.at_level("INFO"):
            assert node_agent.main(node_argv(sourced_config)) == 0

        assert endpoint.tools == ["dispatch_list"]
        tick = tick_body(endpoint.ticks[0])
        assert (tick["fits"], tick["claiming"]) == ([], False)
        assert [record.getMessage() for record in caplog.records][-3:] == [
            "lavender has room, but every project it fits is held by a lease on it; "
            "claiming nothing",
            NOTHING_LAUNCHED,
            NO_WATCH,
        ]
