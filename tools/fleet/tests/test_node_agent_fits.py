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
from platform_core.json_utils import dump_json_str

from fleet.cli import node_agent
from fleet.core import _test_hooks
from tests._node_agent_fixtures import (
    PROBED,
    _credentials_in_env,
    _sourced_config,
    node_argv,
)
from tests._queue_fakes import FakeQueue
from tests._toolchain_fixtures import LAVENDER_2026_09_23
from tests.conftest import DEMO_PROJECT, FakeRun, ok

__all__ = ["_credentials_in_env", "_sourced_config"]


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
    assert [record.getMessage() for record in caplog.records][-1] == (
        "lavender has room for one worker but not for any project's minimum; claiming nothing"
    )
