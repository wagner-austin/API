"""What a node runner's tick records on the queue (MCPs board task 939ec5c7, A4).

On 2026-10-02 thirteen jobs queued behind four running, and finding out why
each node took nothing meant reading seven runner logs on the hub. Every tick
now records one row through ``dispatch_tick``: the tags its probe detected,
the runs, workers and free memory it holds, the projects it fits, whether it
asked the queue, and the verdict line it logged. These drive whole ticks of
lavender's runner through each gate and assert the exact row each records.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, narrow_json_to_str

from fleet.cli import node_agent
from fleet.core import _test_hooks
from tests._node_agent_fixtures import (
    NOTHING_MATCHED,
    PROBED,
    _credentials_in_env,
    _sourced_config,
    node_argv,
)
from tests._queue_fakes import FakeQueue, tick_body
from tests._toolchain_fixtures import LAVENDER_2026_09_23, LAVENDER_STORE_STUB
from tests.conftest import DEMO_PROJECT, FakeRun, failed, ok

__all__ = ["_credentials_in_env", "_sourced_config"]

#: lavender's load with 27.0 GB free and no live run, PROBE_OK's answer.
ROOMY = {"runs": 0, "workers": 0, "freeRamGb": 27.0}


def _ticked(
    config_path: pathlib.Path, runs: list[_test_hooks.CommandResult], replies: list[str]
) -> tuple[FakeQueue, JSONObject]:
    """Run one tick and read the one tick it recorded.

    Args:
        config_path: The workspace document.
        runs: What each ssh call returns, in order.
        replies: What the queue answers, ``dispatch_tick`` aside.

    Returns:
        The queue it spoke to, and its one recorded tick without identity.
    """
    _test_hooks.run = FakeRun(runs)
    endpoint = FakeQueue(replies)
    _test_hooks.http_post = endpoint
    assert node_agent.main(node_argv(config_path)) == 0
    assert len(endpoint.ticks) == 1
    assert endpoint.ticks[0]["agent"] == "fleet-node-lavender"
    return endpoint, tick_body(endpoint.ticks[0])


class TestATickThatClaimsNothing:
    def test_a_silent_node_records_no_load_and_no_tags(self, sourced_config: pathlib.Path) -> None:
        _, tick = _ticked(
            sourced_config,
            [failed(255, "ssh: connect to host lavender: timed out")],
            [dump_json_str({"jobs": []})],
        )

        assert tick == {
            "node": "lavender",
            "elevated": False,
            "tags": [],
            "fits": [],
            "claiming": False,
            "verdict": "did not answer; claiming nothing: ssh to lavender failed while sending "
            "C:/fleet/stage/fleet-capacity-lavender.ps1: ssh: connect to host lavender: timed out",
        }

    def test_a_toolchain_that_did_not_answer_records_the_load_but_no_tags(
        self, sourced_config: pathlib.Path
    ) -> None:
        _, tick = _ticked(
            sourced_config,
            [ok(""), ok("free_ram_gb=27.0\nfree_disk_gb=860.0\n"), ok(""), ok("banner\n")],
            [dump_json_str({"jobs": []})],
        )

        assert (tick["load"], tick["tags"], tick["claiming"]) == (ROOMY, [], False)
        assert tick["verdict"] == (
            "did not answer the toolchain probe; claiming nothing: NODE_TOOL_MISSING: lavender: "
            "a toolchain probe returned nothing recognisable, so the node was never asked: "
            "'banner'"
        )

    def test_a_node_that_cannot_build_records_why(self, sourced_config: pathlib.Path) -> None:
        _, tick = _ticked(
            sourced_config,
            [ok(""), ok("free_ram_gb=27.0\nfree_disk_gb=860.0\n"), ok(""), ok(LAVENDER_STORE_STUB)],
            [dump_json_str({"jobs": []})],
        )

        assert (tick["load"], tick["tags"], tick["fits"]) == (ROOMY, [], [])
        assert narrow_json_to_str(tick["verdict"]).startswith(
            "cannot build; claiming nothing: NODE_TOOL_MISSING: "
        )

    @pytest.mark.parametrize(
        ("free", "verdict"),
        [
            (
                "2.2",
                "has room for nothing; claiming nothing: NODE_OWNER_RESERVED: lavender has 2.2 GB "
                "free",
            ),
            (
                "6.0",
                "has room for one worker but not for any project's minimum; claiming nothing",
            ),
        ],
        ids=["room for nothing", "room for no project's minimum"],
    )
    def test_a_full_node_records_its_tags_and_free_memory(
        self, sourced_config: pathlib.Path, free: str, verdict: str
    ) -> None:
        endpoint, tick = _ticked(
            sourced_config,
            [
                ok(""),
                ok(f"free_ram_gb={free}\nfree_disk_gb=860.0\n"),
                ok(""),
                ok(LAVENDER_2026_09_23),
            ],
            [dump_json_str({"jobs": []})],
        )

        assert endpoint.tools == ["dispatch_list"]
        assert (tick["tags"], tick["fits"], tick["claiming"]) == (["windows"], [], False)
        assert tick["load"] == {"runs": 0, "workers": 0, "freeRamGb": float(free)}
        assert narrow_json_to_str(tick["verdict"]).startswith(verdict)


class TestATickThatAsksTheQueue:
    def test_an_empty_lane_records_the_fitting_projects_it_asked_for(
        self, sourced_config: pathlib.Path
    ) -> None:
        endpoint, tick = _ticked(
            sourced_config,
            list(PROBED),
            [dump_json_str({"jobs": []}), dump_json_str({"claimed": None})],
        )

        assert endpoint.tools == ["dispatch_list", "dispatch_claim"]
        assert tick == {
            "node": "lavender",
            "elevated": False,
            "tags": ["windows"],
            "fits": [DEMO_PROJECT],
            "load": ROOMY,
            "claiming": True,
            "verdict": NOTHING_MATCHED.removeprefix("lavender "),
        }
