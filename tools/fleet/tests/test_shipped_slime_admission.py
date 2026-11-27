"""Book Slime's fixed eight-worker pool before admitting another check.

Slime 6aeaa519's vitest.config.ts starts eight node workers and six browser
files, independent of CORVIS_TEST_MAX_WORKERS. Lavender job b3eda342 was
booked at four beside observer 139aa899's four-core container in an eight
CPU slice: all 4273 cases passed, but check took 355 of 300 seconds with
744 seconds of aggregate dispatch throttling. The same Slime commit on
sedona (a598403c) passed in 193 seconds with no throttling. These cases
exercise the shipped registry and the real admission function, both orders,
for MCPs board task 22f2818e.
"""

from __future__ import annotations

import pytest
from platform_core.errors import FleetErrorCode

from fleet.contracts.node import LiveLoad
from fleet.contracts.tags import node_tags
from fleet.core.capacity import assess
from tests._registry import idle_state, shipped_workspace


@pytest.mark.parametrize("host,quota,workers", [("lavender", 8.0, 8), ("sedona", 18.0, 9)])
def test_idle_node_books_slimes_fixed_pool(host: str, quota: float, workers: int) -> None:
    """Eight fixed workers fit lavender's eight and sedona's eighteen spare cores."""
    workspace = shipped_workspace()
    slime = workspace["projects"]["slime"]
    node = workspace["nodes"][host]
    state = idle_state(node)
    state["check_cpu_quota"] = quota
    assert (slime["minimum_workers"], slime["worker_cores"]) == (8, 1)
    assert assess(node, state, slime, node_tags(node)) == {
        "workers": workers,
        "code": None,
        "reason": "",
    }


@pytest.mark.parametrize("occupied", range(1, 9))
def test_slime_waits_until_all_eight_quota_cores_are_available(occupied: int) -> None:
    """Any occupied core leaves fewer than the fixed eight the suite starts."""
    workspace = shipped_workspace()
    node = workspace["nodes"]["lavender"]
    slime = workspace["projects"]["slime"]
    state = idle_state(node)
    state["check_cpu_quota"] = 8.0
    state["live"] = LiveLoad(runs=1, workers=occupied, cores=occupied, ram_gb=1.5)
    verdict = assess(node, state, slime, node_tags(node))
    assert (verdict["workers"], verdict["code"]) == (0, FleetErrorCode.NODE_OWNER_RESERVED)
    assert f"so {8 - occupied} core(s) are left" in verdict["reason"]
    assert "minimum of 8 worker(s) occupies 8" in verdict["reason"]


def test_observer_cannot_be_admitted_beside_slime() -> None:
    """The real Slime grant consumes the quota before observer asks for four."""
    workspace = shipped_workspace()
    node = workspace["nodes"]["lavender"]
    slime = workspace["projects"]["slime"]
    observer = workspace["projects"]["osrs-lab/observer"]
    tags = node_tags(node) | frozenset(observer["required_tags"])
    state = idle_state(node)
    state["check_cpu_quota"] = 8.0
    grant = assess(node, state, slime, tags)
    state["live"] = LiveLoad(
        runs=1,
        workers=grant["workers"],
        cores=grant["workers"] * slime["worker_cores"],
        ram_gb=grant["workers"] * slime["worker_ram_gb"],
    )
    verdict = assess(node, state, observer, tags)
    assert (verdict["workers"], verdict["code"]) == (0, FleetErrorCode.NODE_OWNER_RESERVED)
    assert "so 0 core(s) are left" in verdict["reason"]


@pytest.mark.parametrize("quota", range(1, 8))
def test_slime_refuses_a_quota_smaller_than_its_fixed_pool(quota: int) -> None:
    """The measured eight-worker demand also holds without another live job."""
    workspace = shipped_workspace()
    node = workspace["nodes"]["lavender"]
    state = idle_state(node)
    state["check_cpu_quota"] = float(quota)
    verdict = assess(node, state, workspace["projects"]["slime"], node_tags(node))
    assert (verdict["workers"], verdict["code"]) == (0, FleetErrorCode.NODE_OWNER_RESERVED)
    assert "minimum of 8 worker(s) occupies 8" in verdict["reason"]
