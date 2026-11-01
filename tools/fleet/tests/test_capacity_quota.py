"""A check gets no more workers than its slice's CPU quota allows (MCPs board task d77b8c73).

Built with ``test_capacity.py``'s builders. The shape is loki's on
2026-11-01: 16 cores, 6 reserved for whoever uses it, and its check slice
capped at one CPU at runtime from 11:01:36Z, after which packages/maketools,
which declares a minimum of 4 workers, was claimed there with 6 and ran 304,
362 and 341 s of its 300 s budget.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode

from fleet.contracts.node import LiveLoad, NodeConfig
from fleet.core.capacity import assess
from tests.test_capacity import LINUX_TAGS, _node, _project, _state


def _loki() -> NodeConfig:
    """loki as the registry declares it, near enough for this.

    Returns:
        The node.
    """
    return _node(host="loki", cores=16, reserved_cores=6)


class TestCheckQuota:
    def test_a_node_whose_check_slice_is_held_to_one_cpu_takes_no_four_worker_suite(
        self,
    ) -> None:
        verdict = assess(
            _loki(),
            _state(host="loki", check_cpu_quota=1.0),
            _project(minimum_workers=4),
            LINUX_TAGS,
        )

        assert verdict == {
            "workers": 0,
            "code": FleetErrorCode.NODE_OWNER_RESERVED,
            "reason": (
                "loki's checks are held to 1.00 CPU(s) by the quota systemd keeps on their "
                "slice, so 1 worker(s) fit where the suite declares a minimum of 4. Whoever "
                "capped it is using the machine, and a check placed here would run past its "
                "budget; it takes the next job when the quota is lifted."
            ),
        }

    def test_the_same_node_with_no_quota_grants_the_suite_its_ceiling(self) -> None:
        verdict = assess(_loki(), _state(host="loki"), _project(minimum_workers=4), LINUX_TAGS)

        assert verdict == {"workers": 5, "code": None, "reason": ""}

    def test_a_quota_s_whole_cpus_bound_the_grant(self) -> None:
        verdict = assess(
            _loki(),
            _state(host="loki", check_cpu_quota=3.5),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )

        assert verdict == {"workers": 3, "code": None, "reason": ""}

    def test_the_live_runs_share_the_quota(self) -> None:
        live = LiveLoad(runs=1, workers=4, ram_gb=4.4)

        granted = assess(
            _loki(),
            _state(host="loki", live=live, check_cpu_quota=6.0),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )
        refused = assess(
            _loki(),
            _state(host="loki", live=live, check_cpu_quota=6.0),
            _project(minimum_workers=4),
            LINUX_TAGS,
        )

        assert granted == {"workers": 2, "code": None, "reason": ""}
        assert refused["reason"] == (
            "loki's checks are held to 6.00 CPU(s) by the quota systemd keeps on their slice, "
            "so 2 worker(s) fit where the suite declares a minimum of 4, and its 1 live fleet "
            "run(s) hold 4 worker(s) and 4.4 GB. Whoever capped it is using the machine, and a "
            "check placed here would run past its budget; it takes the next job when the quota "
            "is lifted."
        )

    def test_a_saturated_disk_is_named_before_the_quota(self) -> None:
        verdict = assess(
            _loki(),
            _state(host="loki", io_pressure_pct=70.0, check_cpu_quota=1.0),
            _project(minimum_workers=4),
            LINUX_TAGS,
        )

        assert verdict["reason"].startswith("loki's disk is saturated")
