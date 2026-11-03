"""A node running on its battery takes no check (MCPs board task 3268afdf).

Built with ``test_capacity.py``'s builders. The shape is loki's on
2026-11-03: 16 cores, off its charger, running back-to-back NixOS VM checks
at 23 to 32 W while its battery fell from 35 to 30 percent in seven minutes.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode

from fleet.contracts.node import NodeConfig
from fleet.core.capacity import assess
from tests.test_capacity import LINUX_TAGS, _node, _project, _state

#: What every refusal of a node on its battery says, after its name.
ON_BATTERY = (
    " is running on its battery: its probe read a power supply discharging, and a check "
    "placed here drains it. It takes the next job once it is back on mains."
)


def _loki() -> NodeConfig:
    """loki as the registry declares it, near enough for this.

    Returns:
        The node.
    """
    return _node(host="loki", cores=16, reserved_cores=2)


class TestOnBattery:
    def test_an_idle_node_on_its_battery_takes_no_check(self) -> None:
        verdict = assess(
            _loki(), _state(host="loki", on_battery=True), _project(minimum_workers=1), LINUX_TAGS
        )

        assert verdict == {
            "workers": 0,
            "code": FleetErrorCode.NODE_ON_BATTERY,
            "reason": f"loki{ON_BATTERY}",
        }

    def test_the_same_node_on_mains_is_granted_workers(self) -> None:
        verdict = assess(
            _loki(), _state(host="loki", on_battery=False), _project(minimum_workers=1), LINUX_TAGS
        )

        assert verdict["code"] is None
        assert verdict["workers"] > 0

    def test_the_battery_refusal_comes_before_every_load_refusal(self) -> None:
        """No load falling clears it, so a reader is told the cause that does."""
        verdict = assess(
            _loki(),
            _state(
                host="loki",
                on_battery=True,
                cpu_pressure_pct=90.0,
                io_pressure_pct=90.0,
                check_cpu_quota=1.0,
            ),
            _project(minimum_workers=4),
            LINUX_TAGS,
        )

        assert verdict["code"] is FleetErrorCode.NODE_ON_BATTERY

    def test_a_node_short_of_disk_is_refused_for_its_disk_first(self) -> None:
        verdict = assess(
            _loki(),
            _state(host="loki", on_battery=True, free_disk_gb=1.0),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )

        assert verdict["code"] is FleetErrorCode.NODE_DISK_EXHAUSTED
