"""A node that is the wrong machine whatever its load is refused (MCPs board task 3268afdf).

Built with ``test_capacity.py``'s builders; :mod:`fleet.core.capacity`'s own
tests reach the same functions through ``assess`` and its ordering.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode

from fleet.core.node_refusal import machine_refusal, wrong_node
from tests.test_capacity import _node, _project, _state


class TestWrongNode:
    def test_a_node_any_project_may_use_is_not_the_wrong_one(self) -> None:
        assert wrong_node(_node(host="lavender"), _project()) is None

    def test_a_project_pinned_elsewhere_is_refused_naming_its_node(self) -> None:
        refused = wrong_node(_node(host="lavender"), _project(node="colossus"))

        assert refused is not None
        assert refused["code"] is FleetErrorCode.NODE_NOT_PINNED
        assert refused["reason"].startswith("lavender is not colossus: ")


class TestMachineRefusal:
    def test_a_node_with_disk_on_mains_is_not_refused(self) -> None:
        assert machine_refusal(_node(), _state(on_battery=False)) is None

    def test_a_node_on_its_battery_is_refused_by_name(self) -> None:
        refused = machine_refusal(_node(host="loki"), _state(host="loki", on_battery=True))

        assert refused == {
            "workers": 0,
            "code": FleetErrorCode.NODE_ON_BATTERY,
            "reason": (
                "loki is running on its battery: its probe read a power supply discharging, and "
                "a check placed here drains it. It takes the next job once it is back on mains."
            ),
        }

    def test_a_short_disk_is_named_before_the_battery(self) -> None:
        refused = machine_refusal(
            _node(max_disk_gb=20.0), _state(free_disk_gb=5.0, on_battery=True)
        )

        assert refused is not None
        assert refused["code"] is FleetErrorCode.NODE_DISK_EXHAUSTED
