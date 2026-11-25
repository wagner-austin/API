"""A node that is the wrong machine whatever its load is refused (MCPs board task 3268afdf).

Built with ``test_capacity.py``'s builders; :mod:`fleet.core.capacity`'s own
tests reach the same functions through ``assess`` and its ordering.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode

from fleet.contracts.egress_reading import EgressAttempt
from fleet.core.capacity import assess
from fleet.core.node_refusal import machine_refusal, wrong_node
from tests._egress_fixtures import LOST, REACHED
from tests.test_capacity import LINUX_TAGS, _node, _project, _state


class TestWrongNode:
    def test_a_node_any_project_may_use_is_not_the_wrong_one(self) -> None:
        assert wrong_node(_node(host="lavender"), _project()) is None

    def test_a_project_pinned_elsewhere_is_refused_naming_its_node(self) -> None:
        refused = wrong_node(_node(host="lavender"), _project(node="colossus"))

        assert refused == {
            "workers": 0,
            "code": FleetErrorCode.NODE_NOT_PINNED,
            "reason": (
                "lavender is not colossus: the project runs on colossus alone, because it reads "
                "what only that host holds, so the answer is that node and never this one."
            ),
        }


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

        assert refused == {
            "workers": 0,
            "code": FleetErrorCode.NODE_DISK_EXHAUSTED,
            "reason": (
                "lavender has 5 GB free and this workspace reserves 20 GB for staged trees. The "
                "first dispatch to a node is a cold stage of a whole monorepo."
            ),
        }


#: What every refusal of a node with no route out says after its failures.
EGRESS_TAIL = (
    ". Every check's setup downloads what its lockfile pins from these indexes, so a check "
    "placed here would fail as its package's own red. It takes the next job once one answers."
)


class TestEgressLost:
    """MCPs board task e70cdf05: lavender lost its IPv4 lease and kept taking checks."""

    def test_a_node_that_reached_neither_index_is_refused_naming_each_failure(self) -> None:
        refused = machine_refusal(_node(), _state(egress=LOST))

        assert refused == {
            "workers": 0,
            "code": FleetErrorCode.NODE_EGRESS_LOST,
            "reason": (
                "lavender cannot reach the internet: https://pypi.org/simple/pip/ could not "
                "resolve its host (curl exit 6); https://registry.npmjs.org/ could not resolve "
                "its host (curl exit 6)" + EGRESS_TAIL
            ),
        }

    def test_one_index_answering_is_reach(self) -> None:
        """One index down alone must not refuse the checks that download from the other."""
        half = (REACHED[0], LOST[1])

        assert machine_refusal(_node(), _state(egress=half)) is None
        assert machine_refusal(_node(), _state(egress=(LOST[0], REACHED[1]))) is None

    def test_an_exit_curl_s_table_is_not_quoted_for_is_named_by_its_number(self) -> None:
        """curl exit 52 is 'the server returned nothing', a failure this module never words."""
        silent = (
            EgressAttempt(target="https://pypi.org/simple/pip/", curl_exit=52),
            EgressAttempt(target="https://registry.npmjs.org/", curl_exit=28),
        )

        refused = machine_refusal(_node(), _state(egress=silent))

        assert refused == {
            "workers": 0,
            "code": FleetErrorCode.NODE_EGRESS_LOST,
            "reason": (
                "lavender cannot reach the internet: https://pypi.org/simple/pip/ failed (curl "
                "exit 52); https://registry.npmjs.org/ did not answer within the probe's bound "
                "(curl exit 28)" + EGRESS_TAIL
            ),
        }

    def test_the_battery_is_named_before_the_lost_egress(self) -> None:
        refused = machine_refusal(_node(), _state(on_battery=True, egress=LOST))

        assert refused == {
            "workers": 0,
            "code": FleetErrorCode.NODE_ON_BATTERY,
            "reason": (
                "lavender is running on its battery: its probe read a power supply discharging, "
                "and a check placed here drains it. It takes the next job once it is back on "
                "mains."
            ),
        }

    def test_the_lost_egress_is_named_before_every_load_refusal(self) -> None:
        """No load falling gives a node a route out, so a reader is told the cause that does."""
        verdict = assess(
            _node(),
            _state(egress=LOST, cpu_pressure_pct=90.0, io_pressure_pct=90.0, check_cpu_quota=1.0),
            _project(minimum_workers=4),
            LINUX_TAGS,
        )

        assert verdict["code"] is FleetErrorCode.NODE_EGRESS_LOST
