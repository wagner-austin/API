"""A node whose CPUs or disk are saturated takes no check (MCPs 2ce5dc13, 814c4555).

Split from ``test_capacity.py``, which sits at its line ceiling, and built
with its builders. The shape is diphtheria's on 2026-10-16: 8 cores, one
check at once, and MCPs/execution-deploy claimed at 42 to 51 percent CPU
pressure running 339 s of its 300 s budget with every test passing, because
its declared cores showed none of the stack's own work.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode

from fleet.contracts.node import LiveLoad, NodeConfig, NodePlatform
from fleet.core.capacity import assess, room_for_any
from tests.test_capacity import LINUX_TAGS, _node, _project, _state


def _diphtheria() -> NodeConfig:
    """diphtheria as the registry declares it, near enough for this.

    Returns:
        The node.
    """
    return _node(
        host="diphtheria",
        cores=8,
        reserved_cores=2,
        platform=NodePlatform.LINUX,
        checks_at_once=1,
    )


class TestCpuPressure:
    def test_a_saturated_node_takes_no_check_however_free_its_cores(self) -> None:
        verdict = assess(
            _diphtheria(),
            _state(host="diphtheria", cpu_pressure_pct=42.6),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )

        assert verdict == {
            "workers": 0,
            "code": FleetErrorCode.NODE_OWNER_RESERVED,
            "reason": (
                "diphtheria's CPUs are saturated: a runnable task waited for one through 42.6% "
                "of the last minute, at or over the 30% a check is placed under. Free cores "
                "counted from its declaration do not show work the fleet does not run, and a "
                "check placed here would run past its budget; it takes the next job when the "
                "pressure falls."
            ),
        }

    def test_the_bound_itself_refuses_and_just_under_it_is_weighed_on_the_pools(self) -> None:
        at = assess(
            _diphtheria(),
            _state(host="diphtheria", cpu_pressure_pct=30.0),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )
        under = assess(
            _diphtheria(),
            _state(host="diphtheria", cpu_pressure_pct=29.99),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )

        assert at["code"] is FleetErrorCode.NODE_OWNER_RESERVED
        assert under == {"workers": 3, "code": None, "reason": ""}

    def test_a_node_with_no_reading_is_weighed_on_the_pools(self) -> None:
        """A Windows node's kernel keeps no such account, so its probe prints none."""
        verdict = assess(
            _node(host="loki", cores=16, reserved_cores=2),
            _state(host="loki", cpu_pressure_pct=None),
            _project(minimum_workers=1),
            frozenset(),
        )

        assert verdict == {"workers": 7, "code": None, "reason": ""}

    def test_the_refusal_names_the_live_runs_beside_the_pressure(self) -> None:
        node = _node(host="colossus", cores=12, reserved_cores=2, platform=NodePlatform.LINUX)
        live = LiveLoad(runs=1, workers=4, ram_gb=4.4)

        verdict = assess(
            node,
            _state(host="colossus", live=live, cpu_pressure_pct=55.0),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )

        assert verdict["reason"] == (
            "colossus's CPUs are saturated: a runnable task waited for one through 55.0% of "
            "the last minute, at or over the 30% a check is placed under, and its 1 live fleet "
            "run(s) hold 4 worker(s) and 4.4 GB. Free cores counted from its declaration do "
            "not show work the fleet does not run, and a check placed here would run past its "
            "budget; it takes the next job when the pressure falls."
        )

    def test_a_node_at_its_check_count_is_refused_for_that_first(self) -> None:
        live = LiveLoad(runs=1, workers=2, ram_gb=2.2)

        verdict = assess(
            _diphtheria(),
            _state(host="diphtheria", live=live, cpu_pressure_pct=60.0),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )

        assert verdict["reason"].startswith("diphtheria's CPUs carry 1 budgeted check(s) at once")

    def test_the_runner_claims_nothing_on_a_saturated_node(self) -> None:
        line = room_for_any(
            _diphtheria(),
            _state(host="diphtheria", cpu_pressure_pct=51.0),
            (_project(),),
            LINUX_TAGS,
        )

        assert str(line).startswith("NODE_OWNER_RESERVED: diphtheria's CPUs are saturated")


class TestIoPressure:
    """A node whose disk is saturated takes no check (MCPs board task 814c4555).

    diphtheria at 17:21:58Z on 2026-10-18: CPU pressure 29.9, under its bound,
    and I/O pressure 71.5 while a database dump and a VM install shared its
    disk; MCPs/transcriber-api claimed there ran five times its usual time.
    """

    def test_a_saturated_disk_takes_no_check_whatever_its_cpus_read(self) -> None:
        verdict = assess(
            _diphtheria(),
            _state(host="diphtheria", cpu_pressure_pct=29.9, io_pressure_pct=71.5),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )

        assert verdict == {
            "workers": 0,
            "code": FleetErrorCode.NODE_OWNER_RESERVED,
            "reason": (
                "diphtheria's disk is saturated: a task stalled on I/O through 71.5% of the last "
                "minute, at or over the 60% a check is placed under. Its CPUs do not show it, and "
                "a check placed here would run past its budget; it takes the next job when the "
                "pressure falls."
            ),
        }

    def test_the_bound_itself_refuses_and_just_under_it_is_weighed_on_the_pools(self) -> None:
        """Checks claimed at 40 to 60 percent ran at their usual time, none over budget."""
        at = assess(
            _diphtheria(),
            _state(host="diphtheria", cpu_pressure_pct=5.0, io_pressure_pct=60.0),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )
        under = assess(
            _diphtheria(),
            _state(host="diphtheria", cpu_pressure_pct=5.0, io_pressure_pct=59.99),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )

        assert at["code"] is FleetErrorCode.NODE_OWNER_RESERVED
        assert under == {"workers": 3, "code": None, "reason": ""}

    def test_saturated_cpus_are_named_before_a_saturated_disk(self) -> None:
        verdict = assess(
            _diphtheria(),
            _state(host="diphtheria", cpu_pressure_pct=42.6, io_pressure_pct=88.9),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )

        assert verdict["reason"].startswith("diphtheria's CPUs are saturated")

    def test_the_refusal_names_the_live_runs_beside_the_pressure(self) -> None:
        node = _node(host="colossus", cores=12, reserved_cores=2, platform=NodePlatform.LINUX)
        live = LiveLoad(runs=1, workers=4, ram_gb=4.4)

        verdict = assess(
            node,
            _state(host="colossus", live=live, cpu_pressure_pct=1.0, io_pressure_pct=65.0),
            _project(minimum_workers=1),
            LINUX_TAGS,
        )

        assert verdict["reason"] == (
            "colossus's disk is saturated: a task stalled on I/O through 65.0% of the last "
            "minute, at or over the 60% a check is placed under, and its 1 live fleet run(s) "
            "hold 4 worker(s) and 4.4 GB. Its CPUs do not show it, and a check placed here "
            "would run past its budget; it takes the next job when the pressure falls."
        )

    def test_the_runner_claims_nothing_on_a_saturated_disk(self) -> None:
        line = room_for_any(
            _diphtheria(),
            _state(host="diphtheria", cpu_pressure_pct=12.7, io_pressure_pct=73.6),
            (_project(),),
            LINUX_TAGS,
        )

        assert str(line).startswith("NODE_OWNER_RESERVED: diphtheria's disk is saturated")
