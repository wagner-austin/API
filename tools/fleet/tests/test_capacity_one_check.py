"""A node that carries one check at once grants it every spare core (MCPs board task b96f9da1).

Split from ``test_capacity.py``, which sits near its line ceiling, and built
with its builders. :func:`fleet.core.capacity.job_ceiling` holds a grant to
half the spare cores so that a second run fits beside it, but a node whose
budget declares ``checks_at_once`` of 1 refuses that second run. On
2026-11-05 serendipity (8 cores, 2 reserved, one check at once) ran
MCPs/mcp-shared at 3 workers with a check phase of 100 to 143 s, at a load of
1.4. The same suite's check phase took a median 72 s at 7 workers on
pendragon. Every expected width below is worked by hand from the cores, the
reservation and the memory each case declares.
"""

from __future__ import annotations

from fleet.core.capacity import assess, fitting_projects, job_ceiling
from tests.test_capacity import LINUX_TAGS, _node, _project, _state


class TestOneCheckAtOnce:
    def test_serendipity_grants_mcp_shared_all_six_spare_cores(self) -> None:
        """8 cores less 2 reserved leave 6. (27.0 - 4.0) GB at 0.5 GB a
        worker holds 46, so the cores bind: 6, where half was 3."""
        node = _node(host="serendipity", cores=8, reserved_cores=2, checks_at_once=1)
        project = _project(minimum_workers=3, worker_ram_gb=0.5)

        assert job_ceiling(node, project) == 6
        assert assess(node, _state(host="serendipity"), project, LINUX_TAGS) == {
            "workers": 6,
            "code": None,
            "reason": "",
        }

    def test_the_same_node_without_the_declaration_keeps_half_for_a_second(self) -> None:
        """No check count: 6 spare halves to 3, and 3 are left for a second run."""
        node = _node(host="serendipity", cores=8, reserved_cores=2)
        project = _project(minimum_workers=3, worker_ram_gb=0.5)

        assert job_ceiling(node, project) == 3
        assert assess(node, _state(host="serendipity"), project, LINUX_TAGS)["workers"] == 3

    def test_a_node_that_carries_two_at_once_keeps_half_as_well(self) -> None:
        """A count of 2 admits the second run the half leaves room for."""
        node = _node(cores=8, reserved_cores=2, checks_at_once=2)

        assert job_ceiling(node, _project(minimum_workers=1)) == 3

    def test_an_odd_spare_is_granted_whole(self) -> None:
        """9 cores less 4 reserved leave 5, all of them, where half rounded up was 3."""
        node = _node(cores=9, reserved_cores=4, checks_at_once=1)

        assert job_ceiling(node, _project(minimum_workers=1)) == 5

    def test_a_worker_of_two_cores_takes_the_spare_two_at_a_time(self) -> None:
        """6 spare cores hold 3 workers of 2 cores each, where half held 1."""
        node = _node(cores=8, reserved_cores=2, checks_at_once=1)

        assert job_ceiling(node, _project(minimum_workers=1, worker_cores=2)) == 3

    def test_memory_still_bounds_the_whole_spare(self) -> None:
        """6 spare cores, but (6.5 - 4.0) GB at 1.1 GB a worker holds only 2."""
        node = _node(cores=8, reserved_cores=2, checks_at_once=1)

        verdict = assess(node, _state(free_ram_gb=6.5), _project(minimum_workers=1), LINUX_TAGS)

        assert verdict == {"workers": 2, "code": None, "reason": ""}

    def test_the_runner_offers_the_whole_spare_to_the_claim(self) -> None:
        """The width the claim's fit reads (MCPs board task 1964fc15) is the
        width the check is granted, so the fit weighs the new one."""
        node = _node(host="serendipity", cores=8, reserved_cores=2, checks_at_once=1)
        projects = {"MCPs/mcp-shared": _project(minimum_workers=3, worker_ram_gb=0.5)}

        offers = fitting_projects(node, _state(host="serendipity"), projects, LINUX_TAGS)

        assert [(offer["project"], offer["workers"]) for offer in offers] == [
            ("MCPs/mcp-shared", 6)
        ]
