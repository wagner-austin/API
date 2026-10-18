"""A node that checks only its pinned projects refuses every other one (MCPs board task d577036f).

Split from ``test_capacity.py``, which sits at its line ceiling, and built
with its builders. The shape is diphtheria's on 2026-10-18: 8 cores, one
check at once, the production stack beside it, and 10 of its 341 checks in
the 7 days before run past the 300 s budget with their tests passing,
against 3 of about 1,500 on the other nodes.
"""

from __future__ import annotations

from platform_core.errors import FleetErrorCode

from fleet.contracts.node import NodeConfig, NodePlatform
from fleet.core.capacity import assess, fitting_projects, room_for_any
from tests.test_capacity import LINUX_TAGS, _node, _project, _state


def _diphtheria(*, pinned_checks_only: bool = True) -> NodeConfig:
    """diphtheria as the registry declares it, near enough for this.

    Args:
        pinned_checks_only: Whether it checks only its pinned projects.

    Returns:
        The node.
    """
    return _node(
        host="diphtheria",
        cores=8,
        reserved_cores=4,
        platform=NodePlatform.LINUX,
        checks_at_once=1,
        pinned_checks_only=pinned_checks_only,
    )


class TestPinnedChecksOnly:
    def test_a_project_any_node_can_run_is_refused(self) -> None:
        verdict = assess(
            _diphtheria(), _state(host="diphtheria"), _project(minimum_workers=2), LINUX_TAGS
        )

        assert verdict["workers"] == 0
        assert verdict["code"] == FleetErrorCode.NODE_NOT_PINNED
        assert verdict["reason"].startswith("diphtheria checks only the projects pinned to it")

    def test_a_project_pinned_to_it_is_granted_its_workers(self) -> None:
        verdict = assess(
            _diphtheria(),
            _state(host="diphtheria"),
            _project(minimum_workers=2, node="diphtheria"),
            LINUX_TAGS,
        )

        assert verdict == {"workers": 2, "code": None, "reason": ""}

    def test_the_same_node_without_the_declaration_takes_either(self) -> None:
        node = _diphtheria(pinned_checks_only=False)
        state = _state(host="diphtheria")
        pinned = _project(minimum_workers=2, node="diphtheria")

        assert assess(node, state, _project(minimum_workers=2), LINUX_TAGS)["workers"] == 2
        assert assess(node, state, pinned, LINUX_TAGS)["workers"] == 2

    def test_a_project_pinned_elsewhere_is_refused_by_its_pin_first(self) -> None:
        verdict = assess(
            _diphtheria(),
            _state(host="diphtheria"),
            _project(minimum_workers=2, node="colossus"),
            LINUX_TAGS,
        )

        assert verdict["reason"].startswith("diphtheria is not colossus")

    def test_the_runner_offers_the_queue_only_the_pinned_projects(self) -> None:
        projects = {
            "MCPs/doc-extract-api": _project(minimum_workers=2, node="diphtheria"),
            "MCPs/watcher": _project(minimum_workers=2),
        }

        assert fitting_projects(_diphtheria(), _state(host="diphtheria"), projects, LINUX_TAGS) == (
            "MCPs/doc-extract-api",
        )

    def test_the_runner_still_claims_when_a_pinned_project_fits(self) -> None:
        pinned = _project(minimum_workers=1, node="diphtheria")

        assert (
            room_for_any(_diphtheria(), _state(host="diphtheria"), (_project(), pinned), LINUX_TAGS)
            is None
        )

    def test_the_runner_claims_nothing_when_no_project_is_pinned_to_it(self) -> None:
        line = room_for_any(_diphtheria(), _state(host="diphtheria"), (_project(),), LINUX_TAGS)

        assert line == (
            "NODE_NOT_PINNED: diphtheria checks only the projects pinned to it, and no project "
            "this runner's tags fit is pinned to it"
        )
