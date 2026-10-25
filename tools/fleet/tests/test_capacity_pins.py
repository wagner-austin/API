"""A project that names its node is weighed by that node alone (MCPs board task bf7a9460).

MCPs' served lanes compare what one host's own search container serves on
127.0.0.1:8090 with the commit's files, so a run on any other node reads
nothing, and no tag singles out one host. Every other node refuses such a
project first, in the one place a runner learns what it may launch.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.tags import NodeTag
from fleet.core.capacity import assess, fitting_projects
from fleet.core.node_choice import first_fit
from tests.test_capacity import LINUX_TAGS, _node, _project, _state


def test_a_pinned_project_is_refused_by_every_other_node_first() -> None:
    """Before its tags or capacity are weighed, even on a node short of both."""
    project = _project(required_tags=(NodeTag.GPU,), node="colossus")

    verdict = assess(
        _node(host="lacrymal"), _state(host="lacrymal", free_ram_gb=1.0), project, LINUX_TAGS
    )

    assert verdict["code"] is FleetErrorCode.NODE_NOT_PINNED
    assert verdict["workers"] == 0
    assert verdict["reason"] == (
        "lacrymal is not colossus: the project runs on colossus alone, because it reads "
        "what only that host holds, so the answer is that node and never this one."
    )


def test_the_named_node_weighs_a_pinned_project_like_any_other() -> None:
    project = _project(required_tags=(NodeTag.LINUX,), node="colossus")
    node = _node(host="colossus")

    assert assess(node, _state(host="colossus"), project, LINUX_TAGS)["workers"] == 7
    lacking = assess(node, _state(host="colossus"), project, frozenset())
    assert lacking["code"] is FleetErrorCode.NODE_LACKS_TAG


def test_a_runner_is_offered_a_pinned_project_only_on_its_node() -> None:
    projects = {"MCPs/served": _project(node="colossus"), "MCPs/any": _project()}

    on_colossus = fitting_projects(
        _node(host="colossus"), _state(host="colossus"), projects, LINUX_TAGS
    )
    elsewhere = fitting_projects(
        _node(host="diphtheria"), _state(host="diphtheria"), projects, LINUX_TAGS
    )

    assert on_colossus == ("MCPs/any", "MCPs/served")
    assert elsewhere == ("MCPs/any",)


def test_a_fleet_without_the_pinned_node_answering_names_the_pin() -> None:
    """Every node that answered is not the named one, which did not."""
    candidates = (
        ("pendragon", _node(host="pendragon"), _state(host="pendragon")),
        ("lavender", _node(), _state()),
    )

    with pytest.raises(AppError) as excinfo:
        first_fit(candidates, _project(node="colossus"))

    assert excinfo.value.code is FleetErrorCode.NODE_NOT_PINNED
    assert "pendragon is not colossus:" in excinfo.value.message
    assert "lavender is not colossus:" in excinfo.value.message


def test_a_pinned_node_of_the_wrong_kind_is_a_tag_answer() -> None:
    """The named node answered and lacks a tag; every other is not it."""
    candidates = (
        ("pendragon", _node(host="pendragon"), _state(host="pendragon")),
        ("colossus", _node(host="colossus"), _state(host="colossus")),
    )

    with pytest.raises(AppError) as excinfo:
        first_fit(candidates, _project(required_tags=(NodeTag.GPU,), node="colossus"))

    assert excinfo.value.code is FleetErrorCode.NODE_LACKS_TAG
    assert "pendragon is not colossus:" in excinfo.value.message
    assert "colossus lacks gpu:" in excinfo.value.message
