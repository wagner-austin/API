"""Choosing among several nodes the one that affords a dispatch the most.

The cases :mod:`fleet.core.node_choice` took with it from ``test_capacity.py``
(MCPs board task 2ce5dc13), built with that file's builders.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.node import NodePlatform
from fleet.contracts.tags import NodeTag
from fleet.core.node_choice import first_fit
from tests.test_capacity import GTX_1630, _node, _project, _state


class TestFirstFit:
    def test_it_picks_the_node_that_affords_the_most(self) -> None:
        """Not the first that fits: the fleet's nodes differ by over 2x."""
        candidates = (
            ("sedona", _node(host="sedona", cores=20), _state(host="sedona", free_ram_gb=11.4)),
            ("loki", _node(host="loki", cores=16), _state(host="loki", free_ram_gb=27.0)),
        )

        assert first_fit(candidates, _project()) == ("loki", 7)

    def test_a_tie_keeps_the_earlier_candidate(self) -> None:
        """Workspace order is a tie-break a person controls."""
        candidates = (
            ("alpha", _node(host="alpha"), _state(host="alpha")),
            ("beta", _node(host="beta"), _state(host="beta")),
        )

        assert first_fit(candidates, _project())[0] == "alpha"

    def test_a_refusing_node_is_skipped_for_one_that_accepts(self) -> None:
        candidates = (
            ("busy", _node(host="busy"), _state(host="busy", free_ram_gb=3.0)),
            ("free", _node(host="free"), _state(host="free")),
        )

        assert first_fit(candidates, _project())[0] == "free"

    def test_when_nothing_fits_every_reason_is_carried(self) -> None:
        """ "No room" and "one is full, one has no disk" want different actions.

        A message naming only the first refusal sends the reader to the wrong
        machine.
        """
        candidates = (
            ("busy", _node(host="busy"), _state(host="busy", free_ram_gb=3.0)),
            ("fullup", _node(host="fullup"), _state(host="fullup", free_disk_gb=1.0)),
        )

        with pytest.raises(AppError) as excinfo:
            first_fit(candidates, _project())

        assert excinfo.value.code is FleetErrorCode.NODE_MEMORY_EXHAUSTED
        assert "busy:" in excinfo.value.message
        assert "fullup:" in excinfo.value.message
        assert "somebody is on this machine" in excinfo.value.message
        assert "reserves 20 GB" in excinfo.value.message

    def test_an_empty_fleet_refuses(self) -> None:
        with pytest.raises(AppError) as excinfo:
            first_fit((), _project())

        assert excinfo.value.code is FleetErrorCode.NODE_MEMORY_EXHAUSTED

    def test_a_fleet_with_no_node_of_the_right_kind_says_so(self) -> None:
        """Every node refused on its tags: nothing is full, and waiting fixes nothing."""
        candidates = (
            ("loki", _node(host="loki"), _state(host="loki")),
            (
                "diphtheria",
                _node(host="diphtheria", platform=NodePlatform.LINUX),
                _state(host="diphtheria"),
            ),
        )

        with pytest.raises(AppError) as excinfo:
            first_fit(candidates, _project(required_tags=(NodeTag.GPU, NodeTag.WINDOWS)))

        assert excinfo.value.code is FleetErrorCode.NODE_LACKS_TAG
        assert "loki lacks gpu:" in excinfo.value.message
        assert "diphtheria lacks gpu, windows:" in excinfo.value.message

    def test_one_full_node_of_the_right_kind_keeps_the_capacity_answer(self) -> None:
        """A tagged node that is merely busy will take the work later, so the reader waits."""
        candidates = (
            ("loki", _node(host="loki"), _state(host="loki")),
            ("lavender", _node(gpu=GTX_1630), _state(free_ram_gb=3.0)),
        )

        with pytest.raises(AppError) as excinfo:
            first_fit(candidates, _project(required_tags=(NodeTag.GPU,)))

        assert excinfo.value.code is FleetErrorCode.NODE_MEMORY_EXHAUSTED
        assert "loki lacks gpu:" in excinfo.value.message
        assert "somebody is on this machine" in excinfo.value.message

    def test_the_tagged_node_is_chosen_over_a_roomier_untagged_one(self) -> None:
        candidates = (
            ("loki", _node(host="loki", cores=32), _state(host="loki", free_ram_gb=60.0)),
            ("lavender", _node(gpu=GTX_1630), _state()),
        )

        assert first_fit(candidates, _project(required_tags=(NodeTag.GPU,))) == ("lavender", 7)
