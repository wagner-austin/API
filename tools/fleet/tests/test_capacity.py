"""The refusal that is the product.

Every case here is a node that would accept work it cannot finish, which is
the failure mode the package was written after: on 2026-09-04 nothing refused
two overlapping suites and they held 77.9 GB of commit doing nothing.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.budget import NodeBudget
from fleet.contracts.node import NodeConfig, NodeState
from fleet.contracts.project import ProjectConfig
from fleet.core.capacity import assess, first_fit, plan_dispatch


def _node(
    *,
    host: str = "lavender",
    cores: int = 16,
    reserved_cores: int = 2,
    reserved_ram_gb: float = 4.0,
    max_concurrent_runs: int = 2,
    max_disk_gb: float = 20.0,
) -> NodeConfig:
    """Build a node declaration.

    Args:
        host: SSH alias.
        cores: Logical processors.
        reserved_cores: Cores left for the owner.
        reserved_ram_gb: Memory left for the owner.
        max_concurrent_runs: Dispatches allowed at once.
        max_disk_gb: Disk reserved for staged trees.

    Returns:
        The node.
    """
    return NodeConfig(
        host=host,
        stage_root="C:/fleet/stage",
        logical_cores=cores,
        ram_gb=32.0,
        gpu=None,
        budget=NodeBudget(
            reserved_cores=reserved_cores,
            reserved_ram_gb=reserved_ram_gb,
            worker_ram_gb=1.1,
            max_concurrent_runs=max_concurrent_runs,
            max_disk_gb=max_disk_gb,
        ),
    )


def _state(
    *,
    host: str = "lavender",
    free_ram_gb: float = 27.0,
    free_disk_gb: float = 800.0,
    live_runs: int = 0,
) -> NodeState:
    """Build a probed state.

    Args:
        host: The node it came from.
        free_ram_gb: Memory free.
        free_disk_gb: Disk free.
        live_runs: Fleet dispatches already live.

    Returns:
        The state.
    """
    return NodeState(
        host=host, free_ram_gb=free_ram_gb, free_disk_gb=free_disk_gb, live_runs=live_runs
    )


def _project(*, minimum_workers: int = 4, worker_ram_gb: float = 1.1) -> ProjectConfig:
    """Build a project declaration.

    Args:
        minimum_workers: Fewest workers worth dispatching with.
        worker_ram_gb: Memory one worker of this suite holds.

    Returns:
        The project.
    """
    return ProjectConfig(
        worker_ram_gb=worker_ram_gb, minimum_workers=minimum_workers, expected_minutes=5
    )


class TestAssess:
    def test_a_healthy_node_accepts_and_names_no_code(self) -> None:
        verdict = assess(_node(), _state(), _project())

        assert verdict["code"] is None
        assert verdict["reason"] == ""
        assert verdict["workers"] == 14

    def test_a_node_at_its_concurrency_limit_is_refused(self) -> None:
        """Three dispatches that each fit alone do not fit together."""
        verdict = assess(_node(max_concurrent_runs=2), _state(live_runs=2), _project())

        assert verdict["code"] is FleetErrorCode.NODE_OWNER_RESERVED
        assert verdict["workers"] == 0
        assert "already holds 2 fleet run(s)" in verdict["reason"]

    def test_a_node_without_room_to_stage_is_refused(self) -> None:
        verdict = assess(_node(max_disk_gb=20.0), _state(free_disk_gb=5.0), _project())

        assert verdict["code"] is FleetErrorCode.NODE_DISK_EXHAUSTED
        assert "reserves 20 GB" in verdict["reason"]

    def test_a_node_whose_owner_is_using_it_is_refused(self) -> None:
        verdict = assess(_node(), _state(free_ram_gb=3.0), _project())

        assert verdict["code"] is FleetErrorCode.NODE_OWNER_RESERVED
        assert "somebody is on this machine" in verdict["reason"]

    def test_a_node_too_small_for_the_suite_is_refused(self) -> None:
        """THE sedona CASE. 6 workers afforded, 8 declared as the minimum.

        Dispatching anyway runs the suite at a fraction of its workers until
        its own lease expires underneath it.
        """
        verdict = assess(_node(cores=20), _state(free_ram_gb=11.4), _project(minimum_workers=8))

        assert verdict["code"] is FleetErrorCode.NODE_MEMORY_EXHAUSTED
        assert "affords 6 worker(s)" in verdict["reason"]
        assert "minimum of 8" in verdict["reason"]

    def test_the_project_cost_overrides_the_node_default(self) -> None:
        """What a worker costs is a property of the suite, not the machine."""
        light = assess(_node(), _state(), _project(worker_ram_gb=0.2))

        assert light["workers"] == 14
        assert assess(_node(cores=200), _state(), _project(worker_ram_gb=0.2))["workers"] == 115


class TestPlanDispatch:
    def test_it_returns_the_worker_count_when_the_node_accepts(self) -> None:
        assert plan_dispatch(_node(), _state(), _project()) == 14

    def test_it_raises_the_verdict_s_own_code(self) -> None:
        with pytest.raises(AppError) as excinfo:
            plan_dispatch(_node(), _state(free_ram_gb=3.0), _project())

        assert excinfo.value.code is FleetErrorCode.NODE_OWNER_RESERVED


class TestFirstFit:
    def test_it_picks_the_node_that_affords_the_most(self) -> None:
        """Not the first that fits: the fleet's nodes differ by over 2x."""
        candidates = (
            ("sedona", _node(host="sedona", cores=20), _state(host="sedona", free_ram_gb=11.4)),
            ("loki", _node(host="loki", cores=16), _state(host="loki", free_ram_gb=27.0)),
        )

        assert first_fit(candidates, _project()) == ("loki", 14)

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
