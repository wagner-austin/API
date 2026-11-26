"""Admission books a project's mutation step as memory (MCPs board task 330b0465).

Fleet job 752810f4 ran MCPs/packages/db's check on pendragon at 7 workers,
booked 3.5 GB, and its mutation step started StrykerJS's TypeScript checkers
beside its runners until systemd killed the unit for memory. The shipped
packages/db row declares a checker charged 2.0 GB, so a run of it is booked
at least 2.0 + 2 x 0.5 = 3.0 GB, the step's smallest run, whatever its
workers: in admission, in a node's live load, and in the smallest job a node
must have room for. pendragon reserves 1.5 GB for its owner. Every figure
below is worked by hand from those three numbers.
"""

from __future__ import annotations

import pathlib

from platform_core.errors import FleetErrorCode

from fleet.contracts.ledger import NO_EXIT_CODE, LedgerEntry, LedgerOutcome
from fleet.contracts.node import LiveLoad, NodeConfig, NodeState
from fleet.contracts.project import ProjectConfig
from fleet.contracts.tags import node_tags
from fleet.core import records
from fleet.core.capacity import assess
from fleet.core.memory_budget import smallest_job_gb
from tests._registry import idle_state, shipped_workspace

#: The shipped row whose check was killed.
DB = "MCPs/packages/db"

#: A shipped row that declares no checker: 0.3 GB a worker.
SHARED_PY = "MCPs/mcp-shared-py"

#: One live run beside the check being weighed: 2 workers on 2 cores, 1.0 GB.
BESIDE = LiveLoad(runs=1, workers=2, cores=2, ram_gb=1.0)


def _pendragon() -> tuple[NodeConfig, ProjectConfig]:
    """pendragon and packages/db, as the shipped registry declares them.

    Returns:
        The node and the project.
    """
    workspace = shipped_workspace()
    return workspace["nodes"]["pendragon"], workspace["projects"][DB]


def _beside(node: NodeConfig, free_ram_gb: float) -> NodeState:
    """The node with that much memory free and :data:`BESIDE` live on it.

    Args:
        node: The node.
        free_ram_gb: Its free memory.

    Returns:
        Its state.
    """
    state = idle_state(node, free_ram_gb=free_ram_gb)
    state["live"] = BESIDE
    return state


def _running(run_id: str, project: str, workers: int) -> LedgerEntry:
    """A running ledger row on pendragon.

    Args:
        run_id: Its id.
        project: Its project.
        workers: Its grant.

    Returns:
        The row.
    """
    return LedgerEntry(
        run_id=run_id,
        node="pendragon",
        host="pendragon",
        project=project,
        agent="opus-mcps-1125-8f88",
        session_id="8f887420-b076-406f-8a6f-6fb640351f42",
        started_unix=1_795_716_200,
        ended_unix=1_795_716_200,
        outcome=LedgerOutcome.RUNNING,
        exit_code=NO_EXIT_CODE,
        workers=workers,
        detail="running",
    )


class TestAdmission:
    def test_memory_left_under_the_floor_is_refused_though_its_workers_fit(self) -> None:
        # 5.0 free - 1.0 live - 1.5 reserved = 2.5 GB: 5 workers of 0.5 fit
        # it, over the minimum of 3, and the 3.0 GB floor does not.
        node, db = _pendragon()
        tags = node_tags(node) | frozenset(db["required_tags"])
        verdict = assess(node, _beside(node, 5.0), db, tags)
        assert (verdict["workers"], verdict["code"]) == (
            0,
            FleetErrorCode.NODE_MEMORY_EXHAUSTED,
        )
        assert verdict["reason"].startswith(
            "pendragon has 2.50 GB left after the owner's 1.5 GB, and its 1 live fleet run(s) "
            "hold 2 worker(s) on 2 core(s) and 1.0 GB, under the 3.00 GB a run of this "
            "project is booked whatever its workers: its mutation step's smallest run"
        )

    def test_memory_left_exactly_at_the_floor_is_granted(self) -> None:
        # 5.5 - 1.0 - 1.5 = 3.0 GB, the floor exactly: 6 workers of 0.5.
        node, db = _pendragon()
        tags = node_tags(node) | frozenset(db["required_tags"])
        verdict = assess(node, _beside(node, 5.5), db, tags)
        assert verdict["code"] is None
        assert verdict["workers"] == 6

    def test_a_project_without_a_checker_is_weighed_by_its_workers_alone(self) -> None:
        # 3.8 - 1.0 - 1.5 = 1.3 GB: 4 workers of 0.3, its minimum of 4.
        workspace = shipped_workspace()
        node = workspace["nodes"]["pendragon"]
        shared = workspace["projects"][SHARED_PY]
        tags = node_tags(node) | frozenset(shared["required_tags"])
        verdict = assess(node, _beside(node, 3.8), shared, tags)
        assert (verdict["workers"], verdict["code"]) == (4, None)


class TestBookings:
    def test_a_live_run_holds_its_floor_and_a_plain_run_its_workers(
        self, tmp_path: pathlib.Path
    ) -> None:
        ledger = tmp_path / "ledger.jsonl"
        records.append_ledger(ledger, _running("db-1", DB, 3))
        records.append_ledger(ledger, _running("py-1", SHARED_PY, 4))
        live = records.live_load(ledger, node="pendragon", projects=shipped_workspace()["projects"])
        # packages/db at 3 workers is booked its 3.0 GB floor, not 1.5;
        # mcp-shared-py at 4 is booked 4 x 0.3.
        assert live == LiveLoad(runs=2, workers=7, cores=7, ram_gb=3.0 + 4 * 0.3)

    def test_the_smallest_job_a_node_holds_is_the_floor_of_its_smallest_checker_project(
        self,
    ) -> None:
        node, db = _pendragon()
        # minimum_workers 3 x 0.5 is 1.5 GB; the floor is 3.0.
        assert smallest_job_gb(node, (db,)) == 3.0
