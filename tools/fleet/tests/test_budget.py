"""The arithmetic that decides how much of somebody's machine we take.

THE CENTRAL ASSERTION is that memory divides and cores bound, never the
reverse. It is the difference between a fleet that works and one that
reproduces the 2026-09-04 incident on a smaller box, and it is invisible in
any test where both quantities are generous -- so the cases below are chosen
to make one of them scarce at a time.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

from fleet.contracts.budget import (
    NodeBudget,
    admissible_workers,
    decode_node_budget,
    encode_node_budget,
)


def _budget(
    *,
    reserved_cores: int = 2,
    reserved_ram_gb: float = 4.0,
    worker_ram_gb: float = 1.1,
    max_concurrent_runs: int = 2,
    max_disk_gb: float = 20.0,
) -> NodeBudget:
    """Build a budget, letting a test name only the field it is about.

    Args:
        reserved_cores: Cores left for the node's owner.
        reserved_ram_gb: Memory left for the node's owner.
        worker_ram_gb: Memory one test worker holds.
        max_concurrent_runs: Dispatches allowed at once.
        max_disk_gb: Disk reserved for staged trees.

    Returns:
        The budget.
    """
    return NodeBudget(
        reserved_cores=reserved_cores,
        reserved_ram_gb=reserved_ram_gb,
        worker_ram_gb=worker_ram_gb,
        max_concurrent_runs=max_concurrent_runs,
        max_disk_gb=max_disk_gb,
    )


class TestAdmissibleWorkers:
    def test_memory_divides_when_cores_are_plentiful(self) -> None:
        """20 GB free, 4 reserved, 1.1 per worker -> 14, not the 30 cores."""
        assert admissible_workers(_budget(), logical_cores=32, free_ram_gb=20.0) == 14

    def test_cores_bound_when_memory_is_plentiful(self) -> None:
        """Memory would afford 87 workers; the machine has 8 cores minus 2."""
        assert admissible_workers(_budget(), logical_cores=8, free_ram_gb=100.0) == 6

    def test_the_sedona_case_that_motivated_the_rule(self) -> None:
        """MEASURED 2026-09-04: 20 cores, 11.4 GB free, a torch suite.

        Dispatching on the core count asks for 20 workers and about 22 GB on
        a machine with 11.4 GB free -- which is how austinpc came to hold
        77.9 GB of wedged commit the same night. The memory-first rule gives
        6, which fits.
        """
        workers = admissible_workers(_budget(), logical_cores=20, free_ram_gb=11.4)

        assert workers == 6
        assert workers * 1.1 <= 11.4 - 4.0

    def test_a_machine_whose_owner_is_using_all_the_memory_affords_nothing(self) -> None:
        assert admissible_workers(_budget(), logical_cores=32, free_ram_gb=3.0) == 0

    def test_memory_exactly_at_the_reservation_affords_nothing(self) -> None:
        """The boundary falls to zero, not to one worker on borrowed memory."""
        assert admissible_workers(_budget(), logical_cores=32, free_ram_gb=4.0) == 0

    def test_a_machine_whose_cores_are_all_reserved_affords_nothing(self) -> None:
        assert admissible_workers(_budget(reserved_cores=8), logical_cores=8, free_ram_gb=64.0) == 0

    def test_a_project_with_a_cheap_worker_gets_more_of_the_same_machine(self) -> None:
        """The same node, two suites: what a worker costs is the variable."""
        node_cores, free = 32, 20.0

        heavy = admissible_workers(
            _budget(worker_ram_gb=1.1), logical_cores=node_cores, free_ram_gb=free
        )
        light = admissible_workers(
            _budget(worker_ram_gb=0.2), logical_cores=node_cores, free_ram_gb=free
        )

        assert heavy == 14
        assert light == 30


class TestDecode:
    def test_a_budget_survives_encoding(self) -> None:
        original = _budget()

        assert decode_node_budget(load_json_str(dump_json_str(encode_node_budget(original)))) == (
            original
        )

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_node_budget([1, 2])

    def test_a_zero_worker_cost_is_refused(self) -> None:
        """It divides free memory; a zero reports every node as infinite."""
        with pytest.raises(JSONTypeError, match="worker_ram_gb must be positive"):
            decode_node_budget(encode_node_budget(_budget(worker_ram_gb=0.0)))

    def test_a_negative_core_reservation_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be negative"):
            decode_node_budget(encode_node_budget(_budget(reserved_cores=-1)))

    def test_a_negative_memory_reservation_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be negative"):
            decode_node_budget(encode_node_budget(_budget(reserved_ram_gb=-1.0)))

    def test_a_node_allowed_no_runs_is_refused(self) -> None:
        """Spelled by leaving the node out, not by a zero."""
        with pytest.raises(JSONTypeError, match="max_concurrent_runs must be at least 1"):
            decode_node_budget(encode_node_budget(_budget(max_concurrent_runs=0)))

    def test_a_zero_disk_reservation_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="max_disk_gb must be positive"):
            decode_node_budget(encode_node_budget(_budget(max_disk_gb=0.0)))
