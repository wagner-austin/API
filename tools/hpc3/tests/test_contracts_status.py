"""Tests for the status contract and the cost calculation.

The cost tests are the point of this file. Slurm's ``billing`` figure is a
rate, and on ``free-gpu`` the usage factor is zero -- so a job there reports
a non-zero billing figure and still costs nothing. Reading the raw figure as
a cost is wrong in both directions, and these assertions pin that.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.status import (
    JOB_STATES,
    TERMINAL_STATES,
    encode_job_status,
    gpu_hours,
    is_terminal,
)
from tests.against_hpc3 import decode_job_status, service_units


def _status(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid status payload with optional overrides.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        A JSON object ready for decoding.
    """
    base: dict[str, JSONValue] = {
        "job_id": "55519937",
        "name": "abl-verify",
        "partition": "gpu32",
        "state": "COMPLETED",
        "elapsed_seconds": 48,
        "billing_tres": 11,
        "gpu_count": 1,
        "cpu_count": 11,
        "node_list": "hpc3-gpu-n54-00",
    }
    base.update(overrides)
    return base


class TestServiceUnits:
    def test_a_billed_job_costs_billing_tres_times_hours(self) -> None:
        """billing=11 for 48s on a UsageFactor 1.0 partition = 0.1467 SU."""
        cost = service_units(decode_job_status(_status()))
        assert round(cost, 4) == 0.1467

    def test_a_free_partition_costs_nothing_however_long_it_ran(self) -> None:
        status = decode_job_status(
            _status(partition="free-gpu", elapsed_seconds=72 * 3600, billing_tres=96)
        )
        assert service_units(status) == 0.0

    def test_the_same_shape_on_a_billing_partition_is_not_free(self) -> None:
        free = decode_job_status(_status(partition="free-gpu", elapsed_seconds=3600))
        billed = decode_job_status(_status(partition="gpu32", elapsed_seconds=3600))
        assert service_units(free) == 0.0
        assert service_units(billed) == 11.0

    def test_the_32gb_free_partition_costs_nothing(self) -> None:
        """Measured against a real RTX6000 job, after this suite spent a day
        asserting the opposite."""
        status = decode_job_status(_status(partition="free-gpu32", elapsed_seconds=3600))
        assert service_units(status) == 0.0

    def test_an_hour_of_the_billing_partition_costs_the_billing_rate(self) -> None:
        status = decode_job_status(_status(elapsed_seconds=3600, billing_tres=4))
        assert service_units(status) == 4.0

    def test_a_pending_job_has_cost_nothing(self) -> None:
        status = decode_job_status(
            _status(state="PENDING", elapsed_seconds=0, billing_tres=0, gpu_count=0, node_list="")
        )
        assert service_units(status) == 0.0


class TestTerminality:
    def test_finished_states_are_terminal(self) -> None:
        for state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
            assert is_terminal(state) is True

    def test_running_states_are_not_terminal(self) -> None:
        for state in ("PENDING", "RUNNING", "SUSPENDED", "COMPLETING"):
            assert is_terminal(state) is False

    def test_requeued_is_not_terminal_because_protection_worked(self) -> None:
        """A requeued job is going back to the queue, not ending."""
        assert is_terminal("REQUEUED") is False
        assert "REQUEUED" not in TERMINAL_STATES

    def test_preemption_and_oom_end_the_run(self) -> None:
        assert is_terminal("PREEMPTED") is True
        assert is_terminal("OUT_OF_MEMORY") is True

    def test_every_terminal_state_is_a_declared_state(self) -> None:
        for state in TERMINAL_STATES:
            assert state in JOB_STATES


class TestDecode:
    def test_a_valid_status_round_trips(self) -> None:
        assert encode_job_status(decode_job_status(_status())) == _status()

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_status("COMPLETED")

    def test_an_unknown_state_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_status(_status(state="VANISHED"))

    def test_a_partition_this_cluster_lacks_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_status(_status(partition="turbo"))
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_UNKNOWN

    def test_an_empty_job_id_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_status(_status(job_id=""))

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_status(_status(name=""))

    def test_negative_elapsed_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_status(_status(elapsed_seconds=-1))

    def test_negative_billing_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_status(_status(billing_tres=-1))

    def test_an_empty_node_list_is_admitted_for_a_pending_job(self) -> None:
        assert decode_job_status(_status(node_list=""))["node_list"] == ""


class TestGpuHours:
    """The only measure of our share on partitions that bill nothing."""

    def test_one_gpu_for_an_hour_is_one_gpu_hour(self) -> None:
        assert gpu_hours(decode_job_status(_status(gpu_count=1, elapsed_seconds=3600))) == 1.0

    def test_four_gpus_multiply(self) -> None:
        assert gpu_hours(decode_job_status(_status(gpu_count=4, elapsed_seconds=3600))) == 4.0

    def test_it_is_charged_on_the_free_partition_too(self) -> None:
        """Unlike service units, which are zero there however long it ran."""
        status = decode_job_status(_status(partition="free-gpu", gpu_count=2, elapsed_seconds=7200))
        assert gpu_hours(status) == 4.0
        assert service_units(status) == 0.0

    def test_a_pending_job_holds_none(self) -> None:
        status = decode_job_status(
            _status(state="PENDING", elapsed_seconds=0, gpu_count=0, billing_tres=0, node_list="")
        )
        assert gpu_hours(status) == 0.0

    def test_a_cpu_only_job_holds_none(self) -> None:
        assert gpu_hours(decode_job_status(_status(gpu_count=0))) == 0.0

    def test_a_negative_gpu_count_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_status(_status(gpu_count=-1))
