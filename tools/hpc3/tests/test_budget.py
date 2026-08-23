"""Tests for the budget: our declared share of a machine 102 people use.

The numbers are the real workload. Six arms of gpt2-large at 9.89 h each is
about 60 GPU-hours; the free partition's own ceiling would happily admit 24
GPUs for three days, which is 1,728. Nothing on the cluster says stop.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.budget import Budget, decode_budget, encode_budget, encode_consumption
from hpc3.contracts.job import JobSpec
from hpc3.contracts.status import JobStatus
from tests.against_hpc3 import (
    check_consumption,
    check_projection,
    decode_job_spec,
    decode_job_status,
    observe,
    project,
)
from tests.conftest import gpus


def _spec(**overrides: JSONValue) -> JobSpec:
    """Build a decoded job spec.

    Args:
        **overrides: Fields to replace.

    Returns:
        A validated spec.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": "arm",
        "partition": "free-gpu",
        "gpu": gpus("A100"),
        "cpus": 8,
        "mem_gb": 96,
        "minutes": 600,
        "requeue": True,
        "checkpoint_steps": 50,
        "accept_billing": False,
        "env_path": "/pub/envs/abl",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"arm": "B"},
        "command": "python train.py",
    }
    base.update(overrides)
    return decode_job_spec(base)


def _status(**overrides: JSONValue) -> JobStatus:
    """Build a decoded accounting row.

    Args:
        **overrides: Fields to replace.

    Returns:
        A validated status.
    """
    base: dict[str, JSONValue] = {
        "job_id": "1",
        "name": "arm",
        "partition": "free-gpu",
        "state": "RUNNING",
        "elapsed_seconds": 3600,
        "billing_tres": 8,
        "gpu_count": 1,
        "cpu_count": 8,
        "node_list": "n1",
    }
    base.update(overrides)
    return decode_job_status(base)


def _budget(gpu_hours: float, units: float) -> Budget:
    """Build a budget.

    Args:
        gpu_hours: GPU-hour cap.
        units: Service-unit cap.

    Returns:
        A validated budget.
    """
    return decode_budget({"max_gpu_hours": gpu_hours, "max_service_units": units})


class TestProjection:
    def test_the_real_six_arm_rung_projects_sixty_gpu_hours(self) -> None:
        """6 arms x 1 GPU x 10 h."""
        projected = project([_spec() for _ in range(6)])
        assert projected["gpu_hours"] == 60.0
        assert projected["jobs"] == 6

    def test_free_partitions_project_zero_spend(self) -> None:
        """GPU-hours are real there; service units are not."""
        projected = project([_spec()])
        assert projected["gpu_hours"] == 10.0
        assert projected["service_units"] == 0.0

    def test_a_billing_partition_projects_cpu_hours_as_spend(self) -> None:
        spec = _spec(partition="free-gpu32", gpu=gpus("L40S"), accept_billing=True, cpus=11)
        assert project([spec])["service_units"] == 110.0

    def test_multi_gpu_jobs_multiply(self) -> None:
        assert project([_spec(gpu=gpus("A100", 4))])["gpu_hours"] == 40.0

    def test_a_cpu_only_job_projects_no_gpu_hours(self) -> None:
        """It still projects spend on a billing partition -- the two are
        separate questions, and only one of them is about GPUs."""
        assert project([_spec(partition="free", gpu=None)])["gpu_hours"] == 0.0

    def test_no_specs_project_nothing(self) -> None:
        assert project([]) == {"gpu_hours": 0.0, "service_units": 0.0, "jobs": 0}

    def test_it_projects_the_requested_limit_not_an_optimistic_runtime(self) -> None:
        """A projection built on hoped-for runtimes admits sweeps that cannot fit."""
        assert project([_spec(minutes=4320)])["gpu_hours"] == 72.0


class TestCheckProjection:
    def test_a_sweep_inside_the_cap_is_admitted(self) -> None:
        projected = check_projection(_budget(100.0, 0.0), [_spec() for _ in range(6)])
        assert projected["gpu_hours"] == 60.0

    def test_exactly_the_cap_is_admitted(self) -> None:
        assert check_projection(_budget(60.0, 0.0), [_spec() for _ in range(6)])["jobs"] == 6

    def test_over_the_gpu_hour_cap_is_refused_before_submission(self) -> None:
        with pytest.raises(AppError) as excinfo:
            check_projection(_budget(50.0, 0.0), [_spec() for _ in range(6)])
        assert excinfo.value.code is Hpc3ErrorCode.BUDGET_PROJECTION_EXCEEDED
        assert "Nothing was submitted" in excinfo.value.message

    def test_over_the_service_unit_cap_is_refused(self) -> None:
        spec = _spec(partition="free-gpu32", gpu=gpus("L40S"), accept_billing=True, cpus=11)
        with pytest.raises(AppError) as excinfo:
            check_projection(_budget(1000.0, 100.0), [spec])
        assert excinfo.value.code is Hpc3ErrorCode.BUDGET_PROJECTION_EXCEEDED
        assert "SU" in excinfo.value.message

    def test_the_free_ceiling_would_admit_what_a_budget_refuses(self) -> None:
        """24 GPUs for 3 days is inside every cluster limit and is 1,728 GPU-hours."""
        flood = [_spec(minutes=4320) for _ in range(24)]
        assert project(flood)["gpu_hours"] == 1728.0
        with pytest.raises(AppError):
            check_projection(_budget(100.0, 0.0), flood)


class TestObservation:
    def test_it_totals_gpu_hours_across_jobs(self) -> None:
        observed = observe([_status(), _status(job_id="2", gpu_count=2)])
        assert observed["gpu_hours"] == 3.0
        assert observed["jobs"] == 2

    def test_free_partition_jobs_observe_zero_spend(self) -> None:
        assert observe([_status()])["service_units"] == 0.0

    def test_billing_jobs_observe_real_spend(self) -> None:
        observed = observe([_status(partition="free-gpu32", billing_tres=11)])
        assert observed["service_units"] == 11.0

    def test_a_pending_job_holds_nothing(self) -> None:
        pending = _status(state="PENDING", elapsed_seconds=0, gpu_count=0, node_list="")
        assert observe([pending])["gpu_hours"] == 0.0

    def test_no_jobs_observe_nothing(self) -> None:
        assert observe([]) == {"gpu_hours": 0.0, "service_units": 0.0, "jobs": 0}


class TestCheckConsumption:
    def test_inside_the_cap_returns_the_reading(self) -> None:
        assert check_consumption(_budget(10.0, 10.0), [_status()])["gpu_hours"] == 1.0

    def test_over_the_gpu_hour_cap_is_reported(self) -> None:
        with pytest.raises(AppError) as excinfo:
            check_consumption(_budget(0.5, 10.0), [_status()])
        assert excinfo.value.code is Hpc3ErrorCode.BUDGET_CONSUMPTION_EXCEEDED

    def test_over_the_service_unit_cap_is_reported(self) -> None:
        status = _status(partition="free-gpu32", billing_tres=11)
        with pytest.raises(AppError) as excinfo:
            check_consumption(_budget(100.0, 5.0), [status])
        assert excinfo.value.code is Hpc3ErrorCode.BUDGET_CONSUMPTION_EXCEEDED

    def test_an_overrun_cancels_nothing(self) -> None:
        """Stopping work destroys it; that decision is the operator's."""
        with pytest.raises(AppError) as excinfo:
            check_consumption(_budget(0.5, 10.0), [_status()])
        assert "Nothing was cancelled" in excinfo.value.message


class TestBudgetContract:
    def test_a_valid_budget_round_trips(self) -> None:
        payload: dict[str, JSONValue] = {"max_gpu_hours": 60.0, "max_service_units": 100.0}
        assert encode_budget(decode_budget(payload)) == payload

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_budget([1.0])

    def test_a_negative_gpu_hour_cap_is_refused(self) -> None:
        """A negative cap admits everything, which is not what a cap means."""
        with pytest.raises(JSONTypeError):
            decode_budget({"max_gpu_hours": -1.0, "max_service_units": 0.0})

    def test_a_negative_service_unit_cap_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_budget({"max_gpu_hours": 1.0, "max_service_units": -1.0})

    def test_a_zero_budget_admits_nothing_that_uses_anything(self) -> None:
        with pytest.raises(AppError):
            check_projection(_budget(0.0, 0.0), [_spec()])

    def test_consumption_encodes(self) -> None:
        assert encode_consumption(observe([_status()])) == {
            "gpu_hours": 1.0,
            "service_units": 0.0,
            "jobs": 1,
        }
