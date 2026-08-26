"""Tests for the budget: our declared share of a machine 102 people use.

The numbers are the real workload. Six arms of gpt2-large at 9.89 h each is
about 60 GPU-hours; the free partition's own ceiling would happily admit 24
GPUs for three days, which is 1,728. Nothing on the cluster says stop.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.clusters.hpc3 import HPC3
from hpc3.contracts.budget import Budget, decode_budget, encode_budget, encode_consumption
from hpc3.contracts.job import JobSpec
from hpc3.contracts.job import decode_job_spec as _decode_job_spec
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
        "env_path": "/pub/envs/abl",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"arm": "B"},
        "command": "python train.py",
        "artifact": None,
    }
    base.update(overrides)
    return decode_job_spec(base)


def _billed_document(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a run document targeting a partition that charges.

    Args:
        **overrides: Fields to replace.

    Returns:
        The document, undecoded -- so a test can choose which budget to
        decode it against, which is the whole variable under test here.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": "billed",
        "partition": "gpu32",
        "gpu": gpus("L40S"),
        "cpus": 4,
        "mem_gb": 16,
        "minutes": 20,
        "requeue": False,
        "checkpoint_steps": 0,
        "env_path": "/pub/envs/abl",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"arm": "B"},
        "command": "python train.py",
        "artifact": None,
    }
    base.update(overrides)
    return base


def _billed_spec(**overrides: JSONValue) -> JobSpec:
    """Decode a billed spec against a workspace that has declared a budget.

    A funded workspace is required to get past the decode-time partition
    rule at all, which is the rule this fixture exists to be on the other
    side of.

    Args:
        **overrides: Fields to replace.

    Returns:
        A validated spec on a billed partition.
    """
    return _decode_job_spec(_billed_document(**overrides), HPC3, max_service_units=1000.0)


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
    return decode_budget(
        {
            "max_gpu_hours": gpu_hours,
            "max_service_units": units,
            "charge_account": "",
        }
    )


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

    def test_free_partitions_still_carry_no_spend(self) -> None:
        """This used to claim that NO submittable job could project a spend,
        and said it was asserted rather than assumed so that it would be the
        test that noticed if a billing partition ever became reachable. One
        did, on 2026-08-26, when a declared service-unit budget started
        admitting them -- so the claim is narrowed to what still holds: a free
        partition's usage factor is zero, so its projected spend is too. The
        broader case is covered by the billed tests below."""
        specs = [
            _spec(partition="free-gpu32", gpu=gpus("L40S"), cpus=11),
            _spec(partition="free", gpu=None, cpus=64),
            _spec(),
        ]
        assert project(specs)["service_units"] == 0.0

    def test_multi_gpu_jobs_multiply(self) -> None:
        assert project([_spec(gpu=gpus("A100", 4))])["gpu_hours"] == 40.0

    def test_a_cpu_only_job_projects_no_gpu_hours(self) -> None:
        """It is still a job and still counted -- the GPU-hour axis is simply
        not the one that measures it."""
        projected = project([_spec(partition="free", gpu=None)])
        assert projected["gpu_hours"] == 0.0
        assert projected["jobs"] == 1

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

    def test_a_free_partition_projects_no_service_units(self) -> None:
        """A free partition has a zero usage factor, so the product is zero and
        a workspace with no budget can still submit there."""
        specs = [_spec(partition="free-gpu32", gpu=gpus("L40S"), cpus=11)]
        assert check_projection(_budget(1000.0, 0.0), specs)["service_units"] == 0.0

    def test_a_billed_partition_still_projects_no_service_units(self) -> None:
        """Not an oversight: Slurm's billing figure is computed from
        per-GPU-model TRESBillingWeights and reported only in accounting, so
        it does not exist before submission. Multiplying by usage_factor alone
        would understate an L40S job 32-fold, and a cap enforced on that would
        admit far more than it claims to. The pre-submission control is the
        decode-time partition rule; the spend itself is measured in
        check_consumption."""
        projected = project([_billed_spec(minutes=20)])

        assert projected["service_units"] == 0.0
        assert projected["gpu_hours"] == pytest.approx(20 / 60)

    def test_a_billed_spec_is_admitted_once_a_budget_is_declared(self) -> None:
        # The rule this whole change exists for: the same document that a
        # zero-budget workspace refuses, a funded one accepts.
        projected = check_projection(_budget(1000.0, 50.0), [_billed_spec(minutes=20)])

        assert projected["jobs"] == 1

    def test_an_unfunded_workspace_cannot_even_decode_a_billed_spec(self) -> None:
        with pytest.raises(AppError) as excinfo:
            _decode_job_spec(_billed_document(), HPC3, max_service_units=0.0)

        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_BILLS
        assert "budget of 0" in excinfo.value.message
        assert "free-gpu32" in excinfo.value.message


class TestCheckConsumption:
    def test_inside_the_cap_returns_the_reading(self) -> None:
        assert check_consumption(_budget(10.0, 10.0), [_status()])["gpu_hours"] == 1.0

    def test_over_the_gpu_hour_cap_is_reported(self) -> None:
        with pytest.raises(AppError) as excinfo:
            check_consumption(_budget(0.5, 10.0), [_status()])
        assert excinfo.value.code is Hpc3ErrorCode.BUDGET_CONSUMPTION_EXCEEDED

    def test_over_the_service_unit_cap_is_reported(self) -> None:
        status = _status(partition="gpu", billing_tres=11)
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
        payload: dict[str, JSONValue] = {
            "max_gpu_hours": 60.0,
            "max_service_units": 100.0,
            "charge_account": "cjmayer_lab",
        }
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
