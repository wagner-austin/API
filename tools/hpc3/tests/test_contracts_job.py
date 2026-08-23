"""Tests for the job contract's five submission rules.

Each rule gets a test proving it rejects, a test proving it admits the valid
neighbour, and an assertion on the error CODE rather than the message text --
a caller branching on a rule needs the code to be stable even when the
wording improves.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.job import (
    PREEMPTION_PROTECTION_THRESHOLD_MINUTES,
    JobSpec,
    encode_job_spec,
)
from tests.against_hpc3 import decode_job_spec
from tests.conftest import gpus


def _spec(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid spec payload with optional field overrides.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        A JSON object ready for decoding.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": "arm-b-42",
        "partition": "free-gpu",
        "gpu": gpus("A100"),
        "cpus": 8,
        "mem_gb": 96,
        "minutes": 30,
        "requeue": False,
        "checkpoint_steps": 0,
        "accept_billing": False,
        "env_path": "/pub/wagnera3/envs/abl-pinned",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"arm": "B", "seed": "42"},
        "command": "python train.py",
    }
    base.update(overrides)
    return base


class TestValidSpec:
    def test_a_valid_spec_round_trips(self) -> None:
        decoded = decode_job_spec(_spec())
        assert encode_job_spec(decoded) == _spec()

    def test_decode_returns_every_field(self) -> None:
        decoded = decode_job_spec(_spec())
        assert sorted(decoded.keys()) == [
            "accept_billing",
            "checkpoint_steps",
            "command",
            "cpus",
            "deterministic",
            "env_path",
            "experiment",
            "gpu",
            "mem_gb",
            "minutes",
            "name",
            "partition",
            "pinned_packages",
            "project",
            "requeue",
        ]


class TestRuleGpuMustBeNamed:
    def test_a_generic_gpu_request_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(gpu=gpus("gpu")))
        assert excinfo.value.code is Hpc3ErrorCode.GPU_TYPE_UNPINNED

    def test_an_empty_gpu_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(gpu=gpus("")))
        assert excinfo.value.code is Hpc3ErrorCode.GPU_TYPE_UNPINNED

    def test_a_named_model_is_admitted(self) -> None:
        assert decode_job_spec(_spec(gpu=gpus("A30")))["gpu"] == {"model": "A30", "count": 1}


class TestRuleGpuRequestOrNoneButNothingBetween:
    """Absence is spelled one way, and a zero-GPU request is not it.

    The nullable object exists so that a model with no count, or a count with
    no model, cannot be written down. These are the states the old flat pair
    permitted and this shape refuses.
    """

    def test_a_cpu_only_job_states_null(self) -> None:
        assert decode_job_spec(_spec(partition="free", gpu=None))["gpu"] is None

    def test_a_zero_gpu_request_is_refused_rather_than_read_as_cpu_only(self) -> None:
        """Two spellings of one state is how they drift apart."""
        with pytest.raises(JSONTypeError):
            decode_job_spec(_spec(gpu=gpus("A100", 0)))

    def test_a_request_missing_its_count_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_spec(_spec(gpu={"model": "A100"}))

    def test_a_request_missing_its_model_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_spec(_spec(gpu={"count": 1}))

    def test_a_bare_string_is_no_longer_a_gpu_request(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_spec(_spec(gpu="A100"))


class TestRulePartitionMustCarryTheGpu:
    def test_asking_free_gpu_for_a_blackwell_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(gpu=gpus("RTX6000")))
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_GPU_MISMATCH

    def test_the_same_gpu_on_its_own_partition_is_admitted(self) -> None:
        decoded = decode_job_spec(
            _spec(gpu=gpus("RTX6000"), partition="free-gpu32", accept_billing=True)
        )
        assert decoded["partition"] == "free-gpu32"

    def test_asking_a_cpu_partition_for_a_gpu_is_refused(self) -> None:
        """Slurm would leave it pending forever rather than reject it."""
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(partition="free", gpu=gpus("A100")))
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_GPU_MISMATCH

    def test_asking_a_gpu_partition_for_no_gpu_is_refused(self) -> None:
        """This one RUNS. Slurm accepts it and hands over a GPU node to do
        CPU work, so nothing surfaces it except this check."""
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(partition="free-gpu", gpu=None))
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_GPU_MISMATCH

    def test_a_cpu_job_on_a_cpu_partition_is_admitted(self) -> None:
        decoded = decode_job_spec(_spec(partition="free", gpu=None))
        assert decoded["partition"] == "free"


class TestRuleBillingNeedsConsent:
    def test_a_billing_partition_without_consent_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(partition="free-gpu32", gpu=gpus("L40S")))
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_BILLS_WITHOUT_CONSENT

    def test_consent_admits_it(self) -> None:
        decoded = decode_job_spec(
            _spec(partition="free-gpu32", gpu=gpus("L40S"), accept_billing=True)
        )
        assert decoded["accept_billing"] is True

    def test_the_free_partition_needs_no_consent(self) -> None:
        assert decode_job_spec(_spec(accept_billing=False))["partition"] == "free-gpu"


class TestRulePreemptibleRunsMustBeProtected:
    def test_a_long_unprotected_preemptible_run_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(minutes=600))
        assert excinfo.value.code is Hpc3ErrorCode.PREEMPTIBLE_RUN_UNPROTECTED

    def test_requeue_without_checkpoints_is_not_protection(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(minutes=600, requeue=True, checkpoint_steps=0))
        assert excinfo.value.code is Hpc3ErrorCode.PREEMPTIBLE_RUN_UNPROTECTED

    def test_checkpoints_without_requeue_is_not_protection(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(minutes=600, requeue=False, checkpoint_steps=50))
        assert excinfo.value.code is Hpc3ErrorCode.PREEMPTIBLE_RUN_UNPROTECTED

    def test_both_together_admit_the_run(self) -> None:
        decoded = decode_job_spec(_spec(minutes=600, requeue=True, checkpoint_steps=50))
        assert decoded["minutes"] == 600

    def test_a_short_run_needs_no_protection(self) -> None:
        decoded = decode_job_spec(_spec(minutes=PREEMPTION_PROTECTION_THRESHOLD_MINUTES))
        assert decoded["requeue"] is False

    def test_a_non_preemptible_partition_needs_no_protection(self) -> None:
        decoded = decode_job_spec(_spec(partition="gpu", minutes=600, accept_billing=True))
        assert decoded["partition"] == "gpu"


class TestRuleTimeLimitFitsThePartition:
    def test_over_the_ceiling_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(minutes=72 * 60 + 1, requeue=True, checkpoint_steps=50))
        assert excinfo.value.code is Hpc3ErrorCode.TIME_LIMIT_EXCEEDS_PARTITION

    def test_exactly_the_ceiling_is_admitted(self) -> None:
        decoded = decode_job_spec(_spec(minutes=72 * 60, requeue=True, checkpoint_steps=50))
        assert decoded["minutes"] == 4320


class TestFieldValidation:
    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_spec([1, 2])

    def test_a_partition_this_cluster_lacks_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_job_spec(_spec(partition="turbo"))
        assert excinfo.value.code is Hpc3ErrorCode.PARTITION_UNKNOWN

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_spec(_spec(name=""))

    def test_an_empty_env_path_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_spec(_spec(env_path=""))

    def test_an_empty_command_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_spec(_spec(command=""))

    def test_zero_cpus_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_spec(_spec(cpus=0))

    def test_zero_memory_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_spec(_spec(mem_gb=0))

    def test_zero_minutes_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_spec(_spec(minutes=0))

    def test_negative_checkpoint_steps_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_job_spec(_spec(checkpoint_steps=-1))


class TestEncodeCpuOnly:
    def test_a_cpu_only_spec_round_trips_through_null(self) -> None:
        """The ledger and the audit trail both re-encode a spec, so a CPU job
        that could not survive the round trip would be unrecordable."""
        payload = _spec(partition="free", gpu=None)
        assert encode_job_spec(decode_job_spec(payload)) == payload

    def test_the_encoded_gpu_field_is_null_not_an_empty_object(self) -> None:
        encoded = encode_job_spec(decode_job_spec(_spec(partition="free", gpu=None)))
        assert encoded["gpu"] is None


class TestEncode:
    def test_encode_preserves_a_billing_spec(self) -> None:
        payload = _spec(partition="gpu32", gpu=gpus("RTX6000"), accept_billing=True)
        spec: JobSpec = decode_job_spec(payload)
        assert encode_job_spec(spec)["partition"] == "gpu32"
        assert encode_job_spec(spec)["accept_billing"] is True
