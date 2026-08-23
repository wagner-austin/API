"""Tests for the sweep contract.

The ceiling checks carry the weight. Slurm queues an over-sized sweep rather
than refusing it, so without these the failure mode is an operator waiting on
jobs that will not start until other jobs of theirs finish.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.clusters.hpc3 import HPC3
from hpc3.contracts.cluster import partition_facts
from hpc3.contracts.sweep import (
    decode_sweep_member,
    encode_sweep_spec,
    expand_sweep,
)
from tests.against_hpc3 import decode_sweep_spec
from tests.conftest import gpus


def _base(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid template payload.

    Args:
        **overrides: Fields to replace.

    Returns:
        A JSON object ready for decoding.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": "rung",
        "partition": "free-gpu",
        "gpu": gpus("A100"),
        "cpus": 8,
        "mem_gb": 96,
        "minutes": 30,
        "requeue": False,
        "checkpoint_steps": 0,
        "env_path": "/pub/envs/abl-pinned",
        "pinned_packages": {},
        "deterministic": False,
        "depends_on": None,
        "experiment": {"rung": "774M"},
        "command": "python train.py",
    }
    base.update(overrides)
    return base


def _members(count: int) -> list[JSONValue]:
    """Build a list of distinct members.

    Args:
        count: How many to build.

    Returns:
        Member payloads, suffixed by index.
    """
    return [{"suffix": f"s{i}", "command": f"python train.py --seed {i}"} for i in range(count)]


def _sweep(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid sweep payload.

    Args:
        **overrides: Fields to replace.

    Returns:
        A JSON object ready for decoding.
    """
    spec: dict[str, JSONValue] = {"base": _base(), "members": _members(3)}
    spec.update(overrides)
    return spec


class TestValidSweep:
    def test_a_valid_sweep_round_trips(self) -> None:
        assert encode_sweep_spec(decode_sweep_spec(_sweep())) == _sweep()

    def test_expansion_names_each_member_from_the_template(self) -> None:
        specs = expand_sweep(decode_sweep_spec(_sweep()))
        assert [spec["name"] for spec in specs] == ["rung-s0", "rung-s1", "rung-s2"]

    def test_expansion_gives_each_member_its_own_command(self) -> None:
        specs = expand_sweep(decode_sweep_spec(_sweep()))
        assert [spec["command"] for spec in specs] == [
            "python train.py --seed 0",
            "python train.py --seed 1",
            "python train.py --seed 2",
        ]

    def test_expansion_shares_every_resource_setting(self) -> None:
        specs = expand_sweep(decode_sweep_spec(_sweep()))
        assert {spec["partition"] for spec in specs} == {"free-gpu"}
        assert [spec["gpu"] for spec in specs] == [{"model": "A100", "count": 1}] * len(specs)
        assert {spec["cpus"] for spec in specs} == {8}
        assert {spec["minutes"] for spec in specs} == {30}
        assert {spec["env_path"] for spec in specs} == {"/pub/envs/abl-pinned"}


class TestGpuCeiling:
    def test_the_six_arm_scale_rung_fits_free_gpu(self) -> None:
        """The real workload: 2 corpora x 3 seeds, 1 GPU each, ceiling 24."""
        decoded = decode_sweep_spec(_sweep(members=_members(6)))
        assert len(decoded["members"]) == 6

    def test_exactly_the_ceiling_is_admitted(self) -> None:
        decoded = decode_sweep_spec(_sweep(members=_members(24)))
        assert len(decoded["members"]) == 24

    def test_one_over_the_gpu_ceiling_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_sweep_spec(_sweep(members=_members(25)))
        assert excinfo.value.code is Hpc3ErrorCode.SWEEP_EXCEEDS_GPU_CEILING

    def test_multi_gpu_members_count_against_the_ceiling(self) -> None:
        """Three members at 2 GPUs each is six GPUs, not three."""
        with pytest.raises(AppError) as excinfo:
            decode_sweep_spec(_sweep(base=_base(gpu=gpus("A100", 3)), members=_members(9)))
        assert excinfo.value.code is Hpc3ErrorCode.SWEEP_EXCEEDS_GPU_CEILING

    def test_a_cpu_sweep_is_bounded_by_cores_not_gpus(self) -> None:
        """`free` caps one user at 3500 cores and declares no GPU ceiling, so
        a wide CPU sweep has exactly one limit that can catch it."""
        payload = _sweep(
            base=_base(partition="free", gpu=None, cpus=64),
            members=_members(60),
        )
        with pytest.raises(AppError) as excinfo:
            decode_sweep_spec(payload)
        assert excinfo.value.code is Hpc3ErrorCode.SWEEP_EXCEEDS_CPU_CEILING

    def test_a_cpu_sweep_inside_the_core_ceiling_is_admitted(self) -> None:
        specs = expand_sweep(
            decode_sweep_spec(
                _sweep(base=_base(partition="free", gpu=None, cpus=8), members=_members(10))
            )
        )
        assert len(specs) == 10
        assert [spec["gpu"] for spec in specs] == [None] * 10

    def test_a_gpu_sweep_is_not_checked_against_an_undeclared_core_ceiling(self) -> None:
        """`free-gpu-part` caps GPUs and says nothing about cores. Inventing a
        core limit for it would refuse sweeps the cluster would have run."""
        specs = expand_sweep(decode_sweep_spec(_sweep(base=_base(cpus=40), members=_members(20))))
        assert len(specs) == 20

    def test_free_gpu32_has_a_much_lower_ceiling(self) -> None:
        payload = _sweep(
            base=_base(partition="free-gpu32", gpu=gpus("L40S")),
            members=_members(5),
        )
        with pytest.raises(AppError) as excinfo:
            decode_sweep_spec(payload)
        assert excinfo.value.code is Hpc3ErrorCode.SWEEP_EXCEEDS_GPU_CEILING


class TestJobCeilingCannotBindFirstOnThisCluster:
    """It used to be tested against ``gpu`` (32 jobs under 40 GPUs), which is
    a billing partition and no longer submittable.

    On every FREE partition of HPC3 the resource ceiling binds first or at the
    same point, so the job check cannot fire on its own here. That is a
    property of the machine, not a gap: it is asserted below rather than left
    implied, and the check itself is exercised on a synthetic cluster shaped
    to reach it in ``test_cluster.py``.
    """

    def test_the_gpu_ceiling_fires_before_the_job_ceiling_on_free_gpu(self) -> None:
        """24 GPUs and 24 jobs -- equal caps, and GPUs are checked first."""
        with pytest.raises(AppError) as excinfo:
            decode_sweep_spec(_sweep(members=_members(25)))
        assert excinfo.value.code is Hpc3ErrorCode.SWEEP_EXCEEDS_GPU_CEILING

    def test_the_core_ceiling_fires_before_the_job_ceiling_on_free(self) -> None:
        """3500 cores and 3500 jobs, so a one-core member hits both at once
        and the core check is the one that reports."""
        payload = _sweep(base=_base(partition="free", gpu=None, cpus=1), members=_members(3501))
        with pytest.raises(AppError) as excinfo:
            decode_sweep_spec(payload)
        assert excinfo.value.code is Hpc3ErrorCode.SWEEP_EXCEEDS_CPU_CEILING

    def test_no_free_partition_lets_the_job_ceiling_bind_alone(self) -> None:
        """The property this class is named for, asserted against the measured
        facts rather than argued in a docstring."""
        for name in ("free", "free-gpu", "free-gpu32"):
            facts = partition_facts(HPC3, name)
            resource = facts["max_gpus_per_user"] or facts["max_cpus_per_user"]
            if resource is None:
                raise AssertionError(f"{name} bounds neither GPUs nor cores")
            assert resource <= facts["max_jobs_per_user"]


class TestMemberValidation:
    def test_a_non_object_member_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_sweep_member("s0")

    def test_an_empty_suffix_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_sweep_member({"suffix": "", "command": "x"})

    def test_an_empty_command_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_sweep_member({"suffix": "s0", "command": ""})

    def test_a_slashed_suffix_is_refused(self) -> None:
        """The suffix reaches a log filename; a separator would escape it."""
        with pytest.raises(JSONTypeError):
            decode_sweep_member({"suffix": "a/b", "command": "x"})
        with pytest.raises(JSONTypeError):
            decode_sweep_member({"suffix": "a\\b", "command": "x"})

    def test_a_valid_member_decodes(self) -> None:
        member = decode_sweep_member({"suffix": "s0", "command": "python x.py"})
        assert member == {"suffix": "s0", "command": "python x.py"}


class TestSweepValidation:
    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_sweep_spec([])

    def test_a_missing_base_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_sweep_spec({"members": _members(1)})

    def test_an_invalid_base_propagates_its_own_code(self) -> None:
        with pytest.raises(AppError) as excinfo:
            decode_sweep_spec(_sweep(base=_base(gpu=gpus("gpu"))))
        assert excinfo.value.code is Hpc3ErrorCode.GPU_TYPE_UNPINNED

    def test_an_empty_member_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_sweep_spec(_sweep(members=[]))

    def test_a_repeated_suffix_is_refused(self) -> None:
        """Two jobs sharing a name interleave into one log file."""
        duplicate: list[JSONValue] = [
            {"suffix": "s0", "command": "a"},
            {"suffix": "s0", "command": "b"},
        ]
        with pytest.raises(JSONTypeError):
            decode_sweep_spec(_sweep(members=duplicate))
