"""The array script: one file that IS the member table, dispatched by index.

What these hold: the script carries every document position whatever subset
gets submitted, carries NO ``--array`` directive (the submitter's argument
owns the selection), and dispatches each task to its own member's payload
with the member's own name exported -- so the payload cannot tell it was an
array task.
"""

from __future__ import annotations

from platform_core.json_utils import JSONValue

from hpc3.contracts.sweep import SweepSpec
from hpc3.core.array_sbatch import NO_SUCH_MEMBER_EXIT, render_array_sbatch
from tests.against_hpc3 import decode_sweep_spec
from tests.conftest import gpus


def _sweep(count: int = 3, **overrides: JSONValue) -> SweepSpec:
    """Build a decoded sweep.

    Args:
        count: How many members.
        **overrides: Template fields to replace.

    Returns:
        A validated sweep.
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
        "experiment": {"rung": "774M"},
        "command": "python train.py",
        "artifact": None,
    }
    base.update(overrides)
    members: list[JSONValue] = [
        {"suffix": f"s{i}", "command": f"python train.py --seed {i}", "artifact": None}
        for i in range(count)
    ]
    return decode_sweep_spec({"base": base, "members": members})


def _render(**overrides: JSONValue) -> str:
    return render_array_sbatch(_sweep(3, **overrides), log_dir="/l", charge_account="")


class TestDirectives:
    def test_the_header_names_the_sweep_and_logs_per_task(self) -> None:
        script = _render()
        assert "#SBATCH -J abl.rung\n" in script
        # %A is the array id and %a the task index -- the exact pair the
        # ledger records, so a log file is findable from its ledger row.
        assert "#SBATCH -o /l/abl.rung-%A_%a.out\n" in script
        assert "#SBATCH -e /l/abl.rung-%A_%a.err\n" in script

    def test_no_array_directive_is_emitted(self) -> None:
        """The selection is the submitter's argument, never the script's:
        a directive would bind the record of what the members ARE to one
        submission's choice of which to run."""
        assert "--array" not in _render()

    def test_an_account_is_emitted_only_when_named(self) -> None:
        bare = _render()
        assert "--account" not in bare
        billed = render_array_sbatch(_sweep(), log_dir="/l", charge_account="lab")
        assert "#SBATCH --account=lab\n" in billed

    def test_a_cpu_only_sweep_emits_no_gres_line(self) -> None:
        script = _render(partition="free", gpu=None)
        assert "--gres" not in script
        assert 'echo "gpu       cpu-only"' in script
        assert "nvidia-smi" not in script

    def test_a_gpu_sweep_emits_the_model_and_the_probe(self) -> None:
        script = _render()
        assert "#SBATCH --gres=gpu:A100:1\n" in script
        assert "nvidia-smi --query-gpu=name,memory.total" in script

    def test_a_dependency_is_paired_with_the_kill_switch(self) -> None:
        """Same pairing as the single-job renderer: without it, an
        unsatisfiable dependency parks EVERY task of the array on
        DependencyNeverSatisfied, each holding a QOS slot."""
        script = _render(depends_on={"kind": "afterok", "job_ids": ["11"]})
        assert "#SBATCH --dependency=afterok:11\n" in script
        assert "#SBATCH --kill-on-invalid-dep=yes\n" in script

    def test_requeue_appears_exactly_when_requested(self) -> None:
        assert "--requeue" not in _render()
        assert "#SBATCH --requeue\n" in _render(requeue=True)


class TestDispatch:
    def test_every_member_gets_a_case_arm_in_document_order(self) -> None:
        script = _render()
        assert 'case "${SLURM_ARRAY_TASK_ID}" in' in script
        arm0 = script.index("0)\n")
        arm1 = script.index("1)\n")
        arm2 = script.index("2)\n")
        assert arm0 < arm1 < arm2
        assert script.index("--seed 0") < script.index("--seed 1") < script.index("--seed 2")

    def test_each_arm_exports_the_members_own_name(self) -> None:
        """The payload sees exactly what it would have seen as a single
        job; the shared -J is a Slurm-side legibility cost, not the
        payload's problem."""
        script = _render()
        assert 'export HPC3_JOB_NAME="abl.rung-s0"' in script
        assert 'export HPC3_JOB_NAME="abl.rung-s1"' in script
        assert 'export HPC3_JOB_NAME="abl.rung-s2"' in script

    def test_an_index_outside_the_table_is_a_loud_refusal(self) -> None:
        """A refusal, not a fallback: an unknown index means the submitted
        --array disagrees with the script."""
        script = _render()
        assert 'echo "no member at array index ${SLURM_ARRAY_TASK_ID}" >&2' in script
        assert f"exit {NO_SUCH_MEMBER_EXIT}" in script

    def test_the_echo_block_names_the_task_identity(self) -> None:
        script = _render()
        assert 'echo "job       ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none}"' in script


class TestScriptShape:
    def test_the_script_is_lf_terminated_and_sets_u(self) -> None:
        script = _render()
        assert "\r" not in script
        assert script.endswith("\n")
        assert "set -u\n" in script
        # The directive itself, not the comment explaining its absence.
        assert "\nset -e\n" not in script
