"""Tests for batch-script rendering.

The assertions are about the submitted text, because that text is what the
cluster acts on. Two of them exist to prove a rule survived the whole way
from contract to script: the GPU model always appears in ``--gres``, and
``--requeue`` appears exactly when the spec carries it.
"""

from __future__ import annotations

from platform_core.determinism_env import (
    CUBLAS_DETERMINISTIC_WORKSPACE,
    CUBLAS_WORKSPACE_ENV_VAR,
)
from platform_core.json_utils import JSONValue

from hpc3.contracts.job import JobSpec
from hpc3.core.sbatch import format_walltime, job_comment, render_sbatch
from tests.against_hpc3 import decode_job_spec

_LOG_DIR = "/pub/wagnera3/logs"


def _spec(**overrides: JSONValue) -> JobSpec:
    """Build a decoded job spec with optional overrides.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        A validated spec.
    """
    base: dict[str, JSONValue] = {
        "project": "abl",
        "name": "arm-b-42",
        "partition": "free-gpu",
        "gpu": "A100",
        "gpu_count": 1,
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
        "command": "python train.py --seed 42",
    }
    base.update(overrides)
    return decode_job_spec(base)


class TestFormatWalltime:
    def test_under_an_hour(self) -> None:
        assert format_walltime(30) == "00:30:00"

    def test_exactly_an_hour(self) -> None:
        assert format_walltime(60) == "01:00:00"

    def test_under_a_day(self) -> None:
        assert format_walltime(23 * 60 + 59) == "23:59:00"

    def test_exactly_a_day_uses_the_day_form(self) -> None:
        assert format_walltime(1440) == "1-00:00:00"

    def test_three_days_is_the_free_partition_ceiling(self) -> None:
        assert format_walltime(72 * 60) == "3-00:00:00"

    def test_days_and_hours_together(self) -> None:
        assert format_walltime(1440 + 90) == "1-01:30:00"


class TestRenderSbatch:
    def test_the_gpu_model_always_reaches_the_gres_line(self) -> None:
        """No code path emits a bare gpu:N -- the spec cannot express one."""
        script = render_sbatch(_spec(), log_dir=_LOG_DIR)
        assert "#SBATCH --gres=gpu:A100:1" in script
        assert "--gres=gpu:1" not in script

    def test_every_resource_directive_is_present(self) -> None:
        script = render_sbatch(_spec(), log_dir=_LOG_DIR)
        assert "#SBATCH -J abl.arm-b-42" in script
        assert "#SBATCH -p free-gpu" in script
        assert "#SBATCH -c 8" in script
        assert "#SBATCH --mem=96G" in script
        assert "#SBATCH -t 00:30:00" in script

    def test_log_paths_carry_the_job_id_pattern(self) -> None:
        script = render_sbatch(_spec(), log_dir=_LOG_DIR)
        assert f"#SBATCH -o {_LOG_DIR}/abl.arm-b-42-%j.out" in script
        assert f"#SBATCH -e {_LOG_DIR}/abl.arm-b-42-%j.err" in script

    def test_requeue_is_absent_when_not_requested(self) -> None:
        assert "--requeue" not in render_sbatch(_spec(), log_dir=_LOG_DIR)

    def test_requeue_appears_when_the_spec_carries_it(self) -> None:
        spec = _spec(minutes=600, requeue=True, checkpoint_steps=50)
        assert "#SBATCH --requeue" in render_sbatch(spec, log_dir=_LOG_DIR)

    def test_the_script_starts_with_a_login_shell_shebang(self) -> None:
        assert render_sbatch(_spec(), log_dir=_LOG_DIR).startswith("#!/bin/bash -l\n")

    def test_it_ends_with_a_newline_and_uses_lf_only(self) -> None:
        script = render_sbatch(_spec(), log_dir=_LOG_DIR)
        assert script.endswith("\n")
        assert "\r" not in script

    def test_the_payload_command_is_present_verbatim(self) -> None:
        script = render_sbatch(_spec(), log_dir=_LOG_DIR)
        assert "python train.py --seed 42" in script

    def test_the_environment_is_put_on_path(self) -> None:
        script = render_sbatch(_spec(), log_dir=_LOG_DIR)
        assert 'export PATH="/pub/wagnera3/envs/abl-pinned/bin:$PATH"' in script

    def test_the_checkpoint_interval_reaches_the_payload(self) -> None:
        spec = _spec(minutes=600, requeue=True, checkpoint_steps=50)
        assert 'export HPC3_CHECKPOINT_STEPS="50"' in render_sbatch(spec, log_dir=_LOG_DIR)

    def test_it_does_not_set_e_so_the_payload_status_survives(self) -> None:
        """`set -e` would let a failed run exit zero through this wrapper.

        Matched against the script's own lines rather than as a substring:
        the rendered comment explains why `set -e` is absent, and a naive
        substring check finds that explanation and fails on it.
        """
        lines = render_sbatch(_spec(), log_dir=_LOG_DIR).splitlines()
        directives = [line.strip() for line in lines if not line.lstrip().startswith("#")]
        assert "set -u" in directives
        assert "set -e" not in directives
        assert not any(line.startswith("set -e") for line in directives)

    def test_it_propagates_the_payload_exit_code(self) -> None:
        script = render_sbatch(_spec(), log_dir=_LOG_DIR)
        assert "rc=$?" in script
        assert script.rstrip("\n").endswith("exit $rc")


class TestJobIsSelfDescribing:
    """What another cluster user, or we in six months, can see about a row.

    HPC3 had 102 distinct users with running jobs when this was measured, and
    ``squeue`` shows all of them to all of them. A row that says only
    ``arm-b-42`` is anonymous; these assertions are the difference.
    """

    def test_the_job_name_is_prefixed_by_its_project(self) -> None:
        assert "#SBATCH -J abl.arm-b-42" in render_sbatch(_spec(), log_dir=_LOG_DIR)

    def test_the_comment_carries_project_hardware_environment_and_experiment(self) -> None:
        assert job_comment(_spec()) == (
            "project=abl;gpu=A100x1;cpus=8;env=/pub/wagnera3/envs/abl-pinned"
            ";det=off;exp=arm=B,seed=42"
        )

    def test_the_comment_states_the_determinism_posture(self) -> None:
        """Two arms differing only in this are two records, not two samples."""
        assert ";det=off;" in job_comment(_spec())
        assert ";det=on;" in job_comment(_spec(deterministic=True))

    def test_a_queue_row_says_which_experiment_it_is(self) -> None:
        """So a row found in squeue answers the question without a ledger."""
        script = render_sbatch(_spec(experiment={"corpus": "07ab4976"}), log_dir=_LOG_DIR)
        assert "exp=corpus=07ab4976" in script

    def test_the_comment_holds_no_space(self) -> None:
        """sbatch takes --comment as one token; a space truncates the rest."""
        assert " " not in job_comment(_spec())

    def test_the_comment_reaches_the_script_as_one_directive(self) -> None:
        script = render_sbatch(_spec(), log_dir=_LOG_DIR)
        prefix = "#SBATCH --comment"
        comment_lines = [line for line in script.splitlines() if line.startswith(prefix)]
        assert comment_lines == [f"#SBATCH --comment {job_comment(_spec())}"]

    def test_the_comment_tracks_a_multi_gpu_request(self) -> None:
        assert "gpu=A100x4" in job_comment(_spec(gpu_count=4))

    def test_the_payload_can_read_its_own_project_and_label(self) -> None:
        """A training script writing checkpoints needs to name them something."""
        script = render_sbatch(_spec(), log_dir=_LOG_DIR)
        assert 'export HPC3_PROJECT="abl"' in script
        assert 'export HPC3_JOB_NAME="abl.arm-b-42"' in script

    def test_two_projects_cannot_produce_the_same_label(self) -> None:
        mine = render_sbatch(_spec(), log_dir=_LOG_DIR)
        theirs = render_sbatch(_spec(project="sirius"), log_dir=_LOG_DIR)
        assert "#SBATCH -J abl.arm-b-42" in mine
        assert "#SBATCH -J sirius.arm-b-42" in theirs


class TestDeterminismIsDeclaredBeforeTheProcessStarts:
    """The half a submitter can guarantee, and the half it cannot.

    cuBLAS reads its workspace variable once, when the handle is created on
    first use, so setting it after CUDA has started is accepted in silence and
    does nothing. Exported from the batch script it cannot be too late. The
    switch that actually enables determinism is a torch call in the payload's
    own process, which this package neither makes nor pretends to.
    """

    def test_a_deterministic_run_carries_the_cublas_workspace(self) -> None:
        script = render_sbatch(_spec(deterministic=True), log_dir=_LOG_DIR)
        assert f'export {CUBLAS_WORKSPACE_ENV_VAR}="{CUBLAS_DETERMINISTIC_WORKSPACE}"' in script

    def test_the_workspace_value_is_the_shared_one_not_a_copy(self) -> None:
        """Trainer and submitter must write the same string or the two runs
        silently stop being comparable; there is one definition."""
        assert CUBLAS_WORKSPACE_ENV_VAR == "CUBLAS_WORKSPACE_CONFIG"
        assert CUBLAS_DETERMINISTIC_WORKSPACE == ":4096:8"

    def test_a_nondeterministic_run_does_not_carry_it(self) -> None:
        assert CUBLAS_WORKSPACE_ENV_VAR not in render_sbatch(_spec(), log_dir=_LOG_DIR)

    def test_the_posture_is_exported_either_way(self) -> None:
        """Absent is a state, not a message. A payload inferring determinism
        from a missing variable would silently train the other record."""
        assert 'export HPC3_DETERMINISTIC="0"' in render_sbatch(_spec(), log_dir=_LOG_DIR)
        assert 'export HPC3_DETERMINISTIC="1"' in render_sbatch(
            _spec(deterministic=True), log_dir=_LOG_DIR
        )

    def test_it_is_exported_before_the_payload_runs(self) -> None:
        """After CUDA starts the variable is accepted and ignored."""
        script = render_sbatch(_spec(deterministic=True), log_dir=_LOG_DIR)
        lines = script.splitlines()
        exported = next(i for i, line in enumerate(lines) if CUBLAS_WORKSPACE_ENV_VAR in line)
        payload = next(i for i, line in enumerate(lines) if line == "python train.py --seed 42")
        assert exported < payload


class TestResumeSurface:
    """The package cannot resume for the payload -- only the payload knows
    what its checkpoint means -- so it surfaces the restart count and leaves
    the decision where the knowledge is.
    """

    def test_the_restart_count_is_exported_from_slurms_own_variable(self) -> None:
        script = render_sbatch(_spec(), log_dir=_LOG_DIR)
        assert 'export HPC3_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"' in script

    def test_it_defaults_to_zero_on_a_first_run(self) -> None:
        """The ':-0' is what makes a first run readable, not an unset variable
        under `set -u`."""
        script = render_sbatch(_spec(), log_dir=_LOG_DIR)
        assert ":-0}" in script

    def test_the_restart_count_is_echoed_into_the_job_log(self) -> None:
        script = render_sbatch(_spec(), log_dir=_LOG_DIR)
        assert 'echo "restart   ${HPC3_RESTART_COUNT}"' in script

    def test_a_protected_run_carries_both_requeue_and_the_count(self) -> None:
        spec = _spec(minutes=600, requeue=True, checkpoint_steps=50)
        script = render_sbatch(spec, log_dir=_LOG_DIR)
        assert "#SBATCH --requeue" in script
        assert "HPC3_RESTART_COUNT" in script
        assert 'export HPC3_CHECKPOINT_STEPS="50"' in script
