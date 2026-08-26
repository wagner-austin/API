"""Tests for batch-script rendering of a job that runs on the cluster itself.

The assertions are about the submitted text, because that text is what the
cluster acts on. Two of them exist to prove a rule survived the whole way
from contract to script: the GPU model always appears in ``--gres``, and
``--requeue`` appears exactly when the spec carries it.

Rendering a job that runs INSIDE AN IMAGE is ``test_sbatch_image``, split off
when this module passed the 600-line ceiling. Both build their specs from
``_sbatch_support`` so the two renderings stay two views of one job.
"""

from __future__ import annotations

from platform_core.determinism_env import (
    CUBLAS_DETERMINISTIC_WORKSPACE,
    CUBLAS_WORKSPACE_ENV_VAR,
    DETERMINISM_ENV_VAR,
    determinism_requested,
)

from hpc3.contracts.job import JobSpec
from hpc3.core.sbatch import (
    _code_provenance_export,
    format_walltime,
    job_comment,
    render_sbatch,
)
from tests._sbatch_support import LOG_DIR, spec
from tests.conftest import gpus


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
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert "#SBATCH --gres=gpu:A100:1" in script
        assert "--gres=gpu:1" not in script

    def test_every_resource_directive_is_present(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert "#SBATCH -J abl.arm-b-42" in script
        assert "#SBATCH -p free-gpu" in script
        assert "#SBATCH -c 8" in script
        assert "#SBATCH --mem=96G" in script
        assert "#SBATCH -t 00:30:00" in script

    def test_log_paths_carry_the_job_id_pattern(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert f"#SBATCH -o {LOG_DIR}/abl.arm-b-42-%j.out" in script
        assert f"#SBATCH -e {LOG_DIR}/abl.arm-b-42-%j.err" in script

    def test_requeue_is_absent_when_not_requested(self) -> None:
        assert "--requeue" not in render_sbatch(spec(), log_dir=LOG_DIR)

    def test_requeue_appears_when_the_spec_carries_it(self) -> None:
        job = spec(minutes=600, requeue=True, checkpoint_steps=50)
        assert "#SBATCH --requeue" in render_sbatch(job, log_dir=LOG_DIR)

    def test_the_script_starts_with_a_login_shell_shebang(self) -> None:
        assert render_sbatch(spec(), log_dir=LOG_DIR).startswith("#!/bin/bash -l\n")

    def test_it_ends_with_a_newline_and_uses_lf_only(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert script.endswith("\n")
        assert "\r" not in script

    def test_the_payload_command_is_present_verbatim(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert "python train.py --seed 42" in script

    def test_the_environment_is_put_on_path(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert 'export PATH="/pub/wagnera3/envs/abl-pinned/bin:$PATH"' in script

    def test_the_checkpoint_interval_reaches_the_payload(self) -> None:
        job = spec(minutes=600, requeue=True, checkpoint_steps=50)
        assert 'export HPC3_CHECKPOINT_STEPS="50"' in render_sbatch(job, log_dir=LOG_DIR)

    def test_it_does_not_set_e_so_the_payload_status_survives(self) -> None:
        """`set -e` would let a failed run exit zero through this wrapper.

        Matched against the script's own lines rather than as a substring:
        the rendered comment explains why `set -e` is absent, and a naive
        substring check finds that explanation and fails on it.
        """
        lines = render_sbatch(spec(), log_dir=LOG_DIR).splitlines()
        directives = [line.strip() for line in lines if not line.lstrip().startswith("#")]
        assert "set -u" in directives
        assert "set -e" not in directives
        assert not any(line.startswith("set -e") for line in directives)

    def test_it_propagates_the_payload_exit_code(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert "rc=$?" in script
        assert script.rstrip("\n").endswith("exit $rc")


class TestJobIsSelfDescribing:
    """What another cluster user, or we in six months, can see about a row.

    HPC3 had 102 distinct users with running jobs when this was measured, and
    ``squeue`` shows all of them to all of them. A row that says only
    ``arm-b-42`` is anonymous; these assertions are the difference.
    """

    def test_the_job_name_is_prefixed_by_its_project(self) -> None:
        assert "#SBATCH -J abl.arm-b-42" in render_sbatch(spec(), log_dir=LOG_DIR)

    def test_the_comment_carries_project_hardware_environment_and_experiment(self) -> None:
        assert job_comment(spec()) == (
            "project=abl;gpu=A100x1;cpus=8;env=/pub/wagnera3/envs/abl-pinned"
            ";det=off;exp=arm=B,seed=42"
        )

    def test_the_comment_states_the_determinism_posture(self) -> None:
        """Two arms differing only in this are two records, not two samples."""
        assert ";det=off;" in job_comment(spec())
        assert ";det=on;" in job_comment(spec(deterministic=True))

    def test_a_queue_row_says_which_experiment_it_is(self) -> None:
        """So a row found in squeue answers the question without a ledger."""
        script = render_sbatch(spec(experiment={"corpus": "07ab4976"}), log_dir=LOG_DIR)
        assert "exp=corpus=07ab4976" in script

    def test_the_comment_holds_no_space(self) -> None:
        """sbatch takes --comment as one token; a space truncates the rest."""
        assert " " not in job_comment(spec())

    def test_the_comment_reaches_the_script_as_one_directive(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        prefix = "#SBATCH --comment"
        comment_lines = [line for line in script.splitlines() if line.startswith(prefix)]
        assert comment_lines == [f"#SBATCH --comment {job_comment(spec())}"]

    def test_the_comment_tracks_a_multi_gpu_request(self) -> None:
        assert "gpu=A100x4" in job_comment(spec(gpu=gpus("A100", 4)))

    def test_the_payload_can_read_its_own_project_and_label(self) -> None:
        """A training script writing checkpoints needs to name them something."""
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert 'export HPC3_PROJECT="abl"' in script
        assert 'export HPC3_JOB_NAME="abl.arm-b-42"' in script

    def test_two_projects_cannot_produce_the_same_label(self) -> None:
        mine = render_sbatch(spec(), log_dir=LOG_DIR)
        theirs = render_sbatch(spec(project="sirius"), log_dir=LOG_DIR)
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
        script = render_sbatch(spec(deterministic=True), log_dir=LOG_DIR)
        assert f'export {CUBLAS_WORKSPACE_ENV_VAR}="{CUBLAS_DETERMINISTIC_WORKSPACE}"' in script

    def test_the_workspace_value_is_the_shared_one_not_a_copy(self) -> None:
        """Trainer and submitter must write the same string or the two runs
        silently stop being comparable; there is one definition."""
        assert CUBLAS_WORKSPACE_ENV_VAR == "CUBLAS_WORKSPACE_CONFIG"
        assert CUBLAS_DETERMINISTIC_WORKSPACE == ":4096:8"

    def test_a_nondeterministic_run_does_not_carry_it(self) -> None:
        assert CUBLAS_WORKSPACE_ENV_VAR not in render_sbatch(spec(), log_dir=LOG_DIR)

    def test_the_posture_is_exported_either_way(self) -> None:
        """Absent is a state, not a message. A payload inferring determinism
        from a missing variable would silently train the other record."""
        assert f'export {DETERMINISM_ENV_VAR}="0"' in render_sbatch(spec(), log_dir=LOG_DIR)
        assert f'export {DETERMINISM_ENV_VAR}="1"' in render_sbatch(
            spec(deterministic=True), log_dir=LOG_DIR
        )

    def test_what_the_script_exports_is_what_the_trainer_reads_back(self) -> None:
        """The two halves joined, against the trainer's own reader rather than
        against a string this file also wrote.

        The launcher writing a variable the trainer does not read is the exact
        failure the shared definition exists to prevent, and it would look
        like success from both sides: the export line is present, the trainer
        pins determinism by default, and a run declared OFF would silently be
        deterministic anyway.
        """
        for deterministic, expected in ((True, True), (False, False)):
            script = render_sbatch(spec(deterministic=deterministic), log_dir=LOG_DIR)
            exported = {
                line.removeprefix("export ").split("=", 1)[0]: line.split("=", 1)[1].strip('"')
                for line in script.splitlines()
                if line.startswith("export ")
            }
            assert determinism_requested(exported.get(DETERMINISM_ENV_VAR)) is expected

    def test_it_is_exported_before_the_payload_runs(self) -> None:
        """After CUDA starts the variable is accepted and ignored."""
        script = render_sbatch(spec(deterministic=True), log_dir=LOG_DIR)
        lines = script.splitlines()
        exported = next(i for i, line in enumerate(lines) if CUBLAS_WORKSPACE_ENV_VAR in line)
        payload = next(i for i, line in enumerate(lines) if line == "python train.py --seed 42")
        assert exported < payload


class TestCpuOnlyJobs:
    """A job that holds no GPU, rendered as one.

    The absences are the assertions here. A ``--gres`` line asking for zero
    is not the same as no ``--gres`` line, and ``nvidia-smi`` on a CPU node
    writes a command-not-found to stderr on every single run -- which teaches
    whoever reads these logs to ignore the one stream a real failure arrives
    by.
    """

    def _cpu(self) -> JobSpec:
        """Build a CPU-only spec on the free CPU partition.

        Returns:
            A validated spec holding no GPU.
        """
        return spec(partition="free", gpu=None, command="sirius --input /pub/data")

    def test_no_gres_line_is_emitted_at_all(self) -> None:
        script = render_sbatch(self._cpu(), log_dir=LOG_DIR)
        assert "--gres" not in script

    def test_nvidia_smi_is_not_called_on_a_node_without_it(self) -> None:
        script = render_sbatch(self._cpu(), log_dir=LOG_DIR)
        assert "nvidia-smi" not in script

    def test_the_log_still_says_what_hardware_it_got(self) -> None:
        """Silence would read as a truncated header, not as a CPU job."""
        script = render_sbatch(self._cpu(), log_dir=LOG_DIR)
        assert 'echo "gpu       cpu-only"' in script

    def test_the_comment_says_cpu_only_rather_than_going_blank(self) -> None:
        assert job_comment(self._cpu()) == (
            "project=abl;gpu=cpu-only;cpus=8;env=/pub/wagnera3/envs/abl-pinned"
            ";det=off;exp=arm=B,seed=42"
        )

    def test_the_payload_and_exit_handling_are_unchanged(self) -> None:
        """A CPU job is a normal job; only the hardware lines differ."""
        script = render_sbatch(self._cpu(), log_dir=LOG_DIR)
        assert "sirius --input /pub/data" in script
        assert "#SBATCH -p free" in script
        assert "#SBATCH -c 8" in script
        assert script.rstrip("\n").endswith("exit $rc")

    def test_a_gpu_job_on_the_same_cluster_still_gets_its_gres(self) -> None:
        """The CPU path must not be reachable by a job that asked for a GPU."""
        assert "#SBATCH --gres=gpu:A100:1" in render_sbatch(spec(), log_dir=LOG_DIR)


class TestDependencyIsNeverEmittedWithoutItsSafetyPairing:
    """The two directives are one decision, so they are tested as one.

    Slurm does not reject an unsatisfiable dependency -- it queues it forever
    on ``DependencyNeverSatisfied``, holding a QOS slot and looking exactly
    like a job waiting its turn. That was 261 of 621 pending GPU jobs on HPC3
    in one sample. ``--kill-on-invalid-dep=yes`` is what makes the failure
    terminal and therefore visible.
    """

    def _waiting(self) -> JobSpec:
        """Build a spec that waits on one job.

        Returns:
            A validated spec carrying an afterok dependency.
        """
        return spec(depends_on={"kind": "afterok", "job_ids": ["55519937"]})

    def test_the_dependency_reaches_the_script(self) -> None:
        assert "#SBATCH --dependency=afterok:55519937" in render_sbatch(
            self._waiting(), log_dir=LOG_DIR
        )

    def test_the_kill_flag_is_emitted_with_it(self) -> None:
        assert "#SBATCH --kill-on-invalid-dep=yes" in render_sbatch(
            self._waiting(), log_dir=LOG_DIR
        )

    def test_neither_appears_when_the_job_waits_on_nothing(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert "--dependency" not in script
        assert "--kill-on-invalid-dep" not in script

    def test_the_kill_flag_never_appears_alone(self) -> None:
        """Alone it is inert, and its presence would imply a wait that is not
        there. Asserted as a relation over both specs rather than as two
        separate absences."""
        for job in (spec(), self._waiting()):
            script = render_sbatch(job, log_dir=LOG_DIR)
            assert ("--kill-on-invalid-dep" in script) == ("--dependency" in script)

    def test_several_ids_render_colon_joined(self) -> None:
        job = spec(depends_on={"kind": "afterany", "job_ids": ["1", "2"]})
        assert "#SBATCH --dependency=afterany:1:2" in render_sbatch(job, log_dir=LOG_DIR)

    def test_the_wait_is_a_directive_not_a_body_line(self) -> None:
        """A `#SBATCH` line after the first command is ignored by Slurm."""
        lines = render_sbatch(self._waiting(), log_dir=LOG_DIR).splitlines()
        dependency = next(i for i, line in enumerate(lines) if "--dependency" in line)
        first_body = next(i for i, line in enumerate(lines) if line == "set -u")
        assert dependency < first_body


class TestResumeSurface:
    """The package cannot resume for the payload -- only the payload knows
    what its checkpoint means -- so it surfaces the restart count and leaves
    the decision where the knowledge is.
    """

    def test_the_restart_count_is_exported_from_slurms_own_variable(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert 'export HPC3_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"' in script

    def test_it_defaults_to_zero_on_a_first_run(self) -> None:
        """The ':-0' is what makes a first run readable, not an unset variable
        under `set -u`."""
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert ":-0}" in script

    def test_the_restart_count_is_echoed_into_the_job_log(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert 'echo "restart   ${HPC3_RESTART_COUNT}"' in script

    def test_a_protected_run_carries_both_requeue_and_the_count(self) -> None:
        job = spec(minutes=600, requeue=True, checkpoint_steps=50)
        script = render_sbatch(job, log_dir=LOG_DIR)
        assert "#SBATCH --requeue" in script
        assert "HPC3_RESTART_COUNT" in script
        assert 'export HPC3_CHECKPOINT_STEPS="50"' in script


class TestCodeProvenance:
    """``git_commit`` was null in every artifact HPC3 has produced.

    The trainer prefers a build-stamped ``GIT_COMMIT`` because a deployed
    environment carries no ``.git``, and nothing set it -- so both MI arms of
    2026-08-25 uploaded 462 MB tarballs that cannot say which trainer built
    them. These assertions are about the emitted text, which is what the
    cluster acts on -- consistent with the rest of this module.
    """

    def test_the_commit_is_exported_from_a_stamp_inside_the_environment(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert (
            'export GIT_COMMIT="$(cat /pub/wagnera3/envs/abl-pinned/GIT_COMMIT '
            "2>/dev/null || echo '')\"" in script
        )

    def test_the_stamp_is_read_from_the_env_not_the_submitters_tree(self) -> None:
        """A submitter's HEAD moves on every edit; the env changes only when a
        wheel is installed. Recording the former as the latter is the
        lock-versus-manifest failure one layer down."""
        script = render_sbatch(spec(env_path="/pub/wagnera3/envs/other"), log_dir=LOG_DIR)
        assert "/pub/wagnera3/envs/other/GIT_COMMIT" in script
        assert "/pub/wagnera3/envs/abl-pinned/GIT_COMMIT" not in script

    def test_the_commit_is_echoed_into_the_job_log(self) -> None:
        script = render_sbatch(spec(), log_dir=LOG_DIR)
        assert 'echo "commit    ${GIT_COMMIT:-<unstamped>}"' in script

    def test_the_export_is_set_before_the_payload_runs(self) -> None:
        lines = render_sbatch(spec(), log_dir=LOG_DIR).splitlines()
        export = next(i for i, line in enumerate(lines) if line.startswith("export GIT_COMMIT="))
        payload = next(i for i, line in enumerate(lines) if line == "python train.py --seed 42")
        assert export < payload

    def test_the_guarded_substitution_cannot_abort_under_set_u(self) -> None:
        """Both halves of the guard are present, in the order that matters.

        ``2>/dev/null`` keeps a missing stamp from writing to the one stream
        a real failure arrives by, and ``|| echo ''`` supplies a value so the
        assignment succeeds -- the generated script runs under ``set -u`` and
        an unset expansion downstream would abort it.

        This asserts the emitted text, not a shell's behaviour on it. An
        execution test was written and removed: on this host, ``bash``
        resolves through a WSL interop shim in which command substitution
        returns empty for a file the same shell can ``cat`` and ``wc -c``
        successfully. Such a test could only be red forever or green for a
        reason unrelated to the artifact.
        """
        line = _code_provenance_export(spec())
        assert "2>/dev/null" in line
        assert line.index("2>/dev/null") < line.index("|| echo ''")
        assert line.endswith("')\"")
