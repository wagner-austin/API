"""Rendering a validated job spec into the batch script Slurm receives.

This module is where the contract's rules stop being assertions and become
the text that gets submitted. Two of them are visible in the output:

* ``--gres`` always carries the GPU model. There is no code path that emits a
  bare ``gpu:N``, because :class:`~hpc3.contracts.job.JobSpec` has no way to
  express one.
* ``--requeue`` appears exactly when the spec requested it, and the spec
  cannot request a long preemptible run without it.

The script body sets no error-handling policy of its own beyond ``set -u``.
It deliberately does NOT ``set -e``: the payload's exit status is what Slurm
records, and swallowing or transforming it here would hide a failed run behind
a successful job.
"""

from __future__ import annotations

from platform_core.determinism_env import (
    CUBLAS_DETERMINISTIC_WORKSPACE,
    CUBLAS_WORKSPACE_ENV_VAR,
    DETERMINISM_ENV_VAR,
    DETERMINISM_OFF,
    DETERMINISM_ON,
)

from hpc3.contracts.cluster import describe_gpu_request
from hpc3.contracts.dependency import dependency_argument
from hpc3.contracts.experiment import comment_fragment
from hpc3.contracts.job import JobSpec
from hpc3.contracts.layout import qualified_name

MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 1440


def format_walltime(minutes: int) -> str:
    """Render a duration in the ``sbatch --time`` format.

    Args:
        minutes: Duration in minutes. Always positive: the job contract
            rejects a non-positive wall clock before rendering.

    Returns:
        ``HH:MM:SS`` under a day, ``D-HH:MM:SS`` at a day or more. Slurm
        accepts both; the day form is used above 24 hours because ``72:00:00``
        and ``3-00:00:00`` are the same duration but only one is legible at a
        glance in ``squeue``.
    """
    days, remainder = divmod(minutes, MINUTES_PER_DAY)
    hours, mins = divmod(remainder, MINUTES_PER_HOUR)
    if days > 0:
        return f"{days}-{hours:02d}:{mins:02d}:00"
    return f"{hours:02d}:{mins:02d}:00"


def job_comment(spec: JobSpec) -> str:
    """Build the provenance string Slurm carries alongside the job.

    Kept to key=value pairs joined by semicolons, with no spaces: ``sbatch``
    takes ``--comment`` as a single token, and a space would silently truncate
    everything after it into a different argument.

    Args:
        spec: The spec being rendered.

    Returns:
        A compact provenance string naming the project, the hardware asked
        for, the environment the payload runs in, and what the run is -- so a
        row found in the queue answers "which experiment is this" without a
        ledger to hand. The experiment fragment is truncated if long.

        Readable through ``scontrol show job <id>`` and ``squeue -o %k``,
        and **only while the job is live**. It does NOT survive into
        accounting on HPC3: ``AccountingStoreFlags`` is unset there, so
        ``sacct -o Comment`` returns empty for every job, finished or not.
        Measured 2026-08-23, after this docstring had claimed otherwise.
        That is precisely why the ledger exists and why nothing in this
        package reads provenance back from the cluster.
    """
    return (
        f"project={spec['project']}"
        f";gpu={describe_gpu_request(spec['gpu'])}"
        f";cpus={spec['cpus']}"
        f";env={spec['env_path']}"
        f";det={'on' if spec['deterministic'] else 'off'}"
        f";exp={comment_fragment(spec['experiment'])}"
    )


def _determinism_exports(spec: JobSpec) -> list[str]:
    """Build the environment lines a deterministic run needs before it starts.

    Args:
        spec: The spec being rendered.

    Returns:
        The export lines. The posture variable is written either way, so the
        payload reads what was asked of it rather than inferring it from the
        presence of the cuBLAS variable -- absent is a state, not a message,
        and a payload that guessed would silently train the other record.

        Its name comes from :mod:`platform_core.determinism_env`, which is
        also where the trainer reads it. One definition, so the two cannot
        drift into a launcher that exports a variable nothing reads.
    """
    if not spec["deterministic"]:
        return [f'export {DETERMINISM_ENV_VAR}="{DETERMINISM_OFF}"']
    return [
        f'export {DETERMINISM_ENV_VAR}="{DETERMINISM_ON}"',
        f'export {CUBLAS_WORKSPACE_ENV_VAR}="{CUBLAS_DETERMINISTIC_WORKSPACE}"',
    ]


def _code_provenance_export(spec: JobSpec) -> str:
    """Build the line that tells the payload which code it is running.

    The trainer stamps ``GIT_COMMIT`` into every manifest it writes, and
    prefers this variable precisely because a deployed environment carries
    no ``.git`` for ``git rev-parse`` to answer from. Nothing set it, so
    ``git_commit`` was null in every artifact HPC3 has produced -- including
    both MI arms of 2026-08-25, whose 462 MB tarballs cannot say which
    trainer built them.

    The value is read from a file INSIDE the environment rather than passed
    down from the submitter. Those are routinely different commits: the
    submitter's working tree moves every time someone edits, while the env
    changes only when a wheel is installed into it. Recording the submitter's
    HEAD would be the lock-versus-manifest failure one layer down -- a record
    of intent presented as a record of fact.

    Args:
        spec: The spec being rendered.

    Returns:
        An export whose value is empty when the environment carries no stamp.
        Empty is correct there: the trainer reads an unset or empty variable
        as "not stamped" and records null, which is true, rather than
        inventing a commit that never built anything.
    """
    inner = f"$(cat {spec['env_path']}/GIT_COMMIT 2>/dev/null || echo '')"
    return f'export GIT_COMMIT="{inner}"'


def render_sbatch(spec: JobSpec, *, log_dir: str) -> str:
    """Render a validated job spec into a batch script.

    Args:
        spec: A spec that has already passed
            :func:`~hpc3.contracts.job.decode_job_spec`. Every rule the
            rendered script relies on was enforced there, so nothing is
            re-checked here.
        log_dir: Absolute directory on the cluster for stdout and stderr.

    Returns:
        The complete script text, LF-terminated. Written with LF regardless of
        the authoring platform: a CRLF shebang makes the cluster's kernel
        report the interpreter as missing, which reads as a broken image
        rather than a line-ending problem.
    """
    label = qualified_name(spec["project"], spec["name"])
    gpu = spec["gpu"]
    directives = [
        "#!/bin/bash -l",
        # The project prefix is what makes `squeue` legible on a cluster 102
        # other people share, and what tells us which body of work a row
        # belongs to when several are running at once.
        f"#SBATCH -J {label}",
        # Slurm surfaces this through `scontrol show job` and `squeue -o %k`
        # while the job is live, so a job carries its own provenance rather
        # than requiring whoever finds it to reconstruct what it was for. It
        # does not reach accounting -- see `job_comment` -- which is what the
        # ledger is for.
        f"#SBATCH --comment {job_comment(spec)}",
        f"#SBATCH -p {spec['partition']}",
        # No --gres line at all for a CPU-only job. An empty or zero-valued
        # one is not the same thing: `--gres=gpu:0` is a GPU request for none,
        # which Slurm may still route to a GPU partition's accounting.
        *([] if gpu is None else [f"#SBATCH --gres=gpu:{gpu['model']}:{gpu['count']}"]),
        f"#SBATCH -c {spec['cpus']}",
        f"#SBATCH --mem={spec['mem_gb']}G",
        f"#SBATCH -t {format_walltime(spec['minutes'])}",
        f"#SBATCH -o {log_dir}/{label}-%j.out",
        f"#SBATCH -e {log_dir}/{label}-%j.err",
    ]
    depends_on = spec["depends_on"]
    if depends_on is not None:
        directives.append(f"#SBATCH --dependency={dependency_argument(depends_on)}")
        # Never emitted without the line above, and never omitted when it is
        # present. Without it Slurm parks an unsatisfiable dependency in the
        # queue forever on `DependencyNeverSatisfied` -- 261 of 621 pending
        # GPU jobs on HPC3 in one sample -- where it holds a QOS slot and
        # looks exactly like a job that is merely waiting its turn.
        directives.append("#SBATCH --kill-on-invalid-dep=yes")

    if spec["requeue"]:
        directives.append("#SBATCH --requeue")

    body = [
        "",
        "# Undefined variables are a bug, not a default. No `set -e`: the",
        "# payload's exit status is what Slurm records, and this wrapper must",
        "# not convert a failed run into a successful job.",
        "set -u",
        "",
        f'export HPC3_PROJECT="{spec["project"]}"',
        f'export HPC3_JOB_NAME="{label}"',
        f'export HPC3_CHECKPOINT_STEPS="{spec["checkpoint_steps"]}"',
        # Determinism is declared here and applied by the payload, because the
        # switch that matters is a torch call this submitter cannot make. What
        # the submitter CAN do is guarantee the half that must precede the
        # process: cuBLAS reads its workspace variable once, when the handle is
        # created, so setting it after CUDA has started is accepted in silence
        # and does nothing. Exported here it cannot be too late, and it cannot
        # be forgotten by a payload that only remembers the torch half --
        # which fails loudly, because deterministic mode raises when the
        # variable is absent.
        *_determinism_exports(spec),
        # Slurm increments SLURM_RESTART_COUNT each time it requeues a job,
        # so a preempted run re-enters here with a non-zero value. This
        # package cannot resume on the payload's behalf -- only the payload
        # knows what its checkpoint means -- so it surfaces the count and
        # leaves the decision where the knowledge is.
        'export HPC3_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"',
        _code_provenance_export(spec),
        f'export PATH="{spec["env_path"]}/bin:$PATH"',
        "",
        'echo "host      $(hostname)"',
        'echo "job       ${SLURM_JOB_ID:-none}"',
        'echo "restart   ${HPC3_RESTART_COUNT}"',
        # Echoed as well as exported: the manifest inside a tarball is the
        # durable record, but a log line answers "what did THIS job run"
        # without unpacking 462 MB, and it is visible while the job is live.
        'echo "commit    ${GIT_COMMIT:-<unstamped>}"',
        # A CPU node has no nvidia-smi, and calling it there writes a
        # command-not-found to stderr on every run -- which trains whoever
        # reads these logs to ignore stderr, on the one stream a real failure
        # arrives by.
        *(
            ['echo "gpu       cpu-only"']
            if gpu is None
            else [
                'echo "gpu       ' + gpu["model"] + '"',
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            ]
        ),
        "date -Is",
        "",
        spec["command"],
        "rc=$?",
        "",
        "date -Is",
        'echo "exit      $rc"',
        "exit $rc",
    ]
    return "\n".join([*directives, *body]) + "\n"


__all__ = [
    "MINUTES_PER_DAY",
    "MINUTES_PER_HOUR",
    "format_walltime",
    "job_comment",
    "render_sbatch",
]
