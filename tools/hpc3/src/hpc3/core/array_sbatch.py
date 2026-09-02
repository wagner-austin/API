"""Rendering a whole sweep into the ONE batch script a job array submits.

The single-job renderer (:mod:`hpc3.core.sbatch`) emits one script per job,
and the sweep loop paid for that shape in SSH round trips: three per member,
~13 seconds each, measured while the cluster scheduled everything instantly
(rusted ab48, 2026-09-01). An array is the same members behind one call.

Two design rules, both downstream of "the script on disk is the record of
exactly what ran":

* **The script always carries the FULL member table**, one ``case`` arm per
  document position, whatever subset is being submitted. Which tasks run is
  the submitter's ``--array`` argument; what each task IS never moves. That
  is what lets a campaign resubmit the sparse gap -- ``--array=3,17-19`` --
  against a byte-identical script, so the task-to-member mapping cannot
  drift between convergence passes.
* **No ``--array`` directive in the script**, for the same reason: a
  directive would bind the record of what the members are to one submission's
  choice of which to run. The submitter passes it explicitly, always.

The renderer takes the sweep document itself rather than expanded job specs.
That is not convenience: a sweep's members share the template by
construction, so handing this function anything else would reintroduce, as a
runtime check, a divergence the type system already makes unrepresentable.

Identity inside a task: ``HPC3_JOB_NAME`` is exported per case arm with the
member's own qualified name, so the payload sees exactly what it would have
seen as a single job. The Slurm-visible job name and ``--comment`` are shared
across the array -- one script, one directive block -- which is the one
legibility cost of the shape: ``squeue`` rows differ only in the
``_<index>`` suffix of their id, and the ledger, which records every task id
against its member name, is where per-member identity durably lives.
"""

from __future__ import annotations

from hpc3.contracts.dependency import dependency_argument
from hpc3.contracts.job import JobSpec
from hpc3.contracts.layout import qualified_name
from hpc3.contracts.sweep import SweepSpec, expand_sweep
from hpc3.core.sbatch import (
    code_provenance_export,
    determinism_exports,
    format_walltime,
    image_digest_export,
    job_comment,
    payload_lines,
    runtime_module_lines,
)

#: Exit status of a task whose index names no member. Distinct from every
#: payload status so accounting shows the dispatch failed, not the match.
NO_SUCH_MEMBER_EXIT = 64


def _member_case_arm(index: int, member: JobSpec) -> list[str]:
    """Render one member's dispatch arm.

    Args:
        index: The member's document position, which is its array index.
        member: The member's expanded spec.

    Returns:
        The ``case`` arm lines, indented for the dispatch block.
    """
    label = qualified_name(member["project"], member["name"])
    lines = [
        f"{index})",
        f'    export HPC3_JOB_NAME="{label}"',
        f'    echo "member    {label}"',
    ]
    lines.extend(f"    {line}" for line in payload_lines(member))
    lines.append("    rc=$?")
    lines.append("    ;;")
    return lines


def render_array_sbatch(spec: SweepSpec, *, log_dir: str, charge_account: str) -> str:
    """Render a validated sweep into the one script its array submits.

    Args:
        spec: A sweep already validated by
            :func:`~hpc3.contracts.sweep.decode_sweep_spec`; every rule the
            rendered script relies on was enforced there.
        log_dir: Absolute cluster directory for the tasks' output. Each task
            logs to its own pair of files via ``%A_%a`` -- the array's id and
            the task's index -- because forty-eight tasks interleaved into
            one file is a log nobody can read.
        charge_account: The Slurm account to bill, or empty for none, with
            the same emit-only-when-non-empty rule the single-job renderer
            documents.

    Returns:
        The complete script text, LF-terminated. Deliberately WITHOUT any
        ``#SBATCH --array`` directive -- see the module docstring.
    """
    base = spec["base"]
    label = qualified_name(base["project"], base["name"])
    gpu = base["gpu"]
    directives = [
        "#!/bin/bash -l",
        f"#SBATCH -J {label}",
        f"#SBATCH --comment {job_comment(base)}",
        f"#SBATCH -p {base['partition']}",
        *([] if charge_account == "" else [f"#SBATCH --account={charge_account}"]),
        *([] if gpu is None else [f"#SBATCH --gres=gpu:{gpu['model']}:{gpu['count']}"]),
        f"#SBATCH -c {base['cpus']}",
        f"#SBATCH --mem={base['mem_gb']}G",
        f"#SBATCH -t {format_walltime(base['minutes'])}",
        # %A is the array's own id and %a the task index -- the same pair the
        # ledger records -- so a log file is findable from its ledger row and
        # vice versa without a mapping step.
        f"#SBATCH -o {log_dir}/{label}-%A_%a.out",
        f"#SBATCH -e {log_dir}/{label}-%A_%a.err",
    ]
    depends_on = base["depends_on"]
    if depends_on is not None:
        directives.append(f"#SBATCH --dependency={dependency_argument(depends_on)}")
        # Paired exactly as the single-job renderer pairs it: without the
        # kill, an unsatisfiable dependency parks EVERY task of the array on
        # DependencyNeverSatisfied, each one holding a QOS slot.
        directives.append("#SBATCH --kill-on-invalid-dep=yes")
    if base["requeue"]:
        directives.append("#SBATCH --requeue")

    members = expand_sweep(spec)
    dispatch: list[str] = ['case "${SLURM_ARRAY_TASK_ID}" in']
    for index, member in enumerate(members):
        dispatch.extend(_member_case_arm(index, member))
    dispatch.extend(
        [
            "*)",
            # A refusal, not a fallback: an index outside the table means the
            # submitted --array disagrees with the script, and running
            # nothing loudly beats running the wrong member quietly.
            '    echo "no member at array index ${SLURM_ARRAY_TASK_ID}" >&2',
            f"    exit {NO_SUCH_MEMBER_EXIT}",
            "    ;;",
            "esac",
        ]
    )

    body = [
        "",
        "# Undefined variables are a bug, not a default. No `set -e`: the",
        "# payload's exit status is what Slurm records, and this wrapper must",
        "# not convert a failed run into a successful job.",
        "set -u",
        "",
        f'export HPC3_PROJECT="{base["project"]}"',
        f'export HPC3_CHECKPOINT_STEPS="{base["checkpoint_steps"]}"',
        *determinism_exports(base),
        'export HPC3_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"',
        *runtime_module_lines(base),
        *image_digest_export(base),
        code_provenance_export(base),
        "",
        'echo "host      $(hostname)"',
        'echo "job       ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none}"',
        'echo "restart   ${HPC3_RESTART_COUNT}"',
        'echo "commit    ${GIT_COMMIT:-<unstamped>}"',
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
        *dispatch,
        "",
        "date -Is",
        'echo "exit      $rc"',
        "exit $rc",
    ]
    return "\n".join([*directives, *body]) + "\n"


__all__ = [
    "NO_SUCH_MEMBER_EXIT",
    "render_array_sbatch",
]
