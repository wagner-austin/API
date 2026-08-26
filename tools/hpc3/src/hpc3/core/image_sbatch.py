"""Rendering the Slurm job that runs an image build on a compute node.

``build.sh`` says how to build; this says where and under what. It was hand
written four times -- once per image version -- and each copy carried the
same thirty lines of module loading, cache exports and connectivity probing.
Two defects entered through that copying and both shipped:

1. The behavioural check wrote its output to a HOST path with no bind
   declared, so a correct image failed its own build job with
   ``OSError: Read-only file system``.
2. That failure printed "the guard is not in this image" -- a cause the job
   had not established and which the traceback directly contradicted. A
   check that names the wrong cause is worse than one that says only that it
   failed, because the reader believes it.

Both are fixed here once. Smoke commands run with no binds and are
documented to write inside the container; the failure message states what
was run and nothing more.

WHY THE RESOURCES ARE CONSTANTS. They describe building, not the image, so
they belong to this renderer rather than to a spec that documents contents.
A build is CPU-only -- ``apptainer build`` has no use for a GPU, and asking
for one queues behind jobs that do.
"""

from __future__ import annotations

import shlex

from hpc3.core.image_exec import APPTAINER_MODULE
from hpc3.core.image_layout import SELFCHECK_NAME, SPEC_DIR

BUILD_PARTITION = "free"
"""Preemptible and unbilled. A build is restartable, so it is the right pool."""

BUILD_CPUS = 8
BUILD_MEM_GB = 32

BUILD_TIME_LIMIT = "02:00:00"
"""Twice the ~25 minutes a torch+CUDA build takes, so a slow mirror does not
kill a build that was working."""

CACHE_DIR = "/pub/wagnera3/apptainer-cache"
TMP_DIR = "/pub/wagnera3/apptainer-tmp"
"""Apptainer's own defaults live under ``$HOME``, a 50 GB volume on HPC3 that
a multi-gigabyte build fills -- failing as though the build were broken
rather than the disk."""


def _preamble(job_name: str, image_dir: str) -> list[str]:
    """Render the SBATCH directives and the shell prologue."""
    return [
        "#!/bin/bash -l",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -p {BUILD_PARTITION}",
        f"#SBATCH -c {BUILD_CPUS}",
        f"#SBATCH --mem={BUILD_MEM_GB}G",
        f"#SBATCH -t {BUILD_TIME_LIMIT}",
        f"#SBATCH -o {image_dir}/build-%j.out",
        f"#SBATCH -e {image_dir}/build-%j.err",
        "# The free partition is preemptible and this build takes ~25 minutes,",
        "# so preemption is the expected case rather than the exceptional one:",
        "# an earlier attempt died at 26 seconds without this. `apptainer build",
        "# --force` overwrites, so a restart is a clean rebuild rather than a",
        "# resume onto a half-written sif.",
        "#SBATCH --requeue",
        "",
        'echo "host      $(hostname)"',
        'echo "job       ${SLURM_JOB_ID:-none}"',
        "date -Is",
        "",
        "# RCIC's module init is NOT nounset-clean -- it reads USERMODULEPATH",
        "# unguarded, so sourcing it under `set -u` aborts the job in under a",
        "# second. Load modules first, then be strict for our own code.",
        "source /etc/profile.d/rcic-modules.sh",
        f"module load {APPTAINER_MODULE}",
        'echo "apptainer $(command -v apptainer)"',
        "",
        f"export APPTAINER_CACHEDIR={CACHE_DIR}",
        f"export APPTAINER_TMPDIR={TMP_DIR}",
        "",
    ]


def _connectivity_probe() -> list[str]:
    """Render the check that this node can reach the package index.

    A compute node without outbound access fails deep inside pip with a
    resolver error, minutes in. Asking first turns that into one line.
    """
    return [
        'echo "--- connectivity probe ---"',
        'if curl -sS -m 20 -o /dev/null -w "pypi %{http_code}\\n" https://pypi.org/simple/; then',
        '    echo "outbound OK"',
        "else",
        '    echo "NO OUTBOUND ACCESS FROM THIS COMPUTE NODE" >&2',
        "    exit 3",
        "fi",
        "",
    ]


def _reverify(image_name: str) -> list[str]:
    """Render the self-check re-run as the unprivileged user.

    ``%post`` runs the self-check as root during the build. That does not
    establish that the user who later runs the image can execute it: a file
    staged mode 640 lands root-owned and unreadable, which happened, and the
    environment imported fine while its own verification could not be run.
    """
    return [
        "if [ $rc -eq 0 ]; then",
        '    echo "--- re-verifying the built image as $(id -un) ---"',
        f"    apptainer exec {image_name} {{env}}/bin/python {SPEC_DIR}/{SELFCHECK_NAME}",
        "    rc=$?",
        '    echo "re-verify exit $rc"',
        "fi",
        "",
    ]


def _smoke_checks(image_name: str, commands: list[str]) -> list[str]:
    """Render the behavioural checks, each run inside the built image.

    No binds are passed. At build time the image is the only thing that
    exists -- there is no job and nothing declaring what to mount -- so a
    command that writes must write inside the container.

    The failure message names the command and says it failed. It deliberately
    does not diagnose: the previous hand-written version guessed a cause,
    printed it as fact, and was wrong.

    THE COMMAND APPEARS TWICE, AND ONLY ONE OF THEM MAY BE RAW. On the ``if``
    line it is a command and the shell must parse its own quoting; in the two
    ``echo`` lines it is TEXT, and interpolating it raw dropped a command's
    quotes inside a quoted string. Two measured consequences, neither of
    which a substring assertion on the rendered text can see:

    * The announcement lies about what ran. Rendered raw,
      ``echo "--- smoke 1/1: python -c "assert L == 'gpt2-tiny'" ---"`` is
      valid shell -- bash concatenates the adjacent quoted runs -- and prints
      ``python -c assert L == gpt2-tiny``. The quotes are gone, so the line
      naming the failing command is not the command.
    * A command carrying a shell metacharacter outside its own quotes breaks
      the script outright. ``bash -n`` on the raw rendering of one containing
      ``$(...)`` fails with "syntax error near unexpected token".

    So the echoes shell-quote it and the ``if`` line does not.
    """
    if not commands:
        return [
            "# The spec declares no smoke commands. The self-check above is",
            "# the only assertion this image makes about itself.",
            "",
        ]
    lines: list[str] = []
    for index, command in enumerate(commands, start=1):
        announce = shlex.quote(f"--- smoke {index}/{len(commands)}: {command} ---")
        failure = shlex.quote(f"SMOKE COMMAND {index} FAILED: {command}")
        lines.extend(
            [
                "if [ $rc -eq 0 ]; then",
                f"    echo {announce}",
                f"    if apptainer exec {image_name} {command}; then",
                f'        echo "smoke {index} OK"',
                "    else",
                f"        echo {failure} >&2",
                '        echo "the image built and self-checked; this command '
                'did not succeed inside it" >&2',
                "        rc=5",
                "    fi",
                "fi",
                "",
            ]
        )
    return lines


def render_build_sbatch(
    *, image_name: str, job_name: str, image_dir: str, env_prefix: str, smoke_commands: list[str]
) -> str:
    """Render the Slurm batch script that builds and verifies an image.

    Args:
        image_name: Filename of the produced ``.sif``.
        job_name: Slurm job name, which is what ``squeue`` shows.
        image_dir: Absolute cluster directory holding the rendered build
            inputs, where logs and the image are written.
        env_prefix: The image's environment prefix, for the interpreter that
            runs the self-check.
        smoke_commands: Commands to run inside the built image, each of which
            must exit 0.

    Returns:
        Shell source, LF-terminated.
    """
    lines = _preamble(job_name, image_dir)
    lines.extend(_connectivity_probe())
    lines.extend(
        [
            f"cd {image_dir}",
            "bash build.sh",
            "rc=$?",
            "",
        ]
    )
    lines.extend(line.replace("{env}", env_prefix) for line in _reverify(image_name))
    lines.extend(_smoke_checks(image_name, smoke_commands))
    lines.extend(
        [
            "date -Is",
            'echo "exit      $rc"',
            "exit $rc",
        ]
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "BUILD_CPUS",
    "BUILD_MEM_GB",
    "BUILD_PARTITION",
    "BUILD_TIME_LIMIT",
    "CACHE_DIR",
    "TMP_DIR",
    "render_build_sbatch",
]
