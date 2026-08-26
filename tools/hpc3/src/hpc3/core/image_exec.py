"""Running one command inside a job's image rather than on the host.

Two different callers need this and they need different SHAPES of it. The
batch script wraps its payload across several continued lines, because a
script is read by people and a single 400-character line is not. A preflight
probe needs the same wrapping as ONE line it can hand to
:func:`~hpc3.core.remote.run_remote`.

What they must not differ on is the module, the binds and the image, because
preflight exists to answer a question about the environment the job will
actually use. This module owns those parts so neither caller builds them
itself.

WHY THIS EXISTS AT ALL. Preflight verified ``env_path`` on the host with no
knowledge of ``image``. For a project that adopted an image, ``env_path`` is
``/opt/env`` -- a path that exists only INSIDE the ``.sif`` and nowhere on
the cluster filesystem. So every image-based job failed preflight with
``ENV_PATH_MISSING`` naming a directory that was never supposed to be there,
and the message said "does not exist on hpc3", which is true and useless: it
sends the reader to look on the host, which is the one place that cannot
answer.
"""

from __future__ import annotations

import shlex

from hpc3.contracts.cluster import GpuRequest
from hpc3.contracts.image import ImageReference

APPTAINER_MODULE = "apptainer/1.4.5"
"""The module that puts ``apptainer`` on PATH.

``which apptainer`` returns nothing on an HPC3 login node until this is
loaded, so any command that reaches for it without loading first fails with
"command not found" on a cluster that has apptainer installed.
"""


def bind_arguments(image: ImageReference) -> list[str]:
    """Render the image's binds as ``--bind`` arguments.

    Each directory is mounted at its own path, so a payload's absolute paths
    mean the same thing inside the container and out. Without them a job
    starts cleanly and then finds nothing: ``/pub`` on HPC3 is a symlink to
    ``/dfs6b/pub``, and apptainer carries the BeeGFS mounts but not the
    symlink.

    Args:
        image: The image whose binds are rendered.

    Returns:
        One ``--bind "path:path"`` argument per declared bind, in declaration
        order. Empty when the image declares none.
    """
    return [f'--bind "{path}:{path}"' for path in image["binds"]]


NVIDIA_FLAG = "--nv"
"""What binds the host's NVIDIA driver and devices into the container.

A container ships CUDA libraries; it does NOT ship the driver, which belongs
to the kernel on the host. Without this flag the payload sees no driver at
all, and torch reports it in a way that reads as a broken node rather than a
missing flag:

    RuntimeError: Found no NVIDIA driver on your system.

Measured 2026-08-25 on hpc3-gpu-l54-05, which had an A100 80GB PCIe attached
and printed it in the job's own prologue -- the prologue runs on the host, so
the log showed a working GPU one line above the traceback saying there was
none. Directly measured both ways on a free-gpu A100:

    apptainer exec      ... torch.cuda.is_available() -> False
    apptainer exec --nv ... torch.cuda.is_available() -> True, A100 80GB PCIe
"""


def gpu_arguments(gpu: GpuRequest | None) -> list[str]:
    """Render the arguments a GPU job needs to reach its card.

    Keyed on whether the job ASKED for a GPU rather than on whether the host
    has one. A CPU job does not need the driver bound, and ``--nv`` on a node
    without one is an error rather than a no-op -- which is also why the
    preflight probes, which run on a GPU-less login node, deliberately do not
    use this.

    Args:
        gpu: The job's GPU request, or None for CPU-only work.

    Returns:
        ``["--nv"]`` for a GPU job, empty otherwise.
    """
    if gpu is None:
        return []
    return [NVIDIA_FLAG]


def run_inside_image(image: ImageReference, command: str) -> str:
    """Rewrite a host command line so it runs inside the image.

    The command is handed to ``sh -c`` inside the container as a single
    quoted argument, so a probe written for a host shell -- pipes, ``&&``,
    quoting and all -- keeps its meaning without being rewritten for
    apptainer's argv form.

    The module load is chained with ``&&`` rather than sequenced with ``;``
    so that a cluster where the module is unavailable fails as
    ``REMOTE_COMMAND_FAILED`` carrying the module's own complaint. Sequencing
    would run ``apptainer`` anyway, fail with "command not found", and let a
    probe that reports absence by printing a token report the environment
    missing -- diagnosing the image instead of the module.

    Args:
        image: The image to run inside.
        command: The command line as it would be run on the host.

    Returns:
        One line suitable for :func:`~hpc3.core.remote.run_remote`.
    """
    parts = [
        f"module load {APPTAINER_MODULE}",
        "&&",
        "apptainer exec",
        *bind_arguments(image),
        f'"{image["path"]}"',
        "sh",
        "-c",
        shlex.quote(command),
    ]
    return " ".join(parts)


def describe_location(image: ImageReference | None, host: str) -> str:
    """Name the filesystem a check just looked at.

    A message that says "on hpc3" when the check ran inside a container
    sends the reader to the one place that cannot answer, which is how
    ``ENV_PATH_MISSING`` on an image job reads as a broken path rather than
    as a preflight looking in the wrong filesystem.

    Args:
        image: The image the check ran inside, or None for a host check.
        host: SSH destination, used when there is no image.

    Returns:
        A phrase naming where the check looked.
    """
    if image is None:
        return f"on {host}"
    return f"inside {image['path']}"


__all__ = [
    "APPTAINER_MODULE",
    "NVIDIA_FLAG",
    "bind_arguments",
    "describe_location",
    "gpu_arguments",
    "run_inside_image",
]
