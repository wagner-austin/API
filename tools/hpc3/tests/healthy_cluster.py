"""What a healthy cluster, environment and project repository answer.

Split out of ``conftest.py`` when it passed the 600-line ceiling (board task
7aab32ef added the image's commit and the repository's HEAD to what every
preflight asks). These are the scripted answers, not the fixtures: the
measured probe outputs a submission's preflight reads, and the two helpers
that install them on a :class:`~tests.conftest.FakeRun`, so a test about the
scheduler does not fail on an environment it never meant to be about.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import JSONValue

from hpc3.contracts.drift import ModuleDrift
from tests.conftest import FakeRun

PREFLIGHT_LINE = (
    "sbatch: Job 1 to start at 2026-08-22T03:23:00 a using 4 processors "
    "on nodes hpc3-gpu-16-02 in partition free-gpu"
)

ABL_PINNED_DISTRIBUTIONS = (
    "torch==2.6.0+cu124\ntransformers==4.46.3\nnumpy==2.1.3\ntyping_extensions==4.12.2\n"
)
"""What ``/pub/wagnera3/envs/abl-pinned`` reports, as measured on the cluster.

The ablation's arms were produced against exactly these versions, so a probe
answering anything else means the run would not be comparable to them. The
``typing_extensions`` entry is here deliberately: it is the underscore spelling
the distribution actually reports, and normalisation has to survive it.
"""

IMAGE_INTERPRETER = "3.11\n/usr/local\nTrue\n"
"""What ``/opt/env``'s interpreter says about itself inside a registered image.

Measured 2026-10-05 inside all eight registered images: ``/opt/env`` is a venv
over the image's own ``/usr/local`` Python, whose base sits on the container's
root filesystem. These are the identity lines every probe now opens with.
"""

IMAGE_PROBE = IMAGE_INTERPRETER + ABL_PINNED_DISTRIBUTIONS
"""The whole probe answer of a healthy imaged ``abl`` environment."""


def host_interpreter(base_prefix: str) -> str:
    """Build the identity lines a host environment's interpreter prints.

    Args:
        base_prefix: The ``sys.base_prefix`` it reports -- the environment's
            own path when it owns its interpreter, another one when borrowed.

    Returns:
        The three identity lines. A host environment's base is on BeeGFS,
        never on ``/``'s device, so the third line is always ``False``.
    """
    return f"3.11\n{base_prefix}\nFalse\n"


HOST_ENV = "/pub/envs/abl-pinned"
"""The host environment the hand-built job specs in these tests name."""

IMAGE_COMMIT = "fab1d991f4d6f39e72c3b21ca5757a072c01f277"
"""The commit the tests' image says built it: v23's, read 2026-10-16."""

HEAD_COMMIT = "51b39a57b0000000000000000000000000000000"
"""The HEAD the tests' project repository is at when a run is compared."""

PROJECT_REPO = pathlib.Path("/work/repos/abl")
"""The project repository the core-level tests hand a submission.

Never touched on disk: every git question about it is answered by the
scripted runner, as :func:`script_healthy_environment` does.
"""


def script_healthy_environment(fake: FakeRun) -> None:
    """Script the tests' environments as present and answering the probe.

    Preflight asks every environment two questions -- does it exist, and what
    is its interpreter and what does it contain -- whether or not the project
    pins anything. Scripting both here keeps a test about the scheduler from
    failing on an environment it never meant to be about.

    Each environment answers for ITSELF, matched by the interpreter path in
    the probe, because the two shapes differ the way the cluster's do: the
    host environment :data:`HOST_ENV` is its own base, and an image's
    ``/opt/env`` is a venv over the image's ``/usr/local``.

    An image also says which commit built it, and the project's repository
    holds that commit and names its HEAD: :mod:`hpc3.core.drift` asks both.
    A command starting no ``python -m`` module is all these tests submit, so
    the stamp line is the whole of the entrypoint probe's answer, and the
    repository tracks nothing, so no source is compared. A test about a
    module scripts the probe itself, BEFORE calling this, since the first
    matching rule answers.

    Args:
        fake: The runner to script.
    """
    fake.add("ENTRYPOINT-COMMIT|%s", stdout=f"ENTRYPOINT-COMMIT|0|{IMAGE_COMMIT}\n")
    fake.add("HEAD^{commit}", stdout=f"{HEAD_COMMIT}\n")
    fake.add("test -d", stdout="PRESENT\n")
    fake.add(f"{HOST_ENV}/bin/python", stdout=host_interpreter(HOST_ENV) + ABL_PINNED_DISTRIBUTIONS)
    fake.add("/opt/env/bin/python", stdout=IMAGE_PROBE)


IMAGE: JSONValue = {
    "path": "/pub/wagnera3/images/v23/abl.sif",
    "sha256": "3bd2e694857821c383c5e90b1c23b63c706ece20069a6bf34431512ef1c041d4",
    "binds": ["/pub/wagnera3"],
}
"""The image a document names for a run whose environment is ``/opt/env``."""

TRAINER_SOURCE = "services/Model-Trainer/src/model_trainer"
"""Where the tests' project repository keeps ``model_trainer``, as the real one does."""

TRAINER_CHANGE = f"{TRAINER_SOURCE}/cli/known_answer_probe.py"
"""The one file of it that changed between :data:`IMAGE_COMMIT` and :data:`HEAD_COMMIT`."""


def script_moved_module(fake: FakeRun, modules: list[str]) -> None:
    """Script an image that starts every module, whose source moved since.

    The image says it was built from :data:`IMAGE_COMMIT` and resolves each
    module inside its own environment; the repository at :data:`HEAD_COMMIT`
    keeps ``model_trainer`` at :data:`TRAINER_SOURCE`, and git names
    :data:`TRAINER_CHANGE` as changed between the two. Call BEFORE
    :func:`script_healthy_environment`, whose stamp-only answer would
    otherwise match the probe first.

    A git command's last argument is what the runner matches: ``ls-tree``
    ends with the HEAD it lists, ``diff`` with the path it compares.

    Args:
        fake: The runner to script.
        modules: The ``model_trainer`` modules the run starts, in the order
            the probe asks about them.
    """
    verdicts = "".join(
        f"ENTRYPOINT|PRESENT|ENV|{module}|/opt/env/lib/python3.11/site-packages/model_trainer\n"
        for module in modules
    )
    fake.add("ENTRYPOINT-COMMIT|%s", stdout=f"ENTRYPOINT-COMMIT|0|{IMAGE_COMMIT}\n{verdicts}")
    fake.add(HEAD_COMMIT, stdout=f"README.md\0{TRAINER_SOURCE}/__init__.py\0{TRAINER_CHANGE}\0")
    fake.add(TRAINER_SOURCE, stdout=f"{TRAINER_CHANGE}\0")


def moved_module_drift(directory: str, module: str) -> ModuleDrift:
    """Build what :func:`script_moved_module` makes one module's comparison.

    Args:
        directory: The directory the run starts it from.
        module: The module.

    Returns:
        Its record: :data:`TRAINER_SOURCE` compared, :data:`TRAINER_CHANGE` changed.
    """
    return ModuleDrift(
        directory=directory,
        module=module,
        compared=[TRAINER_SOURCE],
        changed=[TRAINER_CHANGE],
        not_compared="",
    )


def script_healthy_cluster(fake: FakeRun, *, job_id: str = "55519937") -> None:
    """Script a cluster that admits and accepts everything.

    Submission now preflights unconditionally, so every submit test needs the
    environment probe and the ``--test-only`` verdict scripted, not only the
    ``sbatch``. Centralised here so that adding a step to the submit path
    updates every test at once rather than one failing test at a time.

    Args:
        fake: The runner to script.
        job_id: Id the real submission should report.
    """
    script_healthy_environment(fake)
    fake.add("--test-only", stdout=PREFLIGHT_LINE + "\nrc=0\n")
    fake.add("sbatch", stdout=f"Submitted batch job {job_id}\n")


__all__ = [
    "ABL_PINNED_DISTRIBUTIONS",
    "HEAD_COMMIT",
    "HOST_ENV",
    "IMAGE",
    "IMAGE_COMMIT",
    "IMAGE_INTERPRETER",
    "IMAGE_PROBE",
    "PREFLIGHT_LINE",
    "PROJECT_REPO",
    "TRAINER_CHANGE",
    "TRAINER_SOURCE",
    "host_interpreter",
    "moved_module_drift",
    "script_healthy_cluster",
    "script_healthy_environment",
    "script_moved_module",
]
