"""One project's declared resources, and the rules a declaration must satisfy.

Split from :mod:`hpc3.contracts.workspace` when that module passed the
600-line ceiling. The seam is by role rather than by size: a workspace says
WHERE the cluster is and holds the registry; this module says what ONE
registered body of work asks of it, and carries the rule that a registered
project is an imaged one.
"""

from __future__ import annotations

import pathlib

from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError, JSONValue, require_bool, require_int
from typing_extensions import TypedDict

from hpc3.contracts.budget import Budget, decode_budget, encode_budget
from hpc3.contracts.cluster import (
    ClusterFacts,
    GpuRequest,
    decode_gpu_request,
    encode_gpu_request,
    require_partition,
)
from hpc3.contracts.fields import require_nonempty_str, require_positive
from hpc3.contracts.image import (
    ImageReference,
    decode_image_reference,
    encode_image_reference,
)
from hpc3.contracts.pins import encode_pinned_packages, require_pinned_packages


class ProjectConfig(TypedDict):
    """The resource settings every job in one project starts from.

    Every field is a default, and every one may be overridden by an individual
    run -- but the override goes through the same validation a full spec does,
    so overriding cannot reach a state authoring a spec by hand could not.

    Attributes:
        partition: Partition this project's work goes to.
        gpu: GPUs to pin per job, or None for CPU-only work. Never generic
            when present; see :mod:`hpc3.contracts.job` rule 1.
        cpus: CPU cores per job.
        mem_gb: Host memory per job, in GiB.
        minutes: Wall-clock limit per job.
        requeue: Whether Slurm should resubmit after a preemption.
        checkpoint_steps: Training steps between checkpoints; 0 means none.
        image: Image this project's payloads run inside. REQUIRED, and not
            optional in this type: a project without one cannot be decoded,
            so no reader downstream needs a branch for its absence. This is
            where a project adopts an image: one field, and every run and
            sweep it declares inherits it. See :mod:`hpc3.contracts.image`
            and :func:`_require_project_image` for why the CPU exemption was
            removed. A JOB may still run imageless -- that is how a one-off
            probe works, and :class:`~hpc3.contracts.job.JobSpec` keeps the
            field optional for it. A standing body of work may not.
        env_path: Absolute path to this project's Python environment. Always
            a path INSIDE the image, since every project has one.
        pinned_packages: Distribution versions the environment must actually
            contain, keyed by name. Required as a field, may be empty: a
            project whose payload is a compiled binary has no Python packages
            to pin, and declaring ``{}`` says so deliberately rather than
            leaving the question unasked. When non-empty, preflight asks the
            environment itself instead of trusting its path.
        deterministic: Whether this project's runs are configured for
            kernel-level numerical determinism. Required and explicit,
            because it partitions results rather than improving them:
            measured on this stack, two same-seed runs of a 6-layer model
            diverge by the sixth significant figure of the loss without the
            controls, and the deterministic value is a *different* number
            from the nondeterministic one. Runs on either side of that
            setting form separate records and comparing across them
            reintroduces the confound the controls remove -- so the setting
            has to be part of what a run IS, not a flag someone remembers.
        budget: This project's own caps and charge account. Per project
            rather than per workspace, which is where it lived until
            2026-08-28.

            THE WORKSPACE-LEVEL FIELD DESCRIBED ITSELF AS "shared by every
            project. One pool, because the machine is one machine", AND THE
            PRACTICE HAD ALREADY LEFT IT. Three workspace documents were
            committed, one per project, declaring 0.5, 12.0 and 1.0
            GPU-hours over the same ledger file -- because a cap is the one
            thing that genuinely differs per body of work, and a
            workspace-level field left no way to say so but to fork the
            document. Nothing detected the fork, and two statements in this
            package were false while it stood: the sentence above, and
            ``hpc3-watch``'s claim that "the ceiling this command enforces
            is the same one the submitting command projected against" --
            watching an ``mi`` job with the cleargbm document enforced 0.5
            GPU-hours against work submitted under a declared 12.0.

            ``charge_account`` moves with the caps rather than staying
            behind, and that is the sharpest half. Accounts are per-PI, and
            a job charged to the wrong one spends another lab's allocation
            on work that is not theirs; a single workspace account would
            have made that a one-character mistake.

            Deliberately NOT in :data:`PROJECT_FIELDS`: a run may override
            every resource above, and may override none of this. A cap a run
            can raise is not a cap.
        repo: Where this project's code lives on the workstation, resolved
            against the workspace document's own directory exactly as
            ``ledger`` is, so the declaration stays portable.

            THIS IS THE FIELD THAT MAKES THE TABLE AN INDEX RATHER THAN A
            RESOURCE FILE. Everything above says how a project runs on the
            cluster; nothing said what it IS. That was survivable only while
            every declared project happened to be the same repository -- and
            all three were, so the mapping was implicit and nobody noticed
            it was missing.

            It stopped being survivable the moment the question became
            "what research exists here". A session that opens this workspace
            can now answer it; before, an audit of this very machine found
            four research surfaces and missed two, because ninety directories
            under ``~/PROJECTS`` and no list is not a question anyone can
            answer by reading.

            Also NOT in :data:`PROJECT_FIELDS`. A run may vary what it asks
            of the cluster; it may not relocate the project.
    """

    partition: str
    gpu: GpuRequest | None
    cpus: int
    mem_gb: int
    minutes: int
    requeue: bool
    checkpoint_steps: int
    image: ImageReference
    env_path: str
    pinned_packages: dict[str, str]
    deterministic: bool
    budget: Budget
    repo: str


PROJECT_FIELDS = (
    "partition",
    "gpu",
    "cpus",
    "mem_gb",
    "minutes",
    "requeue",
    "checkpoint_steps",
    "image",
    "env_path",
    "pinned_packages",
    "deterministic",
)
"""The fields a project declares, and exactly the fields a run may override.

Kept as one tuple so the two uses cannot drift: adding a field to
:class:`ProjectConfig` without adding it here would make it undeclarable, and
adding it here without adding it there would make it unreadable.
"""


def _require_project_image(image: ImageReference | None) -> ImageReference:
    """Narrow a decoded image to the one every project must declare.

    A ``require_*`` that narrows, rather than a check that returns None beside
    an optional field. :attr:`ProjectConfig.image` is not optional, so the
    absence has to be refused at the point the type stops permitting it --
    otherwise every reader downstream carries a ``is None`` branch for a state
    the decoder guarantees cannot arrive, and those branches are unreachable
    code that still has to be covered.

    A run's numbers are decided by the whole stack under them -- the CUDA
    runtime, cuDNN and the torch build for GPU work; the compiler, the libc
    and every linked library for CPU work. An image pins all of it and gives
    the run a CONTENT DIGEST, which is the ``image_digest`` axis of
    :class:`~platform_core.comparability.RunFingerprint`. A directory
    environment on a shared filesystem pins nothing and has no digest.

    THIS RULE USED TO EXEMPT CPU-ONLY PROJECTS, and the exemption was wrong
    for the reason the GPU case was already right. What an image pins is not
    the card; it is everything the numbers depend on that nobody re-declares.
    A CPU project's timings move with a compiler version and its results move
    with a BLAS build, and neither is visible in ``env_path`` or recoverable
    from ``pinned_packages`` -- which the caller edits. ``cleargbm`` is the
    demonstration: its timed arm is compiled Rust, so neither a Python package
    list nor a directory path describes what produced a benchmark number, and
    the project has carried ``image: null`` while producing exactly such
    numbers.

    The exemption also worked as a template. ``turkic-lstm`` was onboarded
    unimaged in 2026-08 by copying the shape from the CPU project that was
    allowed to be unimaged; an exemption that exists is an exemption that gets
    copied to where it does not apply.

    WHAT HAPPENED WITHOUT THIS, on 2026-08-28. ``turkic-lstm`` was onboarded
    with ``env_path: /pub/wagnera3/envs/lstm`` and no image, because nothing
    refused it: the CPU project ``cleargbm`` had that shape and it was copied
    without asking whether the reason carried over. Within the hour that
    environment was mutated in place with ``pip install`` to change its torch
    version, and every check still passed, because ``pinned_packages`` was
    simply edited to match. A declaration the caller can bring into agreement
    with reality is not a check. Both GPU projects that had ever actually run
    -- ``mi`` and ``floor`` -- already declared images; this is their
    practice, written where it cannot be skipped.

    WHY HERE AND NOT ON THE JOB SPEC, which was tried first and reverted. A
    job may legitimately run on the host with a GPU: that is how a one-off
    probe works, ``core.sbatch`` has a whole host branch for it, and 47 tests
    in ``test_sbatch`` exercise it. Forbidding it there outlawed a capability
    the tool supports on purpose. What must not happen is ONBOARDING a
    standing body of work that way -- a project is durable, produces numbers
    someone will subtract, and is the thing this registry exists to describe.

    A run may still override ``image`` per submission, which is deliberate
    (an experiment pinning a NEWER image is the normal way an image version
    is rolled). It may not remove one; see
    :func:`~hpc3.contracts.run.resolve_run`.

    Args:
        image: Image the project's payloads run inside, as decoded, or None
            when the document declared none.

    Returns:
        That image.

    Raises:
        AppError: With ``PROJECT_UNIMAGED`` when a project declares no image.
    """
    if image is not None:
        return image
    raise AppError(
        Hpc3ErrorCode.PROJECT_UNIMAGED,
        "Every registered project must declare an 'image'. A directory "
        "environment pins no CUDA runtime, no cuDNN, no torch build, no "
        "compiler and no BLAS, and carries no digest, so two runs of it can "
        "differ in every layer that decides their numbers and still "
        "fingerprint the same -- and an environment can be edited in place "
        "while 'pinned_packages' is edited to match. This applies to CPU-only "
        "projects too: what an image pins is not the card, it is everything "
        "the numbers depend on that nobody re-declares. Produce one with "
        "hpc3-image-capture, hpc3-image, hpc3-stage and hpc3-image-build, "
        "THEN add the project to the workspace carrying the digest that build "
        "produced -- a project is registered once it is reproducible, not "
        "before.",
    )


def decode_project_config(
    value: JSONValue, cluster: ClusterFacts, *, config_dir: pathlib.Path
) -> ProjectConfig:
    """Decode and validate one project's resource defaults.

    The cross-field submission rules -- billing consent, preemption
    protection, partition-carries-GPU -- are NOT checked here. They are
    checked when a run resolves against these defaults, because a run may
    override any field involved in them, and rejecting a project whose
    defaults only become valid once combined would refuse a legitimate shape.

    Args:
        value: Value produced by the JSON loader.
        cluster: The cluster whose partitions and GPUs the defaults must name.
        config_dir: Directory the workspace document was read from. ``repo``
            resolves against it, so a relative declaration means the same
            thing from any working directory.

    Returns:
        Validated defaults.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing,
            mistyped, empty or non-positive.
        AppError: With ``PARTITION_UNKNOWN`` or ``GPU_TYPE_UNPINNED`` if the
            partition or GPU model is not one this cluster has.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"project config must be a JSON object, got {type(value).__name__}")

    checkpoint_steps = require_int(value, "checkpoint_steps")
    if checkpoint_steps < 0:
        raise JSONTypeError(
            f"Field 'checkpoint_steps' must not be negative, got {checkpoint_steps}"
        )

    return ProjectConfig(
        partition=require_partition(cluster, value, "partition"),
        gpu=decode_gpu_request(cluster, value.get("gpu"), "gpu"),
        cpus=require_positive(value, "cpus"),
        mem_gb=require_positive(value, "mem_gb"),
        minutes=require_positive(value, "minutes"),
        requeue=require_bool(value, "requeue"),
        checkpoint_steps=checkpoint_steps,
        image=_require_project_image(decode_image_reference(value.get("image"), "image")),
        env_path=require_nonempty_str(value, "env_path"),
        pinned_packages=require_pinned_packages(value, "pinned_packages"),
        deterministic=require_bool(value, "deterministic"),
        budget=decode_budget(value.get("budget")),
        repo=str(config_dir / require_nonempty_str(value, "repo")),
    )


def encode_project_config(config: ProjectConfig) -> dict[str, JSONValue]:
    """Encode one project's defaults to a JSON object.

    Args:
        config: Defaults to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "partition": config["partition"],
        "gpu": encode_gpu_request(config["gpu"]),
        "cpus": config["cpus"],
        "mem_gb": config["mem_gb"],
        "minutes": config["minutes"],
        "requeue": config["requeue"],
        "checkpoint_steps": config["checkpoint_steps"],
        "image": encode_image_reference(config["image"]),
        "env_path": config["env_path"],
        "pinned_packages": encode_pinned_packages(config["pinned_packages"]),
        "deterministic": config["deterministic"],
        "budget": encode_budget(config["budget"]),
        "repo": config["repo"],
    }


__all__ = [
    "PROJECT_FIELDS",
    "ProjectConfig",
    "decode_project_config",
    "encode_project_config",
]
