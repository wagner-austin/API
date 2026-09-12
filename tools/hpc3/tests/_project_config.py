"""A declared project, for tests that need one without meaning anything by it.

Lifted out of ``test_research_index`` when the claim rules moved to their own
module and both halves still needed a ``ProjectConfig``. Copying it would have
made two fixtures that drift apart the first time ``ProjectConfig`` gains a
field, which is the fork this repository's standards name directly.
"""

from __future__ import annotations

from hpc3.contracts.cluster import GpuRequest
from hpc3.contracts.project import ProjectConfig


def project_config(
    *,
    gpu: GpuRequest | None = None,
    image_sha: str = "b" * 64,
    cpus: int = 4,
    minutes: int = 60,
) -> ProjectConfig:
    """Build a project configuration.

    Args:
        gpu: GPU request, or None for CPU-only work.
        image_sha: Image digest. Not optional, because the configuration it
            builds is not: every project declares an image.
        cpus: Cores per job.
        minutes: Wall clock per job.

    Returns:
        The configuration.
    """
    return ProjectConfig(
        partition="free",
        gpu=gpu,
        cpus=cpus,
        mem_gb=16,
        minutes=minutes,
        requeue=True,
        resumes_from_checkpoint=False,
        image={"path": "/pub/x.sif", "sha256": image_sha, "binds": ["/pub"]},
        env_path="/opt/env",
        pinned_packages={},
        deterministic=True,
        certified_inputs=False,
        budget={
            "self_imposed_gpu_hours": 0.0,
            "max_service_units": 0.0,
            "charge_account": "",
        },
        repo="../../..",
    )


__all__ = ["project_config"]
