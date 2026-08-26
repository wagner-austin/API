"""Fixtures shared by the two batch-script test modules.

``test_sbatch`` covers rendering a job that runs on the cluster directly;
``test_sbatch_image`` covers one that runs inside an image. They were one
module until it passed the 600-line ceiling, and they must keep using the
SAME baseline spec: an image test that quietly drifted onto a different
partition or GPU would stop comparing the two renderings of one job.
"""

from __future__ import annotations

from platform_core.json_utils import JSONValue

from hpc3.contracts.job import JobSpec
from tests.against_hpc3 import decode_job_spec
from tests.conftest import gpus

LOG_DIR = "/pub/wagnera3/logs"

SIF = "/pub/wagnera3/images/abl.sif"

IMAGE: JSONValue = {
    "path": SIF,
    "sha256": "9ed4e27fd0d8207de3f84e833b98e0cf7e6ab09af66726849ca1cf023326cd51",
    "binds": ["/pub/wagnera3"],
}


def spec(**overrides: JSONValue) -> JobSpec:
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
        "gpu": gpus("A100"),
        "cpus": 8,
        "mem_gb": 96,
        "minutes": 30,
        "requeue": False,
        "checkpoint_steps": 0,
        "env_path": "/pub/wagnera3/envs/abl-pinned",
        "pinned_packages": {},
        "deterministic": False,
        "experiment": {"arm": "B", "seed": "42"},
        "command": "python train.py --seed 42",
    }
    base.update(overrides)
    return decode_job_spec(base)


__all__ = ["IMAGE", "LOG_DIR", "SIF", "spec"]
