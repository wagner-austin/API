"""UCI's HPC3, as measured on 2026-08-22.

Every value here was read off the machine with ``sinfo``, ``scontrol``,
``sacctmgr`` and a job on each card -- not from a spec sheet and not from the
Slurm label. Two of those labels are actively misleading and are the reason
this module exists rather than a README paragraph:

* The GRES name ``RTX6000`` reads as the 2018 Turing Quadro. It is a current
  Blackwell server card with 96 GB and compute capability ``sm_120``.
* ``free-gpu32`` is not free. Its QOS carries ``UsageFactor 1.000000`` where
  ``free-gpu`` carries ``0.000000``, so it bills one service unit per
  core-hour.

What is deliberately NOT here: per-card VRAM and compute capability. Both were
measured, and neither is consulted by any rule this package enforces -- nothing
can decide whether a model fits without knowing what the model needs, which
this package never learns. Carrying them as data no code path reads would be
reference material wearing a type's clothing. They live on the wiki page
``hpc3-compute-capability-for-this-ablation`` instead.
"""

from __future__ import annotations

from hpc3.contracts.cluster import ClusterFacts, PartitionFacts

HPC3: ClusterFacts = ClusterFacts(
    slug="hpc3",
    description="UCI HPC3, measured 2026-08-22",
    gpus=("V100", "A30", "A100", "L40S", "RTX6000"),
    partitions={
        "free-gpu": PartitionFacts(
            usage_factor=0.0,
            preemptible=True,
            max_hours=72,
            gpus=("V100", "A30", "A100"),
            max_gpus_per_user=24,
            max_jobs_per_user=24,
        ),
        "free-gpu32": PartitionFacts(
            usage_factor=1.0,
            preemptible=True,
            max_hours=72,
            gpus=("L40S", "RTX6000"),
            max_gpus_per_user=4,
            max_jobs_per_user=16,
        ),
        "gpu": PartitionFacts(
            usage_factor=1.0,
            preemptible=False,
            max_hours=336,
            gpus=("V100", "A30", "A100"),
            max_gpus_per_user=40,
            max_jobs_per_user=32,
        ),
        "gpu32": PartitionFacts(
            usage_factor=1.0,
            preemptible=False,
            max_hours=336,
            gpus=("L40S", "RTX6000"),
            max_gpus_per_user=12,
            max_jobs_per_user=16,
        ),
    },
)
"""The ``free-`` prefix describes priority, not always cost."""


__all__ = ["HPC3"]
