"""UCI's HPC3, as measured on 2026-08-22 and 2026-08-23.

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
    description="UCI HPC3, GPU partitions measured 2026-08-22, CPU added 2026-08-23",
    gpus=("V100", "A30", "A100", "L40S", "RTX6000"),
    partitions={
        "free-gpu": PartitionFacts(
            usage_factor=0.0,
            preemptible=True,
            max_hours=72,
            gpus=("V100", "A30", "A100"),
            max_gpus_per_user=24,
            max_cpus_per_user=None,
            max_jobs_per_user=24,
        ),
        "free-gpu32": PartitionFacts(
            usage_factor=1.0,
            preemptible=True,
            max_hours=72,
            gpus=("L40S", "RTX6000"),
            max_gpus_per_user=4,
            max_cpus_per_user=None,
            max_jobs_per_user=16,
        ),
        "gpu": PartitionFacts(
            usage_factor=1.0,
            preemptible=False,
            max_hours=336,
            gpus=("V100", "A30", "A100"),
            max_gpus_per_user=40,
            max_cpus_per_user=None,
            max_jobs_per_user=32,
        ),
        "gpu32": PartitionFacts(
            usage_factor=1.0,
            preemptible=False,
            max_hours=336,
            gpus=("L40S", "RTX6000"),
            max_gpus_per_user=12,
            max_cpus_per_user=None,
            max_jobs_per_user=16,
        ),
        "free": PartitionFacts(
            usage_factor=0.0,
            preemptible=True,
            max_hours=72,
            gpus=(),
            max_gpus_per_user=None,
            max_cpus_per_user=3500,
            max_jobs_per_user=3500,
        ),
        "standard": PartitionFacts(
            usage_factor=1.0,
            preemptible=False,
            max_hours=336,
            gpus=(),
            max_gpus_per_user=None,
            max_cpus_per_user=2500,
            max_jobs_per_user=2500,
        ),
    },
)
"""The ``free-`` prefix describes priority, not always cost.

``standard`` is HPC3's DEFAULT partition -- it is the one marked ``standard*``
in ``sinfo`` -- and it carries ``UsageFactor 1.0``. A CPU job submitted without
naming a partition bills. That is the same class of mislabel as ``free-gpu32``
pointing the other way, and it is why this package requires the partition to
be stated rather than defaulted.

``free`` is the CPU twin of ``free-gpu``: ``UsageFactor 0.0``, the same
``PreemptMode=CANCEL``, and the same 72-hour ceiling. The preemption-protection
rule therefore applies to it unchanged.

The billing figures are measured rather than read off the QOS name, and the
distinction is load-bearing. A QOS literally named ``free-gpu`` exists and
carries ``UsageFactor 1.000000``; it is NOT what the ``free-gpu`` partition
uses. The partition's QOS is ``free-gpu-part``, at ``0.000000``. Confirmed at
the other end too: after a GPU run and a CPU run, ``sbank balance statement``
reported 0 SUs consumed.

Deliberately absent, each for a measured reason:

* ``standard-hbm`` -- reachable, but only with an explicit
  ``--qos=standard-hbm``. Without it: ``allocation failure: Invalid qos
  specification``. This package emits no QOS directive, and inventing one for
  a single partition would add a surface every other partition ignores.
* ``gpu-hugemem`` -- ``Invalid account or account/partition combination``.
  No access.
* ``highmem``, ``hugemem``, ``maxmem``, ``admin`` -- listed by ``sinfo -a``
  but absent from ``scontrol show partition``, so not available to this
  account.
"""


__all__ = ["HPC3"]
