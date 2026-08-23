"""UCI's HPC3, as measured on 2026-08-22 and 2026-08-23.

Every value here was read off the machine with ``sinfo``, ``scontrol``,
``sacctmgr`` and a job on each card -- not from a spec sheet and not from the
Slurm label:

* The GRES name ``RTX6000`` reads as the 2018 Turing Quadro. It is a current
  Blackwell server card with 96 GB and compute capability ``sm_120``.
* **Billing follows the JOB's QOS, not the partition's.** This module said
  ``free-gpu32`` billed at ``UsageFactor 1.0`` from 2026-08-22 until it was
  measured properly on 2026-08-23. It does not bill. The 1.0 belongs to
  ``free-gpu32-part``, which is the partition QOS and governs *limits*; jobs
  there run under ``low`` (``UsageFactor 0.000000``), because every free
  partition declares ``AllowQos=low,guest``. Reading a factor off the
  partition QOS is how that error was made.

  Verified rather than reasoned: an 8-core, 1-GPU, 2-minute RTX6000 job on
  ``free-gpu32`` moved ``sshare`` ``RawUsage`` by exactly zero, on a meter
  that simultaneously read 33,654,891 for another user in the same account.

So the rule is ``AllowQos``: ``low,guest`` (both 0.0) is free, ``normal,high``
(1.0 and 2.0) bills. The billing partitions below are marked from that rule
rather than from a measurement, deliberately: the safe direction to be wrong
is to refuse a partition that would have been free, never to spend on one
recorded as free.

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
            usage_factor=0.0,
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
"""The three ``free`` partitions are genuinely free; the other three bill.

``standard`` is HPC3's DEFAULT partition -- the one marked ``standard*`` in
``sinfo`` -- and it bills. A CPU job submitted without naming a partition
therefore spends service units, which is why this package requires the
partition to be stated and never defaults it.

``free`` is the CPU twin of ``free-gpu``: ``UsageFactor 0.0``, the same
``PreemptMode=CANCEL``, and the same 72-hour ceiling. The preemption-protection
rule therefore applies to it unchanged.

There is also a QOS literally named ``free-gpu`` carrying ``UsageFactor
1.000000``, which is not what the ``free-gpu`` partition uses. Between that and
the ``-part`` QOS trap above, the lesson is the same one twice: **a name
containing "free" and a QOS containing a factor are two different questions,
and only a job's own accounting answers either.**

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
