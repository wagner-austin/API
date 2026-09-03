# Cluster facts

What was measured off the real machine and how those measurements are held:
partitions and billing, the facts-are-code rule, and what a job looks like
from the cluster's side.

[Billing follows the job's QOS, and free is not a setting](../pages/partitions-and-billing.md) -- the partition table, the AllowQos lesson, the excluded partitions
[Cluster facts are code, never configuration](../pages/facts-are-code.md) -- why a workspace selects but cannot describe a cluster, and how to add one
[What the cluster sees, and why the comment is live-only](../pages/job-identity-on-cluster.md) -- job names, derived directories, HPC3_* env, the comment-vs-ledger split
[Preemption cancels, checkpoints protect, campaigns converge](../pages/preemption-and-campaigns.md) -- PreemptMode=CANCEL, measured on free-gpu
[Small-file boots belong on the node's own scratch, never on BeeGFS](../pages/node-local-scratch.md) -- per-job $TMPDIR measured at 1.9 GB/s; the boot-contention class ten members died to, closed at the command chokepoint
[The interpreter every project needs is not in `module avail python`](../pages/interpreter-availability.md) -- 2.7/3.8/3.10/3.14 as modules and 3.9 as system, while everything here requires 3.11; conda is the door, and the improvised bootstrap left one env symlinked into another
