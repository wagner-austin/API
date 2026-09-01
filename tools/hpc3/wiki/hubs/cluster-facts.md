# Cluster facts

What was measured off the real machine and how those measurements are held:
partitions and billing, the facts-are-code rule, and what a job looks like
from the cluster's side.

[Billing follows the job's QOS, and free is not a setting](../pages/partitions-and-billing.md) -- the partition table, the AllowQos lesson, the excluded partitions
[Cluster facts are code, never configuration](../pages/facts-are-code.md) -- why a workspace selects but cannot describe a cluster, and how to add one
[What the cluster sees, and why the comment is live-only](../pages/job-identity-on-cluster.md) -- job names, derived directories, HPC3_* env, the comment-vs-ledger split
[Preemption cancels, checkpoints protect, campaigns converge](../pages/preemption-and-campaigns.md) -- PreemptMode=CANCEL, measured on free-gpu
