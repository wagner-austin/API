---
title: Small-file boots belong on the node's own scratch, never on BeeGFS
tags: [cluster-facts, storage, performance]
related: ["[[partitions-and-billing]]", "[[preemption-and-campaigns]]", "[[facts-are-code]]"]
source_paths:
  - "src/hpc3/core/sbatch.py"
  - "src/hpc3/core/array_sbatch.py"
source_git_blobs:
  "src/hpc3/core/sbatch.py": "7b378a66a9ab8eaee53a33e06a62494395d259a3"
  "src/hpc3/core/array_sbatch.py": "8ebc37d14ba8fb7d1dcd8b629225fddb57e1cd53"
provenance:
  - "probe job 55675199 on hpc3-l18-04, 2026-09-01"
  - "rusted engine log champion-s2707 (ab48-v7), 2026-09-01"
  - "clients/RustedWarfareBot src/rw_bot/harness/campaign.py@member_command (outside this workspaceRoot)"
fact_checked: 2026-09-01
confidence: high
---

# Small-file boots belong on the node's own scratch, never on BeeGFS

Slurm provisions **`$TMPDIR=/tmp/<user>/<jobid>`** on every HPC3 compute
node: per-job, on local disk, removed with the job. Measured on
`hpc3-l18-04` (probe job 55675199): a 256 MiB write lands at **1.9 GB/s**.
The RCIC docs do not state this anywhere findable — it was established by
probe, which is why it is written down here.

## The failure class this closes

`/pub` is BeeGFS — a parallel filesystem built for large streaming I/O and
poor at concurrent small-file metadata storms. A workload that BOOTS by
reading thousands of small files (a game engine loading `.ini` assets, an
interpreter walking a venv, anything with a file-watcher) crawls when many
jobs boot at once against it, and it crawls in a way that looks like
anything but a filesystem problem: the rusted project lost **ten members
across four batches** to what read as random engine crashes, every one
completing on an uncontended retry. The engine's own log told the truth —
asset lines seconds apart, a failed watch attempt per file, and the
process halted by its own 60-second world-liveness guard, not by any
crash.

Two properties made the class hard to see:

- **Retry succeeds.** An uncontended boot is fast, so every resubmission
  completes and the failure reads as "transient". It is not transient; it
  is deterministic under concurrency.
- **The guard names its symptom, not the cause.** "The live world is
  null after 60s" is a true statement about a slow boot, and nothing in it
  says *filesystem*.

## The rule

Per-job disposable data — clones, working copies, extracted archives,
anything the job creates and the result does not need — goes in
`$TMPDIR`. Reference `$TMPDIR` **unexpanded in the submitted command** so
the batch script's bash expands it on the node after Slurm has provisioned
it; the generated scripts run `set -u`, so a node without it fails loudly
rather than quietly landing on shared disk. What the shared filesystem is
for: the one-time streaming copy IN (BeeGFS is good at that), and the
artifacts a run files OUT.

The rusted project's `member_command` is the worked example: it emits
`--clones $TMPDIR/rw-clones`, the shared-filesystem clone helper was
deleted rather than deprecated, and a test pins that no clone path may
start with the cluster root. Cleanup came free — the node removes
`$TMPDIR` with the job.
