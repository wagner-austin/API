---
title: The five triage conditions that look like health
tags: [operations, triage, ledger]
hubs: [operations]
related: ["[[ledger-closures]]", "[[image-ledger-lessons]]", "[[budget-model]]", "[[command-length-limits]]", "[[job-arrays]]"]
source_paths:
  - "src/hpc3/cli/triage.py"
  - "src/hpc3/core/triage.py"
source_git_blobs:
  "src/hpc3/cli/triage.py": "46fd93061aff3ed3ebd76e0bc9407657d3db5859"
  "src/hpc3/core/triage.py": "d9ffb293b5da749729ff21c83c935d849166f56e"
provenance:
  - "261 of 621 pending GPU jobs on DependencyNeverSatisfied (squeue sample)"
fact_checked: 2026-09-06
confidence: high
---

# The five triage conditions that look like health

`hpc3-triage` reconciles the ledger against the cluster **in both directions**
and exits non-zero if anything is found.

- **blocked** — pending on a reason that will never resolve. On HPC3, 261 of
  621 pending GPU jobs were sitting on `DependencyNeverSatisfied`; in
  `squeue`'s state column that is indistinguishable from waiting on
  `Resources`. A reason the allowlist has never seen is treated as blocked,
  because that is where patience costs a week.
- **unaccounted** — we recorded submitting it and accounting has never heard
  of it. No cluster-side query can find these: the evidence is the *absence*
  of a cluster-side record, which is what the local ledger exists to supply.
  **Absence is only meaningful once you have asked correctly**, and until
  2026-09-06 this check had not: a queued job array is ONE accounting row
  standing for every task the ledger recorded separately, and `sacct -j
  <base>_<index>` returns nothing at all while that task is inside it. Twelve
  healthy queued tasks were reported as jobs the cluster had never heard of.
  The query is now built from array base ids and the rows expanded before
  matching ([[job-arrays]]) — a task genuinely missing from a started array,
  or from a sparse aggregate that does not name it, is still reported.
- **unclaimed** — the cluster is holding it and the ledger has never heard of
  *it*. No ledger-side query can find these either, for the mirror reason:
  the check has to ask the account to enumerate itself (`squeue --me`).
- **silent** — `RUNNING`, holding GPUs, and its log has stopped growing. Log
  age is measured against the cluster's own clock; a few minutes of skew
  would either invent staleness or hide it. Since 2026-09-05 the probe is
  split into batches, and each batch carries its **own** clock reading for
  this reason — see [[command-length-limits]].
- **oversized** — the project asks Slurm for far more wall clock than its
  work has ever taken. Slurm backfills a job into a hole its own size, so an
  oversized request waits for a hole it never needed. Only `COMPLETED` runs
  on the project's **own partition** count as evidence — both exclusions were
  learned from this check's first live run, which counted a cancelled job's
  zero seconds and took evidence from an image build that never used
  `minutes` at all.

**What `oversized` caught the day it was written:** `turkic-lstm` declared
`minutes: 720`; its members finished in 27 minutes, and five more sat
unschedulable for hours. It got past everything because it was never measured
— created before LSTM had ever run on the cluster, inherited from an example,
and finally *ratified* by a budget computed from it (84.0 GPU-hours = 7
members × 12 hours exactly). `floor` was over-requested too, by 10×.

## The mirror check nobody had

**Only three of these existed until 2026-08-28.** `unaccounted` proves every
ledger row is a real job; nothing proved every real job has a ledger row —
and the image-build recipe *told you* to run raw `sbatch`, so twenty-one
builds ran unrecorded ([[image-ledger-lessons]]). The `unclaimed` check found
the twenty-second on its first run, and the finding was answered by closing
the path that produced it.

Two consequences worth knowing before the first red board: an interactive
session is a **true positive** (a job this machine did not submit and cannot
trace — if those become noise, record them, don't teach the check to look
away); and the check cannot see a bypassed job that already finished, because
`squeue` forgets a job minutes after it ends — it catches an unrecorded job
while it is costing something and cancelling is still possible, never
afterwards.
