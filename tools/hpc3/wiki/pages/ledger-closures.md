---
title: Closures, or why the unaccounted check does not rot
tags: [operations, ledger, triage]
hubs: [operations]
related: ["[[triage-conditions]]", "[[job-identity-on-cluster]]", "[[command-length-limits]]", "[[job-arrays]]"]
source_paths:
  - "src/hpc3/cli/triage.py"
  - "src/hpc3/core/ledger.py"
source_git_blobs:
  "src/hpc3/cli/triage.py": "46fd93061aff3ed3ebd76e0bc9407657d3db5859"
  "src/hpc3/core/ledger.py": "94065680790919c0a68e097320d228400805b564"
provenance:
  - "MinJobAge 300s, read from scontrol show config"
fact_checked: 2026-09-06
confidence: high
---

# Closures, or why the unaccounted check does not rot

Two Slurm components forget finished jobs, on very different schedules.

`squeue` drops a job `MinJobAge` after it ends — **300 seconds on HPC3, read
from `scontrol show config`**. Past that, `squeue -j <id>` does not return
empty, it exits non-zero with `Invalid job id specified`. Triage therefore
asks the queue only about ids accounting reports as `PENDING`; nothing else
can be in it.

`sacct` retention depends on `slurmdbd`'s purge settings, which a login node
cannot read — and Slurm's default is to purge nothing, so on this cluster the
expiry is **unverified**. The closure record is built for it anyway: if a
site does enable `PurgeJobAfter`, every job past that window becomes a ledger
entry with no accounting row — the same observation as a job that never
existed — and triage would exit non-zero forever. A board that is always red
is the same as no board, and a closure costs one line per job.

## How a closure works

The moment accounting reports a terminal state, triage writes it to
`<ledger>.closed` and never asks about that job again. The closure is written
*after* the findings are built, so the run that closes a job still reports on
it. Failures close exactly as successes do — accounting forgets both on the
same schedule.

**One closure per TASK, not per accounting row.** A closure is looked up by
the ledger's own id, and the ledger records array tasks individually — but an
array cancelled while still pending reports one aggregate row for the whole
array (`55765275_[0-5]|CANCELLED by 2422328`, HPC3 2026-09-06). A closure
keyed on that expression matches no ledger entry, so its six tasks would never
be filtered out and would be re-queried and re-reported on every run, forever.
The always-red board this record exists to prevent, reached through the record
itself. Terminal rows are expanded ([[job-arrays]]).

A job that vanished before triage ever saw it end has no closure and stays
reportable forever, which is correct: that is the case the finding exists
for. The corollary is that triage has to run at least once inside the
retention window for a job to close cleanly.

**And on 2026-09-05 that corollary met its sharpest case: triage could not
run at all.** The query it builds is sized by the ledger, the ledger reached
6645 rows, and the command outgrew the submitter's argument limit — so the
closure record could not advance for as long as it had been long, and the
backlog was invisible because nothing was failing loudly ([[command-length-limits]]).
The first run after the fix wrote **6345 closures**; the next reported 72
open entries and 0 new ones. Closures are cheap per job and unbounded in
aggregate, and the thing that writes them has to stay buildable.
