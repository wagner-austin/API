---
title: Preemption cancels, checkpoints protect, campaigns converge
tags: [submission, preemption, campaigns]
related: ["[[sweeps-and-artifacts]]", "[[partitions-and-billing]]", "[[submission-rules]]"]
source_paths:
  - "src/hpc3/contracts/job.py"
  - "src/hpc3/core/inflight.py"
  - "src/hpc3/core/campaign.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/contracts/job.py": "45f6be817460501c520ecca58b4f1dbc7341f4d0"
  "src/hpc3/core/inflight.py": "03cf4ad35a0e4c2987b9d5e8ed20e440b3891c55"
  "src/hpc3/core/campaign.py": "661951977b36bde5c56488f996f826c3d23df76d"
  "README.md": "c4cdcc31ae83beaede3c2635a943ddc0bcf0c083"
provenance:
  - "scontrol show partition free-gpu (2026-08-28)"
fact_checked: 2026-09-02
confidence: high
---

# Preemption cancels, checkpoints protect, campaigns converge

`#SBATCH --requeue` is rendered for every preemptible run over an hour, and it
is correct — but it does **not** bring a preempted job back on this cluster:

```
$ scontrol show partition free-gpu
GraceTime=0  PreemptMode=CANCEL
```

Slurm requeues on preemption only under `PreemptMode=REQUEUE`. Under `CANCEL`
the job is cancelled and `--requeue` covers node failure and administrative
requeue instead. Observed 2026-08-28: `turkic-lstm.bases-kk` was preempted at
64 seconds and did not return to the queue.

**What actually protects the work is the checkpoint, not the flag.** The
resume state is written to `/pub` after every completed epoch precisely
because node-local scratch dies with the job — including when it dies by
preemption. A preempted member is resubmitted and continues from its last
completed epoch. That is what `checkpoint_steps` is for, and it is why a long
run on a preemptible partition is viable at all.

## The campaign is the resume mechanism

Run the sweep document again and it submits exactly the members that are
neither finished nor already running:

```
done      turkic-lstm.bases-tr
in flight turkic-lstm.bases-kk <- turkic-lstm.bases-r1-kk
2 done, 5 in flight, 0 submitted, 5 remaining
```

**The artifact is the identity, not the job name.** `bases-uz-r3` is
recognised as covering `bases-uz` because they write the same checkpoint, so a
resume under any name counts. Run it twice in a row and the second run submits
nothing — which is also why it is safe on a schedule.

## Why this exists

On 2026-08-28 `free-gpu` preempted five of seven members inside an hour, and
nothing could say which five. What followed was four hand-written resume
documents describing one experiment, each a transcription of a queue state
that had already changed — and at one point two of them were live, writing the
same `uz_best.pt`. That race is now refused at submit time by
`hpc3.core.inflight`, for every command, and the campaign avoids provoking it
in the first place.

The older shape — a one-member run document naming the job it resumes — still
works and still records the chain in the ledger. It is the right tool for
resuming *one* member deliberately, and the wrong tool for a preemption wave.
Expect waves: the free partitions are free because other people's allocated
work outranks yours.

## Confirmed at array scale, and for the checkpoint-free case

2026-09-02, the plain `free` partition (`PreemptMode=CANCEL`, `GraceTime=0`
verified by `scontrol` the same hour): a wave took 22 of a 96-task array
(`rusted.c0f4vh`) carrying `#SBATCH --requeue` straight to terminal
`PREEMPTED` — nothing returned to the queue. One `hpc3-campaign` converge
pass resubmitted exactly the 22 as a sparse array against the same member
table and left the 49 running and 25 finished members alone. This is also
the `checkpoint_steps: 0` case the deterministic-replay clause admits: the
members are pinned-regime replays, so "resume" is replay-from-zero, and the
campaign is the entire recovery mechanism — no checkpoint file exists to
protect.
