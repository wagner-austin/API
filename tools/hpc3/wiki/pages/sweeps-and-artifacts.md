---
title: Sweeps, and why every member declares its artifact
tags: [submission, sweeps, identity]
hubs: [submission]
related: ["[[preemption-and-campaigns]]", "[[run-documents]]", "[[submission-rules]]"]
source_paths:
  - "src/hpc3/contracts/sweep.py"
  - "src/hpc3/core/sweep.py"
source_git_blobs:
  "src/hpc3/contracts/sweep.py": "7c63b071da5d96248cbee482acb3624a04cb8a25"
  "src/hpc3/core/sweep.py": "b730820ce899c2f022e3c69fd97f78844d77c6cd"
fact_checked: 2026-09-14
confidence: high
---

# Sweeps, and why every member declares its artifact

A sweep is one template run several ways at once:

```json
{
  "project": "abl", "name": "rung-large",
  "minutes": 900, "resumes_from_checkpoint": true,
  "members": [
    { "suffix": "armB-s0", "command": "python -u train.py --arm B --seed 0 --out /pub/wagnera3/abl/s0.json",
      "artifact": "/pub/wagnera3/abl/s0.json" },
    { "suffix": "armB-s1", "command": "python -u train.py --arm B --seed 1 --out /pub/wagnera3/abl/s1.json",
      "artifact": "/pub/wagnera3/abl/s1.json" }
  ]
}
```

Each member states its own `artifact`, or `null` if it writes no file of its
own — six arms writing to one path are five results nobody can read. The
artifact is checked against that member's own command, so a suffix changed in
one and not the other fails here rather than after the run.

`hpc3-sweep` submits each member and records each one as it goes. There is no
rollback: a member that fails leaves the earlier ones running and findable,
because a live job that is fine should not be cancelled for a later job's
failure.

## Sweep vs campaign

A campaign is the same document run repeatedly: it submits exactly the members
that are neither finished nor already running ([[preemption-and-campaigns]]).
"Done" means the member's artifact exists — so a member that declares no
artifact of its own is never done and would be resubmitted forever. That is
the one thing a campaign refuses: `cleargbm`'s sweeps all declare `null`
(correctly, every member runs `--no-save-model`) and are therefore sweeps,
not campaigns. The refusal says so and names `hpc3-sweep`.

## Ceilings are checked before anything is sent

`SWEEP_EXCEEDS_GPU_CEILING` / `SWEEP_EXCEEDS_CPU_CEILING` /
`SWEEP_EXCEEDS_JOB_CEILING`: Slurm does not reject an oversized set; it queues
the excess against `MaxTRESPU`, which reads as a busy cluster and is not.
Which ceiling binds follows from the partition: GPU work pends against
`gres/gpu`, CPU work against `cpu`. A ceiling the QOS **does not declare** is
not checked — `free-gpu-part` caps GPUs and says nothing about cores, and
inventing a core limit for it would refuse sweeps the cluster would have run.

## A sweep larger than the ceiling declares a throttle

Nineteen cloze-floor members on `free-gpu32`, whose QOS lets one user hold
four GPUs, is a sweep the check above refuses and the package had no shape
for until 2026-09-14: a chain is sequential single jobs, a campaign resolves
the whole sweep first, and five documents for one measurement is the failure
[[preemption-and-campaigns]] describes. Slurm's own answer is `--array=0-18%4`,
and the parsers here had understood that `%` from the cluster's side since
probe job 55678543 without a document being able to write it.

A sweep may now carry `"throttle": 4`. Absent or null means what every earlier
document meant, all at once. **The ceiling is checked against the throttle
when one is declared**, because the throttle is what the scheduler will
actually let run: nineteen members at throttle 4 hold four GPUs. Decoding
refuses a throttle below one, above the member count (it cannot bind, so it
is a typo), a boolean, or a non-integer, each by name. The rendered `%N`
travels on both `sbatch` calls, the `--test-only` dry run and the real one,
so the preflight verdict is about the job that gets submitted — and it stays
the submitter's argument rather than a script directive, for the reason
[[job-arrays]] gives.

The wall clock is the other half of fitting. The same document brought its
budget projection from 17.4 GPU-hours to 4.75 by stating `minutes: 15` from a
measured 328-second chunk rather than inheriting the project's 55-minute
default; the cap itself was not touched.
