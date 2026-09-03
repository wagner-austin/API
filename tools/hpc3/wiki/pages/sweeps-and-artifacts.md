---
title: Sweeps, and why every member declares its artifact
tags: [submission, sweeps, identity]
hubs: [submission]
related: ["[[preemption-and-campaigns]]", "[[run-documents]]", "[[submission-rules]]"]
source_paths:
  - "src/hpc3/contracts/sweep.py"
  - "src/hpc3/core/sweep.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/contracts/sweep.py": "0893d3d08329be27f076211775f7f7e7b9faa5cc"
  "src/hpc3/core/sweep.py": "6796f42b6a10dbb1ea223b0ae9f3651ff4f1675e"
  "README.md": "c4cdcc31ae83beaede3c2635a943ddc0bcf0c083"
fact_checked: 2026-09-01
confidence: high
---

# Sweeps, and why every member declares its artifact

A sweep is one template run several ways at once:

```json
{
  "project": "abl", "name": "rung-large",
  "minutes": 900, "checkpoint_steps": 250,
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
