---
title: Chains stop when a stage fails, because Slurm queues the corpse forever
tags: [submission, chains, dependencies]
related: [[run-documents]], [[triage-conditions]], [[budget-model]]
sources: ["squeue sample: 261 of 621 pending GPU jobs on DependencyNeverSatisfied", cli/chain, README.md@4dc63f17]
fact_checked: 2026-09-01
confidence: high
---

# Chains stop when a stage fails, because Slurm queues the corpse forever

Stages run in order, each waiting on the one before it with `afterok`. They
may differ in resources — a training stage holds a GPU, the evaluation reading
its checkpoints often does not — so a stage overrides the same fields a run
can, on top of anything the chain sets once:

```json
{
  "project": "sirius", "name": "batch7",
  "experiment": { "sample_set": "batch7" },
  "stages": [
    { "suffix": "sirius", "command": "sirius ... formula", "cpus": 16, "minutes": 360 },
    { "suffix": "zodiac", "command": "sirius ... zodiac", "cpus": 32, "mem_gb": 128 }
  ]
}
```

**`--kill-on-invalid-dep=yes` is emitted with every dependency, never without
it.** This is the whole reason chains are safe to use here. When a dependency
cannot be satisfied Slurm does *not* reject the dependent job — it queues it
forever on `DependencyNeverSatisfied`, holding a QOS slot, looking in
`squeue` exactly like a job waiting its turn. That was **261 of 621** pending
GPU jobs on HPC3 in one sample. With the flag, a stage whose predecessor
failed is cancelled instead: the slot is freed, and the ledger gets a terminal
state it can close rather than an entry that never resolves.

## What follows from the ids not existing until submission

- **Every stage is validated before the first is sent.** Otherwise a
  misspelled partition in stage three surfaces an hour after stage one
  started running.
- **A chain document may not write `depends_on`.** The chain wires its own,
  so a hand-written one would be silently replaced. It is refused instead.
- **The budget is checked against the whole pipeline, not stage by stage.**
  Stages are sequential in *time* and simultaneous in *commitment*:
  submitting the chain commits every hour of it.

## A chain is not a sweep

A sweep is one template run several ways at once, and is bounded by how many
of your jobs may RUN concurrently. A chain runs one stage at a time, so that
ceiling cannot bind it — what could is the submit ceiling, which on `free` is
3500. No ceiling check is applied to chains for that reason: a check that
cannot fire still reads as protection.
