---
title: Run documents say what is specific to this run
tags: [submission, contracts]
related: ["[[sweeps-and-artifacts]]", "[[submission-rules]]"]
source_paths:
  - "src/hpc3/contracts/run.py"
  - "src/hpc3/contracts/experiment.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/contracts/run.py": "a2b27fcaaabd5c9b1e68c2e69da74f5c953916d9"
  "src/hpc3/contracts/experiment.py": "530e8484b421d13119e951fef3ed8ea8b2706abf"
  "README.md": "c4cdcc31ae83beaede3c2635a943ddc0bcf0c083"
fact_checked: 2026-09-01
confidence: high
---

# Run documents say what is specific to this run

A run document carries only what differs from its project's defaults:

```json
{
  "project": "abl",
  "name": "armB-s42",
  "command": "python -u train.py --arm B --seed 42",
  "experiment": { "arm": "B", "seed": "42", "base_model": "gpt2", "corpus": "armB.txt" }
}
```

`experiment` is required and free-form: it is what the run **is**, as opposed
to which row in the queue it held. It lands in the ledger and in the job's
`--comment`, and `hpc3-trace` searches it. Without it the only link between a
job and the result it produced is a name somebody typed — and `arm-b-43`
mistyped as `arm-b-42` gives two jobs claiming one identity with no error
anywhere.

## Overrides go through the same decoder

Any project default may be restated to override it for this run alone:

```json
{
  "project": "abl", "name": "armC-full", "command": "python -u train.py --arm C",
  "minutes": 900, "checkpoint_steps": 250
}
```

Overriding is not a way around validation — the merged result goes through the
same decoder a fully hand-written spec would, so an override that lengthens a
preemptible run past an hour must also carry `requeue` and `checkpoint_steps`
([[preemption-and-campaigns]]).

## Unrecognised fields are refused, not ignored

`"minute": 600` is a run its author believes is capped at ten hours and that
Slurm will kill at the project default. The decoder refuses the unknown key
instead of silently dropping it.

## depends_on is run-level, never a project default

A run may chain onto a job already queued:

```json
{ "project": "abl", "name": "eval", "command": "...",
  "experiment": { "of": "55519937" },
  "depends_on": { "kind": "afterok", "job_ids": ["55519937"] } }
```

It is never a project default — a default would name ids from a previous
session, and a stale `afterok` on a job that finished last week is satisfied
instantly and silently. Multi-stage pipelines belong to [[chains]].
