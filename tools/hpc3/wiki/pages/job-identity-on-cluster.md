---
title: What the cluster sees, and why the comment is live-only
tags: [cluster-facts, identity, ledger]
related: [[ledger-closures]], [[run-documents]]
sources: ["AccountingStoreFlags = (null), measured 2026-08-23", contracts/layout.py, README.md@4dc63f17]
fact_checked: 2026-09-01
confidence: high
---

# What the cluster sees, and why the comment is live-only

Jobs are not loose. Every one carries its project:

| | |
| --- | --- |
| job name | `<project>.<name>` — self-describing among 102 users' rows |
| `--comment` | project, hardware and environment, readable via `scontrol show job <id>` or `squeue -o %k` **while the job is live** |
| scripts | `<root>/<project>/scripts/<project>.<name>.sbatch` |
| logs | `<root>/<project>/logs/<project>.<name>-<jobid>.{out,err}` |

The payload can read `HPC3_PROJECT`, `HPC3_JOB_NAME`, `HPC3_CHECKPOINT_STEPS`
and `HPC3_RESTART_COUNT` from its environment — enough to name its own
checkpoints and to know whether it is a first run or a requeue.

Directories are **derived from `root` + project, never passed in**. A caller
who can choose a log directory will eventually choose the wrong one, and that
job's output is then findable only by whoever remembers what was typed.

## The comment is live-only, and that is why the ledger exists

`--comment` does not reach accounting on HPC3:

```
AccountingStoreFlags    = (null)
```

Without `job_comment` in that list Slurm never stores it, so `sacct -o
Comment` returns empty for every job — measured 2026-08-23 against both a
finished CPU job and a GPU job from the day before. The README claimed
`sacct -o Comment` worked until that measurement; it never did on this
cluster.

Nothing in the package reads provenance back from the cluster, so no
behaviour depended on the wrong claim. That is the actual point: **the ledger
is the durable record precisely because the comment is not.** The comment is
a convenience for a human looking at a live queue.

## Start estimates are snapshots

A start estimate is a snapshot of the queue, not a reservation. A measured
3.4-hour estimate on this cluster started in 5 seconds.
