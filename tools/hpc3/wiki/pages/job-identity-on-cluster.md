---
title: What the cluster sees, and why the comment is live-only
tags: [cluster-facts, identity, ledger]
hubs: [cluster-facts, operations]
related: ["[[ledger-closures]]", "[[run-documents]]"]
source_paths:
  - "src/hpc3/contracts/layout.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/contracts/layout.py": "cb698fabbd6994fc9b6bc10092e77df3dc7f520e"
  "README.md": "104d4dac210676c88c42b5122146c11687239fc5"
provenance:
  - "AccountingStoreFlags = (null), measured 2026-08-23"
fact_checked: 2026-09-04
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

The payload can read `HPC3_JOB_NAME` from its environment — enough to name
its own output, which is what covenant-radar's optimizer does to keep
concurrent sweep members from overwriting one another's history files.

**Three siblings were removed on 2026-09-04, and the reason is worth more
than the names.** This page said the payload could also read `HPC3_PROJECT`,
`HPC3_CHECKPOINT_STEPS` and `HPC3_RESTART_COUNT` — "enough to name its own
checkpoints and to know whether it is a first run or a requeue". All three
were exported. None was read by any payload in this monorepo or in the LSTM
repository, at any point in the five weeks they existed. They were offers, and
the sentence above described the offer as though it were the wiring.

That is not a documentation slip on its own. `PREEMPTIBLE_RUN_UNPROTECTED`
admits a long preemptible run that declares a positive `checkpoint_steps`, on
the reasoning that such a run survives eviction, and the number reached the
job as `HPC3_CHECKPOINT_STEPS` — so the guard was satisfied by a value whose
only effect was satisfying the guard. `mi` declared 500 and `turkic-lstm`
declares 27,344 against payloads that checkpoint on schedules of their own and
have never seen either number.

`checkpoint_steps` is therefore an **operator assertion the tooling cannot
verify**, not a setting it applies, and nothing in the rendered script pretends
otherwise any more. The restart count still reaches the job log, read from
Slurm's own `SLURM_RESTART_COUNT` rather than re-exported under a name of ours.

When a payload does resume from a checkpoint, the export returns **in the same
change as its reader** — the declaration becomes legal at the moment it becomes
true. `tests/test_exported_env_readers.py` fails if an export is added without
one, and fails equally if a declared name stops being rendered.

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
