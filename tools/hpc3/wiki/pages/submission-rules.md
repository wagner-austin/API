---
title: The submission rules, each with the failure it refuses
tags: [submission, guards]
hubs: [submission]
related: ["[[budget-model]]", "[[preemption-and-campaigns]]", "[[partitions-and-billing]]"]
source_paths:
  - "src/hpc3/contracts/job.py"
  - "src/hpc3/contracts/preflight.py"
  - "src/hpc3/core/preflight.py"
  - "README.md"
source_git_blobs:
  "src/hpc3/contracts/job.py": "45f6be817460501c520ecca58b4f1dbc7341f4d0"
  "src/hpc3/contracts/preflight.py": "e72df28502ee022931ff38610f697c10fef0dbc0"
  "src/hpc3/core/preflight.py": "c642109e539eb3a0a7cfb97c6e44ff161fe7cb0a"
  "README.md": "c4cdcc31ae83beaede3c2635a943ddc0bcf0c083"
fact_checked: 2026-09-03
confidence: high
---

# The submission rules, each with the failure it refuses

Checked when a run resolves, before anything reaches the cluster:

| rule | why |
| --- | --- |
| `PARTITION_UNKNOWN` — the partition exists on this cluster | a workspace written for another machine, or a typo; either way the job would be refused at submission or land somewhere unintended |
| `GPU_TYPE_UNPINNED` — a GPU request names its model and the cluster carries it | a bare `--gres=gpu:1` on `free-gpu` is roughly a two-in-five chance of a V100, whose `sm_70` the pinned torch does not target; the failure reads as a bug in the training code |
| `PARTITION_GPU_MISMATCH` — the partition and the request agree, **both ways** | a GPU on a CPU partition pends forever; no GPU on a GPU partition *runs*, holding a card it never touches, so only this catches it |
| `PARTITION_BILLS` — the partition's `UsageFactor` is zero | this package submits free work only; `standard` is the default partition and charges, so the partition is required and never defaulted ([[partitions-and-billing]]) |
| `DEPENDENCY` fields — a wait names real, distinct, numeric job ids | a typo'd id is not a slow job, it is a job that never existed, and under `--kill-on-invalid-dep` that cancels the dependent stage at once |
| `ENV_PACKAGE_MISMATCH` — the environment contains what the project pinned | `envs/abl` and `envs/abl-pinned` both exist and differ by a transformers major version ([[environment-pins]]) |
| `PREEMPTIBLE_RUN_UNPROTECTED` — long preemptible runs carry requeue paired with **either** checkpointing **or** deterministic replay | `PreemptMode=CANCEL` gives 60 seconds of grace; requeue alone restarts a stochastic run from step zero as a DIFFERENT run, which is not protection. Deterministic replay qualifies because the restart reproduces the same run seed-for-seed (`src/hpc3/contracts/job.py:437`) |
| `TIME_LIMIT_EXCEEDS_PARTITION` — the wall clock fits | rejected at submission otherwise. Bounds a single attempt, not a total: a requeue restarts the clock, and only the GPU-hour budget caps the cumulative spend ([[budget-model]]) |

## Preflight is non-skippable

`hpc3-submit` preflights unconditionally: it probes the environment, uploads
the real rendered script and runs `sbatch --test-only` on it by path. There is
no flag to skip it and no code path that reaches the cluster without it. The
same rendered file is then submitted, so preflight and submission cannot
drift.

## checkpoint_steps is a declaration, not a verification

The contract requires a long preemptible run to carry it; nothing here can
confirm the training script honours it or that resume works, because a
submitter cannot know the trainer. Prove it with one real preempted arm — a
synthetic test cannot schedule its own preemption.
