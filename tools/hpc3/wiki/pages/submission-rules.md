---
title: The submission rules, each with the failure it refuses
tags: [submission, guards]
hubs: [submission]
related: ["[[budget-model]]", "[[preemption-and-campaigns]]", "[[partitions-and-billing]]"]
source_paths:
  - "src/hpc3/contracts/job.py"
  - "src/hpc3/contracts/preflight.py"
  - "src/hpc3/core/preflight.py"
  - "src/hpc3/core/code_claim.py"
  - "src/hpc3/contracts/code_claim.py"
  - "src/hpc3/core/interpreter.py"
source_git_blobs:
  "src/hpc3/contracts/job.py": "9defe2ae9f5f195b2f7d466765669df7ff1977c5"
  "src/hpc3/contracts/preflight.py": "e72df28502ee022931ff38610f697c10fef0dbc0"
  "src/hpc3/core/preflight.py": "84695824400bf77e8994327845bd48e55e43ea3f"
  "src/hpc3/core/code_claim.py": "b7227746e7b2574728de89fc55a9bd63adc7e207"
  "src/hpc3/contracts/code_claim.py": "1de219ba95176a3e343c9c319cb59a7b8841cae7"
  "src/hpc3/core/interpreter.py": "44152c39c9674c78771a2f01288110c0438c5617"
fact_checked: 2026-10-08
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
| `PREEMPTIBLE_RUN_UNPROTECTED` — a long run on a partition that preempts declares work that survives eviction: **either** `resumes_from_checkpoint` **or** `deterministic` replay. Under `PreemptMode=REQUEUE` it must also carry `requeue` | Restarting a stochastic run from step zero is a DIFFERENT run, which is not protection. Deterministic replay qualifies because the restart reproduces the same run seed-for-seed. `requeue` is required only where Slurm honours it: under `CANCEL` the flag is inert (22 array tasks carrying it went straight to terminal `PREEMPTED`, 2026-09-02) and the campaign is the resubmission (`src/hpc3/contracts/job_rules.py`) |
| `TIME_LIMIT_EXCEEDS_PARTITION` — the wall clock fits | rejected at submission otherwise. Bounds a single attempt, not a total: a requeue restarts the clock, and only the GPU-hour budget caps the cumulative spend ([[budget-model]]) |

## Preflight is non-skippable

`hpc3-submit` preflights unconditionally: it probes the environment, uploads
the real rendered script and runs `sbatch --test-only` on it by path. There is
no flag to skip it and no code path that reaches the cluster without it. The
same rendered file is then submitted, so preflight and submission cannot
drift.

Re-read 2026-10-08, preflight refuses two more things before anything is
uploaded. Since 2026-10-05 (`3b76ce978`, board task `2cca4a98`) a run whose
experiment declares a `repo_commit` is refused `REPO_COMMIT_NOT_HEAD` when that
commit is not the HEAD of the cluster checkout its `repo_tree` names, asked
over ssh in one probe and classified with `git merge-base --is-ancestor`; the
gate is equality because a run executes the checkout's HEAD, a merge-base exit
other than 0 or 1 is `REPO_COMMIT_PROBE_UNREADABLE` rather than "not reachable",
and a `repo_commit` without its `repo_tree` is refused when the document is
decoded. And
since the same day (`f5df5fd93`) every environment probe opens with the
interpreter's identity, so an environment running another environment's
interpreter is refused `ENV_INTERPRETER_BORROWED` even when its package list
matches ([[interpreter-availability]]).[^preflight]

## resumes_from_checkpoint is a declaration, not a verification

The contract requires a long run on a preempting partition to carry it or
`deterministic`; nothing here can confirm the training script honours it or
that resume works, because a submitter cannot know the trainer. Prove it with
one real preempted arm — a synthetic test cannot schedule its own preemption.

That proof now exists. The 1.5B extraction-ablation rung took ten preemptions
across six arms on `free-gpu` (2026-09-07/08, `sacct`), and every resubmission
loaded a mid-run checkpoint rather than restarting: `training_checkpoint_loaded`
against first progress lines at epoch 17, 16, 14 and 7 of 20. What the guard
cannot check, a preempted arm can.

The field was an integer step cadence until 2026-09-09. Nothing read the
number — not in this monorepo and not in the separate repository that declared
the largest one — so it asserted a schedule no payload applied. A boolean says
what was meant.

[^preflight]: `src/hpc3/core/preflight.py` § the `preflight` docstring's Raises list, naming `ENV_INTERPRETER_BORROWED`, `REPO_COMMIT_NOT_HEAD` and `REPO_COMMIT_PROBE_UNREADABLE`; `src/hpc3/core/code_claim.py`, which raises `REPO_COMMIT_NOT_HEAD` with `explain_mismatch(claim, reading)`; `src/hpc3/contracts/code_claim.py` § `REPO_TREE_KEY` and its decoder; `src/hpc3/core/interpreter.py` § `check_interpreter_home`. API commits `3b76ce978` and `f5df5fd93` (2026-10-05).
