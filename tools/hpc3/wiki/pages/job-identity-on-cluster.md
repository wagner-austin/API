---
title: What the cluster sees, and why the comment is live-only
tags: [cluster-facts, identity, ledger]
hubs: [cluster-facts, operations]
related: ["[[ledger-closures]]", "[[run-documents]]"]
source_paths:
  - "README.md"
  - "src/hpc3/core/sbatch.py"
  - "src/hpc3/cli/submit.py"
  - "src/hpc3/cli/chain.py"
  - "src/hpc3/contracts/job.py"
  - "src/hpc3/contracts/job_rules.py"
  - "src/hpc3/contracts/preflight.py"
  - "src/hpc3/cli/preflight.py"
  - "tests/test_exported_env_readers.py"
source_git_blobs:
  "README.md": "6d14fcebcfb6a6eb102b927d5855953bcbb192b2"
  "src/hpc3/core/sbatch.py": "bf884f896e0eaaeafd595a034030368467f04026"
  "src/hpc3/cli/submit.py": "bb14dad3f27e50ad5331a1e3dc295429b420684f"
  "src/hpc3/cli/chain.py": "e5dcc40ed100ffa62c4ecda4dbc08ca2f0c86906"
  "src/hpc3/contracts/job.py": "9defe2ae9f5f195b2f7d466765669df7ff1977c5"
  "src/hpc3/contracts/job_rules.py": "9e417bd97f7ea4a62543b8299217e7401e88d7f0"
  "src/hpc3/contracts/preflight.py": "a3803d6b7ab3f3f9de77343d92dcd056ec86f111"
  "src/hpc3/cli/preflight.py": "4bb9e9bcbe89d6b04cf3c531beb67601b94f3776"
  "tests/test_exported_env_readers.py": "eec447f00167b7ed32c5c5861531431809de2887"
provenance:
  - "AccountingStoreFlags = (null), measured 2026-08-23"
  - "platform_core.cluster_layout (libs/platform_core, outside this workspaceRoot)"
  - "API commits c48c3dc59 (2026-08-22, the exports added), 5a52640ba (2026-09-07, three exports removed and test_exported_env_readers.py added), 1106e49af (2026-09-09, checkpoint_steps renamed) and 4ae012649 (2026-10-01, cluster_layout moved to platform_core)"
  - "API services/covenant-radar-api/scripts/optimize/history.py and src/covenant_radar_api/worker/_optimize_common.py read HPC3_JOB_NAME; outside this workspaceRoot"
fact_checked: 2026-11-04
confidence: high
---

# What the cluster sees, and why the comment is live-only

Jobs are not loose. Every one carries its project; the 102-user figure is
the README's reading from when this package was first measured against the
cluster:[^layout][^shared]

| | |
| --- | --- |
| job name | `<project>.<name>` — self-describing among 102 users' rows |
| `--comment` | project, hardware and environment, readable via `scontrol show job <id>` or `squeue -o %k` **while the job is live** |
| scripts | `<root>/<project>/scripts/<project>.<name>.sbatch` |
| logs | `<root>/<project>/logs/<project>.<name>-<jobid>.{out,err}` |

The names and directories are derived in `platform_core.cluster_layout`, not
in this package, since 2026-10-01: a payload placing its results under
`<root>/<project>` computes that directory with the same function this
package submits into, without its compute image carrying the submitter.[^layout]

The payload can read `HPC3_JOB_NAME` from its environment — enough to name
its own output, which is what covenant-radar's optimizer does to keep
concurrent sweep members from overwriting one another's history files.[^jobname]

**Three siblings were removed on 2026-09-07, and the reason is worth more
than the names.** This page said the payload could also read `HPC3_PROJECT`,
`HPC3_CHECKPOINT_STEPS` and `HPC3_RESTART_COUNT` — "enough to name its own
checkpoints and to know whether it is a first run or a requeue". All three
were exported. None was read by any payload in this monorepo in the sixteen
days they existed, from API c48c3dc59 (2026-08-22) to 5a52640ba (2026-09-07).
This page once said the same of the LSTM repository; that repository is not
checked out where this was re-read on 2026-11-04, so that half is no longer
asserted. They were offers, and the sentence above described the offer as
though it were the wiring.[^removed]

That is not a documentation slip on its own. `PREEMPTIBLE_RUN_UNPROTECTED`
admitted a long preemptible run that declared a positive `checkpoint_steps`,
on the reasoning that such a run survives eviction, and the number reached
the job as `HPC3_CHECKPOINT_STEPS` — so the guard was satisfied by a value
whose only effect was satisfying the guard. `mi` declared 500 and
`turkic-lstm` declared 27,344 until the 2026-09-09 rename below, against
payloads that checkpoint on schedules of their own and never saw either
number.[^removed][^renamed]

The declaration is therefore an **operator assertion the tooling cannot
verify**, not a setting it applies, and nothing in the rendered script pretends
otherwise any more. The restart count still reaches the job log, read from
Slurm's own `SLURM_RESTART_COUNT` rather than re-exported under a name of ours.[^restart]

The field itself was renamed on 2026-09-09. `checkpoint_steps: int` became
`resumes_from_checkpoint: bool`, because an integer cadence implies a
configured interval and this one configured nothing: monorepo-wide it appeared
in the guard and the type feeding it, and the guard only ever compared it
`> 0`. A boolean states the claim being made. 389 of the 392 committed
documents declared `0` and asserted nothing at all.[^renamed]

When a payload does resume from a checkpoint, the export returns **in the same
change as its reader** — the declaration becomes legal at the moment it becomes
true. `tests/test_exported_env_readers.py` fails if an export is added without
one, and fails equally if a declared name stops being rendered.[^readers]

Directories are **derived from `root` + project, never passed in**. A caller
who can choose a log directory will eventually choose the wrong one, and that
job's output is then findable only by whoever remembers what was typed.[^derived]

## The comment is live-only, and that is why the ledger exists

`--comment` does not reach accounting on HPC3, read on 2026-08-23:[^comment]

```
AccountingStoreFlags    = (null)
```

Without `job_comment` in that list Slurm never stores it, so `sacct -o
Comment` returns empty for every job — measured 2026-08-23 against both a
finished CPU job and a GPU job from the day before. The README claimed
`sacct -o Comment` worked until that measurement; it never did on this
cluster.[^comment]

Nothing in the package reads provenance back from the cluster, so no
behaviour depended on the wrong claim. That is the actual point: **the ledger
is the durable record precisely because the comment is not.** The comment is
a convenience for a human looking at a live queue.[^comment]

## Start estimates are snapshots

A start estimate is a snapshot of the queue, not a reservation. A measured
3.4-hour estimate on this cluster started in under 5 seconds.[^estimate]

[^layout]: `src/hpc3/core/sbatch.py:344` `render_sbatch` and `:149` `job_comment`; `src/hpc3/cli/submit.py:132` `log_dir`, called as `log_dir(root, project)`; `src/hpc3/cli/chain.py` imports `log_dir` and `script_dir` from `platform_core.cluster_layout`. API `libs/platform_core/src/platform_core/cluster_layout.py` functions `qualified_name`, `script_dir` and `log_dir` lie outside this wiki's workspaceRoot; they moved there in API 4ae012649 on 2026-10-01.
[^shared]: `README.md:6` § "hpc3", the dated reading of 102 distinct users with running jobs.
[^jobname]: `src/hpc3/core/sbatch.py:413` `HPC3_JOB_NAME`; API `services/covenant-radar-api/scripts/optimize/history.py` and `src/covenant_radar_api/worker/_optimize_common.py` read it, outside this wiki's workspaceRoot.
[^removed]: API commit 5a52640ba (2026-09-07), whose message records that nothing in this monorepo read `HPC3_CHECKPOINT_STEPS` and removes the three inert exports from `sbatch.py` and `array_sbatch.py`; `git log -S HPC3_RESTART_COUNT` dates the exports' first commit as c48c3dc59 (2026-08-22).
[^renamed]: API commit 1106e49af (2026-09-09), item 3 of its message: the rename, the guard's only `> 0` comparison, 389 of 392 documents at 0, and `turkic-lstm` the one positive value (27,344 in `runs/hpc3-turkic-lstm.json` before that commit). `src/hpc3/contracts/job.py:176` `resumes_from_checkpoint`; `src/hpc3/contracts/job_rules.py:180` `_check_preemption_protection`, which raises `PREEMPTIBLE_RUN_UNPROTECTED`.
[^restart]: `src/hpc3/core/sbatch.py:438` `SLURM_RESTART_COUNT`, the log line; `:81` `EXPORTED_VARIABLES`, which declares no checkpoint export.
[^readers]: `tests/test_exported_env_readers.py:119` `test_every_rendered_export_is_declared`, `:128` `test_every_declared_export_is_rendered` and `:139` `test_no_exported_variable_is_inert`.
[^derived]: `src/hpc3/cli/submit.py:132` `log_dir`, derived from the workspace root and the project rather than taken from the caller; `src/hpc3/contracts/job.py:77` `JobSpec` carries no log or script directory field (its one path, `env_path`, names the environment).
[^comment]: `src/hpc3/core/sbatch.py:149` `job_comment`, whose docstring records the 2026-08-23 measurement (`AccountingStoreFlags` unset, `sacct -o Comment` empty for every job), that the docstring had claimed otherwise before it, and that nothing in the package reads provenance back from the cluster.
[^estimate]: `src/hpc3/contracts/preflight.py:33` `PreflightResult`, whose `start_estimate` attribute records the 3.4-hour estimate that started in under 5 seconds; `src/hpc3/cli/preflight.py:82` `main` prints that estimates are a snapshot of the queue, not a reservation.
