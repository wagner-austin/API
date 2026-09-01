---
title: Cluster facts are code, never configuration
tags: [cluster-facts, architecture]
related: [[partitions-and-billing]], [[unsupported-shapes]]
sources: [clusters/hpc3.py, clusters/__init__.py, tests/test_cluster.py, README.md@4dc63f17]
fact_checked: 2026-09-01
confidence: high
---

# Cluster facts are code, never configuration

`cluster` selects a module under `src/hpc3/clusters/`. Each one holds facts
read off a real machine: partition names, GPU inventory, per-user QOS
ceilings, walltime caps, and each partition's `UsageFactor`. Every rule is
asked of that module rather than of a constant, so pointing the workspace at
a different cluster changes what is enforced without changing any code that
enforces it.

**A workspace selects a cluster; it cannot describe one.** If
`max_gpus_per_user` were a field you could write, then writing `999` would
not raise the ceiling — it would only disable the check that predicts the
pending job. Committing a cluster module is the act of saying "these numbers
were read off the real thing."

Currently measured: **`hpc3`** (UCI HPC3 — GPU partitions 2026-08-22, CPU
partitions 2026-08-23). See [[partitions-and-billing]] for the numbers.

## Adding a cluster

1. Measure it: `sinfo`, `scontrol show partition`, `sacctmgr show qos`.
2. Write a module beside `clusters/hpc3.py` naming the source and the date.
3. Register it in `clusters/__init__.py`.

Nothing else changes. `test_cluster.py` drives the production decoders
against a synthetic cluster with different partition names, different GPUs, a
half-rate usage factor and much lower ceilings — that test is what keeps the
rules from quietly re-acquiring HPC3's values. A `CLUSTER_UNKNOWN` error
lists what has been measured; the tool never guesses a default, because
submitting to one machine under another machine's ceilings is worse than
refusing.

## Slurm only

`sbatch`, `sbatch --test-only`, `sacct`, `squeue` and `scancel` are wired
into `core/`, so PBS/Torque, LSF and Kubernetes are out of scope rather than
one module away.
