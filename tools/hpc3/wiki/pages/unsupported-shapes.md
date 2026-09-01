---
title: What this package cannot submit, as decisions rather than discoveries
tags: [submission, scope]
related: [[chains]], [[submission-rules]], [[facts-are-code]]
sources: [contracts/job.py, README.md@4dc63f17]
fact_checked: 2026-09-01
confidence: high
---

# What this package cannot submit, as decisions rather than discoveries

`JobSpec` describes **one single-node job**, with GPUs or without. Everything
below is not a missing flag but a shape the contract cannot express.

| shape | status | what it blocks |
| --- | --- | --- |
| **Multi-node / MPI** | no `--nodes`, `--ntasks`, `srun` or `mpirun` anywhere | anything that does not fit one node |
| **Job array** | a sweep is N separate `sbatch` calls | a wide sweep is N ledger rows and N scheduler entries where `--array` would be one; correct, but heavier on the scheduler and on `squeue` |
| **Explicit `--qos`** | not emitted; the cluster auto-selects | `standard-hbm` on HPC3, which refuses the default QOS with `Invalid qos specification` |
| **`--constraint` / `--exclusive`** | not emitted | node features cannot be selected beyond the GPU model |

None of these are hard to add, and the cluster-facts layer already carries
what the checks would need. They are absent because they were never built, not
because they were judged wrong — recorded so the gap is a decision rather than
a discovery.

## Two things left this list

**Job dependencies** were on it and are not any more — see [[chains]].

**CPU-only** was on it and is not any more. `"gpu": null` on a CPU partition
submits, so `cleargbm_rs`, SIRIUS and ZODIAC are reachable. One caveat worth
stating: `pinned_packages` verification runs the environment's own
`bin/python`, and an empty pin map makes **no round trip at all**. A JVM
project is therefore submittable while getting only `test -d` on its
environment — the weakest guarantee here, and exactly the "both paths exist,
both pass, the results aren't comparable" failure the pin check was built for
([[environment-pins]]).
