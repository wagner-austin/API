---
title: Run documents say what is specific to this run
tags: [submission, contracts]
hubs: [submission]
related: ["[[sweeps-and-artifacts]]", "[[submission-rules]]"]
source_paths:
  - "src/hpc3/contracts/run.py"
  - "src/hpc3/contracts/experiment.py"
  - "src/hpc3/contracts/code_claim.py"
  - "src/hpc3/core/code_claim.py"
  - "runs/sweep-cleargbm-p6-rung5.json"
source_git_blobs:
  "src/hpc3/contracts/run.py": "8e526ae04061ef147e61c3afe2d70246f1048665"
  "src/hpc3/contracts/experiment.py": "530e8484b421d13119e951fef3ed8ea8b2706abf"
  "src/hpc3/contracts/code_claim.py": "1de219ba95176a3e343c9c319cb59a7b8841cae7"
  "src/hpc3/core/code_claim.py": "b7227746e7b2574728de89fc55a9bd63adc7e207"
  "runs/sweep-cleargbm-p6-rung5.json": "046840c974fd8fc473e316173e8985f8eaaa6dda"
provenance:
  - "rung 5's cluster facts, read 2026-10-05 over ssh hpc3: `git reflog` in /pub/wagnera3/api ends at `80221ea HEAD@{2026-08-25 01:30:07 -0700}: pull --ff-only`; `git rev-parse --verify --quiet 20d9159^{commit}` there resolves nothing; sacct gives jobs 55571926 and 55571932 submitted 03:05:24 and 03:05:54, ended 03:17:20 and 03:24:46; ledger rows 55571926-55571942 carry image_digest null. The gate's live answer the same day: 80221ea admitted, 20d9159 and 2e8b8e1 refused with REPO_COMMIT_NOT_HEAD."
fact_checked: 2026-10-05
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

## A declared commit is checked against the checkout that runs it

Two keys in `experiment` are not free-form. A run that declares
`repo_commit` must also declare `repo_tree`, the absolute cluster path of the
checkout its payload runs from, or the document does not decode; and
preflight then asks that checkout whether its HEAD IS the declared commit,
refusing with `REPO_COMMIT_NOT_HEAD` when it is not
(`src/hpc3/core/code_claim.py`, wired into both `preflight` and
`array_preflight`).

The incident (board task `2cca4a98`): `sweep-cleargbm-p6-rung5.json`
declared `20d9159`, the workstation's HEAD when it was submitted on
2026-08-25, while all eight members ran from `/pub/wagnera3/api`, whose last
reflog entry put it at `80221ea` at 01:30:07 PDT, ninety-five minutes
before the 03:05 submission. That clone had never fetched `20d9159` at all. The
other five P6 rungs each declared the commit their checkout was at, which
is why one wrong declaration went unnoticed until 2026-09-10: a reader who spot
checks a field that is right five times out of six has no reason to check
the sixth.

The gate is EQUALITY, not `git merge-base --is-ancestor`. A run executes the
checkout's HEAD; ancestry would also admit a commit the checkout has moved
past. Ancestry is still asked, for the refusal's message: exit 0 says the
checkout moved on, exit 1 says the declared commit is code the checkout does
not contain, and any other exit is reported as git declining to answer and
never as a "no". A declared name the checkout cannot resolve is rung 5's case
and is refused as such.

It cannot be a CI check. The checkout lives on the cluster, CI cannot see it,
and it moves with every `git pull` there, so the claim is true or false only
at the instant of submission. What CI does check is the decode half: every
committed document declaring `repo_commit` also names its tree, so the claim
it makes is one preflight can check.

For an imaged run, `repo_tree` covers what the payload reads from the cluster
filesystem (the cleargbm sweeps' `scripts.optimize`), not the image's own
wheels, whose commit is the image spec's `git_commit`.

## Overrides go through the same decoder

Any project default may be restated to override it for this run alone:

```json
{
  "project": "abl", "name": "armC-full", "command": "python -u train.py --arm C",
  "minutes": 900, "resumes_from_checkpoint": true
}
```

Overriding is not a way around validation — the merged result goes through the
same decoder a fully hand-written spec would, so an override that lengthens a
run past an hour on a partition that preempts must declare work that survives
eviction: `resumes_from_checkpoint` or `deterministic`. Under
`PreemptMode=REQUEUE` it must also carry `requeue`; under `CANCEL` that flag is
inert and is not demanded ([[preemption-and-campaigns]]).

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

## exclude_nodes is per submission, for the same reason

```json
{ "project": "mi-cu128", "name": "floorfull-cu128-rtx6000",
  "exclude_nodes": ["hpc3-gpu-n54-00", "hpc3-gpu-n54-01"], ... }
```

Added 2026-09-14 after those two nodes sat IDLE with no reason, advertising
`gpu:RTX6000:4`, while the device an allocation on them bound (`0000:71:00.0`,
GPU 0, the one an idle node hands out first) answered `nvidia-smi` with
"Unable to determine the device handle: Unknown Error". Nineteen tasks died in
under ten seconds and the scheduler would have placed the resubmission on the
same nodes. It renders as `#SBATCH --exclude=` on both the single-job and the
array script, from one helper so the two cannot spell it differently; a sweep's
members and a chain's stages inherit it. A blank entry, a repeat, or anything
that is not a list of names is refused. A project default would be the wrong
shape for the same reason `depends_on` cannot be one: a node fault is a fact
about today's cluster, and a default would keep routing around a node long
after it was repaired.
