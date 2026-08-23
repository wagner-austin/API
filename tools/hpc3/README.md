# hpc3

Submit, watch and account for work on UCI's HPC3 Slurm cluster.

HPC3 is shared — 102 distinct users had jobs running when this package was
measured against it — and it is unforgiving in specific, learnable ways. This
package encodes what it cost to learn them, as rules that run rather than
documentation that has to be remembered. A job that would break one cannot be
constructed.

---

## Quick start

Three files, one directory:

```
runs/
  hpc3.json          the workspace: where the cluster is, what each project needs
  arm-b.json         one run: what is different about this job
  ledger.jsonl       written for you; every job ever submitted from this machine
```

```bash
hpc3-preflight --config runs/hpc3.json --run runs/arm-b.json   # would it start?
hpc3-submit    --config runs/hpc3.json --run runs/arm-b.json   # start it
hpc3-watch     --config runs/hpc3.json --job 55519937          # what is it doing, what did it cost
hpc3-triage    --config runs/hpc3.json                         # is anything wrong that looks fine?
hpc3-trace     --config runs/hpc3.json --match 07ab4976…       # which job trained this?
hpc3-cancel    --config runs/hpc3.json --job 55519937          # stop it, and say what actually stopped
```

Working examples of all three documents are in [`examples/`](examples/).

---

## The workspace

Every command reads one workspace document. It is the only place the cluster's
address, the ledger, the budget and each project's resources are written down.

```json
{
  "cluster": "hpc3",
  "host": "hpc3",
  "root": "/pub/wagnera3",
  "ledger": "ledger.jsonl",
  "quiet_seconds": 1800,
  "budget": { "max_gpu_hours": 120.0, "max_service_units": 0.0 },
  "projects": {
    "abl": {
      "partition": "free-gpu",
      "gpu": "A100",
      "gpu_count": 1,
      "cpus": 8,
      "mem_gb": 96,
      "minutes": 720,
      "requeue": true,
      "checkpoint_steps": 500,
      "accept_billing": false,
      "env_path": "/pub/wagnera3/envs/abl-pinned",
      "pinned_packages": { "torch": "2.6.0+cu124", "transformers": "4.46.3" },
      "deterministic": true
    }
  }
}
```

| field | meaning |
| --- | --- |
| `cluster` | which measured machine's limits every rule is checked against (see [Clusters](#clusters)) |
| `host` | SSH destination — an alias from your `~/.ssh/config`, not a hostname |
| `root` | absolute cluster directory; scripts and logs are **derived** from it |
| `ledger` | local record of every submission; relative paths resolve against this file's directory |
| `quiet_seconds` | how long a running job may write nothing before triage calls it silent (default 1800) |
| `budget` | self-imposed caps, checked before submission and again while running |
| `projects` | resource defaults per body of work |

**There are no `--host` / `--root` / `--budget` / `--ledger` flags.** That is
deliberate. When they existed, nothing tied `hpc3-triage --ledger` to the
ledger `hpc3-submit` had written — and pointing them at different paths gives
you either a clean board while jobs run unwatched, or every job reported as
`unaccounted` while nothing is wrong. Both readings are wrong and neither
looks wrong.

### Adding a project

Add one entry to `projects`. For a **single-node GPU workload** that is the
whole procedure — the LSTM work, covenant-radar's training and the transformer
ablation differ only in their resource line and their environment path:

```json
"turkic-lstm": {
  "partition": "free-gpu", "gpu": "V100", "gpu_count": 1,
  "cpus": 4, "mem_gb": 32, "minutes": 240,
  "requeue": true, "checkpoint_steps": 200,
  "accept_billing": false,
  "env_path": "/pub/wagnera3/envs/turkic",
  "pinned_packages": {},
  "deterministic": false
}
```

**Not every workload is that shape.** See
[What this cannot submit](#what-this-cannot-submit) before assuming a project
entry is all that stands between a workload and the cluster.

Project names are lowercase letters, digits and hyphens, at most 24 characters.

---

## Runs

A run document says only what is specific to this run:

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

Any project default may be restated to override it for this run alone:

```json
{
  "project": "abl", "name": "armC-full", "command": "python -u train.py --arm C",
  "minutes": 900, "checkpoint_steps": 250
}
```

Overriding is not a way around validation — the merged result goes through the
same decoder a fully hand-written spec would, so an override that lengthens a
preemptible run past an hour must also carry `requeue` and `checkpoint_steps`.

An unrecognised field is **refused, not ignored**. `"minute": 600` is a run its
author believes is capped at ten hours and that Slurm will kill at the project
default.

### Sweeps

```json
{
  "project": "abl", "name": "rung-large",
  "minutes": 900, "checkpoint_steps": 250,
  "members": [
    { "suffix": "armB-s0", "command": "python -u train.py --arm B --seed 0" },
    { "suffix": "armB-s1", "command": "python -u train.py --arm B --seed 1" }
  ]
}
```

`hpc3-sweep --config … --run …` submits each member and records each one as it
goes. There is no rollback: a member that fails leaves the earlier ones running
and findable, because a live job that is fine should not be cancelled for a
later job's failure.

---

## Identity: the bytes are the right ones, not just intact

Three checks in this package verify *transport* — that what arrived is what
left. Three others verify *identity* — that what left was the right thing in
the first place. The second kind exists because the first kind cannot catch a
run that completes, reports plausible numbers, and is comparable to nothing.

### The environment is the pinned one

`env_path` proves a directory exists. `pinned_packages` proves what is in it:
preflight runs that environment's own interpreter and holds it to the declared
versions.

This is not hypothetical. `/pub/wagnera3/envs/abl` and
`/pub/wagnera3/envs/abl-pinned` both exist, both pass an existence check, and
they differ by transformers 4.46.3 vs 5.15.1 and torch 2.6.0+cu124 vs
2.11.0+cu128. Seven characters in a path, a major version underneath, and a
McNemar comparison against published arms that silently means nothing.

Declaring `{}` is allowed and deliberate — a project whose payload is a
compiled binary has no Python packages to pin — but the field is required, so
"no pins" is an answer rather than an omission.

### The staged bytes are the published ones

```bash
hpc3-stage --config hpc3.json --manifest runs/stage.json \
    --source-dir runs/corpora --expect-from runs/file_ids.txt
```

A manifest is self-consistent by construction: whoever emitted the files
computed the digests from those same files, so they always agree. That proves
the emitter was deterministic and nothing else.

`--expect-from` is required and points at a record written by a *different act*
— every digest in the manifest must appear in it. That is a real check
precisely because re-emitting a corpus from the wrong source state produces new
digests, and new digests are not in the record. Any text works: a `sha256sum`
listing, a JSON manifest, a run log.

Every manifest also carries a required, non-empty `provenance` block:

```json
{
  "destination": "/pub/wagnera3/abl/corpora",
  "files": [{ "name": "armB.txt", "sha256": "…", "size_bytes": 41943040 }],
  "provenance": {
    "wiki_commit": "176bb8c",
    "emitter": "extraction-eval/emit_corpus.py",
    "emitter_flags": "--seed 0 --dilution oscar_en.txt --dilution-ratio 7.0"
  }
}
```

Free-form because what identifies a source differs per project, and a fixed
schema would mean writing `"none"` into fields that do not apply. It is the
record; `--expect-from` is the enforcement.

### The run's numerical determinism is declared and recorded

`deterministic` is required on every project, and it is not a quality setting —
it **partitions results**. Measured on this exact stack (RTX 3090 Ti, torch
2.6.0+cu124, transformers 4.46.3): two same-seed runs of a 6-layer model
diverge at the sixth significant figure of the loss without the controls, and
the deterministic loss is a *different number* from the nondeterministic one.
Runs on either side form separate records, and comparing across them measures
the setting rather than the thing under test.

So the posture travels with the run — into `--comment` as `det=on|off` and into
the ledger — and two arms that differ only in it can never be silently mixed.

The work itself is split, because only one half is a submitter's to do:

| half | who | why |
| --- | --- | --- |
| `CUBLAS_WORKSPACE_CONFIG` | **this tool**, in the batch script | cuBLAS reads it once when its handle is created; setting it after CUDA has started is accepted in silence and does nothing. Exported from the script it cannot be too late, and cannot be forgotten. |
| `torch.use_deterministic_algorithms(True)`, cuDNN and TF32 flags | **the payload** | they are torch calls in the payload's own process. This tool has no torch and does not pretend to make them. |

The payload reads `HPC3_DETERMINISTIC` (`0` or `1`, always exported) and applies
its half — `platform_ml.apply_determinism` does exactly this, and Model-Trainer
already calls it in `setup_env`. The two halves are safe to split because
PyTorch *enforces* the pairing: deterministic mode raises a `RuntimeError`
naming the missing variable, so a payload that does its half without the
launcher's half fails loudly rather than training quietly non-reproducible
numbers.

The variable's name and value are defined once, in
`platform_core.determinism_env`, and imported by both the trainer and this
submitter. A duplicated literal would be the worst kind: the copies drift,
nothing fails, and the runs stop being comparable.

### The result can be traced back to the run

```bash
$ hpc3-trace --config hpc3.json --match 07ab4976…
101 abl.armB-s42 submitted 2026-08-22T16:00:00+00:00
  arm=B corpus=07ab4976… seed=42
  logs /pub/wagnera3/abl/logs
1 of 6 recorded run(s) match
```

Exits 1 when nothing matches — a question with no answer is not an error, but
it must not read as "nothing was ever run".

---

## What the cluster sees

Jobs are not loose. Every one carries its project:

```
$ squeue -u wagnera3
   JOBID PARTITION       NAME     USER ST  TIME  NODES NODELIST
55519937  free-gpu abl.armB-s42 wagnera3  R  4:21      1 hpc3-gpu-16-02

$ scontrol show job 55519937 | grep Comment
   Comment=project=abl;gpu=A100x1;cpus=8;env=/pub/wagnera3/envs/abl-pinned
```

| | |
| --- | --- |
| job name | `<project>.<name>` — self-describing among 102 users' rows |
| `--comment` | project, hardware and environment, readable via `scontrol` and `sacct -o Comment` |
| scripts | `<root>/<project>/scripts/<project>.<name>.sbatch` |
| logs | `<root>/<project>/logs/<project>.<name>-<jobid>.{out,err}` |

The payload can read `HPC3_PROJECT`, `HPC3_JOB_NAME`, `HPC3_CHECKPOINT_STEPS`
and `HPC3_RESTART_COUNT` from its environment — enough to name its own
checkpoints and to know whether it is a first run or a requeue.

Directories are **derived from `root` + project, never passed in**. A caller
who can choose a log directory will eventually choose the wrong one, and that
job's output is then findable only by whoever remembers what was typed.

---

## Clusters

`cluster` selects a module under `src/hpc3/clusters/`. Each one holds facts
read off a real machine: partition names, GPU inventory, per-user QOS
ceilings, walltime caps, and each partition's `UsageFactor`. Every rule below
is asked of that module rather than of a constant, so pointing the workspace
at a different cluster changes what is enforced without changing any code that
enforces it.

**Facts are code, never configuration.** A workspace selects a cluster; it
cannot describe one. If `max_gpus_per_user` were a field you could write, then
writing `999` would not raise the ceiling — it would only disable the check
that predicts the pending job. Committing a cluster module is the act of
saying "these numbers were read off the real thing."

Currently measured: **`hpc3`** (UCI HPC3, 2026-08-22).

### Adding a cluster

1. Measure it: `sinfo`, `scontrol show partition`, `sacctmgr show qos`.
2. Write a module beside `clusters/hpc3.py` naming the source and the date.
3. Register it in `clusters/__init__.py`.

Nothing else changes. `test_cluster.py` drives the production decoders against
a synthetic cluster with different partition names, different GPUs, a
half-rate usage factor and much lower ceilings — that test is what keeps the
rules from quietly re-acquiring HPC3's values. A `CLUSTER_UNKNOWN` error lists
what has been measured; the tool never guesses a default, because submitting
to one machine under another machine's ceilings is worse than refusing.

This is a Slurm tool. `sbatch`, `sbatch --test-only`, `sacct`, `squeue` and
`scancel` are wired into `core/`, so PBS/Torque, LSF and Kubernetes are out of
scope rather than one module away.

---

## What this cannot submit

`JobSpec` describes **one single-node job that holds at least one named GPU**.
That shape is enforced, not defaulted: `gpu` is required and validated against
the cluster's inventory, and `render_sbatch` always emits
`--gres=gpu:<model>:<n>`. Everything below is therefore not a missing flag but
a shape the contract cannot express.

| shape | status | what it blocks |
| --- | --- | --- |
| **CPU-only job** | cannot be expressed — `gpu` is required | `cleargbm_rs` (Rust/LightGBM), SIRIUS and ZODIAC (JVM, CPU-parallel), any preprocessing or eval pass that needs no GPU |
| **Multi-node / MPI** | no `--nodes`, `--ntasks`, `srun` or `mpirun` anywhere | anything that does not fit one node |
| **Job array** | a sweep is N separate `sbatch` calls | a wide sweep is N ledger rows and N scheduler entries where `--array` would be one; correct, but heavier on the scheduler and on `squeue` |
| **Job dependency** | no `--dependency` | a staged pipeline — train → eval, SIRIUS → ZODIAC — must be driven by hand, one stage submitted after watching the previous finish |
| **`--constraint` / `--exclusive`** | not emitted | node features cannot be selected beyond the GPU model |

None of these are hard to add, and the cluster-facts layer already carries what
the checks would need. They are absent because they were never built, not
because they were judged wrong — recorded here so the gap is a decision rather
than a discovery.

---

## The rules that run

### Submission — checked when a run resolves

| rule | why |
| --- | --- |
| `PARTITION_UNKNOWN` — the partition exists on this cluster | a workspace written for another machine, or a typo; either way the job would be refused at submission or land somewhere unintended |
| `GPU_TYPE_UNPINNED` — the GPU model is named and the cluster carries it | a bare `--gres=gpu:1` on `free-gpu` is roughly a two-in-five chance of a V100, whose `sm_70` the pinned torch does not target; the failure reads as a bug in the training code |
| `PARTITION_GPU_MISMATCH` — that partition carries that model | Slurm leaves the job pending forever rather than rejecting it |
| `PARTITION_BILLS_WITHOUT_CONSENT` — a non-zero `UsageFactor` needs `accept_billing` | `free-gpu32` bills one service unit per core-hour despite its name |
| `ENV_PACKAGE_MISMATCH` — the environment contains what the project pinned | `envs/abl` and `envs/abl-pinned` both exist and differ by a transformers major version |
| `PREEMPTIBLE_RUN_UNPROTECTED` — long preemptible runs carry requeue **and** checkpointing | `PreemptMode=CANCEL` gives 60 seconds of grace; requeue without checkpoints restarts from step zero, which is not protection |
| `TIME_LIMIT_EXCEEDS_PARTITION` — the wall clock fits | rejected at submission otherwise |

### Sweeps — checked before anything is sent

`SWEEP_EXCEEDS_GPU_CEILING` / `SWEEP_EXCEEDS_JOB_CEILING`. Slurm does not
reject an oversized set; it queues the excess against `MaxTRESPU`, which reads
as a busy cluster and is not.

### Budget — ours, because nothing else says stop

The QOS bounds what runs *at once*. Nothing bounds the total, and on the free
partitions nothing bills — a 24-GPU three-day sweep is 1,728 GPU-hours, inside
every limit the cluster enforces, and not a reasonable share of a shared
machine. `BUDGET_PROJECTION_EXCEEDED` before submission,
`BUDGET_CONSUMPTION_EXCEEDED` from `hpc3-watch` while running.

### Preflight — non-skippable

`hpc3-submit` preflights unconditionally: it probes the environment, uploads
the real rendered script and runs `sbatch --test-only` on it by path. There is
no flag to skip it and no code path that reaches the cluster without it. The
same rendered file is then submitted, so preflight and submission cannot drift.

---

## Triage: the three conditions that look like health

`hpc3-triage` reconciles the ledger against the cluster and exits non-zero if
anything is found.

- **blocked** — pending on a reason that will never resolve. On HPC3, 261 of
  621 pending GPU jobs were sitting on `DependencyNeverSatisfied`; in `squeue`'s
  state column that is indistinguishable from waiting on `Resources`. A reason
  the allowlist has never seen is treated as blocked, because that is where
  patience costs a week.
- **unaccounted** — we recorded submitting it and accounting has never heard of
  it. No cluster-side query can find these: the evidence is the *absence* of a
  cluster-side record, which is what the local ledger exists to supply.
- **silent** — `RUNNING`, holding GPUs, and its log has stopped growing. Log
  age is measured against the cluster's own clock; a few minutes of skew would
  either invent staleness or hide it.

### Closures: why `unaccounted` doesn't rot

Two Slurm components forget finished jobs, on very different schedules.

`squeue` drops a job `MinJobAge` after it ends — **300 seconds on HPC3, read
from `scontrol show config`**. Past that, `squeue -j <id>` does not return
empty, it exits non-zero with `Invalid job id specified`. Triage therefore asks
the queue only about ids accounting reports as `PENDING`; nothing else can be
in it.

`sacct` retention depends on `slurmdbd`'s purge settings, which a login node
cannot read — and Slurm's default is to purge nothing, so on this cluster the
expiry is **unverified**. The closure record is built for it anyway: if a site
does enable `PurgeJobAfter`, every job past that window becomes a ledger entry
with no accounting row — the same observation as a job that never existed —
and triage would exit non-zero forever. A board that is always red is the same
as no board, and a closure costs one line per job.

So the moment accounting reports a terminal state, triage writes it to
`<ledger>.closed` and never asks about that job again. The closure is written
*after* the findings are built, so the run that closes a job still reports on
it. Failures close exactly as successes do — accounting forgets both on the
same schedule.

A job that vanished before triage ever saw it end has no closure and stays
reportable forever, which is correct: that is the case the finding exists for.
The corollary is that triage has to run at least once inside the retention
window for a job to close cleanly.

---

## Commands

| command | does |
| --- | --- |
| `hpc3-preflight --config C {--run R \| --sweep S}` | asks the scheduler whether it would start. Nothing is queued. |
| `hpc3-submit --config C --run R` | preflight → submit → record in the ledger |
| `hpc3-sweep --config C --run S` | the same, per member, recording each as it goes |
| `hpc3-watch --config C --job ID[,ID…]` | state, elapsed, real cost, GPU-hours, state tally; one `sacct` call so a sweep's rows share one moment |
| `hpc3-triage --config C` | the three conditions above; exit 1 if any |
| `hpc3-trace --config C {--match V \| --job ID}` | which run produced a result, or what a job was; exit 1 if nothing matches |
| `hpc3-cancel --config C --job ID[,ID…]` | stops jobs and reports which were actually running — `scancel` is silent about one that had already finished |
| `hpc3-stage --config C --manifest M --source-dir D --expect-from R` | places files, verifies sha256 on both sides, and holds every digest against the published record |

Start estimates are a snapshot of the queue, not a reservation. A measured
3.4-hour estimate on this cluster started in 5 seconds.

**`TIME_LIMIT_EXCEEDS_PARTITION` bounds a single attempt, not a total.**
free-gpu's 72 hours is per attempt and a requeue restarts that clock, so
nothing here caps cumulative wall time across requeues. The GPU-hour budget is
the only thing that does, and it projects from *requested* minutes — so a
requeued job can exceed its own projection. Watch it with `hpc3-watch`.

**`checkpoint_steps` is a declaration, not a verification.** The contract
requires a long preemptible run to carry it; nothing here can confirm the
training script honours it or that resume works, because a submitter cannot
know the trainer. Prove it with one real preempted arm — a synthetic test
cannot schedule its own preemption.

---

## Exit statuses

| | |
| --- | --- |
| `0` | the command did what it was asked |
| `1` | it ran and the answer is negative — triage found something, `hpc3-trace` matched nothing |
| `2` | it **refused**; nothing was submitted, staged or run |

A refusal prints one line to stderr carrying its error code:

```
$ hpc3-preflight --config hpc3.json --run runs/arm-b.json
ENV_PACKAGE_MISMATCH: /pub/wagnera3/envs/abl has torch==2.11.0+cu128, but this
project pins torch==2.6.0+cu124. A version difference under a published
comparison is a confound, not a detail.
$ echo $?
2
```

Exactly one place translates — `cli/_fatal.py`, at the process boundary — and
it is **typed**, not an `except Exception`. Three exception types become
messages, because each names something the operator did that the tool declined
to do. Anything else propagates with its traceback intact, because anything
else is a defect in this package rather than a refusal by it, and a defect that
prints one tidy line is a defect nobody debugs.

---

## Architecture

```
clusters/      measured facts, one module per real machine
  hpc3.py        UCI HPC3: partitions, GPUs, QOS ceilings, usage factors
contracts/     types + decode/encode + every validation rule
  cluster.py     what a cluster must declare; the one facts accessor
  workspace.py   the config document; project defaults
  run.py         merging a run onto its project's defaults
  job.py         a fully specified job, and the submission rules
  sweep.py       many jobs from one template, bounded by the QOS
  layout.py      project names, job labels, directory derivation
  pins.py        what an environment must contain
  provenance.py  where staged bytes came from
  experiment.py  what a run IS, as opposed to which queue row it held
  ledger.py  status.py  pending.py  preflight.py  budget.py  stage.py
core/          behaviour: rendering, ssh, parsing, submission, triage
  env_probe.py   asking an environment what is installed in it
  expected.py    holding a manifest against a record written before it
cli/           eight entry points; argument reading and reporting only
```

---

## Development

```bash
make check      # guards → ruff → mypy → pytest, 100% statements and branches
```

Held to the workspace standard: no `Any`, no casts, no `type: ignore`, no
`.pyi`, no mocks, no weak assertions, no fallbacks, no back-compat shims. Every
`TypedDict` has `encode`/`decode` with `require_*` validation; test seams are
`_test_hooks.py` DI, and the production hooks are exercised for real rather
than excluded from coverage.
