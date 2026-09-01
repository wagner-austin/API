# hpc3

Submit, watch and account for work on UCI's HPC3 Slurm cluster.

HPC3 is shared — 102 distinct users had jobs running when this package was
measured against it — and it is unforgiving in specific, learnable ways. This
package encodes what it cost to learn them, as rules that run rather than
documentation that has to be remembered. A job that would break one cannot be
constructed.

**This file is the command reference.** The design record — every rule's
incident, every measured fact's measurement — lives in
[`wiki/`](wiki/index.md): read `wiki/index.md` first, follow the hub for your
topic. New incident narrative goes into the wiki, not here.

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
hpc3-watch     --config runs/hpc3.json --job 1,2,3 --until-done 1   # follow to terminal, emitting state changes
hpc3-triage    --config runs/hpc3.json                         # is anything wrong that looks fine?
hpc3-chain     --config runs/hpc3.json --run runs/pipeline.json # stages, each after the last
hpc3-trace     --config runs/hpc3.json --match 07ab4976…       # which job trained this?
hpc3-cancel    --config runs/hpc3.json --job 55519937          # stop it, and say what actually stopped
```

Working examples of the documents are in [`examples/`](examples/).

---

## The workspace

Every command reads one workspace document. It is the only place the
cluster's address, the ledger, and each project's resources, caps and account
are written down.

| field | meaning |
| --- | --- |
| `cluster` | which measured machine's limits every rule is checked against ([facts are code](wiki/pages/facts-are-code.md)) |
| `host` | SSH destination — an alias from your `~/.ssh/config`, not a hostname |
| `root` | absolute cluster directory; scripts and logs are **derived** from it |
| `ledger` | local record of every submission; relative paths resolve against this file's directory |
| `quiet_seconds` | how long a running job may write nothing before triage calls it silent (default 1800) |
| `projects` | resource defaults, caps and charge account per body of work |
| `projects.<name>.budget` | that project's self-imposed caps, checked before submission and again while running ([budget model](wiki/pages/budget-model.md)) |
| `projects.<name>.repo` | where that project's code lives; relative paths resolve against this file's directory |

**There are no `--host` / `--root` / `--budget` / `--ledger` flags** — the
reasons are measured, not stylistic
([budget model](wiki/pages/budget-model.md)).

**Adding a project** is one entry in `projects` for any single-node job, GPU
or CPU (see `examples/` for complete workspaces). A project that requests a
GPU must declare an `image`
([image build flow](wiki/pages/image-build-flow.md)); `"gpu": null` is the
one spelling of CPU-only. Project names are lowercase letters, digits and
hyphens, at most 24 characters.

**Run, sweep and chain documents** say only what is specific to that work:
see [run documents](wiki/pages/run-documents.md),
[sweeps](wiki/pages/sweeps-and-artifacts.md) and
[chains](wiki/pages/chains.md). Preempted work is resumed by re-running the
sweep document as a campaign
([preemption and campaigns](wiki/pages/preemption-and-campaigns.md)).

---

## Partitions

Measured on UCI HPC3 (GPU partitions 2026-08-22, CPU 2026-08-23):

| partition | GPUs | bills | preemptible | max hours | per-user ceiling |
| --- | --- | --- | --- | --- | --- |
| `free-gpu` | V100, A30, A100 | no | yes | 72 | 24 GPUs |
| `free-gpu32` | L40S, RTX6000 | no | yes | 72 | 4 GPUs |
| `free` | — | no | yes | 72 | 3500 cores |
| `gpu` | V100, A30, A100 | **yes** | no | 336 | 40 GPUs |
| `gpu32` | L40S, RTX6000 | **yes** | no | 336 | 12 GPUs |
| `standard` | — | **yes** | no | 336 | 2500 cores |

**This package submits to the free three and refuses the other three.** How
billing was actually measured, why there is no `accept_billing` flag, and
which partitions are excluded outright:
[partitions and billing](wiki/pages/partitions-and-billing.md).

What the contract deliberately cannot express (multi-node, arrays, `--qos`,
`--constraint`): [unsupported shapes](wiki/pages/unsupported-shapes.md).

---

## Commands

| command | does |
| --- | --- |
| `hpc3-preflight --config C {--run R \| --sweep S}` | asks the scheduler whether it would start. Nothing is queued. |
| `hpc3-submit --config C --run R` | preflight → submit → record in the ledger |
| `hpc3-sweep --config C --run S` | the same, per member, recording each as it goes |
| `hpc3-campaign --config C --run S` | the same sweep document, run repeatedly: submits only the members that are neither finished nor already running |
| `hpc3-watch --config C --job ID[,ID…]` | state, elapsed, real cost, GPU-hours, state tally; one `sacct` call so a sweep's rows share one moment |
| `hpc3-watch … --until-done 1 [--poll-seconds N]` | re-reads accounting until EVERY requested job is terminal, emitting only state transitions, then the ordinary summary. Waits for ids accounting has not learned yet (a member absent from `sacct` is in flight, not done) and rules the ids wrong after five straight empty reads rather than spinning on a typo |
| `hpc3-triage --config C` | the [five conditions that look like health](wiki/pages/triage-conditions.md); exit 1 if any |
| `hpc3-trace --config C {--match V \| --job ID}` | which run produced a result, or what a job was; exit 1 if nothing matches |
| `hpc3-cancel --config C --job ID[,ID…]` | stops jobs and reports which were actually running — `scancel` is silent about one that had already finished |
| `hpc3-stage --config C --manifest M --source-dir D --expect-from R` | places files, verifies sha256 on both sides, and holds every digest against the published record ([staging identity](wiki/pages/staging-identity.md)) |
| `hpc3-chain --config C --chain H` | runs stages in order and stops at the first failure, so a broken stage does not feed the next |
| `hpc3-image-capture --out S …` | reads a live environment and writes the image spec. **Use this rather than writing a spec by hand** |
| `hpc3-image --spec S --out-dir D --image-name N` | renders the spec into definition, requirements, self-check and build script. Pure; builds nothing |
| `hpc3-image-build --config C --project P --name N --image-dir D --image-name I` | preflight → submit the rendered build → record it in the ledger ([why this command exists](wiki/pages/image-ledger-lessons.md)) |

Every command that could break something has a rule that refuses first:
[submission rules](wiki/pages/submission-rules.md). Identity checks —
environment pins, staged bytes, declared determinism, known answers — are
indexed from the [images-and-staging hub](wiki/hubs/images-and-staging.md).

---

## Exit statuses

| | |
| --- | --- |
| `0` | the command did what it was asked |
| `1` | it ran and the answer is negative — triage found something, `hpc3-trace` matched nothing |
| `2` | it **refused**; nothing was submitted, staged or run |

A refusal prints one line to stderr carrying its error code. Exactly one
place translates — `cli/_fatal.py`, at the process boundary — and it is
**typed**, not an `except Exception`. Anything else propagates with its
traceback intact, because anything else is a defect in this package rather
than a refusal by it, and a defect that prints one tidy line is a defect
nobody debugs.

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
cli/           entry points; argument reading and reporting only
wiki/          the design record: incidents, measurements, decisions
```

---

## Development

```bash
make check      # guards → ruff → mypy → pytest, 100% statements and branches
```

Held to the workspace standard: no `Any`, no casts, no `type: ignore`, no
`.pyi`, no mocks, no weak assertions, no fallbacks, no back-compat shims.
Every `TypedDict` has `encode`/`decode` with `require_*` validation; test
seams are `_test_hooks.py` DI, and the production hooks are exercised for
real rather than excluded from coverage.
