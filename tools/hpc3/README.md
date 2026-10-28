# hpc3

Submit, watch and account for work on UCI's HPC3 Slurm cluster and run
digest-bound Python experiments on WRDS Cloud's Grid Engine cluster.

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

### WRDS repeatable experiments

Use the existing `wrds` SSH alias with this package installed locally.
The maintained worker is uploaded automatically; WRDS needs no hpc3 install.
First measure the interpreter and installed package versions:

```bash
python -m hpc3.cli.wrds inspect --host wrds \
  --root /home/uci/awagner14/wrds-experiments/tools \
  --python /usr/local/bin/python3 > runtime.json
```

Write an `experiment.json` with exactly these fields, inserting the complete
object from `runtime.json` and the SHA256 of each source/input file:

```json
{
  "project": "my-research",
  "name": "baseline",
  "host": "wrds",
  "root": "/home/uci/awagner14/wrds-experiments",
  "runtime": {
    "python": "/usr/local/bin/python3",
    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "version": "3.14.5",
    "packages": {}
  },
  "slots": 1,
  "memory_gb_per_slot": 2,
  "seconds": 600,
  "seed": 42,
  "code": [{"path": "main.py", "name": "main.py", "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}],
  "inputs": [],
  "entrypoint": "main.py",
  "arguments": [],
  "outputs": ["metrics.json"]
}
```

The example hashes are placeholders; replace the entire runtime object with
the measured one, including every installed package. `code.path` resolves
against the specification directory; `code.name` is
its staged relative path. Each input has the same three fields, but its
`path` is an existing absolute WRDS path. Pin a research extract before
running experiments against it: an unversioned SQL query is not a frozen
dataset. All declared input contents are checked on the compute node before
and after execution. The specification identity covers code/input hashes,
runtime pins, arguments, seed, resources and declared outputs.

The payload receives `WRDS_SEED`, `WRDS_OUTPUT_DIR`, and `WRDS_INPUTS_JSON`
(an object mapping input names to their absolute paths). It must seed its
own RNGs from `WRDS_SEED` and write every declared output beneath
`WRDS_OUTPUT_DIR`. `metrics.json` must be a nonempty list of finite named
observations, for example `[{"name":"validation_loss","value":0.125}]`.
Python hashing and common CPU thread-count variables are set by the launcher.

```bash
python -m hpc3.cli.wrds preflight --spec experiment.json
python -m hpc3.cli.wrds submit --spec experiment.json --attempt first
python -m hpc3.cli.wrds status --spec experiment.json --attempt first
python -m hpc3.cli.wrds collect --spec experiment.json --attempt first --output results-first
python -m hpc3.cli.wrds submit --spec experiment.json --attempt repeat
python -m hpc3.cli.wrds trace --spec experiment.json --attempt first
python -m hpc3.cli.wrds cancel --spec experiment.json --attempt repeat
```

`preflight` checks local code, remote runtime and input readability. Submit
additionally asks Grid Engine to validate the exact resource request, creates
an exclusive attempt directory, stages and verifies the bytes, submits held,
persists the job ID, then releases it. After a lost submission response,
inspect `status`; use `resume --spec experiment.json --attempt first` only
if it is held. Reusing an attempt never submits a second job. A partially
staged attempt without a job receipt requires inspection and a new attempt
label; `resume` does not manufacture a job ID.

Queue absence is `accounting-pending` until accounting answers; only a zero
scheduler failure and zero process exit count as `completed`. Collection
also requires the worker's matching completion record and verifies each
transferred file. A failed transfer leaves no local completion marker; retry
into a new output directory. Results include the shared
`results.runrecord.json`, so existing research comparison code can consume
them. Lifecycle events go to `wrds-ledger.jsonl` beside an external spec or
at the shared main checkout for repository specs.

```bash
python -m hpc3.cli.wrds report --spec experiment.json --attempt first --report-task TASK_UUID
```

`report` measures one attempt on WRDS and posts the reading on board task
`TASK_UUID` as `load-report-<first eight characters>`, the identity the WRDS
inventories also post under, so an audit selects the program's post rather
than a session's restatement. The post names the job, the experiment identity,
every code and input sha256, the interpreter digest and package-pin digest,
and the scheduler state. A terminal job adds qacct's execution `hostname`,
`failed` and `exit_status`. A completed job adds each attested output's sha256
and the sha256 of the `results.runrecord.json` bytes collect writes, so two
attempts match exactly when those lines do. Any other job adds the
`WRDS_NOT_COMPLETE` refusal collect gives and whether `completion.json` is
present or absent on WRDS. A completed job whose attestation does not match
fails the command with collect's own refusal, and posts nothing.

The backend currently supports individual Python CPU jobs, with conservative
limits of eight slots, 48 GiB total and seven days. It pins interpreter bytes
and installed package **versions**, not an immutable container or every
library byte. A changed runtime fails closed; inspect and deliberately
update the specification to adopt it. A seed alone does not make arbitrary
payloads deterministic. UCI's Slurm sweeps, chains and container workflow
remain separate commands. SSH authentication must be valid when commands
connect; submitted jobs continue if the client disconnects.

### Bounded CRSP extraction

`python -m hpc3.cli.wrds bundle --output extract.pyz` saves the maintained
worker as a portable payload and prints its SHA256. Declare those bytes in
a normal WRDS run's `code`, set `entrypoint` to `extract.pyz`, and set
`arguments` to `["extract", "<selection JSON>"]`. The selection has exactly
`source_table` (`crsp.dsf` or `crsp_a_stock.dsf`), positive `security_id`,
inclusive `start_date` and `end_date` in `YYYY-MM-DD`, and `max_rows`
(3–10,000). The end must precede the extraction day. Use the ordinary
inspect/preflight/submit/status/collect/report lifecycle above.

Declare `data.csv`, `extract.sql`, `extraction.json` and `metrics.json` as
outputs. The extraction uses a read-only repeatable-read transaction with a
120-second statement timeout. It requests one extra row and refuses a result
over the declared cap; it never silently truncates. Missing return values
remain in the CSV for the evaluation adapter to reject explicitly. The receipt
records the scope, start/completion timestamps, row count and both file hashes;
the enclosing worker attests them with its existing completion record.

Collect into private storage. Prepare the existing `covenant_ml.evaluation.wrds`
manifest using its encoder and the receipt's file names, hashes and extraction
time, plus explicitly chosen split fractions and embargo. Stage the manifest,
CSV and query using `hpc3-stage`, then run the existing paired evaluation with
`--wrds-manifest`. Replays use the same saved snapshot; another live extraction
has its own timestamp and identity even when its rows match. The timestamp
records when data were acquired, not when every historical value became known.

### Verify a benchmark replay

```bash
python -m hpc3.cli.execution_report --config runs/hpc3.json \
  --job FIRST_JOB --repeat SECOND_JOB --checksums /private/inputs/checksums.txt \
  --report-task FULL_TASK_UUID
```

This reads both submissions from the shared ledger, queries native Slurm
accounting, and requires completed zero-exit allocations with matching names,
partitions and known execution times. It downloads checksum-verified benchmark
manifests and shared `RunRecord` sidecars, verifies their payload/runtime binding,
and rejects files modified outside their allocation. Each frozen input, runtime
and external entrypoint must still match its checksum and appear as successful
in the compute log. Supply the checksum manifest frozen before either job ran.

The command posts its measured report under `hpc-report-<task prefix>` using
the publication mechanism shared with WRDS inventories. Exit zero means the
complete observations, payloads and result bytes agree; exit two records measured
disagreement. Missing evidence or publication failure fails the command. This
reader understands the paired benchmark's `results` payload convention. It
provides reproducibility evidence, not tamper-proof attestation against an owner
who can rewrite the cluster files, ledger and logs.

### Run admitted validation research

The existing `covenant_ml` proposal journal now drives staging, submission and
reconciliation through this package. See
[`MARKET_RESEARCH.md`](../../libs/covenant_ml/docs/MARKET_RESEARCH.md#frozen-research-proposal-admission)
for the operator-owned trial/cluster configuration and lifecycle commands.
`hpc3.cli.submit --format json` emits the persisted ledger row. Recovery and
single-attempt verification use:

```bash
python -m hpc3.cli.execution_inspect --config runs/hpc3.json \
  --attempt POLICY_SHA256:ATTEMPT_NUMBER --checksums /private/attempt/checksums.txt \
  --report-task FULL_TASK_UUID
```

This resolves one `research_attempt` in the shared ledger. Pending and failed
allocations report their native state; completed allocations must also pass the
same artifact, runtime, timestamp and compute-checksum checks used by replay
verification. The program posts its measurement and emits structured JSON for
the journal controller. Missing or ambiguous submissions never authorize retry.

The installed Ziggurat controller is
`/home/corvis/research-control/wrds-validation-bda8ca2af/bin/python`, with the
proposal entrypoint under the adjacent `entrypoint/` working directory. Its
policy, journal and cluster configuration are under
`/home/corvis/research-data/wrds/experiments/admitted-validation-20261028/`.
It uses installed wheels rather than relying on this temporary worktree.

Use `cluster-4.json` as the installed-controller configuration reference; its
`report_task` must name the task owning a new run. Preserve the existing journal
and use a newly admitted attempt number and exclusive directory for each run.
The [operational verification record](wiki/pages/wrds-installed-controller-verification.md)
documents the measured jobs, controller repair, frozen image and remaining limits.

### WRDS data inventory

The same package can inventory WRDS Cloud through its existing SSH alias:

```bash
python -m hpc3.cli.wrds_catalog --host wrds \
  --remote-dir /home/uci/awagner14/wrds-setup/inventory-UNIQUE \
  --output /existing/local/directory/wrds-catalog.jsonl \
  --report-task <full board task id>
```

Choose a new remote directory and local output for each run. The command
submits a one-hour-bounded Grid Engine job and waits for completion. It
records every file visible beneath `/wrds`, its effective readability and
logical size, plus denied directories, unresolved links and directory aliases.
An unresolved link may have a missing target or an inaccessible target; the
catalog does not infer which from a failed existence check.
It follows each resolved directory once and includes historical trees; summed
sizes are referenced logical bytes, not unique current datasets or disk usage.
It never opens dataset contents. The JSONL catalog and adjacent receipt carry
the scheduler job ID and verified scanner/catalog SHA-256 digests. A missing
completion marker, unexpected filesystem error or failed job fails the command.

This is the filesystem catalog. The PostgreSQL catalog is its own command:

```bash
python -m hpc3.cli.wrds_sql_catalog --host wrds \
  --remote-dir /home/uci/awagner14/wrds-setup/sql-UNIQUE \
  --output /existing/local/directory/sql-relations.csv \
  --report-task <full board task id>
```

It runs two Grid Engine jobs under `psql -X -w -v ON_ERROR_STOP=1`, exporting
the login environment so the job uses WRDS Cloud's ident-authenticated
database host; no password is stored or sent. The first reads `pg_catalog`
in a read-only transaction and writes one CSV row per visible relation: its
schema USAGE and SELECT grants, the `has_table_privilege` guard a WRDS alias
view names, and how many of the relations the view reads are accessible. A
relation counts as accessible only as a granted table, or as an alias whose
guard target and every underlying relation carry both privileges. The second
job opens every accessible relation with `SELECT * LIMIT 0` in one read-only
transaction, so one refused grant fails the run. The CSV, the accessible
shortlist (`<output>.accessible.csv`) and a receipt with both job ids, the
query, validation and catalog digests and every status count are saved
locally. No dataset rows are read.

Both commands post their counts, job ids and catalog digest on the board task
named by `--report-task`, under the service identity
`load-report-<first eight characters of the task id>`, and fail if the board
refuses the post. A malformed task id is refused before any job is submitted.

A readable path or an accessible relation is not permission to redistribute
it. Data extracts are chosen after inventory; neither command downloads data or registers
training datasets automatically. Existing `hpc3-stage` handles manifest-backed
HPC3 transfers, while `libs/covenant_ml` owns Covenant Radar's dataset loaders.
`libs/platform_kaggle` discovers competitions; it is not a WRDS connector.
Grid Engine receipts remain distinct from the Slurm job ledger.

### HPC3 research runs

Three files, one directory:

```
runs/
  hpc3.json          the workspace: where the cluster is, what each project needs
  arm-b.json         one run: what is different about this job
  ledger.jsonl       written for you; every job submitted from this machine, by any worktree
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

**Onboarding a project that does not exist yet** starts one step earlier, because
every command above assumes an environment to run in and a new project has none:

```bash
hpc3-bootstrap --config runs/hpc3.json --project newcomer     --env-path /pub/wagnera3/envs/newcomer --python 3.11   # the first environment
hpc3-image-capture --config runs/hpc3.json --project newcomer     --env-path /pub/wagnera3/envs/newcomer … --out specs/newcomer-image.json
```

`--python 3.11` does NOT come from `module load python`: the cluster's python
modules are 2.7, 3.8, 3.10 and 3.14, and everything here needs 3.11. Bootstrap
goes through `miniconda3` instead, and refuses to hand back an environment that
borrowed its interpreter from another project.

**Registering it** is the last step, once the image is built and the project's
section is written in [`docs/RESEARCH.md`](../../docs/RESEARCH.md):

```bash
hpc3-register --project newcomer --partition free --gpu none --cpus 2 --mem-gb 4 --minutes 60     --requeue yes --resumes no --deterministic yes --certified-inputs no     --image /pub/wagnera3/newcomer/images/v1/newcomer.sif --env-path /opt/env     --pins newcomer==0.1.0 --gpu-hours 0 --billing free --repo ../../clients/Newcomer
```

It reads the image's digest on the cluster, proves the environment and pins
inside it, writes `runs/hpc3-newcomer.json` and regenerates the research
index's table from the workspace documents. Every unmet precondition is named
in one refusal; nothing is written until every check has passed.

From Git Bash, run it with `MSYS_NO_PATHCONV=1`: MSYS rewrites an argument
that begins with `/` into a Windows path, so `--image /pub/wagnera3/…` reaches
the cluster as `C:/Program Files/Git/pub/wagnera3/…` and the digest read
fails (measured registering `cartridge-qa`, 2026-10-05).

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
| `ledger` | local record of every submission; a relative path resolves against this file's directory **as it sits in the repository's main checkout**, so every `git worktree` reads and appends the one ledger ([one ledger per repository](wiki/pages/one-ledger-per-repository.md)) |
| `quiet_seconds` | how long a running job may write nothing before triage calls it silent (default 1800) |
| `projects` | resource defaults, caps and charge account per body of work |
| `projects.<name>.budget` | that project's self-imposed caps, checked before submission and again while running ([budget model](wiki/pages/budget-model.md)) |
| `projects.<name>.repo` | where that project's code lives; relative paths resolve against this file's directory |

**There are no `--host` / `--root` / `--budget` / `--ledger` flags** — the
reasons are measured, not stylistic
([budget model](wiki/pages/budget-model.md)).

**Adding a project** is `hpc3-register` (above), which writes one workspace
document holding one entry in `projects` for any single-node job, GPU or CPU;
a document is not written by hand, because the image digest in it must be read
on the cluster and the research index's table regenerated beside it. Every
project declares an `image` ([image build flow](wiki/pages/image-build-flow.md))
— CPU-only projects included, because what an image pins is not the card, it
is the compiler, the libc and the BLAS build that decide a timing.
`"gpu": null` (`--gpu none`) is the one spelling of CPU-only. Project names are
lowercase letters, digits and hyphens, at most 24 characters.

**Run, sweep and chain documents** say only what is specific to that work:
see [run documents](wiki/pages/run-documents.md),
[sweeps](wiki/pages/sweeps-and-artifacts.md) and
[chains](wiki/pages/chains.md). Preempted work is resumed by re-running the
sweep document as a campaign
([preemption and campaigns](wiki/pages/preemption-and-campaigns.md)).

---

## Documents by example

Every block below is tested against the real decoders, so what is documented
is what decodes (`test_examples.py`). The reasoning behind each shape lives in
the wiki page linked above it.

A full workspace — two projects, one GPU and one CPU:

```json
{
  "cluster": "hpc3",
  "host": "hpc3",
  "root": "/pub/wagnera3",
  "ledger": "ledger.jsonl",
  "quiet_seconds": 1800,
  "projects": {
    "abl": {
      "partition": "free-gpu",
      "gpu": { "model": "A100", "count": 1 },
      "cpus": 8,
      "mem_gb": 96,
      "minutes": 720,
      "requeue": true,
      "resumes_from_checkpoint": true,
      "image": {
        "path": "/pub/wagnera3/images/v20/abl.sif",
        "sha256": "2b89283fccf289e3060b7b66f61315a5ea0922dbad1b63352540d5f9bdd2d1a5",
        "binds": ["/pub/wagnera3"]
      },
      "env_path": "/opt/env",
      "pinned_packages": { "torch": "2.6.0+cu124", "transformers": "4.46.3" },
      "deterministic": true,
      "certified_inputs": false,
      "budget": { "self_imposed_gpu_hours": 120.0, "max_service_units": 0.0, "charge_account": "" },
      "repo": "../../.."
    },
    "sirius": {
      "partition": "free",
      "gpu": null,
      "cpus": 16,
      "mem_gb": 64,
      "minutes": 360,
      "requeue": true,
      "resumes_from_checkpoint": true,
      "image": {
        "path": "/pub/wagnera3/images/sirius-v3/sirius.sif",
        "sha256": "d3e9daf77afae31ca644b8cf9d9d25b604ec35bd34965b94f930e3a3b0a1f88a",
        "binds": ["/pub/wagnera3"]
      },
      "env_path": "/opt/env",
      "pinned_packages": {},
      "deterministic": false,
      "certified_inputs": false,
      "budget": { "self_imposed_gpu_hours": 120.0, "max_service_units": 0.0, "charge_account": "" },
      "repo": "../../../../metabolomics-dashboard"
    }
  }
}
```

A run — only what is specific to it, with the required `experiment` block
([run documents](wiki/pages/run-documents.md)):

```json
{
  "project": "abl",
  "name": "armB-s42",
  "command": "python -u train.py --arm B --seed 42",
  "experiment": { "arm": "B", "seed": "42", "base_model": "gpt2", "corpus": "armB.txt" }
}
```

A run that names its code declares `repo_commit` together with `repo_tree`,
the cluster checkout it runs from; preflight refuses it with
`REPO_COMMIT_NOT_HEAD` unless that checkout's HEAD is the declared commit
([run documents](wiki/pages/run-documents.md)).

Preflight also asks the run's environment, inside its image, whether it can
start every `python -m` module the command names, resolved from the directory
the command enters; a module it cannot start is refused with
`ENTRYPOINT_NOT_INSTALLED`, naming the commit the image was built from, so a
command newer than its image never reaches the queue. An image that cannot
say which commit built it (no readable `GIT_COMMIT`) is refused
`ENTRYPOINT_IMAGE_UNSTAMPED`. A module the image does carry is admitted, and
its package's source is then compared, in the project's `repo`, between the
image's commit and HEAD: what changed prints as a `WARNING` line under the
verdict, every submission records it in its ledger row's `image_drift`, and
`hpc3-trace` prints it, because that run executes the image's older code. An
image commit the repo does not hold is refused
`ENTRYPOINT_IMAGE_COMMIT_UNKNOWN` ([image freshness](wiki/pages/image-freshness.md)).

A run that overrides project defaults — the merged result goes through the
same decoder:

```json
{
  "project": "abl", "name": "armC-full", "command": "python -u train.py --arm C",
  "minutes": 900, "resumes_from_checkpoint": true
}
```

A sweep — each member declares its own `artifact`
([sweeps and artifacts](wiki/pages/sweeps-and-artifacts.md)):

```json
{
  "project": "abl", "name": "rung-large",
  "minutes": 900, "resumes_from_checkpoint": true,
  "members": [
    { "suffix": "armB-s0", "command": "python -u train.py --arm B --seed 0 --out /pub/wagnera3/abl/s0.json",
      "artifact": "/pub/wagnera3/abl/s0.json" },
    { "suffix": "armB-s1", "command": "python -u train.py --arm B --seed 1 --out /pub/wagnera3/abl/s1.json",
      "artifact": "/pub/wagnera3/abl/s1.json" }
  ]
}
```

A stage manifest with its required provenance block
([staging identity](wiki/pages/staging-identity.md)):

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

A chain — stages in order, each waiting on the last
([chains](wiki/pages/chains.md)):

```json
{
  "project": "sirius", "name": "batch7",
  "experiment": { "sample_set": "batch7" },
  "stages": [
    { "suffix": "sirius", "command": "sirius ... formula", "cpus": 16, "minutes": 360 },
    { "suffix": "zodiac", "command": "sirius ... zodiac", "cpus": 32, "mem_gb": 128 }
  ]
}
```

A run chained onto a job already queued — `depends_on` is run-level, never a
project default:

```json
{ "project": "abl", "name": "eval", "command": "...",
  "experiment": { "of": "55519937" },
  "depends_on": { "kind": "afterok", "job_ids": ["55519937"] } }
```

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

## What this cannot submit

`JobSpec` describes **one single-node job**, with GPUs or without. Everything
below is not a missing flag but a shape the contract cannot express.

| shape | status | what it blocks |
| --- | --- | --- |
| **Multi-node / MPI** | no `--nodes`, `--ntasks`, `srun` or `mpirun` anywhere | anything that does not fit one node |
| **Explicit `--qos`** | not emitted; the cluster auto-selects | `standard-hbm` on HPC3, which refuses the default QOS with `Invalid qos specification` |
| **`--constraint` / `--exclusive`** | not emitted | node features cannot be selected beyond the GPU model |

None of these are hard to add; they are absent because they were never built,
not because they were judged wrong. The three shapes that LEFT this list — job
dependencies, CPU-only, and job arrays (a sweep IS one `--array` call now,
[job arrays](wiki/pages/job-arrays.md)) — and the JVM-project caveat are in
[unsupported shapes](wiki/pages/unsupported-shapes.md).

---

## Commands

| command | does |
| --- | --- |
| `hpc3-preflight --config C {--run R \| --sweep S}` | asks the scheduler whether it would start. Nothing is queued. |
| `hpc3-submit --config C --run R` | preflight → submit → record in the ledger |
| `hpc3-sweep --config C --run S` | the same, per member, recording each as it goes |
| `hpc3-campaign --config C --run S` | the same sweep document, run repeatedly: submits only the members that are neither finished nor already running, and refuses with `LIVE_JOB_UNRECORDED` while a live job carries its name that the ledger has no row for |
| `hpc3-watch --config C --job ID[,ID…]` | state, elapsed, real cost, GPU-hours, state tally; one `sacct` call so a sweep's rows share one moment |
| `hpc3-watch … --until-done 1 [--poll-seconds N]` | re-reads accounting until EVERY requested job is terminal, emitting only state transitions, then the ordinary summary. Waits for ids accounting has not learned yet (a member absent from `sacct` is in flight, not done) and rules the ids wrong after five straight empty reads rather than spinning on a typo |
| `hpc3-triage --config C` | the [five conditions that look like health](wiki/pages/triage-conditions.md); exit 1 if any |
| `hpc3-trace --config C {--match V \| --job ID}` | which run produced a result, or what a job was; exit 1 if nothing matches |
| `hpc3-cancel --config C --job ID[,ID…]` | stops jobs and reports which were actually running — `scancel` is silent about one that had already finished |
| `hpc3-stage --config C --manifest M --source-dir D --expect-from R` | places files, verifies sha256 on both sides, and holds every digest against the published record ([staging identity](wiki/pages/staging-identity.md)) |
| `hpc3-chain --config C --chain H` | runs stages in order and stops at the first failure, so a broken stage does not feed the next |
| `hpc3-bootstrap --config C --project P --env-path E --python V` | creates a project's FIRST environment and proves it owns its own interpreter. The step capture assumes has already happened ([the interpreter is not in `module avail python`](wiki/pages/interpreter-availability.md)) |
| `hpc3-image-capture --out S …` | reads a live environment and writes the image spec. **Use this rather than writing a spec by hand** |
| `hpc3-image --spec S --out-dir D --image-name N` | renders the spec into definition, requirements, self-check and build script. Pure; builds nothing |
| `hpc3-image-build --config C --project P --name N --image-dir D --image-name I` | preflight → submit the rendered build → record it in the ledger ([why this command exists](wiki/pages/image-ledger-lessons.md)) |
| `hpc3-register --project P …` | the LAST onboarding step: refuses with every unmet precondition named (already declared, document exists, no section in `docs/RESEARCH.md`, no repo), reads the built image's digest on the cluster, proves `--env-path` and `--pins` inside it, then writes `runs/hpc3-P.json` and regenerates the research index's table from the workspace documents. Every flag is required |

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
place translates — `platform_core.cli_boundary.run_cli`, at the process
boundary, which every console script in the monorepo shares — and it is
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
