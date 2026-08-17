---
title: "The Match Service: Engine Slots Become a Queue"
tags: [harness, infrastructure, design, roadmap]
related:
  - "[[harness-parallel-matches]]"
  - "[[policy-determinism]]"
  - "[[harness-run-lifecycle]]"
source_paths:
  - "src/rw_bot/harness/runner.py"
  - "../../libs/platform_workers/src/platform_workers/rq_harness.py"
  - "../../services/covenant-radar-api/docker-compose.yml"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [headless-harness]
---

# The Match Service

A design page, in the roadmap style of the cleargbm-perf pages: the problem is
measured, the architecture is named, the build has not started.

## The problem, as it presented

On the night of 2026-08-05 three actors shared this working tree at once: a
24-seed panel holding the engine slot on clone `.game-w1`, one AI session
building fast-forward and running its digest gate in the primary dir, and a
second AI session landing the trace income pair and chaining a batch behind
the panel with a shell `until`-loop. Nothing broke -- but only because every
actor happened to respect conventions no allocator enforces. Sweeps claim
clone dirs by worker index (`runner.py::run_worker` -> `prepare_clone(index)`),
so two batches launched concurrently would collide on `.game-w1`; the "engine
slot" is a discipline, not a resource.

## What already exists and is right

The primitives are sound and the service should change none of them: frozen
batch trees (a batch is one experiment), per-invocation randomized ports
(`Makefile` PLAY_PORT, 27600-27999), clone isolation, resumable batches
(`.partial` claims), and uniformly filed artifacts (`runs/sweeps`,
`runs/traces`) that every reader -- ledger, autopsy, analyze_sweep,
export_matches -- already consumes.

## The architecture: covenant's pattern, aimed at matches

The monorepo already ships the shape once:
`covenant-radar-api` = HTTP API + queue + workers. The match service is that
pattern with matches as jobs -- but **Postgres-backed, not Redis-backed**:

* **The queue is a Postgres table**, claimed with `SELECT ... FOR UPDATE
  SKIP LOCKED`, heartbeat column for liveness, lease table for clone
  indices and port ranges. One store instead of two: at this throughput
  (a match takes minutes; queue operations per second are effectively
  zero) Redis buys nothing, and Postgres buys durability plus a job
  history that is just rows -- "how has the champion's win rate moved"
  becomes SQL as well as `scripts/ledger`. `platform_workers.job_store`
  is Redis-backed, so the claim query is new code; it is the standard
  pattern and small.
* **Results and logs land in the same database.** Scorecards and batch
  metadata as rows (kilobytes); traces and engine logs as compressed
  blobs. Measured from real artifacts: a 10k-sample match is ~686KB of
  trace + ~4.5KB scorecard + ~32KB logs, so a 24-seed panel is ~17MB raw
  -- and the fixed-width numeric trace text compresses roughly an order
  of magnitude, so even a year of nightly panels stays around 1GB in the
  database. Retention: compressed traces kept, raw engine logs pruned
  after a window. The `runs/` tree stays the canonical filed record in
  phase 1 (every existing reader keeps working, dual-written); the DB is
  the queryable mirror until a deliberate migration says otherwise.
* **Control plane, dockerized** -- a slim FastAPI/Starlette service
  (python 3.11-slim, multi-stage, poetry-export, non-root, `/readyz`;
  compose service on `platform-network` beside `platform-postgres`, in
  the house style of `services/covenant-radar-api/docker-compose.yml` and
  the MCPs fleet's base-image discipline). Surface: submit a batch (job
  file content + match config), job status, batch results, cancel. Any
  session, AI or human, on any machine, submits through one door.
* **Match workers, phase 1 host-side** -- a Windows worker process that
  polls the queue, holds a **lease** on a clone index + port range from the
  allocator, and plays matches through the existing harness code unchanged,
  filing artifacts exactly where they file today. Host-side because the
  engine currently runs under the Windows JVM in `.game/` -- the same
  precedent as the host-side Outlook MCP. Two batches then interleave
  safely: the allocator owns clone indices, not convention.
* **Match workers, phase 2 containerized** -- the engine is Java and ships
  Linux builds, and its own cross-platform lockstep multiplayer
  ([[multiplayer-portability-invariants]]) implies the simulation is
  OS-portable. The acceptance instrument already exists: the trace's
  world-digest column. One seed, host vs container, byte-compare the digest
  columns -- the same gate fast-forward uses. Digests match -> workers scale
  horizontally in containers; digests diverge -> stay host-side, having
  spent one match to learn it.

## What it composes with

Fast-forward (task #35, built 2026-08-05, gate in progress) multiplies each
worker 3-5x; the allocator multiplies workers. Together a 24-seed panel is
minutes-to-an-hour, not overnight. The exporter
(`scripts/export_matches.py`) hangs off batch completion as an optional
flag, so "play six seeds and file them for rw_matches" is one submission.

**ML stays in the API stack; the match service never grows a GPU.** Match
simulation is CPU-bound Java -- parallelism comes from worker leases, and
the GPU would idle in this service. Training on the exported data is
covenant-radar-api's job, which already has the whole apparatus: eleven
backends, Optuna search, explainability, an RQ worker compose target with
the NVIDIA device reservation, verified against the local RTX 3090 Ti on
2026-08-05 (taiwan/cleargbm end-to-end). The seam between the two services
is a CSV in `data/external/` -- deliberately boring.

**Multi-runner from day one, multi-game by adapter.** The job schema
carries a runner discriminator and workers register what they can run
(`rw-match` first). A TankpitBot runner is a later adapter behind the same
queue -- "run a session with this config, file the artifacts" fits the
shape -- with the honest caveat that Tankpit parallelism is bounded by
live-server accounts, not CPU. The account pool is already lease-shaped:
`TankpitBot/accounts.json` names the accounts and `TANKPIT_ACCOUNT`
selects one per process, so the allocator leases accounts to session jobs
exactly as it leases clone indices to match jobs. Two named accounts exist
today (2026-08-05) plus a `TANKPIT_USERNAME`/`TANKPIT_PASSWORD` override
path, and the pool is expandable -- which makes the ceiling a config fact
the lease table reads, not a design constant.

## The gate the first data run paid for

mltrace24 (2026-08-06) chained behind a panel and froze the working tree
the moment the slot freed — a tree that had changed the simulation regime
overnight (the RNG/timing certification arc). The batch was valid, but a
day of diagnosis was spent proving that, because nothing between "freeze"
and "24 results" states what the tree IS. So the service's job-start
sequence is not "freeze and go" but **freeze, verify, go**: agent selftest
plus one short smoke match against the frozen tree, with the smoke's
scorecard filed beside the batch as its birth certificate. A failing smoke
files a refusal instead of a night of unusable results; a passing one
timestamps which regime the batch belongs to — the cross-regime
comparability trap is exactly what mltrace24's autopsy paid to learn.

## Phase zero: built and live-verified (2026-08-07)

The queue and the allocator exist: ``rw_bot.service`` (queue.py's claim
transaction -- ``FOR UPDATE SKIP LOCKED`` on jobs, and on leases the
insert itself arbitrates; see the race below -- worker.py's
claim-lease-play-finish loop over the unchanged harness seams), submitted
through ``scripts.submit_batch`` and drained by ``scripts.match_worker``,
whose loop exits after two consecutive empty polls by design -- a worker
is a match player, not a daemon, so an idle fleet costs nothing. The
operational consequence, learned across three restarts (logs 2026-08-14
and -15): re-run ``make fleet-up`` after every submission that follows a
drain, because the workers that finished the last batch are gone,
against ``platform-postgres`` on the covenant compose stack. Rows carry the
harness's own encoded config/match/job payloads and decode back through the
same codecs a sweep file does; artifacts file exactly where sweeps file
them. Live smoke: two matches submitted, claimed under lease, played,
verdicts filed, rows ``done``, leases empty (`runs/sweeps/service-smoke/`).
The reaper requeues heartbeat-silent jobs, so a killed worker returns its
job and its clone -- the exact failure tonight's panels hit twice by shell.

## The race the first production panel found (2026-08-07)

navpair48's first minutes killed three of eight workers on
``UniqueViolation: clone_leases_pkey``. The design error: the claim read
the lease table with ``FOR UPDATE`` and then inserted -- but a **free**
clone has no lease row, and a row lock cannot lock a row that does not
exist. Under READ COMMITTED, eight workers polling an empty lease table
all read the same free set, all picked the same lowest index, and every
loser died on the primary key. The failure was contained by construction
-- the transaction rolled back, the job stayed queued, nothing was
orphaned -- but each race cost a worker process, and the panel sagged to
five lanes.

The fix makes the insert the arbiter: ``INSERT ... ON CONFLICT
(clone_index) DO NOTHING RETURNING clone_index``, walking the pool until
a row comes back or the pool is exhausted (a decline, exactly like an
empty queue). The prior read is now only an optimization that skips
known-held indices. Verified live mid-panel: three fixed-code workers
joined five old-code workers on one lease table and filled clones 5-7
without an incident, restoring the panel to eight lanes.

## Phase one: the HTTP door, in the fleet's idiom (2026-08-07)

Built the same day, with one deliberate deviation from this page's original
sketch: not FastAPI-in-a-container but the fleet page's own stdlib pattern
-- ``http.server``, one pure router (``rw_bot.service.http``), requests
parsed by ``ndjson.parse_object`` and responses by ``render_json`` -- because
the repo had already shipped and proven that shape, and consistency beats a
sketch. ``scripts.match_service <dsn> [port]`` serves ``POST /batches``
(name, job-file text, and the sweep knobs; resubmission is resume),
``GET /batches/<name>`` (counts by state), and ``/healthz``, loopback-bound
on 27501. Live-verified against a production panel in flight: the status
endpoint reported navpair48 at 85 queued / 5 running / 6 done while eight
leased workers drained it. Submission configs build in one place
(``rw_bot.service.submit.batch_config``) for the CLI and the wire alike.

## Results mirrored, cards first (2026-08-07)

The "results land in the same database" bullet began paying out the same
night: :func:`finish` now stores the filed scorecard's text on the job row
(``card TEXT``, migrated idempotently by bootstrap -- and the door
bootstraps at startup now too, a gap the live probe found when its results
route met a table an older writer shaped). The worker reads the card from
exactly where ``play_job`` filed it, through a ``read_card`` hook; a failed
match mirrors nothing, because it filed only a ``.partial``. The read side
is ``GET /batches/<name>/results``: one NDJSON line per match -- label,
seed, state, and the verdict word parsed from the card -- which turns a
paired-panel verdict into a wire read instead of an artifact-tree walk.
The ``runs/`` tree stays canonical; rows finished before the mirror
existed carry an empty card, honestly (navpair48's early rows do).
Traces-as-compressed-blobs remains a later step.

## What later phases add

Trace and engine-log blobs (compressed) beside the mirrored cards,
freeze-verify-go smoke certification at batch start, the worker-registration
half of the multi-runner schema, and -- if remote submission ever needs it
-- the dockerized deployment this page originally sketched.
