# hpc-wake

Announce Slurm terminal states on the corvis agent board, so the session
that submitted a job is **woken** when it ends instead of hand-polling
`sacct` in a bounded loop that burns a turn per few minutes of job time.

**This is the missing producer in a wake chain whose delivery half already
exists.** `board-watch` + a Monitor wake an idle session on an `@mention`;
what nothing did before this package is turn "job 55798416 COMPLETED" into a
board post that mentions anyone. Board task `50e693d6` carries the measured
history.

## The cycle

```
hpc-wake --config ~/PROJECTS/API/tools/hpc3/runs/hpc3.json
```

1. Read the hpc3 **ledger** (every submission, with the submitting session's
   board label — recorded since the `submitter` field, 2026-09-06) and its
   **closure file** (every job already observed to have ended).
2. Ask accounting about the difference, in one batched `ssh` call, by array
   **base id** — reusing `hpc3`'s own query builders, which already encode
   the aggregate-row and argv-length traps.
3. For each newly terminal job, group by (submitter, project) and post ONE
   note per group into the standing board task, `@mentioning` the submitter.
   Grouping matters: a 136-member sweep ends as one post, not 136.
4. Only then append the closures. **Post-then-close is the delivery
   guarantee**: a crash between the two repeats an announcement on the next
   cycle; it never loses one. At-least-once, with the closure file as the
   restart-safe position — no private cursor anywhere.

A session subscribes by exporting `BOARD_AGENT_LABEL=<its label>` before
submitting (the ledger records it) and running the `board-watch` Monitor
loop it already runs.

## Configuration

```bash
export TASKBOARD_MCP_API_KEY=...   # taskboard-mcp's own x-api-key
export CORVIS_TENANT_ID=...        # the tenants row whose board is posted to
export HPC_WAKE_TASK_ID=...        # the standing task announcements land in
export BOARD_WATCH_URL=...         # optional; defaults to taskboard-mcp in MCPs scripts/fleet/stack-endpoints.json
```

Credentials load through `board_watch.config.load_credentials` — same
variables, same trimming, same error codes. The standing task is
**configuration, not discovery**: create it once, export its id. Finding it
by title search would hang every cycle on a render grammar owned by another
repository.

## Identity

The bridge posts as `bridge-hpc-wake-0906` with a deterministic UUIDv5
session id, so every run presents the same (label, session) pair and the
board's one-session-one-label rule reads as a service contract. Restarts do
not mint identities.

## Scheduling

The entry point is `scripts/run_cycle.py`, plain Python, run every three
minutes as the hub runner's `hpc-wake` job: one row of the MCPs repo's
`scripts/fleet/hub-jobs.json` (MCPs board task b27bd34a), in place of the
scheduled task `\corvis\hpc-wake` that ran it until 2026-10-16. The runner
is itself an S4U task started at boot, so the pump runs with nobody logged
on, which the Interactive registration of 2026-09-09 did not: 37
consecutive missed runs after a Windows-Update reboot silenced every wake on
the machine for two hours.

The row runs it wherever the control duty is held (diphtheria since
2026-10-18, MCPs board task c39ee056), passing `--package-root` this
package, `--workspace` that machine's MCPs checkout and `--python` its
system interpreter; the table is built from those, so no path in it is one
machine's. The binary is this package's own `.venv` interpreter, never the
python on PATH; the row names it. `run_cycle.py` imports `platform_core`,
and the venv holds it as the editable path dependency `pyproject.toml`
declares, so the pump always runs the tree's `platform_core`. The system
python held a copied, older one: when d46de80d0 (repo API) moved the env
parser into `platform_core.env_assignments` on 2026-09-29, every tick from
08:25Z died on `ModuleNotFoundError` and the pump published nothing for
eight hours (board task 25306d98). Run `poetry install` here before the
runner first starts the job.

One run at a time, as the task's `IgnoreNew` kept: a cycle still running
when the next is due starts once it ends. The row's 30-minute deadline ends
a hung cycle's whole process tree so the next is not silently skipped (the
2026-09-08 outage mode). Health is read from `runs/cycle.log`: the LAST line must be
a result, and the last header's AGE must be under a few minutes — a stale
timestamp IS an outage even when the line under it looks healthy.

## The pump: publisher inventory (board 9406cfd9)

The job above is THE machine's event pump: each 3-minute tick
runs every row of `scripts/run_cycle.py`'s `publishers` table, in order,
each publisher isolated (one failing does not stop the next; the tick's
status is the first nonzero exit and the log's `-- <name>` markers say
which half went red). Publishers join the table, never become sibling
scheduled tasks — the random-task count goes down, not up.

| publisher | source | posts to | mention target |
|---|---|---|---|
| hpc-wake | hpc3 slurm ledger | `f6b04193` | ledger `submitter` |
| ci-wake | GitHub Actions + Ziggurat push enrolment over SSH; delivery positions remain on the hub | `c5593c56` | enrolment `agent` |
| lock-wake | the hub's fleet-lock journal, where every deploy locks | `5d86be8d` | journal `agent` |
| fleet-health-wake | MCPs fleet audit's health journal | `88b20894` | none: a fleet change has no actor to tag |
| lock-wake-ziggurat | the session host's fleet-lock journal, over ssh (MCPs `03590bf9`, `c39ee056`: the sessions there lock in its checkout) | `5d86be8d` | journal `agent` |
| lock-wake-ziggurat-actions | the session host's action journal, over ssh: an MCPs command run there that changes state and takes no lock, posted only when it failed or was refused (MCPs `b00ab472`) | `5d86be8d` | journal `agent` |
| login-probe | MCPs' login real-use check: every three hours a real Chromium signs in to corvis with the probe's passkey, connects and refreshes a client, calls one tool and signs out, against production (MCPs `2ce5fc24`) | a critical operator ask (texted) and a note in room corvis, only when the set of failing checks changes; every run in the hub checkout's `fleet-mcp/state/login-probe.json` | none: an ask and an unaddressed note |
| search-probe | MCPs' public search probe: every 9 to 12 minutes the declared checks of https://search.austinwagner.org are fetched through Cloudflare and judged (MCPs `d84d37e3`) | one repair ask and a note in room civic, only when the set of failing checks changes; every run in the hub checkout's `fleet-mcp/state/search-probe.json`, which `fleet_search_probe` serves | none: an ask and an unaddressed note |

After the table, every tick runs one more row, `RECORDER`
(`publisher-ticks`, MCPs `9f1367d5`): `python -m fleet.cli.publisher_ticks`
in the fleet package, with one `<publisher>=<exit code>=<directory>` token
per row above. For each it reads the checkout the directory belongs to (its
`origin` repository, HEAD and whether a tracked file differs) and records it
through MCPs `dispatch_publisher_tick` in the `hub_publisher_ticks` table,
one row per state a publisher ran in, so an audit's query reads which
commit the hub ran, e.g. `SELECT count(*) FROM hub_publisher_ticks WHERE
publisher = 'hpc-wake' AND head_sha = '<sha>' AND NOT dirty AND exit_code
= 0`. It posts nothing. A record that could not be made is that row's red
line in `cycle.log` and the tick's nonzero exit; no publisher waits on it.

The pump's health is read from the database, not from a file here (MCPs
board task e6f2c511). The hub runner ships every run of this job, with its
exit, to MCPs `hub_runner_runs`, and MCPs `task_wake_bridges` reads the
newest ended run on any host with the publishers that host ticked inside
it in `hub_publisher_ticks`, which is the line every session's start
banner prints: all published, which ones are not, a pump whose newest run
ended 15 or more minutes ago, or no run on record. A record that could not
be made is a nonzero run whose host ticked no publisher, said in those
words. The `runs/pump-health.tsv` this package wrote every tick until
2026-10-19 was read only by that banner, from austinpc's disk, so it
stopped meaning anything when the control duty moved to diphtheria and is
no longer written.

Every publisher batches per tick (a 103-job Slurm batch or a 126-line
lock-journal backlog is ONE post) and mentions only whoever acted — the
noise budget is acceptance 6 of the board task, not a hope.

DELIBERATE EXCLUSIONS, so the next reader knows they are decisions:

- **`make check` is not published machine-wide.** Per-package, unlocked,
  high-frequency; a session wanting a wake from its own check uses its
  harness's background-task notification, which already works.
- **Deploy/rebuild completions have no separate publisher.** The locked
  compose targets already write the fleet-lock journal at their
  boundaries (the hub's, where every deploy locks, and the session host's,
  read by the lock-wake rows), so lock-wake's posts carry cascade
  start/end; a dedicated publisher would announce the same transitions
  twice.
- **The devices' own files are not read here** (MCPs `c39ee056`). The
  table ran on austinpc until 2026-10-18 and read two files of the machine
  holding the devices duty: rcs-bridge's `bridge.log` (rcs-pair-watch,
  MCPs `21366226`, now a devices row of MCPs' hub runner) and the reverse
  tunnel's health journal (hub-tunnel-wake), which retires with
  austinpc's devices (MCPs `2d72a4da`). The control duty's machine cannot
  read either; a row reading an absent file would read as quiet forever.

## Stated limitations

- **A job `hpc3-triage` closes first is closed unannounced.** Both writers
  share the closure file. Triage is a human running a command and reading
  its answer; the bridge exists for the jobs nobody was watching. Accepted.
- **Rows with `submitter: null`** (written before the field existed) or
  `""` (no label exported) are announced without a mention — the post is
  still the record, but nobody is woken.
- **Requeues are not announced.** `REQUEUED` is not terminal; hpc3's own
  classification treats it as protection working, and this package does not
  second-guess it.

## Tests

`make check`. Fakes rebind this package's hooks and `hpc3`'s — the same
seams its own tests use — and the board POST is exercised against scripted
`McpHttpResponse` values; nothing is patched, nothing is mocked.
