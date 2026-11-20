# hpc-wake: the board's event pump

One scheduled tick runs every wake bridge in `scripts/run_cycle.py`'s
`publishers` table, so a machine event (a CI verdict, a fleet-lock boundary,
a fleet health change, a failing login or search probe) becomes a board post
that wakes whoever acted, instead of each session hand-polling a state.

The package keeps the name of its first publisher, which announced Slurm
terminal states from the hpc3 ledger and left the table on 2026-11-20 (MCPs
board task 62765171). The hub runner's job, its service paths and the
publisher ticks all carry the name, so it stays the pump's.

## HPC3 jobs: the submitting session watches its own

Every HPC3 submission is made from the session host, over its own UCI VPN
(MCPs board task 32a11489), so the hpc3 ledger lives there and so does the
only route to the cluster. The control duty's machine, where this pump runs,
has neither: the hpc-wake publisher read a ledger it never had and posted
nothing after 2026-10-05. No fleet machine may relay through the session
host (MCPs board task 1d9ad999), and the session host runs no service, only
Claude Code sessions. So the session that submitted a job is the one
reader that both reaches HPC3 and may, and hpc3 ships the command for it:

```
hpc3-watch --config runs/<workspace>.json --job <ids> --until-done 1
```

run in the background (Claude Code's `run_in_background`), it re-reads
accounting until every named job is terminal, preempted members included,
then prints the summary and exits, and the harness wakes the session with
that output. `hpc3-submit`, `hpc3-sweep` and `hpc3-campaign` print the line
for the jobs they submitted. A Codex session's background exit does not yet
wake it by itself (the board's `background` wait kind says so).

## Scheduling

The entry point is `scripts/run_cycle.py`, plain Python, run every three
minutes as the hub runner's `hpc-wake` job: one row of the MCPs repo's
`scripts/fleet/hub-jobs.json` (MCPs board task b27bd34a). It reads the
machine's credentials from the untracked `runs/credentials.env`, `KEY=VALUE`
lines read by `platform_core.env_file`, the parser the fleet ticks read the
same file with, and hands every line to every publisher's environment.

The row runs it wherever the control duty is held (diphtheria since
2026-10-18, MCPs board task c39ee056), passing `--package-root` this
package, `--workspace` that machine's MCPs checkout and `--python` its
system interpreter; the table is built from those, so no path in it is one
machine's. The binary is this package's own `.venv` interpreter, never the
python on PATH; the row names it. `run_cycle.py` imports `platform_core`,
and the venv holds it as the editable path dependency `pyproject.toml`
declares, so the pump always runs the tree's `platform_core`. The system
python held a copied, older one: when d46de80d0 (repo API) moved the env
parser into `platform_core` on 2026-09-29, every tick from 08:25Z died on
`ModuleNotFoundError` and the pump published nothing for eight hours (board
task 25306d98). Run `poetry install` here before the runner first starts
the job; the project is `package-mode = false`, scripts and tests only.

One run at a time: a cycle still running
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
publisher = 'ci-wake' AND head_sha = '<sha>' AND NOT dirty AND exit_code
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

Every publisher batches per tick (a 126-line lock-journal backlog is ONE
post) and mentions only whoever acted — the noise budget is acceptance 6 of
the board task, not a hope.

DELIBERATE EXCLUSIONS, so the next reader knows they are decisions:

- **HPC3 job endings are not published.** The pump's machine has neither
  the hpc3 ledger nor a route to the cluster; the submitting session
  watches its own jobs, as the section above says (MCPs `62765171`).
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

## Tests

`make check`. The one fake is the process runner rebound through
`scripts._test_hooks`, against a staged package tree with real files; the
production runner is run for real against publishers that fail and hang.
Nothing is patched, nothing is mocked.
