# fleet

Dispatch a project's build to another machine on the tailnet, with a ledger,
per-node budgets, and an event stream a session can subscribe to.

**This exists because of a measured incident, not because distributing builds
is nice.** On 2026-09-04 five AI sessions were working this monorepo at once
and destroyed each other's work twice in one hour:

1. A GPU measurement run died mid-flight with exit −1, no traceback and no
   out-of-memory event. Another session had started `make check` in the same
   project, whose `poetry sync --with dev` uninstalled and reinstalled
   `model_trainer_server` into the **shared** `.venv` — deleting
   `site-packages/model_trainer/**` out from under a live interpreter. Every
   Makefile's `lint` *and* `test` target runs that sync, so one `make check`
   opens the window twice, and 40 of the 48 Makefiles do it against one
   `.venv` per **project**, not per session.
2. Ninety minutes later the box held 66 wedged test processes from two
   overlapping runs of one project, holding **77.9 GB of commit** with an
   aggregate CPU delta of 0.016 s over 5 s — doing nothing, and leaving 22 GB
   of 179 GB free. Nothing had refused either dispatch.

The pre-run reaper (then `scripts/reap-test-processes.ps1`, since 2026-09-20
`tools/maketools`' `reap-stale`) was blamed for the first and is innocent:
its filter requires a command line matching `*pytest*` or `*exec(eval*`, its
sweep only considers processes older than sixty minutes, and an aggregate
CPU-idle gate aborts if any candidate is burning CPU. The collision is
structural — a shared mutable environment with no lock — not a rogue killer.

---

## The three files

```
registry/         the workspace, split by concern (MCPs board task 4a78c916)
  machines/       the fleet's one roster, one file per machine, named by it
                  (machines/colossus.json is colossus): who it is, how it is
                  reached, and what dispatch may take from it (MCPs board
                  task e447de72; see "The roster" below)
  settings.json   fleet-wide settings: data paths, serve and poll seconds,
                  the record paths below, and under identity the fleet's
                  ssh key, Cloudflare zone and account and VPN profile digest
  projects/       one file per project, named by its path (projects/MCPs/
                  mcp-shared.json is MCPs/mcp-shared): what it costs, what it
                  needs and where its commits come from (source)
runs/
  ledger.jsonl    append-only; every dispatch ever made from this machine
  feed.jsonl      append-only; the event stream subscribers tail
  leases.json     live state; who holds which project's environment
  retires.jsonl   append-only; every run owed its directory's retire from its
                  lease, until a retire is done; a collect pass retires each
                  owed run no lease or running ledger row still holds
  mirrors/        one bare git mirror per sourced project; the node runners
                  fetch a submitted sha into it and archive it from there
  host-claims/    live state; each node's claims whose launch is under way,
                  one file each, and the lock its runners claim in turn by
```

**`registry/` is tracked and the three records are not**, because they are
different kinds of thing. The workspace describes the fleet and belongs to
everybody; the records are this machine's own history and live under `runs/`,
which the monorepo gitignores. The workspace sat under `runs/` at first and
was therefore invisible to git — so the one thing every command requires would
not have survived a fresh clone, and the package would have arrived unrunnable.

**One reader assembles the directory**, `fleet.core.workspace_files`, into the
document the decoder has always read, so no command knows the split exists.
It was one `fleet.json` until it reached 2252 lines; every file is now well
under 600. Record paths still resolve against `tools/fleet`, the directory
holding `registry/`.

The ledger and feed are history and are never rewritten. Leases are live
state and are: a release has to make a claim stop existing, and an
append-only log of "taken"/"released" would make every reader replay the file
to answer one question.

## Commands

```bash
fleet-nodes     --config registry                   # what is free right now, and why each lane claims nothing
fleet-preflight --config registry --project P       # would it run, and where
fleet-preflight --config registry --project P --node lavender
fleet-run       --config registry --project P \
                --agent <label> --session <uuid> --repo-root <path>
fleet-collect   --config registry                   # close out what finished
fleet-collect   --config registry --run <run-id>
fleet-watch     --config registry                   # the event stream
fleet-watch     --config registry --run <run-id>
fleet-cancel    --config registry --run <run-id>
fleet-agent     --config registry \                 # one tick of the HUB lane's runner
                --agent <label> --session <uuid> --repo-root <path> [--announce]
poetry run python -m fleet.cli.node_agent \         # one tick of ONE node's runner
                --config registry --node <alias> [--announce]
poetry run python -m fleet.cli.publisher_ticks \    # the hub pump's record of its bridges
                <publisher>=<exit code>=<directory> ...
```

`fleet.cli.publisher_ticks` is the hpc-wake pump's last row each tick (MCPs
board task 9f1367d5): for each wake bridge it ran, it reads the checkout the
bridge ran from (`fleet.core.checkout`) and records the repository, HEAD,
whether a tracked file differed and the exit code through MCPs
`dispatch_publisher_tick`, under the hub runner's identity. That is how an
audit's query learns which commit the hub ran.

`fleet-run` returns as soon as the suite is running and does **not** wait for
it. The build outlives the command because it is launched as a transient
systemd user unit, owned by the node's user manager rather than by the ssh
call (see "One dialect: `sh`" below).

**Something therefore has to go back and ask, and that is `fleet-collect`.**
For every dispatch the ledger still calls running it reads the node's recorded
exit status, and for each that has one it appends the closing row, emits
`passed` or `failed` on the feed, and releases the lease. It is safe to run as
often as you like: an unfinished run is left exactly as it was and a finished
one is closed once.

It exits 0 for a failing suite. The status is whether *collection* worked, not
whether the work passed — otherwise a shell loop stops on the first red build,
which is the one moment somebody wants it still reporting.

```bash
while true; do
  fleet-collect --config registry
  fleet-watch   --config registry
  sleep 30
done
```

`--agent` and `--session` are required and are the board's own identity
fields, so a ledger row and a board post can be matched by whoever reads both.
A default would be one label shared by every session, which is the same as
having none.

## `fleet-runners`: a CI runner host is disposable

The self-hosted GitHub Actions runners live on one roster, `runners.json`:
which runners each host carries, which assets their jobs need, the CI slice
their jobs share, and, in each host's `base`, the apt packages a fleet node
lacks for them. `fleet-runners` holds the hosts to it:

```bash
fleet-runners --spec runners.json                                   # audit every host
fleet-runners --spec runners.json --host lavender --render <dir> --config registry
fleet-runners --spec runners.json --host lavender --onboard owner/repo --config registry
fleet-runners --spec runners.json --host lavender --rebuild yes --config registry
```

The three modes that run the provision take the registry with `--config`,
because the provision holds the runners to the CPUs the host's roster file
reserves for them (`dispatch.budget.reserved_cores`): whole physical cores,
the highest-numbered first, set as the CI slice's `AllowedCPUs` and read back
from its `EffectiveCPUs` (MCPs board task 349e25ee). A weight alone let CI
take every core on lavender, and a check sharing a core with it ran past its
budget.

The operator, on 2026-09-26, while lavender was being recovered by hand from a
full disk: "it justcis a github runner". A machine whose only job is running
CI holds nothing that needs saving, so the answer to any fault on it is to
wipe it and run the recipe again (board task 1aa6a021). `--rebuild yes` is
that recipe. Every runner host is an Ubuntu fleet node since lavender was
reimaged on 2026-10-19 (MCPs board task 1e9ab738), so the recipe starts from
a node a CORVIS stick or a netboot installed and MCPs' `make fleet-provision`
provisioned (Tailscale, the fleet user's key and passwordless sudo, docker,
the GPU driver, the toolchain), because nothing can reach a machine before
that. Each stage is a bash payload sent to the host's `payload_dir` and run
there as root (`remote.run_root_script`), in order:

1. the **base**: refuses a host whose docker is not active, installs the
   roster's apt packages, creates the runner account `gharunner` in the
   docker group and proves it can reach docker; then the **reboot** a
   package asked for (`/var/run/reboot-required`), waited out by comparing
   the kernel's boot id, never by a sleep, and the base again;
2. **registration tokens**, minted with `gh`, one per repository;
3. the roster's own **`provision.sh`**, carrying them;
4. the **audit**, whose findings are the verdict and the exit code.

Every stage is idempotent, so a rebuild cut short by anything is finished by
running the same command again, and a rebuild over a healthy host changes
nothing: an install whose `.runner` exists is not configured twice.

**What a runner host holds, and why none of it needs saving:**

| State | What happens to it on a rebuild |
|---|---|
| Registration tokens | Minted fresh by every run; they expire within the hour |
| Each runner's `.runner`, `.credentials`, service | Written by config.sh; `--replace` takes back the old registration of the same name |
| `_work` trees, poetry venvs | Caches; the next job recreates them. ci-clean removes a venv whose recorded sources are all gone, while jobs run (MCPs board task ecb13d3d) |
| npm's, pip's and poetry's download caches | `~/.cache/fleet-tools/<generation>`: the job-started hook (`runner_tool_caches`) reads or starts the generation, writes it into its runner install's marker under `~/.cache/fleet-jobs`, and hands the job its paths through `GITHUB_ENV`, npm's one per runner install because concurrent writers corrupted a shared one; ci-clean removes a generation whole once no marker names it, the fleet nodes' rule (`tool_caches`). Removing single files instead could part a poetry metadata file from its body and fail every later lock (MCPs board task 4b405239) |
| Docker images and volumes | The pull cache and nothing else: services mount tmpfs (MCPs `check.yml`), and ci-clean prunes daily |
| Docker build cache | What the jobs' `docker build` steps leave; ci-clean prunes daily what no build used for a week, then the reclaimable cache above 10 GB, `BUILD_CACHE_BOUND_GIB` in `runner_base` (MCPs board task 46c7b11b; 60 GB, then 20, until board task 621902c3) |
| Chrome, the gte-small model cache, llama.cpp | Roster assets with a `provision_command`, fetched by provision.sh |
| ci-clean script, service and timer | Written in `ci_clean`, embedded by `runner_render` and installed by provision.sh |
| ci-clean's report | `/var/lib/ci-clean/last-run`, in the state directory systemd makes for the service (`StateDirectory=ci-clean`), world-readable: each run writes the sha256 of the script that ran, when it started, every pass's closing line, the daemon's build cache records before the prunes and after them, which generation each runner install's last job read and what the current one holds. `test_runner_hygiene_host` reads it, because the runner account's home is closed to the fleet's user (MCPs board tasks 46c7b11b and 4b405239) |
| fleet-runner-reaper script, service and timer; each runner unit's `fleet-kill.conf` | Rendered from `runner_reaper_render` and installed by provision.sh. Every 30 s the reaper kills what a finished job left in a runner unit's cgroup: every process outside the unit's own tree while no Worker runs, the orphan trees rooted before the earliest live Worker while one does, and a Worker older than the host's `job_timeout_minutes` with everything under it, because GitHub has ended its job by then. The audit's `orphans:<repo>:<runner>` rows count the same, older than that timeout (MCPs board task 53528106). `--dropped UNIT PID START` ends one Worker the hub's reap pass names as holding a job GitHub dropped: it writes each pid's argv, then kills the Worker and every process of the unit under it, and never the listener (MCPs board task 715b9ec5). Once every unit's leftovers are reaped, a pass restarts each unit whose `Slice=` drop-in names `listeners.slice` while its cgroup is still in `runners.slice`, the moment its Listener runs with no Worker: a unit running a job is read again five times a second for up to `PLACE_WAIT_SECONDS`, 20 s, because a single read missed the seconds between two jobs on every pass for 82 minutes |
| Each unit file the provision writes | Written through `runner_unit_files`, the one writer: a unit or drop-in is rewritten, and systemd reloaded, only when its text differs, so a re-run over a built host reloads nothing |
| The licensed game tree (`manual` asset) | The one exception: no script may fetch it. The rebuild prints it as `PLACE BY HAND`, and the audit fails until it is placed |

**The disk ceiling.** The audit fails a host whose root uses more than
`base.disk.ceiling_gb`, and the row's id carries both numbers, so every line,
passing or failing, says what is allowed and what normal was:
`disk:/:ceiling-210gb:baseline-18gb@2026-10-20`. The baseline is lavender's
root after its reimage and the fleet node provision and before any runner,
18 GB used of 915 GB at 2026-10-20T00:33Z. The ceiling is at least what the
host's own bounds let its root hold, `disk_bounds_gb` in `runner_base`: the
baseline, the 10 GB build cache bound, `cache_ceiling_gb` and one
`work_ceiling_gb` per install, 193 GB for lavender's seven, and the roster
refuses a host whose ceiling sits under it. It was 150 GB, kept from the WSL
guest, until the audit failed lavender at 190 GB on 2026-10-30 with every
bounded area inside its bound (board task 621902c3). It still trips long
before the 536 GB of leaked volumes that filled lavender's disk on
2026-09-25.

## `fleet-agent`: serving the corvis dispatch queue

Everything above is driven from a shell on this machine. `fleet-agent` is how
a session **anywhere** — a phone, claude.ai, another machine — gets work onto
the fleet without any of them being able to reach it.

The queue lives in corvis, as the `dispatch_*` tools on `fleet-mcp` (MCPs
repo, migration 486). A submitter calls `dispatch_submit`; that writes a row
and nothing runs. `fleet-agent` runs **here**, where the ssh keys are, claims
a row and executes it through exactly the engine the commands above use.

**The work travels toward the keys, never the other way.** The corvis server
has no route to the `100.x` tailnet and holds no key, so no inbound path to
the fleet is ever opened and no fleet credential ever lives on an
internet-facing multi-tenant box. That is the whole reason the queue is
inverted.

**Since MCPs board task fd5cabfa the queue has two lanes and two kinds of
runner** (the section after this one). `fleet-agent` is the HUB lane's: it
claims `build-bases` and the five session verbs (restart, revive,
graceful and hard kill, and since MCPs mig 564 `compact-session`, which
runs `session-audit compact` and leaves the session running), runs each on
the hub and closes it in the same tick. The make targets are the NODE lane's, claimed
by one `fleet-node-agent` per enabled node. A `fleet-agent` hub tick is
therefore four passes, in this order:

1. **claim** — take at most one hub-lane job (`dispatch_claim` with
   `lane: "hub"` and no tags), run it here, report it started and closed.
2. **publish** — hand the queue every project the registry declares at the
   commit `refs/fleet/rolled` names (`dispatch_registry_publish`), which
   refuses each queued job naming a project it does not declare (MCPs board
   tasks `1915b042` and `0be88a36`).
3. **observe** — read every enabled worker's Claude Code session records
   (`~/.claude/sessions/<pid>.json`) over ssh and hand them to the board's
   session ledger through `task_session_observe`, keyed by the machine they
   came from (MCPs board task `5a3865bf`). The script goes out as `sh`,
   like every other remote act, the one dialect since the fleet's last
   Windows node left it (board tasks `cd5010c4` and `e705f08c`). The hub
   is not visited: its own directory is recorded by `pcsession-mcp`. The
   machines visited are the roster's enabled workers, vpn-jumps and session
   hosts (`fleet.core.observe.observable`); with none, the tick logs that
   it observed nothing and reads no credential. A node that is asleep is
   one logged line, not a failed tick.
   This pass writes to the **taskboard**, so it needs `TASKBOARD_MCP_API_KEY`
   (mcp-taskboard's `MCP_INTERNAL_KEY`) beside the queue's key.
4. **reap** — read the roster (`runners.json` at the rolled commit) and, for
   each runner host, ask `gh` for each repository's runners and the
   host's job scope helper for each unit's `Runner.Worker`s by pid, start and
   age. A Worker over five minutes old is holding a job GitHub dropped when
   GitHub reads its runner not busy, online or offline, or does not list it,
   or when a newer Worker in its unit superseded it. A runner GitHub reads
   busy keeps its job even while its listener is offline. Where one is named, the pass reads again two
   minutes later and ends each Worker both readings name, by pid, through
   `fleet-runner-reaper --dropped UNIT PID START` as root on the host (MCPs
   board task `715b9ec5`). This pass runs `gh` on the hub, as the runners'
   token minting does.

No loop. The interval belongs to whatever schedules it, where it is visible:

```bash
export FLEET_MCP_API_KEY=...      # mcp-fleet's MCP_INTERNAL_KEY
export TASKBOARD_MCP_API_KEY=...  # mcp-taskboard's MCP_INTERNAL_KEY (observe pass)
export CORVIS_TENANT_ID=...       # the tenant whose queue this serves
while true; do
  fleet-agent --config registry --agent fleet-hub-diphtheria \
              --session <uuid> --repo-root ~/PROJECTS/API
  sleep 30
done
```

**It exits 0 for a refused job**, the same argument `fleet-collect` makes for
a failing suite: the status is whether *the agent* worked. A job no node had
capacity for was handled correctly — the refusal goes back to the queue with
its code and message verbatim, where the submitter reads it. Exiting non-zero
would stop the loop on exactly the condition the loop exists to keep
reporting.

**A job refused after its lease was taken gives the lease back first**
(`fleet.core.run_lease.abandon`, MCPs board task e12affc5). A staging or
launch failure used to close the queue row and leave the lease standing for
the project's whole window, so the resubmission the refusal invites was
refused `LEASE_HELD` by the run that had just died: job 7143a25e, 793 s after
job 4d8b28a9 failed to send its payload to serendipity. The feed gets the run's
terminal `refused` line with the same reason. A `LEASE_HELD` refusal gives
nothing back, because that lease belongs to another run.

**The hub runner is the machine holding the control duty, by name.** Its
label is `fleet-hub-<alias>` and its session the version-5 UUID of
`fleet-agent/hub/<alias>` (`fleet.cli.tick.runner_identity`, which names the
sessions and netboot runners the same way by their lanes), and a base rebuild
or other hub job is reported started on that alias, so moving the duty gives
the new holder its own label (MCPs board task c39ee056). austinpc's former
pair, `fleet-runner-austinpc` and session a850f688, stays bound to the runner
that used it and is never reused.

### Research queue polling

`python -m fleet.cli.tick --api-root /home/corvis/PROJECTS/API
--log-directory /private/logs --lane research` runs a bounded research poll
through the existing rolled launcher. `research-announce` registers the stable
runner identity. The control-duty `fleet-research` scheduler row supplies the
same 180-second cadence and 40-minute outer deadline as the other fleet ticks.
On 2026-10-28, board task `3dff6fbf` verified the deployed scheduler on
Diphtheria against the installed Ziggurat controller: both retained HPC3 and
WRDS pilots reached `passed` through automatic ticks, and repeat submissions
returned the original receipts. The metadata and scope of that check are in
[the installed-controller record](../hpc3/wiki/pages/wrds-installed-controller-verification.md).

The control host reads private `tools/fleet/runs/research.json`, with exactly
`tenant`, `node`, `host`, `python`, `root`, `env_file` and `script` fields.
`node` is the queue's explicit controller node, `host` its SSH alias, and all
four paths are absolute paths on that host. `python` names an installed frozen
fleet runtime; `root` holds tenant-authorized requests; `env_file` supplies
native-report environment assignments; `script` is the private remote tick path.
Credentials are never included in queue payloads or controller transcripts.
The current environment file sets the runner label only; maintained native
reporters read their credentials through the existing workspace configuration.

Submit `command: "research"`, `project: "research/hpc3"` or `"research/wrds"`,
an explicit `node` and board `taskId`, and
`researchRequest: {backend, specSha256, attempt}` through `dispatch_submit`.
The digest resolves only to `root/<specSha256>/request.json`. That document
must bind the authenticated tenant, backend and attempt. Every referenced
immutable input uses `{path, sha256}` and is rehashed on every poll.

An HPC3 request additionally names `python`, pinned `script`, `policy`, `trial`
and `cluster`, plus absolute mutable `ledger` and `directory` paths. It requires
an explicit `phase`; private requests without it are refused and must be rebuilt
and re-pinned. `phase: "validation"` requires the `native_attempt` identity.
It invokes the maintained proposal controller:
`admitted` submits once, `submitting` or `running` reconciles, and `complete`
requires a fresh native inspection matching the journal's allocation and
RunRecord. Admission budgets remain in the existing research journal.

`phase: "final"` instead requires `remote_manifest`, the frozen final ciphertext
manifest path. `directory` names the consumed local grant whose verified private
delivery receipt binds the original journal's one-shot release. The adapter calls
`submit-final` once, then `reconcile-final` after the grant's exclusive `native`
directory exists. The controller reuses native staging, submission, accounting
and dataset verification; completion retains the original validation record.
The cluster's report task must equal the shared queue task. A missing receipt or
failed final submission never restores the grant or authorizes another allocation.

A WRDS request additionally names `python`, pinned `spec`, absolute `output`
and boolean `adopt`. `adopt: true` reconciles a pre-existing native attempt.
Otherwise durable submission intent is recorded before invoking the maintained
held-submission command. Subsequent polls inspect the receipt and resume a held
job. Completed work is collected and checked against a freshly measured native
completion digest, local artifact hashes and the shared RunRecord.

Four requests may be active; one least-recently-renewed receipt is polled per
tick. The existing shared fleet lease file protects `research/controller` on
the configured node and `research-controller@<node>` only for the bounded
controller operation. Native scheduler waiting consumes no local worker slot.
Cluster resource requests and allocation limits stay with the native controllers.
Transport failure records progress and preserves the receipt for the next poll.
Missing submission receipts, changed inputs and partial collections refuse
success; they never authorize a replacement native job. Stop an active native
job through its controller and reconcile; generic queue cancellation is only
available before a research request is claimed.

**The netboot runner arms machines, and nothing is typed** (MCPs board task
83102d9d). A `netboot-arm` row (MCPs migs 807 and 809, queued only by
`dispatch_netboot_arm`) names one roster machine as its requested node. The
`fleet-netboot` hub-runner row runs `fleet.cli.tick --lane netboot` on the
machine holding the netboot duty, as `fleet-netboot-<alias>`
(`fleet.cli.tick.runner_identity`), with the installed netboot release, the
node binary, the served root and the workspace env file. The agent claims the
lane with no node filter, since the machine an arm names is its target and
never where it runs. Before the job is reported started it checks the
release's seal with that release's own verifier (`fleet.core.sealed_release`,
the check the session verbs make too) and reads the tailnet writer OAuth
client, `TAILNET_WRITER_CLIENT_ID` and `TAILNET_WRITER_CLIENT_SECRET` (MCPs
board task 3b4c0931), from the
env file through `platform_core.env_file`, the one reader of the stack's
`.env`; then it runs the release's own `fleet-netboot arm --node <machine>
--registry <the registry directory the tick loaded>`, whose `machines/` names
each machine's boot MAC, with the seal's revision as
`--commit` and the key in the arm's environment alone
(`fleet.core.netboot_arm`). Exit 0 with one `NETBOOT ARMED` line closes it
`passed` with that line, which names the key's id and never the key. A refusal
raised before anything is minted (`NETBOOT_ARM_MACHINE_UNKNOWN`,
`NETBOOT_ARM_MACHINE_NOT_ARMABLE`, `NETBOOT_ARM_LIVE`) closes it `refused` with
the arm's own text, and any other exit `failed` with its output's tail. Output
carrying a `tskey-` is never copied into a detail.

**The ledger row names the submitter, not the runner.** A row stamped with
the hub runner's label would record only that the runner ran something; the
queue job carries the enqueueing session's label and UUID, and those are what
`run_lease.take` is given.

**What a runner holds is asked of the queue, never remembered locally.** A
runner that crashed between launching a suite and writing itself a note would
otherwise strand a job on a node with nothing left that knows to collect it.

## Two lanes, one runner per node: `fleet-node-agent`

Until 2026-09-21 the queue had ONE runner, the hub's `fleet-agent`, claiming
the oldest row once per tick whatever it was. Two defects followed from that
shape and MCPs board task fd5cabfa names them: nine nodes behaved as one slow
node, because a tick claimed one job and the next tick was three minutes
away; and a session revive waited behind a queue of checks, because the FIFO
had one head. Two changes end both.

**The queue is partitioned into lanes** (MCPs migration 532, `packages/db`).
A row's `command` decides its lane: `build-bases` and the session verbs are
the `hub` lane (the session verbs have their own `sessions` lane since mig 795,
and `netboot-arm` the `netboot` lane since mig 807), `check`/`lint`/`test` are
the `node` lane, and
`dispatch_claim` takes `lane` and claims only from that one. A revive never
queues behind a check because no runner ever asks for both at once
(`test_agent.py` pins the hub claim; `test_node_agent.py` the node claim).

**Every enabled node has a runner of its own, on the hub.** `fleet-node-agent
--config registry --node <alias>` is one tick for one node
(`fleet.cli.node_agent`). Every tick is a job of the MCPs hub runner (MCPs
board task b27bd34a): one resident loop, `session_audit.hub_runner`, kept
alive on diphtheria by the account's `corvis-hub-runner.service` user unit
(MCPs `scripts/host/lib/hub-runner.sh`, board task c39ee056; austinpc's two
Windows instances are retired, board task edaabb43), runs every row of MCPs
`scripts/fleet/hub-jobs.json` its machine's duties name, and its fleet rows
expand to one job per node `registry/machines/` dispatches to and the hub's
own. Each job runs `poetry run --
python -m fleet.cli.tick` in `tools/fleet` as a transient unit bound to the
runner's, fires on the multiples of 180 s since the epoch, runs one at a
time, and is ended at its row's 2400 s deadline; each writes its log under
the directory the runner passes as `--log-directory`. MCPs `maketools
ship-hub-runner` cuts and installs the runner's release, and its release-sync
row keeps it current.
The runners live on the hub and not on the nodes because the hub holds the
git credentials and the ssh keys, and the tailnet policy lets a
`tag:corvis-node` open no socket to it. Two checks submitted in one breath
are claimed by two runners in one tick, which is the measurement A1 asks for
and the wiki page (`fleet-check-runner`, mcps-codebase) carries.

**A tick fills its node's room, not one slot of it** (MCPs board task
48842bfd). After every job it launches, a runner runs its whole gate again
(`fleet.cli.node_ready.ready_state`: the ledger, which already charges the
run just launched, then the probe) and claims again, until the node has no
room or the lane nothing it fits (`fleet.cli.node_agent.fill_pass`). It stops
at the first job it refuses, so a node that stops answering mid-staging
refuses one job, not the lane. Each pass leaves out of its claim what a lease
on the node holds now (`fleet.core.run_lease.held_on_node`, below), so the
pass after a launch never claims a second job of the project still running
and refuses it `LEASE_HELD`; that job waits in the lane instead. Each
pass logs `<alias> launched N job(s) this pass`.

**A node runner serves across its starts: a run is closed when it ends, a
job claimed when it arrives** (MCPs board tasks c1d48330 and 8993c306,
`fleet.cli.node_serve`, `fleet.cli.node_watch`). Measured 2026-10-04, a
finished row was closed only by the start after it ended and a queued job
claimed only by a start, so every session waiting on a fleet verdict waited
up to two 3-minute ticks beside its work. The 100-second watch that followed
covered 100 s of each 180 s, and a fill running past the next fire skipped
it: loki's 77fee544 ended 05:53:35Z on 2026-10-05 and closed at 05:57:44Z.
Now a start runs its collect and fill passes and keeps serving. Its watch,
on a thread of its own, is handed the runs the queue says this runner holds
running and each run the fill pass launches, reads each held run's result
off the node every `node_poll_seconds` (5 in `registry/settings.json`) and settles one
that has ended through the same `collect_one_job` the collect pass uses,
under one lock so a run is settled once. At the same pace the main thread
lists the queued jobs (one `dispatch_list`) and runs the fill pass again
when a job naming this node or no node has arrived since its last fill, or
the watch has settled a run since then: the same fill pass, so a claim still
goes through the probe, the tags, the leases and the owner reservation. At
each fire boundary, a multiple of 180 s since the epoch
because the MCPs hub runner fires each tick there, as the scheduled tasks it
replaced did from local midnight every 3 minutes, it runs the collect pass again, which renews every running
job's queue lease as each start did. It hands over only 10 s before a
boundary and only while no settle is under way, once it has served
`node_serve_seconds` (1200 in `registry/settings.json`, at most 1800 so a drain ends
inside the launcher's 39-minute wall) or `refs/fleet/rolled`, read through
git from its records directory, has moved, so a roll still takes effect at
the next boundary; a serve past its time claims nothing more. The fires that
arrive while it serves are skipped by `IgnoreNew` on purpose, and the node
goes unwatched only for those 10 s and the next start's few, once a serve. A
serve of zero seconds is one start's shape: its opening passes, no listing,
no read of the roll, and a handover at the first boundary. A poll that finds
a run still going writes nothing to the queue, and a watch holding no run
waits on its condition with no call. A serve's log ends `<alias> served N s
from <time>: F fire(s), P fill pass(es), Q poll(s), C run(s) closed, K still
watched; handed over at <time> before the <time> fire: <reason>`.

**A node that misses a read costs a poll, never the serve, and a renewal is
written once a fire** (MCPs board task c1d48330, `fleet.cli.node_collected`).
Between 2026-10-06T20:16Z and 2026-10-07T03:48Z, 3 of 265 node rows closed
more than 30 s after their checks through the runner's own doing, besides an
outage of the shared ledger (board task 9fcc78a3) and mcp-fleet's deploys:
lavender-wsl stopped answering ssh at 02:53Z, a settle's tail read raised
and ended the serve, and the watch had counted an unanswered collect as a
closed run (1d5d0a1a and 24c4e934, 192 s); serendipity adopted run 220c2a9e
and never handed it to the watch (47.5 s); and twice a start report landing
during the collect pass made a just-started run look lost to two jobs, so
DISPATCH_CLAIM_AMBIGUOUS ended the serve. Nine lease renewals came under
60 s after the one before, from collect passes rerun while the queue
recovered. Now every collect says what it did (settled, renewed, still
running, unreachable, not live, not held) and only a settle counts as
closed; the settle reads the transcript's tail first, with a read the node
may miss, so an unreachable node changes nothing and is read again at the
next poll; a run the collect pass adopts goes to the watch and is the
pass's own; a job that names a run is the one that launched it when this
runner holds it, and another runner's run when that runner does
(serendipity's two runners share one node); a run still going has its
lease renewed only once it was last set 70 s or more ago (`leaseExpiresAt`,
decoded to whole seconds, so 60 s let a pair land 59.2 s apart), so the
boundary's pass renews and a pass seconds later writes nothing; a removal of
cache generations the node does not answer is left to the next one; and a run
whose node missed the boundary's read is owed a renewal, which the watch
makes at its first read that reaches the node.

**A settle closes the queue job before it removes cache generations.** The
node's cache generations no run reads (`fleet.core.cache_generations`) are
removed after the row and the queue job are closed, not inside the retire,
since housekeeping's script round trip stood between a finished check and
its close (11 to 24 s per settle on loki and sedona); a removal that fails
after the close leaves nothing a later one does not remove.

**A run that never settles is retired too** (MCPs board task 3d9eec0d). The
lease records the run's retire owed in `runs/retires.jsonl` before anything is
staged, so a run refused while staging, which gives its lease back, and a
launch whose runner ended, whose lease lapses, are retired by the next collect
pass on their node as a settled run whose node missed the retire is. Before:
12 run directories on six nodes, measured 2026-10-18, from runs the feed ends
at `refused` or at `leased` or `staged`, none of them in the record.

A retire the node answered and failed is recorded `failed` and no pass sends
it again, since its fault would fail it again. Once the fix is rolled, the
session that made it runs `python -m fleet.cli.owe_again --config registry
--run <run> --because "<what fixed it>"` on the hub, and the node's next
collect pass retires the run. The hub's stage-root case
(`tools/fleet-execution-hub`) names every such run, since its directory is
held by no live run and no owed retire.

**Every cache on a node has an owner that bounds it** (MCPs board task
346abc45), measured 2026-10-18 before the change:

| State | Owner and bound |
|---|---|
| Poetry virtualenvs | In the project (`POETRY_VIRTUALENVS_IN_PROJECT`), inside the run's directory, so the retire removes them with the run. Before: execdocker's cache held 168 (19 GB) on lavender-wsl, 77 (8.5 GB) on diphtheria and 33 (3.7 GB) on colossus, never swept |
| npm's and poetry's download caches and Playwright's browsers | `<cache root>/tools/<generation>`, the generation named in `tools/current`; a build starts a new one when the current is 14 days old (`fleet.core.tool_caches`) and writes the one it reads into its run directory. After each retire, every generation that is not current, that no run directory names and that stopped being current over an hour ago is removed whole, so no tool finds half of one of its entries. Inside a generation, every `playwright install` also removes each browser no live link under `.links` names. Before: poetry's held 10 to 23 GB per node, per-file removal was measured to break both tools, and `<cache root>/ms-playwright`, outside every generation, was bounded only by those installs |
| A docker project's copy in execdocker's home | Removed by the build when it runs to its end, and by the retire on a node that declares `docker`. Before: lavender-wsl held eight stopped builds' copies, 11 GB |
| MCPs' prepared installs and venvs (`npm-installations`, `python-venvs`, `fleet-prepare`) | MCPs maketools' prepare store: the three newest entries per pool are kept, and an older one is removed once it has gone an hour unused |

**A node claims only what its tags admit.** The claim sends the node's
derived tags (`fleet.contracts.tags.node_tags`: its platform, plus `gpu` for
a CUDA device, `testdb` for a node whose `test_database` is true, and `rust`
or `cxx` for each toolchain the node declares a version of) and the
queue's `required_tags <@ tags` containment returns
only jobs the node satisfies. A job's `required_tags` should be the
project's from its registry file, but what a job NEEDS is its project's
declaration: every claim names the projects the runner fits, each assessed
against that declaration and the runner's tags, so the job's own tags only
route it. The runner refuses `PROJECT_TAGS_MISMATCH` on claim only when a job
requires a tag its project does not declare; a job naming fewer runs under
the declaration. This workspace's declared `gpu` is held to
the CUDA devices the fleet audit's inventory measures through `nvidia-smi`
on every pass (its `roster.gpu` row, MCPs board task e447de72), so the
tags the hub plans by are the devices the nodes have.

**`testdb` is the one tag the roster does not measure** (MCPs board task
6bbfd171). A node declares `test_database: true` when it runs
`corvis-fleet-testdb`, the loopback postgres container on tmpfs that MCPs
`scripts/host/lib/fleet-testdb.sh` defines once and every Linux testdb
node's provisioning sources (diphtheria's `scripts/host/diphtheria/provision.sh`,
and every other Linux node's `scripts/host/fleet-node/provision.sh`, which
replaced lavender-wsl's `scripts/host/lavender/provision-fleet-node.sh` of
MCPs board task c4fc4f3e), and that MCPs
`scripts/testdb-setup.sh --container corvis-fleet-testdb` restarts empty and
migrates before a run, as a CI job's service container is fresh. Every MCPs
project whose suite starts from `packages/db`'s global test setup names that
step as an install step and requires `testdb`, so it only reaches a node where
the setup can succeed. No node is pointed at the production cluster's
`corvis_test` (measured 2026-09-26: sedona and serendipity both refuse the
cluster's 6432 anyway), which would put the `corvis_app` credential on it.

**Every testdb node is a Linux one.** From 2026-10-02 (MCPs board task
daae17f2) until board task e705f08c a Windows node could hold the same
container through Docker Desktop, installed by `install-windows-testdb.ps1`;
that installer, the Windows build and the Windows probe left with the
fleet's last Windows node, so the container exists only where a Linux
provisioner put it.

**A testdb node also carries the gte-small model directory** (MCPs board
tasks 1d8e2e0e, f7085447 and f28396f8). MCPs `packages/embed` and
mcp-proxy's live tool-search block load `Xenova/gte-small` from
`<stage root>/cache/models` with remote fetch off, and a node whose directory
lacks the model fails the block by name instead of waiting on
huggingface.co. Every node's directory was filled by hand on 2026-10-07, and
the two nodes that joined after it, colossus and loki, each failed their
first mcp-proxy check with no directory at all on 2026-10-09. The linux
provisioners fill it (MCPs `scripts/host/lib/fleet-node.sh`,
`step_fleet_model_cache`).

**`corvis-fleet-testdb` is node-local** (MCPs board task c4fc4f3e). The
workspace's `node_local_resources` lists the exclusive resources each node
runs its own copy of, and a lease holds such a name as `<name>@<node>`, so
it contends only with a lease on the same node, while every other declared
name stays one thing in the fleet (`fleet.contracts.resources.scoped`). The
check made before a node is chosen asks only about fleet-wide names. Until
this, the name read as fleet-wide, and on 2026-09-29 lavender-wsl closed
every testdb job it claimed while diphtheria ran one as `RESOURCE_HELD ...
held fleet-wide`, so two testdb nodes still ran one testdb suite at a time.

**lavender-wsl was the second testdb node** (MCPs board task c4fc4f3e), from
2026-09-29 until lavender was reimaged to Ubuntu on 2026-10-19 and the WSL
guest went with Windows (MCPs board task 1e9ab738); lavender itself carries
the lane since. Until 2026-09-29 diphtheria was the only node carrying
`linux` and `testdb`, and that day 17 queued jobs could land nowhere else, the
tail about six hours behind, while diphtheria sat at load 42 to 57 on 8
cores. lavender-wsl was the WSL2 Ubuntu on lavender that already hosted the
GitHub runners: 16 logical cores and 25.4 GB given to the distro, g++ 13.3.0,
Node 24.21.0 from NodeSource. The hub reached it as `Host lavender-wsl`
through lavender's own sshd, which ran `nc` to `127.0.0.1:2222` inside the
distro over the session's stdin and stdout, so no port opened on the tailnet;
the first way in, `ProxyJump lavender` through WSL's localhost relay, kept a
dead mapping for the port after an sshd restart on 2026-09-29. Half its cores
and memory were reserved for the runners beside it, as half of lavender's
cores are now.

**lavender-wsl was the second docker node** (MCPs board task c4fc4f3e), over
the same span. On
2026-09-29 diphtheria's stack grew past the half of its memory the fleet
leaves it (14.2 GB available against the 16 GB reservation after the 09:54Z
reboot), so it claimed nothing, and every project requiring `docker`,
`tools/fleet-execution-linux` among them and so every roll, had nowhere to
run. lavender-wsl's system daemon was Ubuntu's docker.io, the runners' rootful
one, which ships no rootless tooling, so MCPs
`scripts/host/lib/fleet-exec-docker.sh`, the definition diphtheria's
provisioning sources too, installs Docker's `docker-rootless-extras` archive
of the same release (29.1.3, pinned by its sha256, since Docker publishes
none), gives `execdocker` its own rootless daemon, grants the runner account
the sudo the isolated build uses, and holds docker.io, because Ubuntu's
unattended upgrades would otherwise move it away from the declaration. Its
`docker` was `29.1.3`, the ServerVersion that daemon reported as rootless.

**`rust` is declared as a version and re-measured every tick** (MCPs board
task 1e2da299). `services/covenant-radar-api` builds the maturin crate
`libs/cleargbm_rs` from source in `poetry sync`, and on 2026-09-26 no node
had cargo, so the project requires `rust`. A node's `rust` is the version
`cargo --version` printed there (`1.98.1` on diphtheria, installed by rustup
at user scope, reversible by removing `~/.rustup` and `~/.cargo`), or null.
The toolchain probe asks cargo on every runner tick, and the runner claims the
`rust` tag exactly when cargo answers a version, whatever the declaration says
(see *Tags are detected every tick* below), so the tag cannot outlive the
toolchain. The Linux dialect puts `~/.cargo/bin` on every script's PATH beside
`~/.local/bin`, so the probe and the build find the same cargo.

`libs/covenant_ml` builds the same crate and requires `rust` alone. From
2026-10-03 (board task eaf80425) it also required `windows`, because its
`benchmarking/power.py` tests called the Win32 power-throttling boundary
through `ctypes.WinDLL`. Once serendipity was reimaged to Linux on 2026-10-20,
no runner carried `windows`, so no node would take the check. The module was
removed for MCPs board task 2e880b79, under the operator's 2026-10-20 ruling
that Windows code is converted to Linux.

**`cxx` is the same contract for node-gyp's C++ toolchain** (MCPs board task
3f19c136). Every MCPs TypeScript project installs with a root `npm ci`, which
rebuilds `hnswlib-node` under node-gyp, so every project in the registry
whose install runs a plain `npm ci` requires `cxx` (`MCPs/execution`, which
installs with `--ignore-scripts`, does not). A node's `cxx` is what
`g++ -dumpfullversion` prints. Measured 2026-09-27: diphtheria has g++
13.3.0, and no Windows node had the VC tools then, so those projects queued
for diphtheria instead of failing in `npm ci` on a Windows node.

**Tags are detected every tick, and the declaration is only compared** (MCPs
board task 939ec5c7). A runner claims with what its node's toolchain probe
answers this tick (`fleet.contracts.detection`): its platform, `gpu` when
nvidia-smi lists a device, `testdb` when the `corvis-fleet-testdb` container
exists, `rust`, `cxx`, `docker` and `stack` when each answers a version, and
`ffmpeg` when the executable is found, and `hooks` when the build account
carries `~/.claude/corvis-hooks.json` and its system interpreter imports
ruff, mypy, pytest, xdist and pytest_cov in one `-c` (MCPs board task
ec895824). `hooks` routes MCPs `packages/claude-hooks`, whose check runs on
the system interpreter as Claude Code runs the hooks and whose live suites
reach the board through that route file; measured 2026-10-02, sedona carried
the file and pendragon and serendipity did not. The package is one project,
`MCPs/packages/claude-hooks-linux`, since its Windows twin left the registry
(MCPs board tasks c1832d51 and 5b040fdb); diphtheria is the node carrying
the tag, its pins installed by MCPs `scripts/host/lib/hooks-tools.sh`. A node with pip but without
the tools is offered `python -m pip install --user` at the versions the hub
ran that check with; the route file only `install-hooks-node.py` writes.
`go` is found the same way, asked `go version` because go has no `--version`
flag (MCPs board task 1da15750). It routes MCPs `rcs-bridge`, the Go half of
the SMS gateway, which became a fleet project with `MCPs/sms-gateway` once an
outage on 2026-10-05 showed a host process needed a check a closure could
cite; no fleet node had go that day, and a node without it is offered
`sudo apt-get install -y golang-go` (any go from 1.21 fetches the 1.27.1
toolchain the module names). `chrome` is Google Chrome at the path
Playwright's channel `chrome` launches, `/opt/google/chrome/chrome`, never
looked for on a PATH it is not on (MCPs board task 2f596185).
It routes MCPs `web-scraper`, whose check launches that channel in three
real-browser cases: diphtheria, a `cxx` node without Chrome, failed exactly
those three on 2026-10-04 (fleet job 874c160c) and 2026-10-09 (56d7fbb5)
and passed the other 490, so the verdict depended on which node claimed the
suite. A node without it is offered the .deb Google ships, the same command
`runners.json` provisions the asset with. `pdftotext` routes MCPs
`packages/wiki-check`, whose audit rules read PDF sources through poppler's
pdftotext and depend on its exact page breaks (MCPs board task 6dd73ada). It
is found on the PATH and asked `-v`. Every live runner had it on 2026-10-11,
but nothing declared it and the roster's apt packages lacked poppler-utils;
a node without it is offered `sudo apt-get install -y poppler-utils`.
lavender, the runner host since its reimage to Ubuntu, gets go, ffmpeg and
pdftotext from `runners.json`'s base `apt_packages` and Chrome from its
`assets`, so a `--rebuild` keeps all four tags, and
`test_runner_contracts.py` fails if a tagged tool apt can supply is missing
from that list or Chrome's path from those assets. Installing a tool therefore makes a
node eligible on its next tick with no file edited, and a toolchain that
disappears takes its tag with it. Each difference from the node's machine file is logged
on every tick, for example `pendragon (pendragon) declares cxx none but its
probe reports cxx '17.14.37710.0', so it claims with the cxx tag; set cxx to
'17.14.37710.0' in its registry/machines file`, because the hub's own planning (`fleet-run`,
preflight) still reads the declaration without asking the node. Until this
change a node whose answer differed from its declaration claimed nothing at
all, for every job. Measured 2026-10-02, every dispatchable node's answer
matched its declaration.

**Each writer has its own probe scripts on a node** (MCPs board task
939ec5c7). The capacity and toolchain probes are written to the node's stage
root and run there, as `fleet-capacity-<writer>` and `fleet-toolchain-<writer>`
(`fleet.core.names.capacity_probe_stem`): a runner writes as its runner name
(`serendipity`), and `fleet-run`, `fleet-nodes`,
`fleet-preflight` and `fleet-bootstrap` each as their own command name. Until
2026-10-03 every caller wrote one `fleet-capacity.ps1` and one
`fleet-toolchain.ps1` per node, and serendipity's two runners, which tick on
the same minute, collided on the toolchain script every tick: the node runner
read `The process cannot access the file ... because it is being used by
another process`, the elevated one `Set-Content : Stream was not readable`,
and the node claimed nothing with 11.5 GB free. The two old files stay on
each node's stage root until removed by hand; nothing writes or runs them.

**The identity is derived, never configured.** The label is
`fleet-node-<alias>` and the session id is the version-5 UUID of
`fleet-node-agent/<alias>`, so every tick of one node's runner is one session
on the board's ledger and `held_by` finds its own claims across ticks. Each
node job's boot command runs the tick once in its `announce` lane before its
first tick, which posts the check-in that registers that session (MCPs mig
530) and claims nothing. The hub's job does the same through the tick's
`hub-announce` lane: the queue's close posts a task-naming job's
outcome under the closer's label, and an exit-session job names the finished
task (MCPs board task 2fecad69).

### What a node-lane job carries: a commit, not a working tree

`dispatch_submit` for a `check`/`lint`/`test` REQUIRES a `sha` (forty hex; the
queue's `fleet_dispatch_jobs_sha_pin` CHECK refuses a node-lane row without
one) and the runner exports THAT commit:

1. the project's `source` in its registry file names its `remote`, the `path`
   inside the repository the recipe runs in, and the `install` steps a clean
   export needs before `make check` (argv lists under a grammar that admits
   no shell syntax, rendered into the node's script as spelled);
2. the sha is fetched from the remote into a bare mirror on the hub
   (`runs/mirrors/<key with / as ->.git`, `git fetch --no-tags --force
   <remote> +<sha>:refs/fleet/fetched`, so the next fetch offers that ref's
   history and receives only new objects, inside the mirror's fetch turn,
   so a second export of one commit waits and then finds it held: MCPs board
   task 86f70457) BEFORE any lease is taken, so a sha the remote has never seen is
   refused `SHA_NOT_ON_REMOTE` with nothing held, and a project with no
   `source` is refused `PROJECT_REMOTE_MISSING` by name; a fetch git loses
   (a reset connection, a TLS drop, a timeout) is sent again, up to three
   attempts within 900 s, before the job is refused `EXPORT_FAILED` naming
   each attempt (`fleet.core.mirror_fetch`, MCPs board task dfcdfb48: six
   such refusals in three days to 2026-10-18, one of which passed on
   resubmission at the same sha);
3. `git archive --format=tar.gz <sha>` is staged through the same verified
   transport every dispatch uses, so the tree on the node equals the commit
   by construction, and the staged tree is made a one-commit repository
   (`git init`, `git add --all --force`, one commit as `fleet`) because MCPs
   `packages/db`'s migrator admits test migrations against `HEAD`;
4. every repository the project's `source` declares under `companions` is
   exported BESIDE it, from its own mirror at the TIP of the declared `ref`,
   into `<stage_root>/<directory>` — which is `../<directory>` from the
   export root, the layout a workstation already has — and made a one-commit
   git repository there;
5. the build script points npm, poetry and Playwright at the current
   generation of the node's caches
   (`<stage_root>/cache/tools/<generation>/{npm,pypoetry,ms-playwright}`), keeps poetry's
   virtualenv in the project, runs the install steps at the export root,
   and `make check` in `path`.

**Why a companion exists and why a node cannot fetch one itself** (MCPs board
task 0515040d). A repository whose check reads a second repository has
nothing to read on a node, which is handed that project's commit and nothing
else; every `check-fleet` job for it died in `make lint` before a single
test. The node cannot go and get it: both repositories are private and a node
holds no git credential, measured on sedona 2026-09-22 as `git ls-remote`
exiting 128 with "could not read Username for 'https://github.com'". So a
companion travels from the hub with the work, through the same mirror,
archive and digest-verified transport as the project's own commit. It names a
`ref` for checks that intentionally follow an upstream branch; the resolved
commit is recorded on the feed, so a verdict can be read against the workspace it was
measured with. The directory is REPLACED each run rather than unpacked over,
and the archive stays in a `<directory>.stage` beside it, so what the recipe
reads is the export of one commit and nothing else. Every project declares
`companions`. All current projects declare `[]`: slime now verifies a committed
integrity lock offline instead of reading a second checkout (20e977b6).
The companion transport remains available for projects that genuinely need it.

**Every remote script ends at the first command that failed**
(`fleet.core.dialect.checked_script`): `sh` scripts carry `set -e`. Measured
on sedona 2026-09-22, when the fleet still sent PowerShell: `powershell -File`
exited 0 even when a native command in the script exited non-zero, so a `tar`
that unpacked nothing and a `git init` that never ran were both reported as a
staged tree.

`fleet-run` still dispatches the working tree (its own tarball, no sha) for
the interactive case; the queue path never does.

### The verdict is posted where the closure can cite it

When the collect pass finds the suite finished it reads the result and the
last 200 lines of the transcript off the node, composes one line
(`fleet.core.verdict`) and closes the job with it as the detail. A job that
names a task gets its row from the queue itself: the close writer in MCPs
`packages/db` (`fleet-dispatch/outcome.ts`) posts every terminal outcome,
passed, failed, refused or cancelled, to the task's thread in the
transaction that closes the job, opening `@<submitter> FLEET JOB <id>
<status>`, under the closing runner's label (MCPs board task 2fecad69). The
mention is what wakes a submitter above the board-wake context ceiling, and
until it existed a refused job posted nothing at all. A job that names no
task gets this runner's own post, in the `fleet` room addressed
`@<submitter>`:

```
FLEET-CHECK 3f2a9c1e MCPs/packages/wiki-search sha=<40 hex> node=lavender exit=0 banner=yes tests=887p/0f coverage=statements=100% branches=100% budget=within:41s/300s ended=2026-09-21T10:04:12Z log=lavender:/home/corvis/fleet/stage/logs/<run>.log run=<run>
```

`banner` is whether `=== ALL CHECKS PASSED ===` appeared; `tests` and
`coverage` are read from vitest's or pytest-cov's own lines and say `unread`
when neither printed them. `budget` is maketools' five-minute check budget
verdict (MCPs board task 1b152218): `over:<took>s/<budget>s` from its
`CHECK OVER BUDGET` line, which fails a check whose every suite passed,
`within:<took>s/<budget>s` from its `CHECK BUDGET` line, and `unread` when
the run printed neither. `ended` is when the check ended, UTC to the
second: the time the node's result records, or when the runner stopped a
run past its lease (MCPs board task c1d48330), so the row's `closed_at`
minus it is how long a finished check waited to be closed, which no other
record carried. The same line is the job's close `detail` on the
queue, so a closure may quote it as its check evidence and the manager audit
resolves it against the `fleet_dispatch_jobs` row it names.

From an MCPs package, `make check-fleet` submits HEAD for the package with
its tags and prints the job id (`packages/maketools/README.md`); `make check`
stays local.

## Subscribing from a Claude session

`fleet-watch` prints one line per event and exits. That is the whole
integration:

```
Monitor({command: "fleet-watch --config registry", description: "fleet events"})
```

There is deliberately no `--follow`. A polling loop belongs in the shell where
its interval and filter are visible, not hidden inside the command.

**Filter for every terminal kind, not just the happy one.** `refused`,
`passed`, `failed`, `cancelled` and `lost` all end a run;
`fleet.contracts.feed.TERMINAL_KINDS` names them together so a subscriber
cannot enumerate the successes and think it is done. Silence is not success —
that is exactly what the two wedged suites looked like from outside.

## What the budgets are for

**Memory divides; cores only bound.** Every xdist worker importing torch
reserves about 1.1 GB of commit while its working set stays at a few
megabytes, so Task Manager makes a wedged run look harmless. `sedona` has 20
logical cores and 11.4 GB free: dispatching on the core count asks for ~22 GB
and reproduces incident (2) on a smaller machine. `admissible_workers`
divides free memory by what a worker costs and clips to the cores left after
the owner's reservation — never the other way round.

`reserved_cores` and `reserved_ram_gb` have no cluster analogue at all. Slurm
never has to leave a core for the person sitting at the node.

**The limit is the host's, not each runner's, and a node may say how many
checks its CPUs carry** (MCPs board task a85ef09e). A node that declared
`elevated` had two runner identities, two processes on the hub, until the
fleet's last Windows node left it (board task e705f08c), and each used to
charge the launches it had under way from its own memory, for the 40 to 80
s before a run reaches the ledger. On 2026-10-07 serendipity (8
cores, 4 reserved) held three runs and 6 workers that way, and fleet job
c858b46b's sms-gateway check ran 321 s of its 300 s budget. Now every claim
under way is written to `runs/host-claims/<node>/<job>.json` until its
launch ends (`fleet.core.host_claims`), every runner of the host charges all
of them, and a claim pass runs inside the host's claim turn
(`runs/host-claims/<node>.lock`), so two processes claiming for one host,
one runner's overlapping ticks included, claim one at a time and never both
read the room one claim is about to take. The pool alone was
still too generous there: fleet job 25d07c7c's sms-gateway check, granted
2 of the 4 spare cores beside an execution-elevated run granted the other 2,
took 306 s (build 73.7 s, suite 195.6 s) where it takes 162 s alone (build
48.8 s, suite 109.5 s), because a check's install, build and lint phases run
wider than its test workers. So `budget.checks_at_once` is how many
budgeted checks a node's CPUs carry at once, across its runners, measured:
serendipity's is 1, and `null` leaves a node to its pools until someone
measures it.

**A check yields memory to the stack; free memory is `MemAvailable`** (MCPs
board tasks a282400d and d111690a). diphtheria reserved 16 GB from
2026-09-20, sized when it read 26.4 GB free; once the stack's backends moved
in, `MemAvailable` already excluded them and the reservation counted them a
second time, so the node refused every dispatch (179 `NODE_OWNER_RESERVED`
ticks by 18:30Z on 2026-09-29). From then the Linux capacity probe
(`fleet.core.linux_capacity_probe`) also took off what every memory-capped
stack container could still grow into, so a job would never meet a worker
growing to its cap; on colossus that held back 17.13 of an idle host's
19.59 GiB available (8d3e4b16, 2026-10-15). The kernel decides instead now:
every check starts through MCPs' fleet-check-unit at `OOMScoreAdjust=1000`
with `OOMPolicy=stop`, so when the host runs out the check is ended before
any stack container, and the whole of it, with the result `oom-kill`
(`fleet.core.linux_unit_end`). `fleet.cli.node_settle` then gives such a job
back to the queue rather than closing it failed, and the queue requeues it
once and fails it the second time (`fleet.cli.job_close.give_back`), so a job
too big for any node never loops. `free_ram_gb` is `MemAvailable`, and a
node's `reserved_ram_gb` is the only memory dispatch holds back, for whoever
uses the machine. The compose caps stay, as runaway containment.

**That every stack host's compose caps can fire is summed in MCPs** (MCPs
board task 8d3e4b16). On 2026-10-15 a fifth doc-extract replica took the
caps diphtheria declared past what the probe then read free, and nothing
had counted the caps against the host. MCPs' `host-memory-budget.test.ts`
fails when the caps a stack host's compose files declare could not all be
reached before the host runs out, from `scripts/fleet/host-memory.json`,
which `make host-memory-measure` there measures on each host in one read:
its `MemAvailable` and each capped container's cap and resident memory.
colossus's `reserved_ram_gb` is 1.5 since 2026-10-15: its 6.0 held memory for
`glimmer.service` and a desktop session, and the operator stopped glimmer
and ruled colossus a server. Its `reserved_cores` fell from 8 to 4 the same
day for the same reason: four were glimmer's and the desktop's (aa8a6d0e1),
four the OCR service's two torch processes at `OMP_NUM_THREADS=4`
(3f384639a), which stay.

**How many runs a node holds is live, not declared** (MCPs board task
939ec5c7). Every node used to declare `max_concurrent_runs`, and every node
declared 1: on 2026-10-02 thirteen jobs queued behind four running while the
probes showed memory a second run could have used. The field is gone from
`NodeBudget`, the registry and every reader. `capacity.assess` now charges a
node's live runs first, from the ledger (`records.live_load`): the workers each
was granted come off the node's cores and `workers x worker_ram_gb` comes off
its free memory, whether or not those workers have spawned yet, and what is left
is granted. One grant is capped at `capacity.job_ceiling`, half the cores the
owner leaves rounded up and never below the project's `minimum_workers`, so the
first job leaves room for a second. On the fleet as declared on 2026-10-20 the
ceiling was sedona and lacrymal 9, pendragon 7, loki 5, colossus and lavender
4, serendipity 3 and diphtheria 2, unless a project's minimum is higher;
loki left the roster on 2026-10-25 (board task e705f08c).

**A node running several jobs is never offered one its own lease file would
refuse.** The projects a runner hands `dispatch_claim` are the ones that fit by
room, less every one `run_lease.held_on_node` finds held on that node now: a
project already running there (`LEASE_HELD`), and every testdb project while
one testdb suite runs there, since each declares the node's own
`corvis-fleet-testdb` (`RESOURCE_HELD`). A claimed job refused for either is
closed for good, so the job stays queued for another node or a later tick, and
the runner logs what it left out.

A node shared with CI keeps its headroom through its reservations, not a run
count: lavender became a CI runner on 2026-09-06 and `opus-artifact-sweep-0902`
saw a heavy job saturate it until ssh connections were **reset** (TCP 22 and 445
open, tailscale direct: alive and overloaded, not down). That is why
`lavender` reserves 8 of its 16 cores and the linux probe reads the CI
slice, and both still bound every grant.

## The roster, and `enabled`

**The fleet is written down once**, one file per machine under
`registry/machines/`, named by the machine (MCPs board task e447de72).
Until 2026-10-19 it was written down twice: MCPs `fleet-mcp/fleet-nodes.json`
held every machine on the tailnet with its identity and reach, and
`registry/nodes.json` here held the machines dispatch may use with what each
can take. Both declared `enabled` and `platform`, each declared a `gpu` and a
`test_database`, and nothing held them together but a comparison a session
ran by hand. On 2026-09-05 they disagreed about whether loki should answer,
and every auto-select dispatch paid an ssh timeout rediscovering it; on
2026-10-19 lacrymal's `test_database` read true here and false there until a
session noticed. The operator, that day: "One roster. Fold fleet-nodes.json
into the API registry and have everything read it." It lives here because
every dispatch must work with no MCPs checkout present, and MCPs already
reads this registry. Each runner stages the registry it read beside every
run and exports its path as `CORVIS_FLEET_REGISTRY`, so a suite on a node
reads the roster the runner that launched it read.

**A machine's file** carries what this package decodes
(`fleet.contracts.machine`) and what MCPs `fleet-mcp` decodes beside it
(`fleet-mcp/src/registry-types.ts` documents every field):

| field | what it says |
|---|---|
| `role` | `hub` (reaches everything else, runs the docker fleet), `vpn-jump` (holds the UCI VPN and proxies to hpc3; MUST have a Cloudflare tunnel, because a node holding the VPN loses its tailnet endpoint), `worker`, `session-host` (runs the operator's Claude Code sessions), or `client` (reaches the fleet, never provisioned by us: the phone) |
| `platform`, `user` | `linux`, the one platform this package decodes since board task e705f08c, and the account the fleet provisioned (null for a client) |
| `enabled`, `disabledUntil` | whether it is expected to answer at all, and the date a time-bound disable ends |
| `tailnetIp`, `tunnel`, `reachKey`, `bootMac`, `duties`, `ciWorkersPerRunner` | how it is reached and what it holds, read by MCPs `fleet-mcp` |
| `ledger_observer` | which arm records its sessions in the agent session ledger |
| `guest` | null, or `{host, sshPort}` for a machine reached through the one it runs inside |
| `boot` | what it must bring back after a reboot beyond sshd and Tailscale |
| `dispatch` | the reason it takes no dispatched work, or what dispatch may take from it with `held` |
| `notes` | prose for a reader; nothing parses it |

The fleet-wide identity (the fleet's ssh key and its fingerprint, the
Cloudflare zone and account, and `uciVpnProfileSha256`, the SHA-256 of
`uci-duo-profile.xml` measured byte-identical on loki and sedona on
2026-09-04, so a vpn-jump whose profile differs is known not to run the
working nodes' VPN config) is `settings.json`'s `identity`.

**Add a machine here first, as one new file**, which MCPs `fleet-mcp`'s
`fleet-enroll` writes, then audit it. On the new machine, at its console,
the `fleet-node-setup` skill from a CORVIS stick (repo `corvis-stick`,
`stick-templates/skills/fleet-node-setup/`) provisions sshd, the key, the
ACLs, hardening and power policy, then re-measures all of it and exits
non-zero if any did not take.

**Enabling or disabling a machine is one edit and an audit.** `enabled` is
said once, in its file, and decides both what the audit expects and whether
work may be sent there. Then re-run the host-side `fleet-audit` on the hub:
the corvis `fleet_*` tools serve a stored snapshot and probe nothing (a
container on the compose network has no route to a 100.x address), so until
the audit re-runs they keep reporting the old state and are not wrong to. A
scheduled task also runs it every twenty minutes.

**A time-bound disable goes in `disabledUntil`, never in `notes`.** A date in
prose has no reader: loki's note said "re-enable on 2026-09-08", and on
2026-09-09 it was still disabled while a session reported HPC3 transport down
and named that very disable as why the fallback jump host was unavailable.
`disabledUntil` is compared against the clock on every audit: on and after
that date the machine reports as DISABLE EXPIRED and the run exits non-zero
until somebody re-enables it or moves the date. Null is a standing gap, a box
that cannot answer and has no known return date.

**`ledger_observer` declares which arm records the machine's sessions** in
the agent session ledger (MCPs board task c90b6fb7): `pcsession`
(pcsession-mcp's observer on the hub, every 30 s), `fleet-arm` (this
package's observe pass, every 3 minutes) or `none`. The audit reads the
machine's watermark from `agent_session_observer_passes` (MCPs mig 529) and
reports `LEDGER NOT OBSERVED <node> <age>`, exiting non-zero, when it is
older than three of that arm's ticks or was never stamped.

**`guest` names the machine a node runs inside** (MCPs board task c4fc4f3e).
The first was lavender-wsl, lavender's WSL2 Ubuntu, until lavender's reimage
to Ubuntu on 2026-10-19 (MCPs board task 1e9ab738), and no machine declares a
guest now: the hub's ssh config reached it through lavender's sshd, which ran
`nc` to `127.0.0.1:<sshPort>` inside the distro, and its sshd bound that
loopback address alone. A guest has no tailnet device, so its `tailnetIp` is its
host's, the tailnet policy owes it no hosts entry, and the audit judges its
sshd bind and AllowUsers against 127.0.0.1. This package read the guest's
host as the node's `wsl_host` until board task e705f08c, which removed the
field with the Windows platform; MCPs `fleet-mcp` still decodes `guest`.

**`boot` is what a machine must bring back after a reboot** beyond sshd and
Tailscale, which the audit holds every owned machine to without a
declaration (MCPs `fleet-mcp/src/standard-boot.ts`, board task ebc80a03).
Each unit is a windows-service, scheduled-task, systemd, wsl-systemd or
listener, and says in `purpose` what breaks without it, because that is what
the alert has to say. Declared from a measurement of the machine, never from
memory; `units: []` says it carries nothing else.

**`dispatch` is required, and silence is not a reason.** A string is why the
machine takes no dispatched work and declares no capacity: `austinpc`, the
hub, sat for weeks enabled, 24 logical cores and invisible to the scheduler,
with no way to tell a decision from an oversight, and its file now says why.
An object is what dispatch may take from it (stage root, cores, RAM, CUDA
device, test database, toolchain, budget), with `held` null when its runners
claim work or the reason they do not. A node dispatches exactly when its
machine is enabled and not held, and a held reason on a machine that is off
is refused, since a machine that is off claims nothing already. lavender's
Windows lane was held that way on 2026-09-29 (MCPs board task 7e467416): it
kept its capacity because lavender-wsl's `guest` named it, and stayed
enabled because the audit and the session observer still visited it, so a
runner whose scheduled task outlived the retirement logged `lavender is
disabled in registry/machines/lavender.json; claiming nothing` and asked the
node nothing, until the reimage to Ubuntu on 2026-10-19 replaced both.

Every machine declares `enabled`, and the decoder refuses one that leaves it
out. Neither default is safe: defaulting it to `true` is the defect this
field was added for, and defaulting it to `false` would silently shrink the
fleet with nothing to notice the missing capacity.

**"Was never asked" is not "did not answer."** Three states, three codes,
three different next actions:

| state | what happened | what to do |
|---|---|---|
| `NODE_DISABLED` | deliberately off; not probed at all | edit its `registry/machines` file |
| `NODE_UNREACHABLE` | should have answered, did not | check the tailnet |
| `NODE_MEMORY_EXHAUSTED` | answered, no room | wait |

A reader handed one code for all three goes to the wrong place twice. A named
`--node` that is disabled is refused rather than rerouted, for the same reason
an unreachable one is: running somewhere else answers a question nobody asked.
In auto-select a disabled node is skipped **without an ssh call**, which is the
whole saving — the alternative costs a ten-second `ConnectTimeout` per node per
dispatch.

## The lease is the part that fixes incident (1)

A dispatch takes a lease on `(node, project)` and holds it for the run, so two
`make check`s on one project on one node cannot overlap and the environment
mutation is serialised **by construction** rather than by everyone
remembering. Different projects on one node run concurrently.

Expiry is required and has no "forever" spelling: the failure being designed
against is a wedge, and a wedge holding an unexpiring lease turns one stuck
suite into a project nobody can ever build again.

## How a dispatch is put on a node

Tar, because no node has `rsync` and all three have `tar`. Not git: uncommitted
work is the normal state here and the standing rule is no branches, so "push
and pull" would refuse to dispatch the thing you are actually working on.

The archive crosses over **scp**, as bytes. Until 2026-09-26 it crossed as one
base64 line streamed into PowerShell, because raw bytes do not survive ssh
INTO a shell; scp uses ssh's file transfer, where no shell reads them. The
base64 route's cost grew with the archive through a pipeline that reads line
by line: an 83 MB archive did not land on serendipity inside the 120 s ssh
deadline, and scp copied it in 7.3 s with its digest intact (MCPs board task
140e7042).

The node digests the landed archive and reports. **Only then is it told to
unpack.** Verifying after extraction would mean an unverified tree had
already landed where the build looks, and a truncated tree builds and fails in
a way that reads as the code's fault.

`.venv` leads the exclusion list, and not for tidiness: it is the thing this
package exists to stop two dispatches sharing, and one machine's has absolute
paths baked into it. The node builds its own from the lockfile that is sent.

**The archive itself is built outside the repository**, and getting that wrong
twice is instructive. Left beside `fleet.json` it was staged by the next
dispatch, which grew 185 KB → 1.4 → 4.5 → 13.5 → 20.7 → 42.5 MB, each carrying
its predecessors. Excluding the directory it sat in fixed the size and broke
something worse: `tools/hpc3` commits 294 run documents under
`tools/hpc3/runs`, force-added past the monorepo's `**/runs/` ignore, and its
suite reads them — so the exclusion failed four of its tests for a reason that
read as hpc3's fault. A scratch file inside the tree being staged is the whole
error, and excluding a directory to compensate is how a fix becomes a second
bug.

**A project is not the unit of staging, and the first real dispatch proved
it.** `tools/fleet` went to sedona as one directory, which is what "dispatch a
project" reads as and cannot build: its `pyproject.toml` resolves
`platform-core` at `../../libs/platform_core`, its Makefile includes
`../../scripts/make/shell.mk` and calls `../../tools/maketools/scripts/run.py`
(which puts `libs/platform_core/src` on `sys.path`), its `scripts/guard.py` imports from
`<root>/libs/monorepo_guards/src`, and those rules then read
`monorepo-guards.toml` from the root. `fleet.core.manifest` computes the set:
the project, its transitive **path dependencies read out of the manifests that
declare them**, and the shared paths the monorepo asserts about its own layout.

The path dependencies are read from `pyproject.toml` rather than declared
beside each project in the registry, because poetry already reads the
authoritative list every time anybody builds and a second copy drifts silently
in the direction of staging too little — which surfaces as a lockfile error on
a node and reads as the project's fault.

## The other kind of lease: a thing there is one of

A `(node, project)` lease serialises a project's environment on one machine.
Some suites contend for something a node does not own. MCPs `packages/db` runs
`migrate-test` before vitest, and that applies migrations to a single shared
`corvis_test` — two of them deadlock on an `AccessExclusiveLock` whether they
are on one machine or three. Distributing that suite moves the CPU contention
off one box and leaves the database contention exactly where it was, and makes
it worse: two nodes cannot see each other's processes, and a per-node capacity
check admits both while neither node is short of anything.

So a project may declare what it needs exclusively:

```json
"packages/db": {
  "worker_ram_gb": 0.5, "minimum_workers": 4, "expected_minutes": 6,
  "exclusive_resources": ["corvis_test"]
}
```

The names are free strings, because what is exclusive is a fact about the
world rather than about this package — `corvis_test` is one database because
there is one, and no introspection here would discover that.

**It is the same lease record**, not a second kind of claim. One thing to
expire, one to release, no way for two to disagree about whether a run is
still going.

`RESOURCE_HELD` is a separate code from `LEASE_HELD` because the two send a
reader in opposite directions. An environment is per node, so the answer is
another node. A fleet-wide resource has no second copy, so the answer is to
wait — and the refusal says so rather than leaving somebody hunting for
capacity that could not have helped:

```
cannot dispatch: corvis_test is held fleet-wide by opus-weight-injection-0902
(run tools-fleet-1788562637, tools/fleet on sedona), 570s remaining;
there is one of it in the fleet, so no other node is an alternative
```

It is checked **before any node is probed**, by both `fleet-run` and
`fleet-preflight`, through the same function `leases.acquire` enforces with.
Probing three nodes to collect three identical refusals costs three round
trips and produces a message shaped like a capacity problem.

## What a project requires of a node: `required_tags`

Some suites cannot run on every kind of node, whatever is free on it. The
first was `slime` (MCPs board task 41f45bd7, 2026-09-21): its browser project
drives a real scene through headless Chromium on the machine's GPU with ANGLE
over Direct3D 11, refuses every other platform by name
(`slime/scripts/chromium-launch.ts`), and under the software rasteriser draws
a frame every few seconds, so on a CPU-only or linux node it would not fail —
it would time out, in a way that reads as a regression in the change under
test. So a project declares what it needs, and the declaration is REQUIRED,
`[]` for the ordinary suite any node may run. slime required `gpu` and
`windows` until board task e705f08c; with no Windows node left it requires
`gpu`, and whether its check passes on a Linux node's card is that check's
own answer, a Windows dependency it finds removed from slime, never answered
with the tag:

```json
"slime": {
  "worker_ram_gb": 0.5, "minimum_workers": 4, "expected_minutes": 15,
  "required_tags": ["gpu"]
}
```

**A node never declares tags.** Its tags are DERIVED from the two measured
fields it already carries: `platform` gives `linux`, and a `gpu`
that is a device rather than `null` gives `gpu` (`fleet.contracts.tags`). A
declared `tags` column would be a second copy of those facts, and the copy is
the one that drifts: a box whose card was pulled would keep its `gpu` tag
until somebody remembered the list. Derived, the tag is exactly as true as the
measurement, and `gpu` keeps the meaning it has always had here — a CUDA
device `nvidia-smi` reports, never an integrated adapter. The fleet audit's
`roster.gpu` row holds each declaration to the devices its inventory reads
through `nvidia-smi` on every pass (MCPs board task e447de72), which is how
"which nodes may carry this tag" is a measured fact: sedona, lavender and
diphtheria, as of 2026-10-20.

The planner checks tags **first**, before disk, cores and memory, and
refuses with `NODE_LACKS_TAG` naming the tags missing:

```
colossus lacks gpu: the project requires gpu and this node carries linux.
Tags come from the node's declaration and from the tools its toolchain probe
found, so the answer is another node, or this one once the missing tool is
installed.
```

Auto-select skips such nodes, and when EVERY node that answered refused on
its tags the fleet-wide refusal is `NODE_LACKS_TAG` too — nothing is full and
waiting fixes nothing — while one tagged node that is merely busy keeps the
capacity answer, because that node will take the work later. A project
naming `windows` or `elevated` is refused at decode, since no node carries
either after board task e705f08c.

**`slime` is a project of another repository**, `github.com/wagner-austin/slime`
(private, the same account as this one), not a path under `--repo-root`, so
`fleet-run` (which stages the working tree under that root by its
`pyproject.toml`) refuses it at staging with `PROJECT_MANIFEST_MISSING`. The
queue path runs it: its `source` names that remote with `path: ""` and the
two install steps a clean export needs (`npm ci`, then `npx playwright
install chromium webkit`), and `fleet-node-agent` exports the submitted sha
from it (fd5cabfa, A2). Its source declares no companions: the vendored code
and published-revision integrity lock travel in slime's own export, and its
normal lint does not read or fetch MCPs (20e977b6).
`fleet-preflight --project slime --node sedona`
answers with a worker count (measured 2026-09-21: `slime would run on sedona
with 10 worker(s)`). Its `expected_minutes` is the full `make check` w1 timed
on austinpc that night (03:45Z to 03:53Z); its `worker_ram_gb` is an estimate
for vitest's jsdom workers and should be re-pinned from the first fleet run's
memory reading. The three MCPs packages declared beside it
(`MCPs/packages/wiki-search`, `MCPs/mcp-shared-py`, `MCPs/packages/maketools`)
are the same shape: another repository's remote, the package's directory as
`path`, and the install its checkout needs at the repository root.

**`MCPs/packages/wiki-search` requires `linux`** (MCPs board task fd5cabfa,
2026-09-25). Its install builds `hnswlib-node` from source with node-gyp,
which needs a C++ toolchain, and no Windows node carries Visual Studio's:
dispatch job dd0b2ab3 failed on sedona with node v24.20.0 at `gyp ERR!
find VS`, and de32d61e on serendipity the same way under node 18. Its only
passes, 6c81f413 among them, ran on diphtheria, whose linux toolchain builds
it. The tag said so rather than letting a Windows node claim a check it
could not install, until no Windows node was left (board task e705f08c).

**`MCPs/execution-twin` requires `docker`** (MCPs board task 83b5c38f,
2026-10-21). It is the execution workspace again, ending with MCPs
fleet-prepare's `twin-lane` phase, whose marker makes its `make check` run
`tests/passkey-twin.test.ts` alone: mcp-proxy's `make test-twin`, which
starts Keycloak and mailpit, so the build runs as execdocker against that
user's rootless daemon. A push from ziggurat touching one of mcp-proxy's
Keycloak twins queues it, because a node runs a project's `make check`
whatever the command (`MAKE_TARGET`), and only `vitest.twin.config.ts`
includes the twins. Since MCPs board task 94d23062 sent a push's container
suites to their project's fleet `make test`, the twins, which that run never
includes, were refused as UNRUNNABLE; before it they ran on ziggurat and
failed EACCES on `/var/run/docker.sock`. It names no node, so any node
carrying the tag runs it, and it is no deploy gate: `MCPs/execution-deploy`
runs the same case.

## A project that names its node: `node`

A tag says what KIND of node a suite needs; a few suites need one particular
host, because they read what only that host holds, and no tag singles a host
out. The first were MCPs' served lanes (MCPs board task bf7a9460,
2026-10-15): `MCPs/execution-served-diphtheria` and
`MCPs/execution-served-colossus` each compare the files that host's own
search container serves on `127.0.0.1:8090` with the commit's, and a run on
any other node would read nothing. So a project may name its node:

```json
"MCPs/execution-served-colossus": {
  "required_tags": ["linux", "cxx"], "node": "colossus", ...
}
```

The key is optional and absent for every other project. The workspace
refuses a name it does not declare, and a node whose `host` is another name,
because `capacity.assess` matches the pin against the weighing node's host.
Every other node refuses the project with `NODE_NOT_PINNED`, first, before
its tags or capacity are weighed, so a runner is never offered it
(`fitting_projects`) and nobody who submits the check has to name the node.

`tools/fleet-execution-runners` is this package's own (MCPs board task
715b9ec5, A3). It was pinned to `lavender-wsl` and left the registry with
that guest on 2026-10-19, and it is declared again, pinned to `lavender`,
once runners run there. Its `make check` is this package's
`execution-runners`, which passes `--runner-host=lavender` to the suite and so
runs only the `host_runner_units` cases (`tests/_host.py`): they read, on the
host and as the fleet user, that every runner unit `runners.json` declares
for lavender is active in `listeners.slice`, that slice unbounded and
`runners.slice` at the roster's `memory_high_gb`, and every live
`Runner.Worker` in its own job scope; and they fill a slice of the fleet
user's systemd manager past its high, swap denied, beside an unbounded
sibling, and require the probe inside it to stall and the sibling's not to.
Each prints what it read as a `host reading:` line. It is no roll or deploy
gate: a runner a job holds mid-move fails it, which is what it is for.

`tools/fleet-execution-hub` is this package's too (MCPs board task 3d9eec0d),
pinned to `diphtheria`, the hub. Its `make check` is this package's
`execution-hub`, which passes `--fleet-hub=$(HOME)/PROJECTS/API`, the checkout
the hub's ticks run with, and so runs only the `host_fleet_hub` case: it
lists every enabled node's stage root over the hub's ssh and fails on any
run's directory, staging directory or root script whose run is neither live
(a held lease or a running ledger row in that checkout's records) nor still
owed a retire in `runs/retires.jsonl`, and on any node it cannot read. It
prints each node's count as a `host reading:` line, and it is no roll or
deploy gate.

`MCPs/execution-netboot-diphtheria` (MCPs board task d387271d) is pinned to
`diphtheria` because its case arms and serves a clean Secure Boot VM's
install through MCPs' `fleet-netboot`, which mints the VM its own tailnet
key, and then reaches the VM over ssh at its tailnet address, which
diphtheria's tailnet grant allows and a `tag:corvis-node` machine's does not.
Its last install step, `netboot-lane ENV_FILE=/home/corvis/PROJECTS/MCPs/.env
VM_HOST=lavender`, copies only the tailnet writer client's two lines from that host's
workspace `.env` into the export, after the steps whose untracked files the
node stores, so the client is never stored with the tree, and names the host
whose KVM boots the VM. The VM is not diphtheria's: its 7.5 GB never fit
beside the stack, which on 2026-10-18 left 3.6 GB after its capped
containers' promises (job 86c7afcc sat queued three hours declared at
8.5 GB), and mapped from a file instead it was what the kernel reclaimed
with the swap full, its qemu reading 156 GB and writing 75 GB to the SSD
before job a6c3dec8 missed its check-in. So the case runs qemu on
lavender over ssh (on its WSL guest lavender-wsl until the reimage to Ubuntu
on 2026-10-19), reserving its four cores there in
`.fleet/reservations` and forwarding the server's port back, and its
`worker_ram_gb` of 1.0 is what it holds on diphtheria. Its server listens on
the definition's fixed port, so `netboot-http-8780` is its exclusive
resource and two runs never start at once.

`MCPs/execution-deployed-demure` (MCPs board task 03b683d2) is pinned to
`demure` because its case runs MCPs maketools `deploy-kept` against the
database host's own checkout, `/home/corvis/PROJECTS/MCPs`: its newest
deploy record and the journaled hold that wrote it, held against the
postgres and every container of the stack running there, which only that
host's fleet user reads. It was `MCPs/execution-deployed-diphtheria` until
the database moved to demure (MCPs board task 43fa3a6c): diphtheria's deploy
now migrates nothing, so only demure's hold journals the kept database. Its
last install step, `deployed-lane`, writes the marker that makes the
execution workspace collect `tests/deployed` alone, so an audit's `command
MCPs execution-deployed-demure` line carries no other case's verdict. It
takes no exclusive resource: it only reads.

`MCPs/execution-egress-diphtheria` (MCPs board task 523a7445) is pinned to
`diphtheria` because its case runs one probe inside `mcp-watcher` and one
inside `mcp-doc-extract-api`, the stack's containers there, to reach
Garden Grove's archive at ggcity.org through the host's own address. Its
last install step, `egress-lane`, writes the marker that makes the execution
workspace collect `tests/egress` alone. The site has dropped that address
since 2026-10-19 (MCPs `fleet-mcp/README.md`, Known gaps), so the case is red
until the block lifts, and a lane of its own keeps that red out of every
served check and the probe off every served run. It takes no exclusive
resource: two requests a run are all it sends.

`MCPs/execution-lead-lacrymal` (MCPs board task 770837b7, A1) is pinned to
`lacrymal` because its case runs one real planning pass, and the session
host admits that node's own key. The case starts `taskboard-mcp` over
`corvis_test` on the node's loopback, carries the port to the session host's
loopback with `ssh -R` (the node's firewall admits no inbound port), and
runs session-audit's planning program, which launches a `claude -p` there
whose only MCP server is that board. Its last install step, `lead-lane`,
builds session-audit's venv and writes the marker that makes the execution
workspace collect `tests/lead` alone, so no other lane spends model usage.
It holds `corvis-fleet-testdb` exclusively, because the pass reads and files
there.

## The staged tree is made a git repository, and that is not decoration

Ruff honours `.gitignore` and applies it **only inside a git repository**. A
tree that carries the file but no `.git` therefore lints every path the
repository deliberately excludes.

Measured dispatching `tools/hpc3` to lavender: 902 ruff errors, all in
`tools/hpc3/runs`, which `.gitignore` line 170 excludes as build artifacts
while explicitly tracking the run documents beside them. The same tree with
`git init` run in it reports `All checks passed`. Locally `ruff check .`
passes and `ruff check . --no-respect-gitignore` reports exactly 902 — the
same number, which identifies the mechanism rather than suggesting it.

So `staging.stage` initialises a repository after extracting, and
`.gitignore` travels with the tree. Neither is any use without the other.

**The index is filled as well, since the first fleet verdict** (board task
`fd5cabfa`, 2026-09-21T10:03Z): `MCPs/packages/maketools` on diphtheria read
`1 failed, 416 passed`, the one failure asserting that `git ls-files
CLAUDE.md` names the file, which in a repository with an empty index it does
not. A checkout's suite reads its own tracked set (maketools' `lint-makefiles`
lints exactly the tracked Makefiles), so the init script indexes the tree and
commits it under the export identity (`packages/db`'s migrator reads `HEAD`,
board task 6bbfd171).

**The index is forced, since API board task 0b3591d7** (2026-10-03). The
staged tree is exactly the archive of the commit, so every file in it is one
the commit tracks, and `git add --all --force` indexes all of them. A plain
`--all` applied `.gitignore` instead and dropped every tracked file it
ignores, which is every force-added one: `tools/hpc3` at f75c4a94 on
lavender-wsl read `1 failed`, its run-document audit reading `git archive
HEAD` and finding only the one sweep a `.gitignore` negation re-includes,
with the nine force-added sweeps beside it missing. Ruff reads `.gitignore`
patterns, not the index, so the lint the repository was made for is
unchanged.

**The alternative was an `exclude` in the project's own ruff config, and it
would have been wrong.** The repository already states which paths are build
output; a ruff `exclude` restating it is a second copy of one policy, and the
copy that drifts is the one nobody looks at. Reproducing the environment a
build is defined against is this package's job, not the project's.

## A project's suite may read outside its own directory

`tools/hpc3` was dispatched to lavender and four of its tests failed: its
suite reads `docs/RESEARCH.md` at the monorepo root, the index of every body
of work on the machine, and a staged tree did not have it. Same class as the
guard rules that resolve their declaring module from the root — except the
guards can be *asked* (`monorepo_guards.external_inputs`) and a project's
tests cannot, because nothing but the project knows what they open.

So the project declares it:

```json
"tools/hpc3": {
  "worker_ram_gb": 0.25, "minimum_workers": 4, "expected_minutes": 5,
  "external_paths": ["docs"]
}
```

The declaration can drift from the code, and that is accepted rather than
solved: the drift surfaces as a loud remote failure naming the missing file,
which is exactly how the field was found. A declared path that has since been
renamed is refused at dispatch rather than at tar, because `tar: docs: Cannot
stat` names a path and the reader's actual fix is to edit the workspace.

### Why `tools/hpc3` left the registry on 2026-09-06, and why it is back

`tools/hpc3` was a declared project until 2026-09-06, when it was removed. Five
dispatches of it failed over two days, every one recorded in the ledger as
`failed`, and none of them was an hpc3 defect.

Its suite asserts over the **whole monorepo**, not over its own directory.
`tests/test_exported_env_readers.py` walks `("libs", "services", "clients",
"tools")` to prove every exported variable is read by something, and
`tests/test_committed_specs.py` resolves the `model_trainer.*` symbols an
image spec names. On a staged tree those tests do not merely fail — **they
answer wrongly**, reporting `HPC3_JOB_NAME: exported, read by nothing` when the
reader exists and simply was not staged. The counts settle it: 1484 tests
locally, and on lavender 1478 passed + 4 failed + 2 skipped, the same 1484,
with only the tree differing.

`external_paths` could not express this. The four working trees measured
**139 GB** (`services` alone was 130 GB) against the ~34 MB archive an hpc3
dispatch shipped then, so no declaration made the suite correct on a node,
and it was removed rather than left to answer wrongly.

The premise no longer holds. A node-lane check now ships `git archive` of
the submitted commit, and since board task 140e7042 that archive is the
WHOLE tracked tree less the data directories `data_paths` declares
(`fleet.core.archive_scope`): every package's source under `libs`,
`services`, `clients` and `tools` travels, and the 139 GB was untracked
working-tree data that was never going to. The readers
`test_exported_env_readers.py` walks for and the `model_trainer` modules
`test_committed_specs.py` resolves are tracked source, so they are on the
node, and `tools/hpc3/artifacts`, the one declared data directory the suite
could want, is hpc3's own and so is kept. It was re-declared on 2026-10-02
for board task 0b3591d7, beside `clients/RustedWarfareBot`, because that
task's close needs a fleet verdict of both at the commits that moved the
sweep, stage and layout contracts into `platform_core`.

## Cancelling

`fleet-cancel` kills exactly one dispatch by name. There is no sweep, no
"cancel everything on this node", and no age heuristic — a direct response to
what this fleet is for, since the incident behind it was work destroyed by
something that was not trying to destroy it.

It **will** cancel a run whose lease has already lapsed. That is precisely the
wedge case, and refusing because the lease is gone would leave the only tool
that can stop it unable to.

**A stop ends the build's whole process tree, not only its task.** The
build is a transient systemd user unit, and stopping it stops its control
group, every process the build started whatever became of its parent. A
Windows build needed a recorded `$PID`, `taskkill /T` and a kill-on-close
job object for the same (measured on sedona 2026-09-23 and in MCPs board
task e40bca34); all three left with the fleet's last Windows node (board
task e705f08c).

**The node runner stops two kinds of run by itself**, through the same
`fleet.core.stop` path (MCPs board task fd5cabfa):

- **One whose queue job was cancelled under it.** `dispatch_cancel` marks the
  queue row terminal and has no route to a node, and a cancelled job is not
  live, so until this nothing looked at it again: slime-1790104328 was
  cancelled at 2026-09-22 20:22Z and closed here only at 00:49Z, by hand. A
  tick now lists the jobs cancelled while its runner held them (a cancel keeps
  `claimedBy` and `runId`) whenever this machine's ledger has a live run on
  the node that no held job names, and stops each one it finds, closing it
  `cancelled`. A live run no cancelled job names, a `fleet-run` by hand, is
  left alone.
- **One still running past its lease.** A renewal used to have no deadline,
  so a hung suite held its node for as long as it stayed hung. Past
  `started + expected_minutes × 60 × 2` the environment is unprotected, so the
  runner stops the build and closes the job `failed` with exit 124 and
  `LEASE_NOT_HELD` in its verdict. A suite that legitimately needs longer
  needs a larger `expected_minutes`; slime's went from 12 to 15 on
  2026-09-23 because its measured runs took 1258 to 1446 s against a 1440 s
  deadline.

## Why this is not `tools/hpc3`

hpc3's contracts are Slurm-shaped — partitions, `sbatch`, GRES, requeue,
preemption, service units — and a workstation over SSH has none of those.
Forcing them in would mean optional fields, and `hpc3.contracts.workspace`
deliberately makes the illegal state unrepresentable rather than branching on
it. What lifts is the *pattern*, and it lifts wholesale: one strictly decoded
workspace, preflight before dispatch, an append-only ledger row per dispatch,
declared budgets checked before and during.

## The one rule about talking to a node

**Never interpolate a remote command.** Probing three nodes that way failed
twice on 2026-09-04: quotes are stripped passing through the local shell, ssh,
and `cmd` into `powershell`, and the second attempt arrived as
`@(python,poetry,git,...)` — unquoted bare words, a parser error on the far
side. `fleet.core.remote` renders the script, sends it over stdin, and runs it
by path, so the bytes that run are the bytes that were sent.

**And the rule applies one layer further in.** The first Windows version
passed the whole build inside a scheduled task's `-Argument` string, and
PowerShell ended it at the first inner quote, so the task could not start
and the ledger recorded a run that did not exist. So the build is its own
script file, sent and named by path, and the launch interpolates one path
and no code.

## One dialect: `sh`

Every node declares `platform` `linux`, and the decoder refuses any other
since board task e705f08c (the operator, 2026-10-25: "so no windows shit
then ok"). The identity registry declares the same field; the fleet audit's
`roster.agrees` row fails a node the two disagree on. `fleet.core.dialect`
renders every act as `sh`: how a file is written over ssh (`mkdir -p … &&
cat >`), how a script is run by path (`/bin/sh`), how the archive scp landed
is digested (`sha256sum`), how capacity is read (`/proc/meminfo` and `df`),
and how the suite is detached from the connection. Until that task it was
one of two dialects behind a protocol, the other PowerShell. The detach is a
**transient systemd user unit**
(`systemd-run --user --unit=fleet-<run>`), the user manager's equivalent of a
scheduled task: owned by the machine rather than the connection, stoppable by
name, started or refused synchronously. It needs the account to **linger**
(`sudo loginctl enable-linger <user>`), because a user manager stops with the
user's last session otherwise and the unit would die with the ssh that started
it; the launch script checks first and names that command. The toolchain
probe asks `python3` (what `scripts/make/shell.mk` calls) and reports it
under `python`, the name the toolchain contract requires.

## What a lease can and cannot tell you

Whether a run was **protected** is a question about whether its lease covered
the run. Whether a lease is held **now** is a question about how promptly
somebody came to collect. Conflating them refused a healthy run that had
finished three minutes inside its window, twenty minutes after it ended.

So the node reports *when* the build finished as well as what it exited with,
and `fleet-collect` compares that against the deadline recomputed from the
ledger row and the project's declared duration. A run that outlived its lease
is refused — during that window a second dispatch could have entered the same
environment, which is the corruption this package exists to prevent. A run
collected late is closed normally.

## Development

```bash
make lint     # guards, ruff, mypy --strict over src tests scripts
make test     # pytest -n auto, 100% statements and branches
make check    # both
```
