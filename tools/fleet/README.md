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

`scripts/reap-test-processes.ps1` was blamed for the first and is innocent:
its filter requires a command line matching `*pytest*` or `*exec(eval*`, its
sweep only considers processes older than sixty minutes, and an aggregate
CPU-idle gate aborts if any candidate is burning CPU. The collision is
structural — a shared mutable environment with no lock — not a rogue killer.

---

## The three files

```
fleet.json        the workspace: where the nodes are, what each project costs
runs/
  ledger.jsonl    append-only; every dispatch ever made from this machine
  feed.jsonl      append-only; the event stream subscribers tail
  leases.json     live state; who holds which project's environment
```

**`fleet.json` is tracked and the three records are not**, because they are
different kinds of thing. The workspace describes the fleet and belongs to
everybody; the records are this machine's own history and live under `runs/`,
which the monorepo gitignores. It sat under `runs/` at first and was therefore
invisible to git — so the one file every command requires would not have
survived a fresh clone, and the package would have arrived unrunnable.

The ledger and feed are history and are never rewritten. Leases are live
state and are: a release has to make a claim stop existing, and an
append-only log of "taken"/"released" would make every reader replay the file
to answer one question.

## Commands

```bash
fleet-nodes     --config fleet.json                 # what is free right now
fleet-preflight --config fleet.json --project P     # would it run, and where
fleet-preflight --config fleet.json --project P --node lavender
fleet-run       --config fleet.json --project P \
                --agent <label> --session <uuid> --repo-root <path>
fleet-collect   --config fleet.json                 # close out what finished
fleet-collect   --config fleet.json --run <run-id>
fleet-watch     --config fleet.json                 # the event stream
fleet-watch     --config fleet.json --run <run-id>
fleet-cancel    --config fleet.json --run <run-id>
```

`fleet-run` returns as soon as the suite is running and does **not** wait for
it. The build outlives the command because it is launched through the node's
task scheduler rather than as a child of the ssh call — Windows OpenSSH puts
that child in a job object that dies with the connection.

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
  fleet-collect --config fleet.json
  fleet-watch   --config fleet.json
  sleep 30
done
```

`--agent` and `--session` are required and are the board's own identity
fields, so a ledger row and a board post can be matched by whoever reads both.
A default would be one label shared by every session, which is the same as
having none.

## Subscribing from a Claude session

`fleet-watch` prints one line per event and exits. That is the whole
integration:

```
Monitor({command: "fleet-watch --config fleet.json", description: "fleet events"})
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

The archive crosses as **base64**. Raw bytes do not survive ssh into
PowerShell — the stream is decoded as text at more than one layer, and one
mangled byte is a corrupt gzip that extracts partially. Base64 costs a third
more bytes and removes the failure mode instead of making it rarer.

The node reassembles the archive, digests it, and reports. **Only then is it
told to unpack.** Verifying after extraction would mean an unverified tree had
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
`platform-core` at `../../libs/platform_core`, its Makefile calls
`..\..\scripts\run-tests.ps1`, its `scripts/guard.py` imports from
`<root>/libs/monorepo_guards/src`, and those rules then read
`monorepo-guards.toml` from the root. `fleet.core.manifest` computes the set:
the project, its transitive **path dependencies read out of the manifests that
declare them**, and the shared paths the monorepo asserts about its own layout.

The path dependencies are read from `pyproject.toml` rather than declared
beside each project in `fleet.json`, because poetry already reads the
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

So `staging.stage` initialises an empty repository after extracting, and
`.gitignore` travels with the tree. Neither is any use without the other.

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

## Cancelling

`fleet-cancel` is the only command that kills anything, and it kills exactly
one dispatch by name. There is no sweep, no "cancel everything on this node",
and no age heuristic — a direct response to what this fleet is for, since the
incident behind it was work destroyed by something that was not trying to
destroy it.

It **will** cancel a run whose lease has already lapsed. That is precisely the
wedge case, and refusing because the lease is gone would leave the only tool
that can stop it unable to.

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

**And the rule applies one layer further in.** The first version passed the
whole build to `New-ScheduledTaskAction -Argument '-Command "cd ''{path}''…"'`.
PowerShell ended that single-quoted argument at the first inner quote, so the
registered task carried `-Command "cd` as its arguments and the remaining two
hundred characters as its **working directory** — a path that does not exist,
so the task could not be started at all. Nothing failed loudly:
`Start-ScheduledTask` reports a refusal as a *non-terminating* error,
PowerShell exited 0, and the ledger recorded a run that did not exist.

So the build is its own script file, sent and named by path, and the
registration interpolates one path and no code. It then waits for the task to
leave `SCHED_S_TASK_HAS_NOT_RUN` before reporting a launch, because a
registration that cannot start is not a dispatch.

`-AllowStartIfOnBatteries` and `-DontStopIfGoingOnBatteries` are on the
settings for the same reason: `New-ScheduledTaskSettingsSet` defaults both to
refusing, and two of the three nodes are laptops.

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
