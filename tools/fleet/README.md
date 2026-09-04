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
runs/
  fleet.json      the workspace: where the nodes are, what each project costs
  ledger.jsonl    append-only; every dispatch ever made from this machine
  feed.jsonl      append-only; the event stream subscribers tail
  leases.json     live state; who holds which project's environment
```

The ledger and feed are history and are never rewritten. Leases are live
state and are: a release has to make a claim stop existing, and an
append-only log of "taken"/"released" would make every reader replay the file
to answer one question.

## Commands

```bash
fleet-nodes     --config runs/fleet.json                 # what is free right now
fleet-preflight --config runs/fleet.json --project P     # would it run, and where
fleet-preflight --config runs/fleet.json --project P --node lavender
fleet-run       --config runs/fleet.json --project P \
                --agent <label> --session <uuid> --repo-root <path>
fleet-watch     --config runs/fleet.json                 # the event stream
fleet-watch     --config runs/fleet.json --run <run-id>
fleet-cancel    --config runs/fleet.json --run <run-id>
```

`fleet-run` returns as soon as the suite is running and does **not** wait for
it. The build outlives the command because it is launched through the node's
task scheduler rather than as a child of the ssh call — Windows OpenSSH puts
that child in a job object that dies with the connection. The result arrives
on the feed, so follow it with `fleet-watch --run <the printed id>`.

`--agent` and `--session` are required and are the board's own identity
fields, so a ledger row and a board post can be matched by whoever reads both.
A default would be one label shared by every session, which is the same as
having none.

## Subscribing from a Claude session

`fleet-watch` prints one line per event and exits. That is the whole
integration:

```
Monitor({command: "fleet-watch --config runs/fleet.json", description: "fleet events"})
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

## Development

```bash
make lint     # guards, ruff, mypy --strict over src tests scripts
make test     # pytest -n auto, 100% statements and branches
make check    # both
```
