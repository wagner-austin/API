# lock-wake

The fourth wake bridge: fleet-lock transitions to the agent board, so a
cascade starting or ending WAKES the sessions that care instead of each one
tailing the journal (board task `9406cfd9`, publisher b).

```
hpc-wake     slurm ledger       -> board
fleet-wake   dispatch ledger    -> board
ci-wake      GitHub Actions     -> board
lock-wake    fleet-lock journal -> board   THIS PACKAGE
```

## What it reads

`scripts/with-fleet-lock.ps1`'s append-only journal in the MCPs repo root
(`.fleet-events.jsonl`): one JSON object per transition — requested,
waiting, acquired, step, released, failed, timeout — written by every
locked compose target. Since MCPs `66b85d32` each row carries `agent`, the
session label behind the invocation when its shell exported
`BOARD_AGENT_LABEL`; history rows without the key read as unlabelled.

It also reads a second journal beside it, `.check-events.jsonl`: the MCPs
per-package check lock's (`packages/maketools` `check_lock`) record of
finished `make test` runs, one `checked` row each, in the same record
shape, so check results and rebuilds share one stream (MCPs board task
`ea2ea29c`). Its `label` is the package and its `detail` the outcome, exit
code, seconds, HEAD and tree. The rows are their own file because every
checkout's older reader of `.fleet-events.jsonl` refuses a kind it does not
declare: for six minutes on 2026-09-24 they went into the fleet journal and
the hub's `make fleet-status` died on them. A lock transition found in the
check journal is refused, not announced. Each journal keeps its own cursor
beside it, and both advance only after the tick's post.

The position is a byte offset kept beside the journal
(`.fleet-events.jsonl.lock-wake-offset.json`), per the journal's own
subscription contract. Torn tails from a mid-append writer are left for the
next cycle; a complete line that fails to decode is fatal, never skipped.

## The noise budget (board 9406cfd9, acceptance 6)

One cycle produces AT MOST ONE post, covering every hold that crossed a
boundary — acquired, released, failed, timeout — with requested/waiting/step
folded in as counts. Progress-only slices advance the offset silently: a
30-minute deploy's step lines are not posts. Worst case is therefore one
post per pump tick (3 minutes), and in practice one per cascade phase.

Finished check runs join that same post under a `CHECKS` heading, one line
each, naming the runner without an `@`: the runner saw its own output, so
the post reaches only the sessions subscribed to the standing task.

## Running

```
lock-wake --journal C:\Users\Test\PROJECTS\MCPs\.fleet-events.jsonl --check-journal C:\Users\Test\PROJECTS\MCPs\.check-events.jsonl
```

One cycle, then exit. The interval belongs to the pump
(`tools/hpc-wake/scripts/run_cycle.py`, PUBLISHERS table) — this package
registers no scheduled task of its own, per the pump's rule.

Environment, exported where the pump runs (`tools/hpc-wake/runs/env.ps1`):

```
TASKBOARD_MCP_API_KEY   taskboard-mcp's own x-api-key
CORVIS_TENANT_ID        the tenants row whose board is posted to
LOCK_WAKE_TASK_ID       the standing task announcements land in
BOARD_WATCH_URL         optional; defaults to taskboard-mcp in MCPs scripts/fleet/stack-endpoints.json
```

Announcements POST before the offset advances — at-least-once, the family
guarantee. A refused post ends the cycle nonzero for the pump's history and
is retried whole on the next tick.

## Identity

The bridge posts as `bridge-lock-wake-0909` with a deterministic UUIDv5
session id, pinned as a literal in `tests/test_identity.py`. Restarts do
not mint identities.

## Tests

`make check`. Fakes rebind this package's hooks; the journal fixtures write
bytes in the lock wrapper's own form, torn tails and pre-`agent` history
included; the board POST is exercised against
`platform_core.mcp_testing.FakeHttpPost`. Nothing is patched, nothing is
mocked.
