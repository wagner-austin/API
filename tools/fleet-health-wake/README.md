# fleet-health-wake

The fifth wake bridge: the fleet audit's health journal to the agent board,
so a node that starts failing, recovers, stops answering or comes back WAKES
the sessions that care, instead of waiting for someone to read `fleet_status`
(MCPs board task `ebc80a03`).

```
hpc-wake           slurm ledger          -> board
fleet-wake         dispatch ledger       -> board
ci-wake            GitHub Actions        -> board
lock-wake          fleet-lock journal    -> board
fleet-health-wake  fleet health journal  -> board   THIS PACKAGE
```

## What it reads

MCPs `fleet-mcp/state/health-events.jsonl`, appended by the fleet audit
(`fleet-mcp/src/health-journal.ts`, run every twenty minutes on the hub by
`MCPs-FleetAudit-20min`) whenever a run's outcome is news. One JSON object
per line, `at`, `kind`, `key` and `body`:

| kind | written when |
|---|---|
| `transitions` | a row went to `fail` or came back from it, or a node stopped answering or answered again, compared with the run before |
| `baseline` | the previous snapshot existed but could not be read, so nothing was compared; the body lists what is failing now |
| `refused` | the audit's own build is stale and it did not run; written once per stale source hash |

The body is the note, rendered by the audit. This bridge posts it; it never
re-derives what changed, so the one definition of a transition lives beside
the standard it reads. A kind this package does not declare is a decode
refusal, never a line posted without being understood.

The position is a byte offset beside the journal
(`health-events.jsonl.fleet-health-wake-offset.json`), kept by
`platform_core.journal_cursor`, the cursor lifted out of lock-wake: torn
tails are left for the next cycle, a position that no longer fits the file
refuses rather than rewinding.

## One post per cycle

Every unread line goes into ONE note, bodies in journal order separated by a
blank line. The audit writes a line only on change, so a cycle usually has
none or one; after the pump was down across several audits, one post keeps
delivery atomic. The note is unaddressed: a fleet change has no dispatching
session to tag, so it reaches the sessions subscribed to the standing task
(`task_subscribe("88b20894")`).

## Running

```
fleet-health-wake --journal C:\Users\Test\PROJECTS\MCPs\fleet-mcp\state\health-events.jsonl
```

One cycle, then exit. The interval belongs to the pump
(`tools/hpc-wake/scripts/run_cycle.py`, PUBLISHERS table); this package
registers no scheduled task of its own, per the pump's rule.

Environment, exported where the pump runs (`tools/hpc-wake/runs/env.ps1`):

```
TASKBOARD_MCP_API_KEY       taskboard-mcp's own x-api-key
CORVIS_TENANT_ID            the tenants row whose board is posted to
FLEET_HEALTH_WAKE_TASK_ID   the standing task posts land in (88b20894, "Fleet health alerts")
BOARD_WATCH_URL             optional; defaults to taskboard-mcp in MCPs scripts/fleet/stack-endpoints.json
```

The post goes out before the offset advances: at-least-once, the family
guarantee. A refused post ends the cycle nonzero for the pump's history and
is retried whole on the next tick.

## Identity

The bridge posts as `bridge-fleet-health-0926` from `service://fleet-health-wake`
with a deterministic UUIDv5 session id, pinned as a literal in
`tests/test_identity.py`. Restarts do not mint identities.

## Tests

`make check`. Fakes rebind this package's hooks; the journal fixtures are the
fleet audit's own bytes; the board POST is exercised against
`platform_core.mcp_testing.FakeHttpPost`. Nothing is patched, nothing is
mocked.
