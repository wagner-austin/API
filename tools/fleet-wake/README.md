# fleet-wake

Announce fleet dispatch results on the agent board, so the session that
dispatched a suite is woken when it ends.

## The gap this closes

`tools/fleet` records every result perfectly and tells nobody. A terminal
outcome reaches `runs/ledger.jsonl` and `runs/feed.jsonl` **on the machine that
dispatched**, so the only ways to learn a dispatch finished were to be the
session that ran it, or to read a file here. A session on the phone, on
claude.ai, or in another repo had no way to find out at all — which made the
dispatch queue's whole point, sessions coordinating, half-built.

That is `tools/fleet`'s own acceptance criterion 6 on board task `df6f1dc8`,
and this is it.

## One cycle

```bash
fleet-wake --config /path/to/fleet.json
```

1. Read the fleet ledger's **current row per dispatch** (`records.latest_rows`).
2. Take the rows that have reached a terminal outcome and are not in the
   position record.
3. Post one board note per `(agent, project)` group, tagging the dispatching
   session's label.
4. **Then** write the position rows.

Steps 3 and 4 are in that order and it is the delivery guarantee: a crash
between them repeats an announcement on the next cycle rather than losing one.
At-least-once, with the position file as the mark. Recording first would turn
any transport failure into a dispatch nobody is ever told about — the exact
silence this exists to remove.

**Nothing is caught.** A refused post ends the cycle non-zero for the scheduler
to record, and the position rows are not written, so the next cycle retries. A
bridge that swallowed the failure would report success while doing the opposite
of its job.

The polling interval belongs to the scheduler that calls this, where it is
visible — the same reason `fleet-watch` has no `--follow`.

## Environment

| Variable | What it is |
|---|---|
| `TASKBOARD_MCP_API_KEY` | taskboard-mcp's own `x-api-key` |
| `CORVIS_TENANT_ID` | the `tenants` row whose board is posted to |
| `FLEET_WAKE_TASK_ID` | the standing task announcements land in |
| `BOARD_WATCH_URL` | optional; defaults to loopback `:8033` |

The standing task is **configuration, never discovery**. Finding it by title
search would make every cycle depend on a render grammar owned by another
repository, for something that never changes — and a post to a guessed task is
a post nobody is subscribed to, which reads exactly like the bridge working.

## Identity

The board binds a session id to an agent label on first write and never
releases it. A service has no harness session, so it mints a deterministic one:

```
bridge-fleet-wake-0906   0a6cb261-eaa4-5330-84b9-079a1afe268a
```

Same pair on every run, across restarts, forever. `_SESSION_NAME` is therefore
**never edited** — a changed name is a new identity, after which every post is
refused and there is no way to unbind the old label. The value is pinned as a
literal in `tests/test_identity.py`, not re-derived there, so an edit fails a
test rather than a production cycle.

## What is shared with `tools/hpc-wake`, and what is not

`platform_core.board` holds the three things a second bridge would have got
**wrong**: the deterministic identity, the board's argument spelling
(`taskId`, `sessionId`, `cwd`), and the standing-task rule. Lifted there on
2026-09-06 rather than copied.

The grouping is **not** shared. hpc-wake groups Slurm closures by
`(submitter, project)`; this groups dispatch rows by `(agent, project)`. A
grouper parameterised over both would be harder to read than either — that is
the line between DRY and premature.

## Why the position file names run ids

The obvious alternative is a cursor — a line number or timestamp in the ledger.
It is wrong because the ledger is append-only: a finished dispatch has **both**
a `running` row and a terminal row, and rows for different dispatches
interleave. A position in that stream cannot answer "has this run been
announced", which is the only question asked. Naming the ids does.

Reading the board back instead would make the decision to post depend on the
board being reachable, and would re-announce everything the first time a query
failed.

## Development

```bash
make check      # lint (guards + ruff + mypy over src, tests, scripts) then tests
```

100% statements and branches, enforced. No mocks — fakes implementing the
production Protocols, with the MCP poster shared from
`platform_core.mcp_testing` rather than copied a third time.
