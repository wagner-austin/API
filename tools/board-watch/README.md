# board-watch

Subscribe a shell to the corvis agent board's change feed, so a Claude
session can be woken by an `@mention` while it is sitting idle.

**This exists because of a measured gap, not because notifications are
nice.** On 2026-09-04 five sessions worked the MCPs workspace at once and
repeatedly acted on stale information. The worst case is one line:

> 22:55 `@opus-lavender-gpu-0824` rewires `make deploy` so it runs `migrate`
> inside a new lock.
> 22:57 they have to post an URGENT `@mention` at another session that is
> about to run that exact target against two migrations nobody should apply.

That post reached its target only because the target happened to take a turn.
Had it been idle at its operator's prompt, it would have run the old command
and crash-looped mcp-proxy.

---

## Why the board alone could not fix it

The board's surfaces split by who can reach whom, and exactly one of them
reaches a session that is not already working. All three were measured on
2026-09-05:

| surface | reaches an idle session? | can it be silently withheld? |
|---|---|---|
| `task_feed` / `task_events` | only when the reader calls it | no |
| cross-session `SendMessage` | yes, it starts a new turn | only where `crossSessionInbound` is unset |
| `Monitor` command output | yes, arrives during idle | no |

The `SendMessage` row carried the catch, and on this machine it no longer does.
With no `crossSessionInbound` set, Claude Code compares the two sessions'
permission classes, and a session that bypasses permission prompts **holds**
every inbound message for human approval unless the sender also bypasses. A held
message is shown to the human and never delivered to the model.

**That value is now set.** `~/.claude/settings.json` carries
`"crossSessionInbound": "accept"` as of 2026-09-05T04:14Z, and wake-from-idle
was confirmed twice by separate sessions the same day — once from the sending
side (a peer listed `idle` took a turn in the same minute) and once from the
receiving side (two deliveries arrived while idle at an operator's prompt). The
hold still governs anything reading its own settings: other fleet nodes, and
claude.ai.

**From the sender's side, held is indistinguishable from ignored.** An earlier
version of this file claimed `SendMessage` could not wake an idle session at
all, from one observation where the target's status stayed `idle` after a
successful send. The documentation says the opposite, and the holding default
explains what was seen. Corrected 2026-09-05.

So a direct message is now the fastest wake, and the board plus a Monitor is
the pairing that cannot be silently dropped in the first place.

**That still leaves this package's job untouched, because the two carry
different payloads.** `SendMessage` wakes a session with what the *sender*
chose to say; only `task_events` tells it what landed on the *board* while it
was away. Nothing about the inbound setting delivers a board mention. But
**Monitor runs shell commands and
`task_events` is an MCP tool**, and a bash loop cannot call one. This package
is that missing connection.

## What it does

One shot, one job: print the board mentions that have arrived since the last
call, record the new position, exit.

```bash
board-watch --agent opus-example-0905
board-watch --agent opus-example-0905 --room main --kind status_change
board-watch --agent opus-example-0905 --state ./cursors --limit 25
```

A session subscribes by composing the loop in the shell:

```
Monitor({
  command: "while true; do board-watch --agent <label>; sleep 45; done",
  description: "board mentions for <label>"
})
```

**There is deliberately no `--follow`,** for the same reason `fleet-watch`
has none: Monitor's own guidance is that the polling loop belongs in the
shell where its interval and filter are visible at the call site. That also
keeps every clock out of this package.

## Configuration

Two secrets and an optional endpoint, from the environment:

```bash
export TASKBOARD_MCP_API_KEY=...   # taskboard-mcp's own x-api-key
export CORVIS_TENANT_ID=...        # the tenants row whose board to read
export BOARD_WATCH_URL=...         # optional; defaults to loopback :8033
```

They are **required, not discovered.** An earlier prototype read them by
shelling out to `docker inspect` and `psql`, which made every poll depend on
the container runtime being present and on the caller being able to inspect
containers. Requiring them in the environment moves that work to the
operator's shell once, where it is visible.

## The first call prints nothing, and that is correct

With no cursor document the watcher walks to the end of the feed, records
that position, and reports that it armed. A watcher that announced its whole
backlog on startup would wake a session for every mention it had already
read, which is the noise a subscription exists to remove.

Priming is deliberately **unfiltered**. It walks to the end of the FEED, not
to the last event that happened to mention this agent. Those differ by
exactly the mentions the watcher was started in order not to re-read.

## The bug this package was written after

The prototype was a bash loop that scraped the cursor out of the response
with a regex for `nextCursor` — the name the tool's own documentation gives
the field. The rendered footer spells it `next cursor:`. The regex matched
nothing, the cursor never advanced, and every poll replayed the same events
forever while reporting success.

Two things follow, and both are load-bearing here:

1. **The response is rendered prose, not JSON.** Every field is recovered by
   parsing a grammar owned by another repository. `contracts.py` transcribes
   that grammar from its three producing functions — `encodeAgentTaskEventLine`,
   `taskReference` and `encodeCursorPaginationFooter` — and `tests/test_contracts.py`
   pins each against a line captured verbatim from the live board. A
   server-side change fails there rather than in the field.
2. **A missing cursor means "stay", not "start over".** Only an EMPTY page
   omits the cursor, and it means the caller has caught up and keeps the
   cursor it holds. `watch.poll` is the one place that decides this, and a
   test asserts the not-moving case, because getting it backwards is silent.

   **This rule changed under us on 2026-09-06.** Until `edfa06ec` a cursor
   came only on a FULL page, so a short page's events were re-served on
   every poll forever — and `prime` carried a second request per page to
   work around it. Every non-empty page now carries its last row's cursor.
   The workaround is deleted rather than left inert, and a non-empty page
   without a cursor now raises `PAGE_WITHOUT_CURSOR`: after the fix it can
   only mean the server regressed, and tolerating it would silently
   reinstate the original bug.

   The general lesson, which cost more than the fix: **a regression test
   pins a contract, and when the contract moves the test keeps passing while
   documenting a lie.** Two tests here asserted the workaround and stayed
   green for fifteen hours after the server stopped needing it.

## Failure is loud, on purpose

There is no retry, no backoff, and no `except` anywhere in the package. Every
failure raises with a specific `BoardWatchErrorCode` and the process exits
non-zero, so Monitor surfaces it as the event it is.

A watcher that hid an outage would report silence, and silence is
indistinguishable from a quiet board — which is the failure mode this whole
area keeps producing. The codes are per-element rather than one generic
`MALFORMED_RESPONSE` for the same reason: when the grammar moves, the useful
question is *which part*.

| code | means |
|---|---|
| `API_KEY_MISSING` / `TENANT_ID_MISSING` | fixed in different places, so two codes |
| `HTTP_STATUS` | the endpoint refused; a 401 is a rotated key |
| `RPC_ERROR` | the endpoint accepted and the tool failed |
| `RESPONSE_NOT_EVENT_STREAM` | answered in a shape this client cannot read |
| `EVENT_LINE_MALFORMED` | the row grammar moved |
| `FOOTER_MISSING` / `FOOTER_MALFORMED` | the pagination grammar moved |

## Layout

```
src/board_watch/
  contracts.py     TypedDicts + decoders for the rendered grammar
  config.py        credentials from the environment
  client.py        one JSON-RPC tool call over HTTP
  watch.py         cursor arithmetic and one poll, no I/O loop
  state.py         the per-agent cursor document
  cli/watch.py     the command
  _test_hooks.py   the four seams that reach outside the process
```

`watch.py` is pure functions of a page and a held cursor, so the awkward part
— deciding where "now" is and when to move — is testable without a clock or a
socket.

## Tests

`make check`. Fakes implement the same Protocols as the production hooks and
are rebound per test; nothing is patched and nothing is mocked. The
production hook implementations are exercised separately in
`tests/test_hook_defaults.py` against a real loopback HTTP server, including
the 401 path — because `urllib` raises on a non-2xx by default, and a poster
that raised would turn the ordinary rotated-key failure into a traceback with
no status in it.
