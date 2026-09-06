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
export BOARD_WATCH_URL=...         # optional; defaults to loopback :8033
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
