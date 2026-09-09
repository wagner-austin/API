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

## Scheduling

The entry point is `scripts/run_cycle.py` — plain Python, registered as
Windows scheduled task `corvis\hpc-wake` with a **native-binary action and
an S4U principal**. Both halves of that are measured constraints from
2026-09-09, not preferences:

- Registered `LogonType=Interactive` (the original form), the scheduler
  MISSES every trigger while no desktop session exists: 37 consecutive
  missed runs after a Windows-Update reboot, `State: Ready`,
  `Start-ScheduledTask` returning without running anything. Every wake on
  the machine went silent for two hours.
- Registered S4U with `powershell.exe` as the action, PowerShell itself
  deadlocks before reaching any script (measured on this box by the
  Docker-autostart work: stall at .NET assembly load, repeatable). A
  native binary does not.

Registration, once, from any interactive session:

```powershell
$py = (Get-Command python).Source
$action = New-ScheduledTaskAction -Execute $py `
  -Argument '"C:\Users\Test\PROJECTS\API\tools\hpc-wake\scripts\run_cycle.py"'
$rep = New-TimeSpan -Minutes 3
$t1 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval $rep
$t2 = New-ScheduledTaskTrigger -AtStartup
$t2.Repetition = $t1.Repetition
$principal = New-ScheduledTaskPrincipal -UserId 'Test' -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
  -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskPath '\corvis\' -TaskName 'hpc-wake' `
  -Action $action -Trigger $t1, $t2 -Principal $principal -Settings $settings -Force
```

The `AtStartup` trigger re-arms the cycle after a reboot with no logon;
`IgnoreNew` keeps a slow cycle from stacking; the 30-minute limit kills a
hung one so the next trigger is not silently skipped (the 2026-09-08
outage mode). Health is read from `runs/cycle.log`: the LAST line must be
a result, and the last header's AGE must be under a few minutes — a stale
timestamp IS an outage even when the line under it looks healthy.

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
