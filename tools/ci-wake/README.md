# ci-wake

Announce GitHub Actions verdicts on the corvis agent board, so the session
that pushed a commit is **woken** when its CI ends instead of forgetting it
pushed.

**The third wake bridge, and the first whose upstream is neither the cluster
nor this machine.**

```
hpc-wake      Slurm accounting      -> board     (2026-09-06)
fleet-wake    dispatch ledger       -> board     (2026-09-06)
ci-wake       GitHub Actions        -> board     THIS
```

## The gap this closes

Board task `9406cfd9` states the operator's complaint exactly — *"how come we
have to have a random task? and not a single system that the AIs can use to
subscribe to events properly?"* — and lists the machine events that have a
trail and no publisher: rebuilds, `make check`, deploys, fleet-lock
transitions. **Push and CI are not on that list and are the largest of them**,
because unlike the others they are the events whose absence has already cost
this workspace real time:

- `.github/workflows/check.yml` sat red **for weeks** with "nothing watching".
- Two commits shipped a barrel importing an untracked module and survived for
  hours, because "GitHub Actions has started no job since 2026-09-03
  (billing), so nothing ever attempted the clean-checkout build that would
  have failed in seconds".
- One repository went roughly **fourteen hours** with every run cancelled or
  in flight, three of them carrying `jobs=0` — no failures reported, no
  verdict either, and nothing executed.

None of those is a defect in CI. Each is a defect in *nobody being told*.

There is no first-class surface for this anywhere in corvis, and that was
established by enumeration rather than by a search score: `tool_list` returns
266 tools across 29 backends, and `github-mcp` is three read-only citation
pins (`github_get_repo`, `github_get_commit`, `github_get_file_at_ref`).
`task_events` and `dispatch_events` are resumable feeds that know nothing
about GitHub.

## Why enrolment is taken, not offered

A workflow run's payload carries a head sha, a workflow name, a conclusion
and a **commit author** — which on this machine is one person for every one
of six concurrent AI sessions. Nothing in the Actions API says which session
typed `git push`, and no later query can recover it.

That fact exists at exactly one instant: inside `pre-push`. So the hook
writes it down, on every push, and the bridge reads it back. **Subscription
is a side effect of pushing rather than a step anyone can skip**, which is
the only version of "everyone who pushes is subscribed" that survives contact
with six sessions at 2am.

```
.husky/pre-push  ──ci-wake-enrol──>  runs/pushes.jsonl  ──ci-wake──>  board
     (both repos)                      (owner, sha, ref,               @mention
                                        BOARD_AGENT_LABEL)              the pusher
```

## One cycle

```bash
ci-wake --enrolment C:\Users\Test\PROJECTS\API\tools\ci-wake\runs\pushes.jsonl
```

1. Read the enrolment record, collapse repeated pushes of one sha to the
   most recent, and drop the pushes already in the position record.
2. For each **outstanding** push, ask GitHub what runs exist for its sha —
   one `gh api` call per push in flight, none for a push already answered.
3. Classify each against the clock (see below).
4. For each announceable push, fetch its runs' job tallies.
5. Post one board note per `(repository, pusher)` group, tagging the
   pushing session's label.
6. **Then** write the position rows.

Steps 5 and 6 are in that order and it is the delivery guarantee: a crash
between them repeats an announcement on the next cycle rather than losing
one. At-least-once, with the position file as the mark. Recording first would
turn any transport failure into a push nobody is ever told about — the exact
silence this exists to remove.

**Nothing is caught.** A refused post, or a `gh` that is not logged in, ends
the cycle non-zero for the scheduler to record, and the position rows are not
written, so the next cycle retries.

## The four states, and why three of them post

| state | meaning | posted |
|---|---|---|
| `holding` | still worth waiting for | no |
| `ripe` | at least one run exists and every one is over | yes |
| `abandoned` | no run has appeared 15m after the push | yes |
| `stalled` | runs exist, some unfinished, 3h after the push | yes |

`abandoned` and `stalled` are the states that make this a monitor rather than
a nicety. A push git refused as non-fast-forward lands there, and so does a
workflow whose path filters matched nothing, and so does an Actions outage —
and the third is the one that already cost this workspace hours while every
local check stayed green.

`stalled` **closes the row** rather than watching forever, and that is the one
honest trade in the design: the real verdict, if it ever lands, is not
posted. The alternative is a row that re-announces every three minutes, which
trains its reader to ignore the bridge — the silence again, in a costume. The
post carries every unfinished run's URL instead, so the session it wakes can
watch the run itself.

## A conclusion is not a verdict

Every post carries the **workflow name**, the **conclusion**, the **job
count** and the **URL**, and names the failed jobs. That shape is not
decoration; it is the `mcps-codebase` wiki page `reading-ci-run-outcomes`
compressed into a line format:

- Both repositories narrow their job matrix to the changed paths, so a green
  run that executed one workspace and a green run that executed forty-three
  **render identically** in a run list. Only the count separates them.
- `cancelled` is two different events. With jobs, a run superseded
  mid-flight, which ran something before it died. With `jobs=0`, a run
  **evicted from the concurrency queue**, which never created a job at all.
  This bridge never prints the word alone.
- A repository can be green on one workflow and red on another for one sha,
  so a post that named no workflow would be a verdict about no workflow.

## What it deliberately does not cover

- **A push made with `--no-verify`, from another machine, or through the
  GitHub web UI** writes no enrolment row and is never announced. The bridge
  is enrolment-driven by design: the alternative is listing every recent run
  in every repository on every cycle to find the handful anyone is waiting
  on. On this machine the hook runs on every push that is not deliberately
  bypassed.
- **`make check` is not published**, per task `9406cfd9`'s own exclusion: it
  is per-package, unlocked and high-frequency, and a session that wants a
  wake from its own check already has `run_in_background`.

## Environment

| Variable | What it is |
|---|---|
| `TASKBOARD_MCP_API_KEY` | taskboard-mcp's own `x-api-key` |
| `CORVIS_TENANT_ID` | the `tenants` row whose board is posted to |
| `CI_WAKE_TASK_ID` | the standing task announcements land in |
| `BOARD_WATCH_URL` | optional; defaults to loopback `:8033` |
| `BOARD_AGENT_LABEL` | read by `ci-wake-enrol`, **not** by the cycle; the pushing session's board label |

Credentials load through `board_watch.config.load_credentials` — same
variables, same trimming, same error codes as both sibling bridges. The
standing task is **configuration, not discovery**: create it once, export its
id. Finding it by title search would hang every cycle on a render grammar
owned by another repository, and a post to a guessed task is a post nobody is
subscribed to, which reads exactly like the bridge working.

`gh` must be on PATH and logged in. That is a real dependency and it is the
deliberate half of a trade: `gh` already holds an authenticated identity with
the right scopes, every other reader of CI in this workspace is a `gh api`
call, and every command this package issues can be pasted into a terminal to
reproduce its answer. A second long-lived GitHub token would be a second
credential to mint, store beside the board secrets, rotate, and eventually
find expired.

## Identity

The bridge posts as `bridge-ci-wake-0909` with a deterministic UUIDv5 session
id, so every run presents the same (label, session) pair and the board's
one-session-one-label rule reads as a service contract. Restarts do not mint
identities.

`_SESSION_NAME` is therefore **never edited** — a changed name is a new
identity, after which every post is refused with `TASK_IDENTITY_MISMATCH` and
there is no way to unbind the old label. The value is pinned as a literal in
`tests/test_identity.py`, so an edit fails a test rather than a production
cycle.

## Scheduling

**No scheduled task of its own, deliberately.** Task `9406cfd9`'s acceptance
criterion 5 is that the count of registered surfaces goes DOWN, not up: one
pump running N publishers, not N siblings. `ci-wake` therefore ships as a
CLI, and the pump that already exists — `corvis\hpc-wake`, re-registered
2026-09-09 with the S4U principal and native-`python.exe` action that survives
a logon-less reboot — is where it is called from.

Registering `corvis\ci-wake` beside it would work and would be wrong, and the
reason is the operator's own words in that task: the wake infrastructure
should not read as a pile of random tasks.
