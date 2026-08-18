---
title: "Run Lifecycle: Why Ad-Hoc Jobs Stopped Being Enough"
tags: [harness, experiments, process, verification]
related:
  - "[[harness-parallel-matches]]"
  - "[[harness-nodisplay]]"
  - "[[policy-determinism]]"
source_paths:
  - "scripts/sweep.py"
  - "src/rw_bot/harness/runner.py"
  - "sweeps/run-xmap-1v1.sh"
source_git_blobs:
  "scripts/sweep.py": "8e99b999502b279e76b928e656f9e0261ad2680a"
  "src/rw_bot/harness/runner.py": "ace02192fa040da0cadc67d3b486300ab03112c5"
  "sweeps/run-xmap-1v1.sh": "4bf70377b380d60fae8fa09c1dd5509ddd483f5f"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [headless-harness]
---

# Run Lifecycle: Why Ad-Hoc Jobs Stopped Being Enough

Every sweep, probe and panel today is a detached shell job: `make sweep ... &`,
chained `sh` scripts with `until`-loops, background probes. That worked at one
run at a time. On 2026-08-05 the project ran two panels, a probe series, and a
four-sweep chain concurrently, and one evening produced the full failure
catalogue this page exists to record.

## The measured incidents (2026-08-05, one session)

- **Killing a sweep took three rounds**, because its pieces die separately: the
  `sh` chain loop, the `make`/`poetry` wrappers, the sweep runner, the planner,
  and the engine are five processes, and stopping any subset lets the rest
  respawn or run on. The cross-map chain was "stopped" twice and kept going.
- **A surviving runner silently blocked the next panel.** The killed chain's
  `duel-big_island` runner lived on, relaunched its match on clone `.game-w1`,
  and `ledger-solo24` -- whose single worker leases the same clone -- sat
  stalled for twenty minutes with no error anywhere. Silent dir contention
  looks identical to a slow launch.
- **Two sweeps measured a broken condition before anyone checked.** The
  seating regression (players 4 -> 2 on a map asked to seat 2) was visible in
  the FIRST sample of the first match, but nothing reads a batch's opening
  sample before committing hours to it.
- **Strays from a five-day-old session** were only discovered because a person
  asked what was running (2026-07-30 processes found 2026-08-05).

Prior sessions contributed the rest of the catalogue: background-task teardown
killed watchers and a live sweep (2026-08-02); parallel engine loads off one
`.game` tripped the safe-mode counter and broke menus until staggered launches
were adopted as folklore rather than enforced ([[harness-nodisplay]]).

## The design that answers it (task #34)

One supervisor, `scripts/runs.py`, as the only launch surface:

1. **A registry of whole process trees.** Every run records its name, runner /
   planner / engine PIDs, game dir, frozen-tree identity and start time.
   `runs stop <name>` kills the tree in one call; `runs list` shows everything
   alive, including orphans from dead sessions, at every session start.
2. **Clone-pool leasing.** Game dirs are leased, not assumed. A launch that
   wants a busy dir queues visibly instead of stalling silently.
3. **Preflight gates.** Before a batch commits hours: the lease is free, the
   safe-mode counter reads zero, and the first sample's `players` line matches
   the requested seating -- the check that would have stopped the 1v3 confound
   at launch #1 instead of after eight sweeps and a retraction
   ([[policy-determinism]]).
4. **Queues instead of shell chains.** "Run these four after that one" becomes
   supervisor state that survives kills, not an orphanable `until`-loop.

## The rules already in force (before the supervisor lands)

- **Stopping a sweep means stopping its runner**, not its match: kill the
  `scripts.sweep` python and its children, then verify no new match log
  appears within a sweep interval.
- **Every scorecard read starts at the `players` line.** A verdict whose
  seating was never checked is a verdict about an unknown experiment
  ([[policy-determinism]]).
- **One engine per game dir.** The safe-mode counter
  (`numLoadsSinceRunningGameOrNormalExit`) is checked after any unclean kill,
  and clones are inspected before reuse.
- **Frozen trees are the run's identity.** A batch's `.tree` records what it
  measured; comparing batches means comparing tree identities first
  ([[harness-parallel-matches]]).
