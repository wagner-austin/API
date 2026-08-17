---
title: "Playing Matches in Parallel: What Actually Had to Be Separated"
tags: [harness, experiments, determinism, verification]
related:
  - "[[harness-nodisplay]]"
  - "[[harness-run-lifecycle]]"
  - "[[policy-determinism]]"
  - "[[policy-loop]]"
  - "[[runtime-split-java-agent-python-brain]]"
source_paths:
  - "src/rw_bot/harness/sweep.py"
  - "src/rw_bot/harness/clone.py"
  - "src/rw_bot/harness/runner.py"
  - "scripts/sweep.py"
  - "Makefile"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [headless-harness, bot-architecture]
---

# Playing Matches in Parallel

Every question this bot has needed answering has been answered by a batch of matches: three seeds against each of two or three arms, six to nine matches, about seven minutes each. Run one at a time that is forty-five to sixty-five minutes per question, and the cost of asking is what decides how many questions get asked.

They now run several at a time. What that took was smaller than it looks, and the reason is worth recording: **`make play` was already isolated in every respect but one.**

## What was already per-invocation

- **The agent jar.** Built into a GUID-stamped path per invocation, and deliberately so — a jar held open by a running game cannot be replaced, so a shared build makes playing impossible whenever any game is already up.
- **The channel port.** Drawn per invocation from 27600–27999.
- **The logs.** Named by the caller; the game's own log and the agent's stdout are already separate files.

## What was not

The game's own directory. A running match writes three fixed-name paths inside it:

| path | what it is |
|---|---|
| `preferences.ini` | settings, rewritten on every boot |
| `saves/autosave.rwsave.tmp2` | autosave scratch file |
| `cache/mods-info.cachedata/` | mod metadata cache |

Two matches launched from one directory race on all three. Cloning the directory is therefore the whole of what concurrent matches require, and `GAME_DIR` was already a command-line override, so no launcher change was needed at all.

## What a clone contains, and what it must not

At 0.44 GB a copy is cheap against 521 GB free, but two exclusions matter.

**`saves` and `cache` are created empty rather than copied.** They are the trees the game rewrites, so copying them risks reading one while a match already in flight is part-way through writing it, and the game rebuilds both on boot.

**`jvm` is not copied at all.** It is the 32-bit JVM, 118.5 MB, and nothing names it: the launcher runs `jvm64/bin/java.exe` for the game and `jvm64/bin/javac.exe` and `jvm64/bin/jar.exe` for the agent. Copied once per worker it was 474 MB of dead weight at four workers, scaling linearly.

Everything else is copied by **exclusion rather than an allow-list**, so a directory the game gains in a future patch is copied rather than silently dropped — a dropped directory shows up as a match that will not boot, which is an expensive way to find out.

The clone is verified before a worker is given it. A copy missing the JVM fails ninety seconds later as *"the agent never opened port N"*, which reads like a fault in the agent rather than a truncated copy.

Verification checks the entries a match needs to LAUNCH; it cannot know which map the batch will ask for, and reuse made that gap expensive: maps added to the pinned copy after a clone was made never reached it, the engine's load failed with an unread alert, and the match drifted into the boot sandbox ([[policy-determinism]], the seating section -- the whole xmap family was voided this way). Since 2026-08-06 `prepare_clone` re-syncs the pinned copy's skirmish maps into every reused clone and reports what it copied.

## Settings are reset before every match, not once per clone

The game rewrites `preferences.ini` on each boot, so without a reset the second match a worker plays starts from the first one's leavings, and two workers that have played different numbers of matches start from different settings. Measured across four clones, the files had already diverged into three distinct versions.

The only key that moved was `nextBackgroundMap` — a main-menu counter, and the harness runs `-nodisplay`, so nothing measured before the reset landed is affected. It is reset anyway. The property an experiment needs is not "the state that differs happens to be harmless" but "the state does not differ", and nothing guarantees the next key the engine writes there is a cosmetic one.

## Lockstep is not optional for a batch

`PLAY_LOCKSTEP` defaults to `0`, which free-runs the sample exchange on a wall clock. Under CPU contention several matches would then sample at different game-times — **the act of running them in parallel would change their results**, which is the one failure a batch harness must not have. Sweeps pass a frame count on every job.

It costs nothing. Measured, a locked match still runs at ~297 fps against the engine's 300 fps `Display.sync` cap, and the frame count comes out exact: 120 samples at 75 frames gave 8,925 engine frames against 9,000 requested.

## Two properties that come from one decision

A match's result is a file named after its job. That single choice gives both:

- **Resumable.** A batch killed part way through is continued by issuing the same command; jobs whose result file exists are skipped. Nothing tracks progress separately, so nothing can disagree about it.
- **Crash-isolated.** A match that dies takes its own result and no other. A batch is never a single unit of work.

A match that printed no verdict did not finish. Its transcript is kept as `.partial` and **no result file is written**, which leaves the job outstanding rather than filing a blank as though it were a measurement.

## What actually limits the worker count

Memory, not cores — and by more than the per-process figure suggests.

| resource | measured | ceiling |
|---|---|---|
| JVM working set | ~430 MB | — |
| all-in per match, incl. `make` and PowerShell parents | ~725 MB | — |
| free RAM (with ~27 of 31.8 GB held by other applications) | 4.4 GB | **~6 matches** |
| logical cores | 24, roughly one per match | ~20 |
| disk per clone | 0.44 GB against 521 GB free | irrelevant |

An earlier estimate of ~10 was taken from the JVM working set alone and was wrong by the weight of each match's launcher processes.

## Why not containers

The pinned game is a Windows binary with native libraries — `lwjgl64.dll`, `freetype.dll`, `OpenAL64.dll` — and its own bundled JVM. A Linux container means Wine, which changes the runtime substrate underneath every engine claim in this wiki; those pages are pinned to one build for exactly that reason. Windows containers or VMs each carry an OS image, and since the binding constraint is memory, an OS per instance makes density *worse* rather than better. Orchestration across machines solves a scheduling problem that does not exist here.

## Running one

    make sweep SWEEP_JOBS=sweeps/wave-mass.txt SWEEP_NAME=wave-mass SWEEP_WORKERS=4

A job file is one match per line — `label | seed | goals | max_workers | samples | mass` — with blanks and `#` comments skipped so an arm can be commented out of a batch rather than deleted from it. The format is positional and narrow deliberately: a missing field is an error naming the line, not a default quietly changing what the arm means.
