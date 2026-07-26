---
title: The Policy Loop
tags: [policy, planner, architecture, scoring]
related:
  - "[[command-channel]]"
  - "[[building-structures]]"
  - "[[engine-entity-model]]"
  - "[[runtime-split-java-agent-python-brain]]"
  - "[[mechanics-resource-pools]]"
source_paths:
  - "wiki/sources/m9-policy/plan-completed.txt:5"
  - "wiki/sources/m9-policy/plan-stalled-on-laboratory.txt:5"
  - "wiki/sources/m9-policy/engine-refused-the-laboratory.txt:4"
  - "wiki/sources/m9-policy/visible-includes-opponents.ndjson:1"
  - "wiki/sources/m11-pools/pool-build-run.log:403"
  - "wiki/sources/m11-pools/builder-travel-timing.txt:13"
  - "src/rw_bot/policy/build_order.py"
  - "src/rw_bot/policy/runner.py"
  - "scripts/play.py"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: high
hubs: [bot-architecture, game-mechanics]
---

# The Policy Loop

The bot plays a build order unattended: it reads the world, decides what to build next, orders it, watches for the result, and reports a scorecard. Five structures, five orders, no waste ([[command-channel]]).

## Deciding is pure

`decide` takes a sample, a plan, the unit catalogue and the placement rules, and returns what to do. It opens no socket, reads no clock and mutates nothing, which is why the playing logic can be tested exhaustively without a game running — the same separation the agent holds on its own side ([[runtime-split-java-agent-python-brain]]).

The loop around it is thin by design: read, ask, act, repeat.[^5] Everything that could be judged wrong lives in the pure half, where a world state goes in and a decision comes out.

## Progress is read, never counted

Plan progress comes from the roster rather than a counter.[^6] A structure that was destroyed stops counting and gets rebuilt, and a planner that reconnects mid-match sees the same progress as one that watched the whole thing — both fall out of reading rather than remembering.

Two ownership traps sit here, and the stream makes both live. A sample carries **every visible entity**, not just the player's: nineteen entities across five teams in one capture, of which three were ours.[^1] Without the ownership check an opponent building the same structure in view would advance our plan, and an enemy builder could be selected to receive an order it would never accept ([[engine-entity-model]]).

## Ordering once is necessary and not sufficient

Structures take time to appear and the roster is what reports them, so re-deciding every sample would re-order the same structure repeatedly while the first was still going up — spending credits the policy believes it still holds. Each plan position is therefore ordered at most once.[^5]

That protection has a cost, and a live run found it. A plan ending in a laboratory ran for three hundred samples and eighty-nine thousand frames, banked eleven thousand credits, and reported *"building laboratory (3 of 3)"* the entire time.[^2] Nothing was wrong with credits, placement or the channel. The engine had refused the order outright and said so only in its own log: **`Unit 'builder' can not queue build:laboratory`**.[^3]

A builder cannot construct a laboratory. That is not derivable from the unit catalogue, which carries prices and stats but no build lists ([[building-structures]]), and the refusal produces no roster change and no error the planner can see.[^3] So the loop now stops after a bounded number of samples with no progress and reports `stalled`, naming what it was waiting on.[^5] A once-only order without that check is a bot that looks like it is working forever.

## Two placement rules, because the engine has two

Most structures go on a ring of offsets around a fixed anchor — the oldest owned immobile entity, which is the Command Center at the start of a match. The anchor has to be something that holds still: the builder walks to every site it is sent to, so a ring centred on it collapses onto itself by the third structure.

Extractors do not use the ring at all. The engine allows them on resource pools and nowhere else, so their legal sites are exactly the free pools in view, and offering a ring position would produce an order the engine refuses without saying so. Which types are bound that way is read out of the live engine rather than assumed, and the pools themselves ride in the world stream as terrain ([[mechanics-resource-pools]]).

## The stall window measures standing still, not elapsed time

A window counted from the moment an order is sent silently caps how far the bot can build, and that is not a hypothetical: at a measured 11.7 world units per sample, forty-five samples reach 527 units, and a perfectly good order to a pool 588 units out was reported as refused while the builder was still walking to it. It finished seconds after the run gave up.[^8]

Timing one far build settled the fix. The structure appeared on the very sample the builder stopped moving — travel is the whole of the delay at this sampling rate — so a builder still moving is an order still in flight, and the clock only runs while it stands still.[^5][^8] That needs no speed constant, no frame rate, and no assumption about map size, which matters because the catalogue's speed figure turns out not to be a distance per unit time ([[mechanics-resource-pools]]).

## The scorecard

A run is judged, not just performed:[^7]

```
outcome        done (all 5 structures built)
completed      5/5
orders sent    5
samples seen   170
frames elapsed 12703
credits left   1710
```

`orders sent` against `completed` is the figure that matters most — equal means nothing was wasted, higher means orders were re-issued, refused or lost. The stalled run scored three orders for two structures, which is what a refusal looks like in the numbers.[^2]

## What it still is not

This policy executes a fixed sequence. It does not scout, fight, react to an opponent, or choose what to build from anything but a hardcoded list — and it plays against five opponents who are doing all of those things.[^1] It has no notion of winning. It now expands toward resources in the narrow sense that extractors go to pools, but it picks them by distance alone and never upgrades one.

What it does have is the shape a real policy needs: observe, decide, act, verify, score, and fail loudly when the world disagrees with it ([[building-structures]]).

[^1]: `wiki/sources/m9-policy/visible-includes-opponents.ndjson:1` — a frame declaring `"visible":19` followed by entities on teams 5 and 1 with `"mine":false`; the full capture at `wiki/sources/m6-wire/world-sample.ndjson` carries teams 0, 1, 3, 5 and 7.
[^2]: `wiki/sources/m9-policy/plan-stalled-on-laboratory.txt:5` — `outcome sample_limit (building laboratory (3 of 3))` with `completed 2/3` at `:6`, `orders sent 3` at `:7` and `credits left 11258` at `:10`.
[^3]: `wiki/sources/m9-policy/engine-refused-the-laboratory.txt:4` — `Unit 'builder' can not queue build:laboratory`, followed at `:5` by `isValidNewWaypoint==false on: builder(pos:4247,2767 id:214 t:0)`; the order the planner sent is at `:3`.
[^5]: `src/rw_bot/policy/runner.py` — `run` reads a sample, calls `decide`, and dispatches; `ordered_positions` bounds each plan position to one order, and `stall_samples` converts an unchanging position into a `stalled` outcome only once `_has_moved` reports the builder stationary.
[^6]: `src/rw_bot/policy/build_order.py` — `completed_count` walks the roster, skipping entities whose `mine` is false, and matches each plan entry at most once.
[^7]: `wiki/sources/m11-pools/pool-build-run.log:403` — the five `channel: build` lines of the pool-aware run, three `extractorT1` and two `landFactory`, scoring `completed 5/5, orders sent 5` over 170 samples. It supersedes the earlier three-factory run at `wiki/sources/m9-policy/plan-completed.txt:5`, which is what the scorecard block showed before extractors were in the plan.
[^8]: `wiki/sources/m11-pools/builder-travel-timing.txt:13` — `RESULT travel_samples=52 total_samples=52`, with `construction_samples=0` at `:14` and `units_per_sample=11.72` at `:16`, over the 609.3-unit order stated in the header at `:5`.
