---
title: The Policy Loop
tags: [policy, planner, architecture, scoring]
related:
  - "[[command-channel]]"
  - "[[building-structures]]"
  - "[[engine-entity-model]]"
  - "[[runtime-split-java-agent-python-brain]]"
  - "[[mechanics-resource-pools]]"
  - "[[wire-contract-ndjson]]"
  - "[[policy-budget]]"
  - "[[policy-verdict]]"
source_paths:
  - "wiki/sources/m9-policy/plan-completed.txt:5"
  - "wiki/sources/m9-policy/plan-stalled-on-laboratory.txt:5"
  - "wiki/sources/m9-policy/engine-refused-the-laboratory.txt:4"
  - "wiki/sources/m9-policy/visible-includes-opponents.ndjson:1"
  - "wiki/sources/m11-pools/pool-build-run.log:403"
  - "wiki/sources/m11-pools/builder-travel-timing.txt:13"
  - "wiki/sources/m12-produce/produce-timing.txt"
  - "wiki/sources/m12-produce/produce-run.log:411"
  - "wiki/sources/m12-produce/produce-run.log:437"
  - "src/rw_bot/policy/build_order.py"
  - "src/rw_bot/policy/runner.py"
  - "src/rw_bot/policy/campaign.py"
  - "src/rw_bot/policy/budget.py"
  - "scripts/play.py"
source_git_blobs:
  "wiki/sources/m9-policy/plan-completed.txt": "b6f1fad82d73c97f2766ba2612c9b4e8c7b20d7b"
  "wiki/sources/m9-policy/plan-stalled-on-laboratory.txt": "42b0264abde9c5f1be10ae3eabd436d2485fd3d0"
  "wiki/sources/m9-policy/engine-refused-the-laboratory.txt": "88891e856780548f80c6165cf10e4146e36359ff"
  "wiki/sources/m9-policy/visible-includes-opponents.ndjson": "c8404de03d2eaa122169110db186d75eac2c83e3"
  "wiki/sources/m11-pools/pool-build-run.log": "d661b6813fdcc17b1cdc08da7fc390fe22ce67b6"
  "wiki/sources/m11-pools/builder-travel-timing.txt": "c73ead926dfb2d281ba4ec591f56d6545ce2e948"
  "wiki/sources/m12-produce/produce-timing.txt": "fe366f914e87811134a45ac1f7fe0ac041f3a711"
  "wiki/sources/m12-produce/produce-run.log": "9e6dbf9b8d9f7867ed36868664fdfd7210f46a79"
  "src/rw_bot/policy/build_order.py": "e07a055e86b45ac976c4eab4cea48cfe0f860c3c"
  "src/rw_bot/policy/runner.py": "749de3dbe47cc72f97571d57ee21677c4eb4fba6"
  "src/rw_bot/policy/campaign.py": "ae3700f5a5c413b05f2909de398d1154d8262b2f"
  "src/rw_bot/policy/budget.py": "06e3cb9d18cf4b87be4d309da6b5a9b52a0c226f"
  "scripts/play.py": "c95267440b13b56f48a32def2a00b99e2f9efe72"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [bot-architecture, game-mechanics]
---

# The Policy Loop

> **[2026-07-26] There is one loop now.** What follows describes the rules the
> loop applies; the two-phase shape it used to have is gone. The build loop ran
> the opening plan to completion and handed over to a fight loop, and that seam
> was the bot's largest structural defect rather than a separation of concerns:
> while building there was no army and no economy, once fighting there was no
> build policy at all — so `extractorT1` was the only structure that could ever
> be placed again and the factory count was frozen for the rest of the match —
> and a plan that stalled meant a match that never fought, because the handover
> was conditional on the plan finishing.
>
> `rw_bot.policy.campaign.play` is now the whole loop: perceive, arbitrate,
> dispatch, acknowledge, every observation, for the whole match. Spending across
> the layers is arbitrated by [[policy-budget]] rather than raced. What survived
> of `runner.py` is `OrderTracker` — the judgement about whether an order
> already given is still being carried out, which was never about looping.
>
> **One builder, one owner.** The plan holds the builder for as long as it wants
> something placed, including while waiting to afford it. A live 400-sample run
> with both the plan and the economy ordering it had four expansions issued and
> the plan still stuck at 3 of 8; the engine runs whichever waypoint arrived
> last, so neither order was carried out. With the rule in place, same seed and
> same budget: plan 4 of 8, one expansion, factory up and producing.
>
> **The match ends when the engine says so.** "No army left" and "nothing
> hostile in sight" were both dropped as exit conditions. The second is the
> opening position of every match — the map is fogged — so it would have ended
> the run on its first observation. The first is no longer terminal now that
> production runs every tick. Both were proxies for a verdict the engine states
> outright ([[policy-verdict]]).

The bot plays a build order unattended: it reads the world, decides what to make next, orders it, watches for the result, and reports a scorecard. Six entries, six orders, no waste -- five structures placed and one unit produced ([[command-channel]]).

## Deciding is pure

`decide` takes a sample, a plan, the unit catalogue and the placement rules, and returns what to do. It opens no socket, reads no clock and mutates nothing, which is why the playing logic can be tested exhaustively without a game running — the same separation the agent holds on its own side ([[runtime-split-java-agent-python-brain]]).

The loop around it is thin by design: read, ask, act, repeat.[^5] Everything that could be judged wrong lives in the pure half, where a world state goes in and a decision comes out.

## Presence is not completion

A building joins the roster the moment construction starts, so counting on presence reported a plan finished while a factory was still a shell -- and a shell produces nothing, so the next entry could be ordered against a building that could not accept it.[^6] Progress therefore counts only finished structures.

That correction has a consequence which has to land with it. The builder stops moving the instant it arrives, and the structure appears unfinished at about the same moment, so movement stops being evidence of progress exactly when construction starts. Without the rising structure counting as in-flight too, the fix would trade a wrong scorecard for a false stall. The run is visibly slower for it -- the same six entries take 559 samples where presence-counting took 215, because it now waits for each one to finish.[^12]

## Progress is read, never counted

Plan progress comes from the roster rather than a counter.[^6] A structure that was destroyed stops counting and gets rebuilt, and a planner that reconnects mid-match sees the same progress as one that watched the whole thing — both fall out of reading rather than remembering.

Two ownership traps sit here, and the stream makes both live. A sample carries **every visible entity**, not just the player's: nineteen entities across five teams in one capture, of which three were ours.[^1] Without the ownership check an opponent building the same structure in view would advance our plan, and an enemy builder could be selected to receive an order it would never accept ([[engine-entity-model]]).

## Ordering once is necessary and not sufficient

Structures take time to appear and the roster is what reports them, so re-deciding every sample would re-order the same structure repeatedly while the first was still going up — spending credits the policy believes it still holds. Each plan position is therefore ordered at most once.[^5]

That protection has a cost, and a live run found it. A plan ending in a laboratory ran for three hundred samples and eighty-nine thousand frames, banked eleven thousand credits, and reported *"building laboratory (3 of 3)"* the entire time.[^2] Nothing was wrong with credits, placement or the channel. The engine had refused the order outright and said so only in its own log: **`Unit 'builder' can not queue build:laboratory`**.[^3]

A builder cannot construct a laboratory. That is not derivable from the unit catalogue, which carries prices and stats but no build lists ([[building-structures]]), and the refusal produces no roster change and no error the planner can see.[^3] So the loop now stops after a bounded number of samples with no progress and reports `stalled`, naming what it was waiting on.[^5] A once-only order without that check is a bot that looks like it is working forever.

## Who can make it is asked, not assumed

The planner used to find a unit to order by looking for one whose type name was `builder`. That was a guess wearing the clothes of a constant, and it cost a three-hundred-sample run: a builder cannot construct a laboratory, the engine refuses the waypoint silently, and nothing in the catalogue says so ([[building-structures]]).

The engine answers the question itself, per unit, and the answer now rides in every sample ([[wire-contract-ndjson]]). `find_producer` reads it, so a plan entry nothing owned can make is `blocked` before an order is spent rather than after three hundred samples of reporting progress.[^9]

One exclusion is load-bearing. The map editor's placeholder is an owned entity in every sample, parked off-map at (-1000, -1000), and it answers for **108** types against the real Builder's **13** — a superset of everything the Builder can make, plus 95 more including the laboratory.[^10] Counting it would make the check above pass for types nothing playable can build, and the order would go to a unit that is not in the game. The check would look like protection while removing it.

## Two verbs, because the engine has two

A structure is placed at a position the planner chooses. A unit rolls out of the building that made it, and the engine decides where — so a produce order carries no coordinate at all. Which verb applies is read from the action rather than guessed from the produced type's speed.[^9]

## One stall rule, two kinds of evidence

The clock only runs while nothing observable is happening. What counts as observable differs by verb, and neither is a deadline the planner invents.

A **placed build** is in flight while the builder walks to the site, and then while the structure itself is going up. A **produced unit** is in flight while the producing building holds it in its queue, which the building reports directly.[^9] That keeps the rule uniform: a factory never moves, so the movement test alone would call a working one refused.

Two worse answers were tried first and are worth recording, because both look reasonable. Bounding production by **elapsed samples** caps what the bot can afford — production time scales with price, so any fixed window silently forbids expensive units, exactly as the pre-fix travel window forbade distant ones. Bounding it by **falling credits** does not work at all: measured through one production run the balance read 4243, 3678, 3813, 3849 — *rising* through most of it, because income outpaced the drain.[^11]

The queue settles it, and it was measured rather than assumed: ordering a Scout, the Command Center reported `queued: 1` for all forty-five samples the unit took and dropped to zero on the sample it appeared.[^11] Production time itself is linear in price — a $500 Builder took 34 samples and a $700 Scout 45, or 14.7 and 15.6 credits per sample — which is why no constant derived from it could have been right for both.[^11]

## Two placement rules, because the engine has two

Most structures go on a ring of offsets around a fixed anchor — the oldest owned immobile entity, which is the Command Center at the start of a match. The anchor has to be something that holds still: the builder walks to every site it is sent to, so a ring centred on it collapses onto itself by the third structure.

Extractors do not use the ring at all. The engine allows them on resource pools and nowhere else, so their legal sites are exactly the free pools in view, and offering a ring position would produce an order the engine refuses without saying so. Which types are bound that way is read out of the live engine rather than assumed, and the pools themselves ride in the world stream as terrain ([[mechanics-resource-pools]]).

## The stall window measures standing still, not elapsed time

A window counted from the moment an order is sent silently caps how far the bot can build, and that is not a hypothetical: at a measured 11.7 world units per sample, forty-five samples reach 527 units, and a perfectly good order to a pool 588 units out was reported as refused while the builder was still walking to it. It finished seconds after the run gave up.[^8]

Timing one far build settled the fix. The structure appeared on the very sample the builder stopped moving — travel is the whole of the delay at this sampling rate — so a builder still moving is an order still in flight, and the clock only runs while it stands still.[^5][^8] That needs no speed constant, no frame rate, and no assumption about map size, which matters because the catalogue's speed figure turns out not to be a distance per unit time ([[mechanics-resource-pools]]).

## The scorecard

A run is judged, not just performed:[^7]

```
outcome        done (all 6 plan entries satisfied)
completed      6/6
orders sent    6
samples seen   559
frames elapsed 41814
credits left   12682
```

Six entries, six orders: three extractors and two factories placed, and one Scout produced.[^12]

`orders sent` against `completed` is the figure that matters most — equal means nothing was wasted, higher means orders were re-issued, refused or lost. The stalled run scored three orders for two structures, which is what a refusal looks like in the numbers.[^2]

## What it still is not

This policy executes a fixed sequence. It does not scout, fight, react to an opponent, or choose what to build from anything but a hardcoded list — and it plays against five opponents who are doing all of those things.[^1] It has no notion of winning. It now expands toward resources in the narrow sense that extractors go to pools, but it picks them by distance alone and never upgrades one.

It knows what it *could* make and does not use that to plan. The build tree is read every sample, so the planner could derive that a Mammoth Tank needs a Land Factory first and insert it — but the plan is still a list a human wrote, and an entry whose prerequisite is missing is reported blocked rather than solved.

What it does have is the shape a real policy needs: observe, decide, act, verify, score, and fail loudly when the world disagrees with it ([[building-structures]]).

[^1]: `wiki/sources/m9-policy/visible-includes-opponents.ndjson:1` — a frame declaring `"visible":19` followed by entities on teams 5 and 1 with `"mine":false`; the full capture at `wiki/sources/m6-wire/world-sample.ndjson` carries teams 0, 1, 3, 5 and 7.
[^2]: `wiki/sources/m9-policy/plan-stalled-on-laboratory.txt:5` — `outcome sample_limit (building laboratory (3 of 3))` with `completed 2/3` at `:6`, `orders sent 3` at `:7` and `credits left 11258` at `:10`.
[^3]: `wiki/sources/m9-policy/engine-refused-the-laboratory.txt:4` — `Unit 'builder' can not queue build:laboratory`, followed at `:5` by `isValidNewWaypoint==false on: builder(pos:4247,2767 id:214 t:0)`; the order the planner sent is at `:3`.
[^5]: `src/rw_bot/policy/runner.py` — `run` reads a sample, calls `decide`, and dispatches; `ordered_positions` bounds each plan position to one order, and `stall_samples` converts an unchanging position into a `stalled` outcome only once `_has_moved` reports the builder stationary.
[^6]: `src/rw_bot/policy/build_order.py` — `completed_count` walks the roster, skipping entities whose `mine` is false, and matches each plan entry at most once.
[^7]: `wiki/sources/m11-pools/pool-build-run.log:403` — the five `channel: build` lines of the pool-aware run, three `extractorT1` and two `landFactory`, scoring `completed 5/5, orders sent 5` over 170 samples. It supersedes the earlier three-factory run at `wiki/sources/m9-policy/plan-completed.txt:5`, which is what the scorecard block showed before extractors were in the plan.
[^9]: `src/rw_bot/policy/build_order.py` — `find_producer` reads the sample's own option list; `decide` blocks on no producer, waits on an unavailable action, and branches to `produce` when the option reports `placed` false.
[^10]: `wiki/sources/m6-wire/world-sample.ndjson:1` — in the first sample of the archived capture the option records divide 2 / 13 / 108 across units 213 (`commandCenter`), 214 (`builder`) and 217 (`editorOrBuilder`); all 13 of the Builder's types also appear under 217, and `laboratory` appears only under 217.
[^11]: `wiki/sources/m12-produce/produce-timing.txt` — the two timings, the derived per-sample rate, and the credit series showing the balance rising mid-production.
[^12]: `wiki/sources/m12-produce/produce-run.log:411`–`:437` — five `channel: build` lines followed by `produce: scout via action 'u_scout' on com.corrodinggames.rts.game.units.a.l` and `channel: produce scout by 213`, six dispatches for six plan entries with nothing re-issued. The action class is the same `a.l` the bytecode identified as unit production, which is what closes the loop between the reading and the dispatch.
[^8]: `wiki/sources/m11-pools/builder-travel-timing.txt:13` — `RESULT travel_samples=52 total_samples=52`, with `construction_samples=0` at `:14` and `units_per_sample=11.72` at `:16`, over the 609.3-unit order stated in the header at `:5`.
