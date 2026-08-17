---
title: "The Economy: What a Pool Is Worth, and Why the Bot Only Ever Took Three"
tags: [policy, economy, extractors, measurement, verification]
related:
  - "[[policy-combat]]"
  - "[[policy-loop]]"
  - "[[policy-threat]]"
  - "[[ai-opponent-strategy]]"
  - "[[mechanics-resource-pools]]"
  - "[[mechanics-build-tree]]"
source_paths:
  - "wiki/sources/m19-income/measured-rates.txt"
  - "wiki/sources/m19-income/income-windows.ndjson"
  - "wiki/sources/m19-income/economy-run.txt"
  - "wiki/sources/m15-production/before-after.txt"
  - ".game/assets/units/extractor/extractor.ini"
  - ".game/assets/units/fabricator/fabricatorT1.ini"
  - "src/rw_bot/policy/economy.py"
  - "src/rw_bot/policy/ledger.py"
  - "wiki/sources/m28-holding/diag-post-worker-fix.ndjson"
  - "src/rw_bot/mechanics/income.py"
  - "scripts/income.py"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [bot-architecture, engine-internals]
---

# The Economy: What a Pool Is Worth, and Why the Bot Only Ever Took Three

For most of this bot's life its income was a constant. `DEFAULT_GOALS` asked for three extractors, the build plan finished about thirty seconds in, and income never changed again for the rest of the match — on a map carrying **46 resource pools**.

That was not a bug in any one function. `sustain` orders produced units and refuses placed structures, and `reinforcements` drops anything that needs a pool. Both are right: choosing a site is a placement decision and a producer queue cannot express one.[^1] The gap was a policy nobody had written.

## The measurement that should have come first

An extractor costs 700 and the catalogue says only "-Generates credits". No rate. So every trade between credits now and income later — the reserve held back for the army, the fourth extractor against two more tanks — had been settled by argument.

Two sources now answer it, and they agree to within 0.1%.

**Measured.** One match, four extractors built one at a time, standing completely still for 200 samples between each so that every window's credit slope is income and nothing else:[^2]

| extractors | credits/s |
|---|---|
| 0 | 26.95 |
| 1 | 39.00 |
| 2 | 51.08 |
| 3 | 63.09 |
| 4 | 75.01 |

Successive differences: +12.05, +12.08, +12.01, +11.92. **12.01 credits/s per extractor**, and a 700-credit extractor pays for itself in **58.3 seconds**.

**Read from the game.** Extractors and fabricators are plain INI files shipped with the game, not obfuscated bytecode, and they state the figure outright: `generation_resources: credits=8`.[^3] Eight generation units measured 12.01 credits/s, so `credits/s = generation_resources × 1.5` — which is exactly what a 60-tick simulation gives against the parser's default `generation_delay` of 40.[^4]

The 26.95/s at zero extractors is the Command Center, a built-in engine unit with no readable INI. Under the same conversion its generation value is 17.97, i.e. 18.

## Spread, do not upgrade

The upgrade chain is real — an `extractorT1` lists `extractorT2` as a build edge — and it is a trap while pools remain.

| action | cost | +generation | cost per +1 | payback |
|---|---|---|---|---|
| **new extractorT1** | **700** | **+8** | **87.5** | **58s** |
| upgrade T1 → T2 | 1400 | +4 | 350.0 | 233s |
| upgrade T2 → T3 | 4000 | +8 | 500.0 | 333s |
| upgrade T3 → overclocked | 8000 | +10 | 800.0 | 533s |
| new fabricatorT1 | 2200 | +2 | 1100.0 | 733s |

A new T1 extractor is **four times** more credit-efficient than upgrading one and **twelve and a half times** more efficient than a fabricator.

This reading is not imposed on the data. It is how the game's own authors annotated the files: `#price per credit: $87` in `extractor.ini`, `#price per credit: $800 (+10)` beside the overclocked upgrade, `#price per credit: $1100` in `fabricatorT1.ini`.[^3] Those are exactly the marginal figures above.

So the rule is: **take every free pool before upgrading anything, and never build a fabricator while a pool is free.** Upgrades and fabricators are what you buy when the map runs out — the fabricator's whole value is that it needs no pool.

One exception that does not belong in the table: the reinforced T3 buys *no* income (20 → 20) for 3,000 credits. It buys hit points, 2,000 → 4,700. That is a survivability purchase.

A community analysis reaches the same conclusion by a different route: *"At the beginning, where most resource pools are still free, it is smarter to build more T1 extractors instead of fewer T2 or T3 extractors."*[^8] Treat that as corroboration of the **method**, not of the numbers — the same guide tabulates "credit factories" at 25,000/47,500/70,000 credits, and no such building exists in this build. Only extractors and fabricators carry `generation_resources`, and nothing is priced anywhere near that.[^3] It is describing a mod or much older content.

Both that guide and this page do share one real blind spot: neither prices the builder's walk to a distant pool, which an in-base upgrade does not pay.

## What the opponents were doing

The shipped AI's own configuration, from `extractor_common.ini`:[^3]

```
buildPriority: 0.4
recommendedInEachBaseNum: 2
maxEachBase: 99
```

**`maxEachBase: 99`.** The opponents are not aiming at a number; they take what they can reach ([[ai-opponent-strategy]]). A bot capped at three was not playing a different economic strategy — it was playing none.

And they are doing it on the same income we get. The difficulty enum runs `-2` Very Easy through `3` Impossible, and `.game/preferences.ini` sets `aiDifficulty:0` — **Medium**.[^9] The room's `incomeMultiplier` defaults to 1.0.[^10] Community lore holds that Hard and above give the AI an income boost; no such table was found anywhere in the decompiled tree, and at difficulty 0 it does not apply to us regardless. There is no handicap to blame: every run this bot has lost was lost on equal income against the baseline difficulty.

## Where the decision lives

`expand_economy` is pure and answers one question: should another pool be claimed, and which. It does not re-derive what a good pool is — `survey_pools` already rejects occupied pools, pools on another land mass, and pools whose approach runs through hostile fire, and ranks the survivors by distance from the base so the economy grows outward ([[policy-threat]]).[^5]

Three gates, cheapest first:

1. **An order in flight blocks another.** There is one builder and it walks to one pool. A builder in transit is an order still being carried out, so expansion asks the world — is it moving, is an extractor going up — rather than counting samples ([[policy-loop]]). Without this the fight loop re-tasks the builder every sample and it never arrives, the same defect that produced 743 attack orders against 24 targets before commitment fixed it ([[policy-combat]]).
2. **A reserve protects the army.** Expansion spends the credits reinforcement needs. The caller sets the reserve, because what it costs to replace a loss is the caller's business.
3. **The pool must be worth having**, per the survey above.

There is deliberately **no cap** on how many pools to take. The map's pool count, the reserve and the threat filter bound this already, and a number written here would be a guess overriding three measurements.

A lost builder used to end the economy permanently, so the fight loop now asks the Command Center for another when nothing owned can place an extractor. It is requested *last* in the preference order, which is what keeps the factories on tanks: a producer takes the first type it can make, and only the Command Center — which cannot make a tank — falls through to a builder.[^6]

## What the first live run showed, and what it did not

extractors **3 → 9**, ten expansion orders, six finished with a seventh going up at the budget.[^7] Income roughly doubled, 63 → 135 credits/s including the Command Center.

It does **not** show the bot wins more, or even fields a bigger army. It is one unseeded run, and the honest comparison is the three committed lockstep runs with the same combat code and the old fixed economy: army 4 → 14, 4 → 6, 4 → 3. This run's 4 → 10 sits inside a range that already spanned 3 to 14 with no economy change at all. The opponents' unit mix is weighted-random, so no two unseeded runs are the same experiment.

The claim stops at: **the economy now grows, and each pool is worth a measured 12 credits/s.** An outcome claim needs the seeded A/B — three seeds, expansion on and off.

## The bottleneck moved

The build phase ended holding 6,975 credits and the run banked more. Nine extractors against a single Land Factory means credits arrive faster than one producer can spend them: a 350-credit tank is 2.6 seconds of income, and a factory cannot start one every 2.6 seconds.

Income is no longer the binding constraint — production throughput is. Fixing the economy moved the bottleneck rather than removing it.

## The economy was switched off for most of every match, and nothing could say so

Three findings in sequence, and the order matters because the first is what made the other two visible at all.

### First, the bot's own reasoning was being thrown away

`Budget.claim` has always recorded what each request wanted and why it was refused — *"expand:extractorT1 wanted 700 of 305 available past a 0 reserve"* — and `format_ledger` has always rendered it. Neither was ever called outside its own tests. The loop reduced a whole tick of that to `sum(1 for claim in ledger if not claim["granted"])`: at roughly one refusal per sample across four thousand samples, about four thousand sentences nobody kept.[^11]

Two records now survive to the report. **What was asked for**, totalled by purpose, and **which spender was even reached**. The second is the one that matters, because "declined three thousand times" and "never asked once" were previously the same number: zero.

### Second: one busy worker switched off every spender

`Expander.step` opened with a gate that returned nothing at all when the opening plan held a worker. The plan holds **one**. The bot runs four to eight. Instrumented over 800 samples with six workers alive:

    reach  plan-holds-worker   reached  572  acted  0
    reach  no-free-worker      reached   69  acted  0
    reach  income              reached  159  acted  7

**572 of 800 samples — 71.5% of the match — the economy was not declining to spend. It was never asked.** The gate was written when there genuinely was one builder and silently outlived that. It also fires while the plan is merely *waiting to afford* something, so a plan parked at "extractorT1 costs 700, holding 130" switches the economy off for the rest of the match.

The gate existed for a real defect — two spenders each ordering the same builder, the engine running whichever waypoint arrived last, so neither order arrived ([[policy-loop]]). The fix keeps that guarantee and drops only the plan's own worker, which meant a `"wait"` decision had to start naming the unit it was holding.

Same seed, same 800 samples, before and after:

| | before | after |
|---|---|---|
| expander skipped, plan held a worker | **572** | 269 |
| every worker genuinely busy | 69 | **434** |
| income reached / **acted** | 159 / **7** | 97 / **32** |
| expansion orders | 4 | **21** |
| workers | 6 | **8** |

`no-free-worker` rising is the healthy state: the workforce is working rather than barred.

### Third: with the workers freed, they all walked to the same pool

The unblocked economy immediately showed a second defect it had been hiding. A pool is judged occupied by **what stands on it**, so a pool a builder is walking toward still reads as free. With one worker in flight at a time that was nearly harmless. With six:

    spend  expand:extractorT1   asked  32  got  20  spent  14000
    extractors: peak 4  end 4  gains 4  drops 0
    losses by type: none

**Twenty-three granted extractor orders, nothing lost all match, four extractors standing.** Nineteen orders never became anything. The credits were not burnt — a granted claim is intent, and the engine simply built one structure — but every duplicate cost a worker its travel time.

This is the shape [[policy-holding-ground]] recorded as *"275 expansion orders against a single surviving extractor"* and could not explain, because expansion had never run freely enough for the ratio to be observable. The workforce had recorded every assigned site all along; `survey_pools` was never asking. A pool under orders now counts as occupied.

**Outcome not yet established.** All three are mechanism findings, verified by instrumentation rather than by a scoreboard. Whether they win matches is a separate measurement, and the figures to judge on are wins and extractor drops — not expansion orders, since *fewer* orders is the point of the third fix and a naive reading would call that a regression.

[^1]: `src/rw_bot/policy/production.py` — `sustain` skips any option the engine reports as `placed`; `scripts/play.py` `reinforcements` skips any type whose placement rule sets `needs_pool`.
[^2]: `wiki/sources/m19-income/measured-rates.txt`, with all 1,000 readings archived as `income-windows.ndjson`. Produced by `make income`, which builds one extractor per stage and idles 200 samples between them.
[^3]: `.game/assets/units/extractor/extractor.ini`, `extractorT2.ini`, `extractorT3.ini`, `extractorT3_overclocked.ini`, `extractorT3_reinforced.ini`, `.game/assets/units/fabricator/fabricatorT1.ini`, and `extractor_common.ini` for the `[ai]` block. Shipped as plain text with the game.
[^4]: `.decompiled/com/corrodinggames/rts/game/units/custom/ag.java:1500-1506` — `generation_delay` defaults to 40 and the rate is scaled by `40.0f / generation_delay`. The 60-tick simulation is inferred from the measured ×1.5 rather than read directly.
[^5]: `src/rw_bot/policy/economy.py` — `expand_economy`, which calls `survey_pools` from `src/rw_bot/policy/build_order.py` rather than re-implementing pool selection.
[^6]: `src/rw_bot/policy/campaign.py` — `fight` appends `BUILDER_TYPE` to the wanted list only when `find_builder` returns None.
[^7]: `wiki/sources/m19-income/economy-run.txt`, from `runs/econ1.planner`.
[^8]: Steam Community guide "A scientific approach to the economy system (kind of)" by Ionics, posted 2020-05-14, https://steamcommunity.com/sharedfiles/filedetails/?id=2095871975 — game version unstated, and its unit table does not match this build (see above). Carried only because its extractor conclusion agrees with the INI arithmetic, and explicitly NOT relied on for any number. Full notes in `wiki/sources/m19-income/community-research.txt`.
[^9]: `.game/assets/translations/Strings.properties` lines 55-61 for the enum; `.game/preferences.ini` line 2 for the setting in force.
[^11]: `runs/sweeps/diag/duel-s12345.txt` and `wiki/sources/m28-holding/diag-post-worker-fix.ndjson` — one 800-sample match before and after, chosen over a full batch because these are mechanism figures a single run settles.
[^10]: `.decompiled/com/corrodinggames/rts/gameFramework/j/ah.java` — the room settings object, whose `b()` dumps `startingCredits`, `fogMode`, `aiDifficulty`, `startingUnits`, `incomeMultiplier` and `randomSeed`; `incomeMultiplier` is field `h`, initialised to `1.0f`.
