---
title: "What a Credit Buys: The Unit Value Table, and the Arithmetic Error That Nearly Rewrote It"
tags: [mechanics, catalogue, combat, measurement, verification]
related:
  - "[[mechanics-unit-catalogue]]"
  - "[[mechanics-combat-profile]]"
  - "[[mechanics-build-tree]]"
  - "[[policy-production]]"
  - "[[policy-combat]]"
source_paths:
  - "wiki/sources/m0-probe/printunits.log"
  - "wiki/sources/m11-pools/type-flags.ndjson"
  - "src/rw_bot/mechanics/catalogue.py"
  - "src/rw_bot/mechanics/combat_profile.py"
  - "src/rw_bot/mechanics/upgrades.py"
source_git_blobs:
  "wiki/sources/m0-probe/printunits.log": "d5ef7237bc6cc175a0e75014b1dafd521806a1e0"
  "wiki/sources/m11-pools/type-flags.ndjson": "f1d519832e75306a2497669e479224b26e731f3a"
  "src/rw_bot/mechanics/catalogue.py": "9a079803d1cde17f271a3106c0e9449c0a91a562"
  "src/rw_bot/mechanics/combat_profile.py": "4de76d4850ace0fd0cdcb25c121ff7ad7a299698"
  "src/rw_bot/mechanics/upgrades.py": "cb442808ca3d6d3a20203e4ad4b6b87ec06c3213"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [engine-internals, bot-architecture]
---

# What a Credit Buys

Two dumps answer every question about what a unit is worth, and both come from the same registry pass so they cannot drift against different builds: `printunits.log` carries prices, hit points and weapons, and `type-flags.ndjson` carries reach and which movement layers a weapon can touch.[^1]

Joined, they say what the bot's army is actually buying.

## The table

Everything reachable from a Command Center that has a weapon, ranked by damage per second per 100 credits. `dps = volley_damage / shoot_delay × 60`.

| type | price | hp | range | dps | dps/100c | hp/100c | hits air | built by |
|---|---|---|---|---|---|---|---|---|
| c_turret_t1 | 500 | 700 | 165 | 82.0 | **16.40** | **140** | · | builder |
| c_turret_t2_gun | 1500 | 1100 | 185 | 132.0 | 8.80 | 73 | · | c_turret_t1 |
| c_antiAirTurret | 600 | 800 | 250 | 45.0 | 7.50 | 133 | **Y** | builder |
| c_interceptor | 600 | 250 | 170 | 37.5 | 6.25 | 42 | **Y** | airFactory |
| **c_tank** | 350 | 210 | **130** | 20.0 | 5.71 | 60 | · | landFactory |
| c_antiAirTurretT2 | 1800 | 1400 | 320 | 102.9 | 5.71 | 78 | **Y** | c_antiAirTurret |
| c_laserTank | 1600 | 400 | 190 | 70.0 | 4.38 | 25 | Y | experimentalDropship |
| c_mammothTank | 3900 | 2600 | 190 | 156.0 | 4.00 | 67 | · | experimentalDropship |
| hoverTank | 450 | 150 | 140 | 15.3 | 3.41 | 33 | Y | landFactory |
| scout | 700 | 350 | 110 | 20.4 | 2.91 | 50 | Y | commandCenter |
| c_turret_t1_lightning | 2700 | 1500 | 210 | 75.0 | 2.78 | 56 | · | c_turret_t1 |
| c_helicopter | 700 | 150 | 130 | 17.0 | 2.43 | 21 | Y | airFactory |
| **c_artillery** | 900 | 140 | **290** | 20.2 | 2.25 | **16** | · | landFactory |

## The arithmetic error, recorded because it was nearly acted on

`Weapon` carries damage twice: `direct_damage` is per shot and `direct_damage_volley` is per full volley. The engine prints a separate volley total **only when it differs**, and the decoder copies the per-shot figure across when it does not.[^2] So for every single-barrel unit in the game the two fields hold the same number.

A derived table multiplied them. That squares the damage, and squaring is not a monotone transform of a *ratio* — it reorders any two units whose damage and firing rate differ in opposite directions. Under it `c_artillery` scored 2.96 against `c_tank`'s 2.38 and read as the better buy; corrected, it is 2.25 against 5.71 and is less than half as good. The conclusion drawn from the bad table — "the tank is the worst thing we can build" — was exactly backwards: **the tank is the best thing the land factory makes.**

The check that caught it was printing the raw fields instead of the derived ones, and noticing `direct_damage == direct_damage_volley` in all seven units sampled. The corrected figures reproduce the numbers already recorded from an independent earlier pass (`c_turret_t1` 16.40, `c_tank` 5.71), which is what makes them trustworthy now.

## A conversion is not priced at what the result costs to build

The extractor line is the one place where a credit buys income rather than force, and it carries **two prices per step that differ by up to 50%**. The engine prints both, side by side, and reading the wrong one is a bug the type system cannot catch because both are ints called "price".[^5]

| type | credits/s | maxHp | build price | cost to convert *into* it |
|---|---|---|---|---|
| extractorT1 | 8 | 800 | 700 | — (placed on a pool) |
| extractorT2 | 12 | 1,000 | 2,100 | **1,400** |
| extractorT3 | 20 | 2,000 | 6,100 | **4,000** |
| extractorT3_overclocked | **30** | 1,100 | 14,100 | 8,000 |
| extractorT3_reinforced | 20 | **4,700** (+800 shield) | 10,100 | 3,000 |

Nothing in this game can *build* a tier two — the builder places tier ones and nothing above.[^6] So `extractorT2`'s 2,100 is a price for a transaction that never happens, and the only figure that is ever charged is the 1,400 conversion. The spending policy claimed the 2,100, over-reserving 700 credits on every upgrade and refusing the purchase outright on any tick where the balance sat between the two. That band is not rare: this budget already refuses 1,185–1,685 claims a match.[^7]

**The fix reads the price off the holder, not the target**, and the position rather than the label. The first entry of a unit's `upgrade_prices` is the cost of its own next conversion — 1,400 on the tier one, 4,000 on the tier two, 8,000 on the tier three — each matching the corresponding `convertTo` action in the asset. The *labels* cannot be trusted for this: the dump prints a tier three's overclock cost under `T2 Upgrade Price`, and a tier two carries both `T2 Upgrade Price: $4000` and `T3 Upgrade Price: $4000` for what is a single declared action.

**The line forks, and the fork is not a ranking.** `extractorT3.ini` declares two conversions off the tier three and neither leads to the other; both carry only an `action_refund` back down. Overclocking buys 50% more income and *reduces* hit points below the tier three it came from; reinforcing holds income flat and more than doubles survivability. On a map where the opponents finish holding 44 of the 46 pools, which of those is worth more is an open question rather than an obvious one — so `next_tier` walks only as far as the two paths agree and returns nothing at the fork.

This was originally modelled as one five-long chain, which asserted that an overclocked extractor was an upgrade of a reinforced one. It is false in both directions.

## What the table says about the bot's army

**The land factory is a narrow shop.** It makes five things: `builder`, `c_tank`, `c_artillery`, `hoverTank`, `scout`. Of those only `c_tank` is competitively priced. Everything above it in the table is built by a **builder**, not a factory — which means the bot's spending ceiling is set by how many builders it has, not by how many factories.

**Nothing the bot builds can shoot at aircraft.** `c_tank` has `hits_air = false`. Across three 1500-sample matches roughly 15 of ~99 visible enemies were unreachable for exactly this reason.[^3] The two cheap answers are `c_antiAirTurret` (600, static, builder-built, better value than a tank) and `c_interceptor` (600, mobile, needs a 1,000-credit airFactory first).

**`c_tank` has the shortest reach of anything worth fielding.** At 130 it is out-ranged by every static defence in the game:

| enemy defence | range | overmatch vs c_tank |
|---|---|---|
| c_turret_t1 | 165 | 1.27× |
| c_turret_t2_gun | 185 | 1.42× |
| c_turret_t1_lightning | 210 | 1.62× |
| c_antiAirTurret | 250 | 1.92× |
| c_turret_t1_artillery | 350 | 2.69× |
| c_turret_t2_artillery | 460 | 3.54× |

A tank army attacking a defended position absorbs unanswered fire for the whole approach. That is the loss table's finding stated in units instead of corpses: 96% of losses occur more than 2,000 world units from home and none at all within 900 of it.[^3]

**`c_artillery` is the only reach the bot has.** Its 290 out-ranges every turret except the two artillery ones. It is also the most fragile unit in the game at 16 hp per 100 credits, which is why fielding it *alone* lost decisively — an all-artillery army has nothing to absorb a charge. Screening it is a composition question, and until the production policy learned to hold a ratio there was no way to ask for one.[^4]

[^1]: `wiki/sources/m11-pools/type-flags.ndjson` — one pass over the live type registry emitting `unittype`, `unitcombat` and `buildedge` records. See [[mechanics-combat-profile]].
[^2]: `src/rw_bot/mechanics/catalogue.py`, `_damage` — the engine writes either `12` or `12 (total:24.0)`.
[^3]: `wiki/sources/m21-losses/where-units-die.txt`.
[^4]: See [[policy-production]].
[^5]: `.game/assets/units/extractor/` — `extractor.ini`, `extractorT2.ini`, `extractorT3.ini`, `extractorT3_overclocked.ini`, `extractorT3_reinforced.ini`. Each `[action_*]` block carries `convertTo` and its own `price`; `[core]` carries the build `price` and `generation_resources: credits=N`. The same figures appear in `wiki/sources/m0-probe/printunits.log` as `Price:` and `T<n> Upgrade Price:`.
[^6]: `wiki/sources/m26-upgrades/structure-offers.txt` — the capture has the builder offering `extractorT1` and nothing above it. See [[mechanics-build-tree]].
[^7]: See [[policy-holding-ground]] for the refusal counts and the full-length scorecards.
