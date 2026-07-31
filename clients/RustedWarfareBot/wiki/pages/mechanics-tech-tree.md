---
title: "The Tech Tree, Priced — Every Builder-Reachable Unit"
tags: [mechanics, tech, units, upgrades, registry, measured]
related:
  - "[[mechanics-unit-value]]"
  - "[[mechanics-combat-profile]]"
  - "[[policy-holding-ground]]"
  - "[[community-play-strategies]]"
source_paths:
  - "wiki/sources/m11-pools/type-flags.ndjson"
  - "wiki/sources/m0-probe/printunits.log"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-31"
confidence: high
hubs: [game-mechanics]
---

# The Tech Tree, Priced — Every Builder-Reachable Unit

The registry's own answer to "which units, when to upgrade, which path" —
mined from the archived dumps, not from guides, after the guide corpus went
three-for-five refuted ([[community-play-strategies]]). Every figure below is
the engine's.

## Producers the Builder can place directly

| producer | cost | makes |
|---|---|---|
| landFactory | 700 | builder 500, c_tank 350, hoverTank 450, c_artillery 900, scout 700 |
| mechFactory | 1,000 | builder 500, mechGun 600, mechMissile 900, mechArtillery 1,400, mechBunker 4,500 |
| airFactory | 1,000 | lightGunship 250, c_interceptor 600, c_helicopter 700 |
| seaFactory | 1,000 | gunBoat 300, lightSub 450, builderShip 500, hovercraft 600, attackSubmarine 800, missileShip 900, battleShip 1,500 |
| experimentalLandFactory | 11,000 | fireBee 12,000, c_experimentalTank 14,000, experimentalHoverTank 21,000, experimentalDropship 30,000, experimentalSpider 70,000 |

## Turret upgrade chains (the defence the bot has never bought)

The ground turret and AA turret each upgrade **in place** — the same
mechanism the extractor walk uses, needing no builder trip:

| from | to | cost | reach | hp | hits air |
|---|---|---|---|---|---|
| c_turret_t1 (500) | c_turret_t2_gun | 1,500 | 185 | 1,100 | no |
| c_turret_t1 | c_turret_t2_flame | 1,200 | 155 | 1,600 | no |
| c_turret_t1 | c_turret_t1_lightning | 2,700 | 210 | 1,500 | no |
| c_turret_t1 | **c_turret_t1_artillery** | 2,100 | **350** | 1,000 | no |
| c_antiAirTurret (600) | c_antiAirTurretT2 | 1,800 | 320 | 1,400 | air |
| c_antiAirTurret | antiAirTurretFlak | 4,600 | 200 | 2,200 | air |

**The standout is the artillery turret: 350 reach**, longer than every mobile
unit in the game and second only to the enemy's own T2 artillery turret
(460). A turret the bot already builds, one upgrade from outranging every
raider that has ever killed an extractor. Never bought — `expand_defence`
knows one type ([[policy-holding-ground]]).

## Notable units the bot has never fielded

- **mechMissile (900, mechFactory)**: 190 reach, hits air, 500 hp at 0.8
  speed — more reach, triple the hp, and anti-air where the current
  anti-air answer (hoverTank: 140 reach, 150 hp) is fragile and short. The
  mech-family speed refutation (log 2026-07-28) was about `mechGun` as the
  *core* composition; a missile *component* is untested.
- **c_artillery (900, landFactory)**: 290 reach at 0.9 speed, 140 hp. The
  arty *composition* arm was refuted twice, but as siege support behind a
  tank line it is untested — the refuted arm made it a reinforcement ratio,
  not a role.
- **c_experimentalTank (14,000)**: 6,000 hp, 310 reach, hits air, 0.3 speed.
  The value-per-fight play if any economy ever supports an 25,000-credit
  entry ticket (factory + unit). The affordability guard exists for exactly
  this climb.

## What this table says about the Impossible problem

At 3.7x income the opponent out-produces any symmetric composition, so the
paths that remain are: **end it early** (the rush verb — cheap units, first
contact fastest), **hold ground at extreme range** (artillery-turret walls
cost 2,600 per emplacement and outrange everything that raids), or **win
per-fight value** (experimental tier, needing the economy the first two
protect). These are the three arms the ladder work tests, in that order.
