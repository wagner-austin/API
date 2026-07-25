---
title: "Unit Catalogue and the Mobility Predicate"
tags: [mechanics, units, economy, catalogue, planner]
related:
  - "[[engine-entity-model]]"
  - "[[issuing-orders]]"
  - "[[wire-contract-ndjson]]"
  - "[[harness-nodisplay]]"
source_paths:
  - "wiki/sources/m0-probe/printunits.log:403"
  - "wiki/sources/m0-probe/printunits.log:973"
  - "wiki/sources/m0-probe/printunits.log:976"
  - "wiki/sources/m0-probe/printunits.log:762"
  - "wiki/sources/m0-probe/printunits.log:1308"
  - "wiki/sources/m7-mobility/mobility-predicate.txt:9"
  - "wiki/sources/m7-mobility/mobility-predicate.txt:15"
  - "src/rw_bot/mechanics/catalogue.py"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: high
hubs: [game-mechanics, bot-architecture]
---

# Unit Catalogue and the Mobility Predicate

The engine prints its own unit catalogue on request and exits before the game loop ([[harness-nodisplay]]). It carries 90 units with prices, hit points, speed and weapons, and it is decoded into typed records rather than transcribed.[^6] These are the engine's numbers, not a table copied from a community wiki, and they regenerate in one command against any build.

## The join key

Each block is tagged `unit:<name>`, and that name is the same string three separate surfaces already use: the `type` field of a live entity on the world stream ([[wire-contract-ndjson]]), the argument the type registry accepts when placing a building ([[issuing-orders]]), and the key here.[^1] A planner can therefore price what it is looking at without a second mapping table.

**The join is not total, and the exception is checked rather than assumed.** Of the three types the archived world capture reports, two are in the catalogue and `editorOrBuilder` is not — it is the map editor's placeholder, parked off-map, and not a buildable unit.[^7] The test asserts that exact intersection, so a new unpriced type appearing on the stream fails rather than passing unnoticed. An earlier version of it checked two hand-picked types while calling itself "all priced", which is what let the stronger claim stand unexamined.

## Speed is the mobility predicate

38 of the 90 units have a speed of exactly zero, and the Command Center is one of them.[^2][^3]

That number is the catalogue-level explanation for a failure already recorded from the other direction. The first real order this project issued moved nothing, and the subject was the Command Center; it was diagnosed by sampling position before and after, and the conclusion was that the roster has to be legible before selection can be ([[issuing-orders]]). The catalogue is what makes it legible. `speed > 0` is a read mobility predicate, so a planner no longer has to discover immobility by ordering a building to walk.

**Tested, not merely asserted.** The same order and the same offset were issued to both roster entries on a fresh run. The Command Center, at catalogue speed 0, held position through all three samples; the Builder, at catalogue speed 0.6, moved and arrived 2.5 world units from its destination.[^8] The prediction could have failed in either direction and did not.

What that establishes is a correlation confirmed on two units, one of each kind — not a law over all 90. Nothing here rules out speed being merely correlated with some other property that decides mobility, and the 38 zero-speed units have not been individually ordered. It is enough to select on and not enough to assert as mechanism.

Keeping that predicate in the planner rather than the agent is deliberate: a mobility test embedded in the dispatch layer would be exactly the decision logic the agent must not carry ([[issuing-orders]]).

## What the numbers say

Prices span from $250 for the Light Gun Ship to $90,000 for the Modular Spider — a factor of 360, so economic decisions dominate unit choice rather than trailing it.[^6] The Builder costs $500 with 170 HP and no weapon at all; the Command Center costs $3,000 with 4,000 HP and does attack, at range 280.[^1][^3] 61 units are armed and 17 have tier upgrades.[^6]

## Two shapes that had to be modelled, not flattened

**A unit is armed only if the engine prints an attack range**, which 61 of 90 do. Damage of a kind that is not printed is zero — that is a fact about the unit, not a missing reading.

**Per-shot and per-volley damage are separate figures and neither derives from the other.** The engine writes `Direct Damage: 12 (total:24.0)` for multi-barrel weapons.[^4] The ratio is not fixed: observed values include 2x, 4x, 6x and one unit at 1.84x.[^6] A decoder that assumed a barrel count, or that kept only one of the two numbers, would be wrong for most multi-barrel units — so both are recorded.

The decoder found this rather than a survey of the keys: reading the key names alone suggested a plain number, and only running it against the real log surfaced the parenthesised total.[^6]

## Range is long relative to the map

The longest-ranged unit is the T2 artillery turret at 460 world units.[^5] Entity positions on the world stream are in the same units, so range comparisons need no conversion — the Builder at `(4250, 2610)` and the Command Center at `(4250, 2550)` are 60 apart, well inside every weapon in the catalogue ([[wire-contract-ndjson]]).

[^1]: `wiki/sources/m0-probe/printunits.log:403` — `<img src="unit:builder" />`, opening the Builder block whose `<pre>` reports `Price: $500`, `Hp: 170`, `Speed: 0.6` and no attack range.
[^2]: `wiki/sources/m0-probe/printunits.log:973` — `<img src="unit:commandCenter" />`, with `<h4>Command</h4>` following.
[^3]: `wiki/sources/m0-probe/printunits.log:976` — `Price: $3000`, with `Hp: 4000` and `Speed: 0` on the two lines after, and `Attack Range: 280` in the same block.
[^4]: `wiki/sources/m0-probe/printunits.log:762` — `Direct Damage: 12 (total:24.0)`, the multi-barrel shape; `:804` carries `Direct Damage: 65 (total:130.0)`.
[^5]: `wiki/sources/m0-probe/printunits.log:1308` — `<img src="unit:c_turret_t2_artillery" />`, whose attack range of 460 is the largest in the catalogue.
[^6]: `src/rw_bot/mechanics/catalogue.py` — the decoder and its typed records; the aggregate counts (90 units, 61 armed, 17 upgradable, 38 immobile, price range $250–$90,000, damage ratios) are produced by running it over the archived log and are asserted in `tests/test_catalogue.py`.
[^7]: `tests/test_catalogue.py` — `test_every_live_roster_type_is_checked_against_the_catalogue` decodes the archived world capture and the archived catalogue together, and asserts the unpriced set is exactly `["editorOrBuilder"]`.
[^8]: `wiki/sources/m7-mobility/mobility-predicate.txt:9` — the Command Center still at `(4250.0, 2550.0)` ten seconds after being ordered to `(4550.0, 2550.0)`; `:15` shows the Builder at `(4547.5493, 2611.7517)` against a destination of `(4550.0, 2610.0)`.
