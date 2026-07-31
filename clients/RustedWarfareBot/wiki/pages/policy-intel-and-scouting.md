---
title: "Intel and Scouting — Remembering the Fog, Carefully"
tags: [policy, scouting, intel, fog, measured]
related:
  - "[[community-play-strategies]]"
  - "[[mechanics-combat-profile]]"
  - "[[policy-production]]"
source_paths:
  - "src/rw_bot/policy/intel.py"
  - "src/rw_bot/policy/scouting.py"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-29"
confidence: high
hubs: [bot-architecture]
---

# Intel and Scouting — Remembering the Fog, Carefully

Two pieces. `Intel` remembers hostile sightings per engine identity for a
9,000-frame trust window — the opponent's wave tempo, so intel a cycle old no
longer describes their army ([[engine-ai-triggers]]).[^1] `ScoutRunner` keeps
one scout walking the pool circuit farthest-first — the far pools are the
opponent's side — kept alive by the composition exactly as the builder
is.[^2] The counter tilt reads the remembered picture, so the mix reacts to
what was *seen* rather than only to what is already shooting.

## V1 was refuted, and the refutation designed v2

Twelve seeds at Very Hard, one field from control: wins level at 3/12 and
everything else worse — rivals at 97,000–113,000 against the control's
33,000–82,000. The intel arrived and poisoned both consumers:

1. **The tilt drowned.** A scouted base is mostly buildings and boats;
   feeding everything remembered collapsed the flying share toward zero, and
   the arm finished with *less* anti-air than not scouting. V2 filters the
   threat set to **mobile units** (catalogue speed over zero).[^3]
2. **The scout starved the economy.** It dies often on the circuit, so it was
   permanently the furthest-behind share, and the Command Center — the one
   producer of builders — spent whole matches replacing scouts. One match ran
   on a single builder. V2 makes the scout **yield below two workers**, the
   expander's own floor for diverting labour.[^3]

The standing caveat this bought: the corpus's "scouting is the win condition"
is true only of what the consumers do with the intel
([[community-play-strategies]]).

## V2 measured: the catastrophes cured, the price still wrong

Twelve seeds at Very Hard against an in-batch control: **3/12 against 5/12**,
and the mechanism of the loss moved exactly where v1's fixes said it would —
out of catastrophe, into cost. No worker starvation below the floor and no
drowned tilt; but the control peaks at 7 extractors in 11 of 12 matches while
the scout arm peaks at 4–6, income medians 46/s against 62–78/s. The scout's
replacement stream still competes with the economy for Command Center
production, and at Very Hard that margin is the match.[^4]

The result worth keeping: **the scout arm won seed 555, which control lost —
the only challenger flip of a control loss in the whole five-arm batch.**
The intel pays when the economy survives buying it. V3's question is
therefore posed precisely: eyes that cost no production slot — a scout
drafted only from surplus throughput, or intel taken from combat contact
alone.

[^4]: `runs/sweeps/all-arms-veryhard`; log entry "the all-arms batch", 2026-07-29.

## Corrections flow back

The memory cannot see a kill through the fog; a caller standing where a
sighting said, seeing nothing, reports the death (`Intel.forget`) rather than
letting the ghost stand until the window expires. The raid is the first such
caller ([[policy-raid]]).

[^1]: `src/rw_bot/policy/intel.py` — `INTEL_WINDOW_FRAMES`, `Sighting` (deliberately a `counter.Threat`), identity-ordered `remembered()`.
[^2]: `src/rw_bot/policy/scouting.py` — the farthest-first route, the once-per-leg order rule, the circuit reset on replacement.
[^3]: `runs/sweeps/scout-ab-veryhard`; log entry "scouting v1 refuted", 2026-07-29. The v2 fixes: `campaign._mobile_threats` and `ScoutRunner.need`'s worker floor.
