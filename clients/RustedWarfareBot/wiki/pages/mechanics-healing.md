---
title: "Healing: Repair Bays, Combat Engineers, and the Verification Tiers"
tags: [mechanics, healing, decompile, community]
related:
  - "[[community-play-strategies]]"
  - "[[mechanics-unit-catalogue]]"
  - "[[policy-holding-ground]]"
source_paths:
  - "runs/decompiled/com/corrodinggames/rts/game/units/d/r.java"
  - "runs/legion.out"
  - "wiki/sources/m0-probe/printunits.log"
  - "wiki/sources/m17-community/namu-units.txt"
game_version: "1.15 (code 176, build #28)"
confidence: medium
hubs: [game-mechanics]
---

# Healing: what repairs units, how fast, and what is still inferred

The community meta the Impossible campaign keeps rediscovering is built on
healing — fabricator-class support behind the mass, wounded units pulled out
and returned ([[community-play-strategies]]) — and until 2026-08-01 the bot
fielded none and the wiki said nothing mechanical about it. This page is the
decompile's answer, with the verification tier of each claim stated.

## The repair bay heals automatically — verified

The repair bay's class is `units/d/r`. Three readings pin its behaviour:[^1]

- `y()` returns **230** — the repair radius, in world units.
- `c(am)` returns **0.2f** — the per-application heal amount per target.
- Its own update `a(float)` runs the repair on a timer: `this.d += f2; if
  (this.aq() && this.d > 40.0f) { this.d = 0; r.a(this, …) }` — an area
  application roughly every 40 delta-units, via a spatial query at the bay's
  own radius. **No order is involved anywhere on the path**: a damaged
  friendly unit inside 230 of a finished bay heals, full stop.

Eligibility is `a(am) = !am2.q()` — broad, with the one exclusion behind
`q()` not yet named from this read.[^1]

Consequences for policy, already in code: the creep verb lays a bay every
third structure (`rw_bot/policy/creep.py`, retained behind `creep 0` --
log 2026-08-09), the hospital arm stands one at the base
the flee reflex runs toward, and both work by proximity alone.

## The combat engineer — offer verified, auto-heal inferred

The tier-two land factory offers the combat engineer: measured live,
`produce:combatEngineer asked 181` in the legion probe's ledger.[^2] Its
catalogue entry reads 3,500 credits, 1,000 hp, armed, "Self repair with
built-in Fabricator".[^3] NamuWiki's meta page states it heals units at
roughly twice the repair bay's rate and is the mid-game support staple.[^4]

**Whether it heals automatically by proximity like the bay, or requires a
repair order, is inferred and not yet verified** — the medic channel
([[policy-budget]]) hires them into the composition on the proximity
assumption, and one live probe (a medic beside a damaged unit, hp read off
the sample series) settles it. Until that probe runs, this paragraph is the
claim's honest tier.

## Builders repair buildings only — verified by catalogue

The builder's own description: "Constructs and repairs buildings. Can not
attack."[^3] Unit healing is not in its vocabulary, which is why the medic
channel exists at all.

## Not verified in this page

The engineer auto-heal question above; the repair bay's `q()` exclusion;
whether healing draws credits (the community treats it as free, and no
ledger line has ever shown a heal charge); stacking behaviour of multiple
bays; and every healing figure for modded or naval units.

[^1]: `runs/decompiled/com/corrodinggames/rts/game/units/d/r.java` — `y()`
    at the 230 return, `c(am)` at the 0.2f return, the timer block inside
    `a(float)`, `a(am)` for eligibility.
[^2]: `runs/legion.out` — spend ledger line `produce:combatEngineer asked
    181 got 0` (log 2026-07-31).
[^3]: `wiki/sources/m0-probe/printunits.log` — the Combat Engineer and
    Builder stat blocks.
[^4]: `wiki/sources/m17-community/namu-units.txt` — the combat engineer
    paragraphs ("they can heal units", "produces a lot of combat
    engineers... excellent performance"), captured 2026-07-31.
