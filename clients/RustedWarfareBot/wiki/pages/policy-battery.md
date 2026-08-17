---
title: "The Artillery Battery: a Shore Turret That Outranges the Fleet"
tags: [policy, navy, defence, experiments, quartermaster]
related:
  - "[[mechanics-combat-profile]]"
  - "[[policy-exact-timing]]"
  - "[[policy-budget]]"
  - "[[policy-holding-ground]]"
  - "[[campaign-ledger]]"
source_paths:
  - "wiki/sources/m28-battery/probe-run.txt:18"
  - "wiki/sources/m28-battery/pilot1-card.txt:43"
  - "wiki/sources/m28-battery/pilot5-card.txt:79"
  - "wiki/sources/m28-battery/pilot7-card.txt:43"
  - "src/rw_bot/policy/battery.py"
  - "src/rw_bot/policy/quartermaster.py"
  - "src/rw_bot/policy/convert.py"
source_git_blobs:
  "src/rw_bot/policy/battery.py": "0b647333fa48f5f7694bddf0fc1dff41452bccfc"
  "src/rw_bot/policy/quartermaster.py": "2e82e5dc22e984e1096bad96dacfc2a4aaab86d0"
  "src/rw_bot/policy/convert.py": "6930c93a62cba09ebade77e6cbfa17516831ee8b"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [bot-architecture]
---

# The Artillery Battery: a Shore Turret That Outranges the Fleet

The naval hole's cheapest response yet tried: the ground turret's
artillery fork reaches 350 against the battleship's 240
([[mechanics-combat-profile]]), so one $2,100 structure on the shore-most
land shells the parked fleet from ground no ship can answer -- the attack
submarine's standoff logic (rejected at -8; [[campaign-ledger]]) without
the factory, the escort, or the recurring hull bill. A builder cannot
place the fork directly: the chain is a $500 `c_turret_t1` at the shore,
then the engine's $1,600 in-place conversion, priced by the option row.

**The mechanism is proven live.** The probe placed, converted, and drew
attributed blood -- an enemy hoverTank's `damaged_by` naming the battery
is the engine's own testimony.[^1] Pilot seven stood the fork in a full
integrated match: `c_turret_t1_artillery x1` in the owned-peak census
with both claims granted at the engine's prices.[^4]

**The economics are REJECTED at Very Hard.** battery96 (48 fresh pairs,
2026-08-15): the arm won 6 to the control's 18 -- net -12 against the +4
bar -- with the battery standing in 47 of 48 arm matches, so the panel
measured the real thing. The counter works and still loses: the $2,100
plus a builder held at the shore through construction is paid out of the
tempo that decides these games, and the arm wiped 25 times to the
control's 15 -- a uniform drag, not a trade. Law ten's third derivation
([[campaign-ledger]]): cost-in-tempo ranks responses, not cost. The
channel stays behind `battery 0`, banked for any map or rung where the
arithmetic differs.

## The five defects the pilots bought (one match each)

Every one was an interaction with the running bot, invisible to the
standalone probe, whose world held no competitor for credits, builders,
or the holder:

1. **Channel competition** -- the flame converter's $700 claim funds
   before the $1,600 fork and took the shore turret every time it stood;
   pilot one's peak carried `c_turret_t2_flame x2` and no battery.[^2]
   Fix: the site turret is spoken for (both converters take an
   `exclude`), and an overridden holder is EVICTED from a converter's
   underway count -- it no longer offers the target and never will.
2. **Unfunded rebuilds** -- a turret that dies mid-conversion is rebuilt,
   and the engine charges per attempt, so the books re-fund per attempt.
3. **Acceptance latency** -- the engine may accept a candidate after
   patience moved the walk on; the walk remembers EVERY offered point,
   and a structure within snap of any of them is its own.
4. **Spending-chain position** -- a claim at the tail of the tick binds
   nobody (the `fund_cover` rule), and the walk starved through 4,866
   refusals while the army spent first.[^3] Fix: `Battery.fund` claims
   both halves FIRST in the quartermaster's produce pass.
5. **Construction abandonment** -- the expander re-tasks the builder the
   tick the walk goes silent, and the abandoned turret dies unfinished;
   the walk now re-sends the build at the standing incomplete structure,
   winning the builder back every tick until the engine reports
   completion.

All five fixes are lifted to their twins: the `Shipyard` gained the fund
step, the construction hold and re-funded rebuilds; the `TurretLadder`
gained the eviction; the builder-pin avoidance is symmetric.

## Where it lives

The channel is `rw_bot/policy/battery.py`; it is constructed, funded and
dispatched by the quartermaster (`rw_bot/policy/quartermaster.py`), the
standing-purchases seam split from the campaign loop, whose two stated
orderings are policy: the fleet guard hires before the submarines, and
the battery's fork order sends LAST so its re-send wins a contested
holder. One battery per match by design: a fork that stood and died is a
loss the panel measures, not a rebuild.

[^1]: `wiki/sources/m28-battery/probe-run.txt:18` — "[battery] DREW
    BLOOD: hoverTank (HOVER) at 194 hp=78/150"; the probe run on
    duel_lake s424242, difficulty 3, terrain-by-attempt siting at
    fraction 0.14.
[^2]: `wiki/sources/m28-battery/pilot1-card.txt:43` — the owned peak
    carrying `c_turret_t2_flame x2` and `c_turret_t1 x1`, with the fork
    refused all match on the same card.
[^3]: `wiki/sources/m28-battery/pilot5-card.txt:79` — "battery:c_turret_t1
    asked 4867 got 1 spent 500 held: ... wanted 500 of 31 available past
    a 900 reserve".
[^4]: `wiki/sources/m28-battery/pilot7-card.txt:43` — the owned peak
    carrying `c_turret_t1_artillery x1` (seed 424253, VH, flame-battery1).
