---
title: "The Raid — Remembered Income as an Objective"
tags: [policy, raid, offence, intel, attack-move]
related:
  - "[[policy-intel-and-scouting]]"
  - "[[policy-holding-ground]]"
  - "[[issuing-orders]]"
  - "[[community-play-strategies]]"
source_paths:
  - "src/rw_bot/policy/raid.py"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [bot-architecture]
---

# The Raid — Remembered Income as an Objective

A first-wave-sized party — the engine's own first-group size, below which its
AI calls a force a trickle ([[engine-ai-triggers]]) — drafted from the army by
lowest id and attack-moved at the remembered enemy extractor nearest our
anchor: the frontier one, reachable before the deep ones.[^1]

## Why it exists

Every Very Hard non-win ends the same way: our economy holds, theirs
compounds, and five to eight enemy builders rebuild whatever the waves kill
([[policy-holding-ground]]). The waves cannot reach the rebuild engine — they
attack what is visible near the army, and it stands in the fog. The raid is
the first policy that makes the opponent's economy the target, which the
community corpus treats as ordinary play and this bot had never once done
([[community-play-strategies]]).

## The rules, and why each

- **Income types only.** Raiding the army is the waves' job; raiding defences
  is what waves die to.
- **One unit, one commander.** The party is withheld from the wave
  controller; assignment is the arbitration, the AI's own zone invariant
  ([[engine-ai-zones]]).
- **Ghosts are reported.** A raider standing on the memory of an extractor
  and seeing none calls `Intel.forget` and the raid advances — without it,
  the party assaults a dead sighting until the trust window expires.
- **Attack-move, not move** — the party fights its way there, which the
  probe proved live ([[issuing-orders]]).

## V1 refuted: 0/12, and the mechanism was never the problem

The A/B — `aa-counter-guard-raid`, one field from control, twelve seeds at
Very Hard, in-batch control — came back **0 wins against the control's 5**,
far outside the noise floor's 7/5 replica split.[^2]

The traces acquitted the obvious suspect first: withholding the party doubles
the effective first-wave gate, but army growth is production-limited and both
arms' early games are identical (army of 6 at sample ~708, rival ~22,000 at
1,000 in both). The conviction is mid-game: **the party is an attrition
conveyor.** `Raider.strike` replaces each fallen member by drafting one
recruit, which attack-moves across the map *alone* — a one-unit trickle into
a fortified base, issued forever. Raid arms reinforced as much as control and
ended with half the army value, kills no higher; the drain weakened
interception, extractors died, income halved, and the opponent snowballed to
57,000–75,600 worth where control held it near 24,000. Seed 777 sharpens it:
72 kills, double the control's, and the economy race still lost — kills that
do not protect income are not progress.

Every mechanical part worked live — the fog memory, the attack-move, the
ghost confirmation. What failed is arbitration: nothing asked whether the
army could *spare* a party.

## V2, designed by the refutation — and measured cost-neutral

Three rules: **a party or nothing** — survivors below strength disband and
attack-move home, and only a full party is ever drafted; **drafted whole,
from the gathered** — recruits come from inside the rally radius of the
anchor, so a party starts together the way a wave does; and **surplus only**
— the draft is gated on the army exceeding the current wave rung's need plus
the party size (`WaveController.need()`), judged in the campaign where the
withholding already lives. A `marches` report line rides along, because the
`raids` count (2–6) hid a conveyor of dozens of re-drafts.

The re-measure, same twelve seeds and an identical doctrine file: **3/12
against control's 3/12, drops 30 against 33** — dead even on every figure
that convicted v1, with `marches = raids × 3` in all twelve matches. The
economy stopped paying (income 38–78/s against v1's starved 46/s) and the
rival mostly stopped snowballing.[^3]

**Standing state: free, not yet decisive — and size is measured out.** The
party size became a doctrine knob (`raid N`) and the five-unit arm answered:
4/12 against 5/12, inside the floor, with three matches never raiding at all
because a heavier party's surplus gate (`need + 5`) sometimes never opens at
Very Hard.[^4] A bigger party raids more rarely and still does not convert.
The open knobs that remain: the objective set (enemy *builders* are the
rebuild engine the raid was conceived against; extractors are only its
fuel), and timing.

[^1]: `src/rw_bot/policy/raid.py` — `Raider.strike`, `income_objectives`, `_confirmed_dead`; `tests/test_policy_raid.py` pins each rule.
[^2]: `runs/sweeps/all-arms-veryhard`, `runs/traces/all-arms-veryhard`; log entry "raid v1 refuted at 0/12", 2026-07-29.
[^3]: `runs/sweeps/raid2-ab-veryhard`; log entry "raid v2: from 0/12 to cost-neutral", 2026-07-30.
[^4]: `runs/sweeps/cap-raid5-veryhard`; log entry "cap refuted with its mechanism attached; raid5 says size is not the bite", 2026-07-30.
