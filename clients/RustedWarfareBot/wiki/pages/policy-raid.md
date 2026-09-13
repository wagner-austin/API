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
  - "src/rw_bot/policy/party.py"
  - "src/rw_bot/policy/hunt.py"
  - "src/rw_bot/policy/dive.py"
source_git_blobs:
  "src/rw_bot/policy/raid.py": "6bf7705d0a39f09a5e36b3cb5f753f843d5c11e7"
  "src/rw_bot/policy/party.py": "66f07f551bbb437600a967a1fe1699d0df1a328f"
  "src/rw_bot/policy/hunt.py": "e00e6a8ab543a94899205b49795d4f2174fb2ed7"
  "src/rw_bot/policy/dive.py": "34d7708d42fdf25125419f04f77800038f64fa32"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-09-13
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
anchor, so a party starts together the way a wave does (the draft and the
homeward walk were lifted to `policy/party.py` on 2026-09-05 when the hunt
adopted the same discipline; the rules here are unchanged, shared rather
than copied); and **surplus only**
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

## Three holders, one bookkeeping (2026-09-13)

The hunt was the second holder of the party rules and copied the raid's
between-observation state -- who is in the party, which objective, what
has been ordered -- rather than sharing it; the dive would have been the
third copy. `Detachment` (`party.py`) now carries that state for all three:
`survivors`, `disband` (home fighting, everything forgotten), `muster`,
`advance` (one objective counts once and re-sends every member; a standing
objective sends nothing to a member already on it) and `forget_objective`
(the raid's confirmation path). `Raider`, `Hunter` and `Diver` subclass it
and own only their quarry and their draft; the report's `raids`, `hunts`
and `dives` read the shared `objectives` counter. The refactor is priced
the way this wiki prices every refactor: byte-identical cards against the
control on seeds where the verb fired (`dive16`'s three champion members,
the raid-heaviest seeds of pinbase48).[^5]

## The dive: the gun, not the income (2026-09-13)

The loss-table anatomy of pinbase48 (`wiki/log.md`, 2026-09-13) reads every
Very Hard opener seed as the same shape: three to twelve `c_artillery`
behind a screen of scouts and hover tanks, the army stalling at eleven to
thirteen pieces because the kill-groups take the nearest target -- the
screen -- and trade into the 290-reach guns behind it, while the enemy's
income compounds past 76 by sample 1,500. No winning seed faces a single
piece. Every composition answer measured flat or traded evenly across four
trigger generations. The dive is the tactical answer the community corpus
plays against a battery: a party of the FASTEST gathered units
(`draft_fastest`, the hover slot's leg) closes on the visible hostile ground
mover whose land gun outranges every land gun the army fields -- the
outranged clause's own membership test, lifted as `counter.outranges` and
`counter.land_reach` rather than restated -- nearest the party's own
centre. No memory fallback and no lesser objective: with no outranging gun
in sight a standing party goes home and none is raised, which is what makes
a zero-artillery seed the champion's match bit for bit -- the identity
property the static merge could not have and the conditional join could
only approximate. Arbitrated like the hunt, against the opening rung,
because the battery stands at frames 60k-110k when the army holds eight to
thirteen pieces. `dive N` is the doctrine field; `D` is its trace code.[^6]

[^1]: `src/rw_bot/policy/raid.py` — `Raider.strike`, `income_objectives`, `_confirmed_dead`; `tests/test_policy_raid.py` pins each rule.
[^5]: `src/rw_bot/policy/party.py` — `Detachment`, `draft_fastest`; `tests/test_policy_party.py` pins the shared contract, `tests/test_policy_hunt.py` and `tests/test_policy_raid.py` pass unchanged on the rewritten holders.
[^6]: `src/rw_bot/policy/dive.py` — `Diver.dive`, `outranging_guns`; `tests/test_policy_dive.py` pins the quarry, the draft and the stand-down; `tests/test_campaign_dive.py` pins the loop identity (a tank in sight raises no party, order for order).
[^2]: `runs/sweeps/all-arms-veryhard`, `runs/traces/all-arms-veryhard`; `wiki/log.md:930`, "raid v1 refuted at 0/12", 2026-07-29.
[^3]: `runs/sweeps/raid2-ab-veryhard`; `wiki/log.md:986`, "raid v2: from 0/12 to cost-neutral", 2026-07-30.
[^4]: `runs/sweeps/cap-raid5-veryhard`; `wiki/log.md:1020`, "cap refuted with its mechanism attached; raid5 says size is not the bite", 2026-07-30.
