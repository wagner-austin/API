---
title: "Fighting: the Attack Verb, and Keeping an Army Alive"
tags: [policy, combat, production, orders, verification]
related:
  - "[[policy-loop]]"
  - "[[ai-opponent-strategy]]"
  - "[[issuing-orders]]"
  - "[[mechanics-build-actions]]"
  - "[[mechanics-unit-catalogue]]"
source_paths:
  - "wiki/sources/m15-production/before-after.txt"
  - "wiki/sources/m15-production/sustained-run.log"
  - "wiki/sources/m15-production/target-churn.txt"
  - "wiki/sources/m15-production/committed-run.log"
  - "wiki/sources/m13-expand/idle-after-plan.txt"
  - "agent/src/rwbot/agent/Orders.java"
  - "src/rw_bot/policy/combat.py"
  - "src/rw_bot/policy/production.py"
  - "src/rw_bot/policy/campaign.py"
source_git_blobs:
  "wiki/sources/m15-production/before-after.txt": "a141155176d9ceb1631fbf3d3004244fc252261b"
  "wiki/sources/m15-production/sustained-run.log": "259d966dc1a8164be5252977c78bda56dd0a38c5"
  "wiki/sources/m15-production/target-churn.txt": "441a9b23b29cb6195a9f4c7cd0fe4075d2731de3"
  "wiki/sources/m15-production/committed-run.log": "79995a9896181a7382a4285b315b85c1670b2fec"
  "wiki/sources/m13-expand/idle-after-plan.txt": "dce97ba64dd0c0e37bb6cc4fd3c3385a9f45691f"
  "agent/src/rwbot/agent/Orders.java": "846c66b42fcf439dc5ad3534424b42d0da6d598a"
  "src/rw_bot/policy/combat.py": "8afff52d08a953cafe435d3543c337652cab3f47"
  "src/rw_bot/policy/production.py": "3ccbb9f5aec7bffa5fece236bf8a1d9684ebc110"
  "src/rw_bot/policy/campaign.py": "ae3700f5a5c413b05f2909de398d1154d8262b2f"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [engine-internals, bot-architecture]
---

# Fighting: the Attack Verb, and Keeping an Army Alive

The bot built an economy and did nothing with it — five minutes past a completed plan it had taken no damage, lost nothing, and banked 21,164 credits while the enemy doubled ([[policy-loop]]). This is what it took to make it fight, and what fighting revealed.

## Which setter attacks was not obvious

The command class carries six setters that each take a unit, and none of their names says which one attacks. The mode is `av.b`, reached through `e.a(am)`.[^1]

Two things pointed at it and a third settled it. The engine's own waypoint renderer draws `av.b` **red** and the build mode blue, which is the game's UI convention for an attack. Then a controlled probe: one Scout ordered onto an enemy Land Factory 3,600 world units away travelled for about 245 samples, and the target went from **1200 to 1183** hit points.[^2]

Seventeen damage is exactly the Scout's declared `direct_damage` in the catalogue ([[mechanics-unit-catalogue]]). That figure is what makes it proof rather than correlation — it identifies the gun, not merely the fact that something took damage.

The weaker evidence is worth naming, because it was tempting. Ordering an attack and watching the target vanish proves nothing: a unit that merely *moves* at an enemy auto-engages on arrival, and a target that retreats into fog reads identically to a dead one. That is also why the run report counts **"engaged gone"** rather than kills.[^3]

## Deciding is split from doing, again

The build policy decides what to make; the combat policy decides what to do with it; and the two share nothing but the sample they read.[^4][^5] Neither opens a socket — the campaign module runs the phases and is the only place orders are sent.[^6]

The army is the set of owned, finished, armed, mobile units that are not the editor placeholder. Each exclusion is load-bearing: an unfinished unit does not exist, an unarmed Builder sent at a tank is a Builder thrown away, a turret cannot travel to a fight, and the placeholder is not a unit ([[policy-loop]]). Armed and mobile are read from the catalogue rather than guessed from type names.

The whole army commits to one target, chosen nearest the army's **centre** rather than nearest each unit. Two tanks on one target kill it in half the time and take half the return fire; two tanks on two targets kill neither quickly.

## Reinforcement is what turned a sortie into a fight

Without it, the bot committed a fixed force and was finished when that force was: **army 4 → 0, and the phase ended at 466 samples with nothing left**.[^7]

Production now keeps idle producers busy with the same units the goals asked for. Repeating a stated goal avoids inventing a combat-worth ranking, which would be a guess with a number attached. Two constraints come free from the engine rather than being modelled here: an option reports itself unavailable at the unit cap and under tech gating, because the agent asks the engine's own predicate ([[mechanics-build-actions]]). Credits are budgeted across the whole batch, since two factories that can each afford a tank cannot always afford two.

Measured on the same map and opening: **army 4 → 2, the phase ran its full 1,500-sample budget instead of ending in defeat, and 45 reinforcements were produced.** Engaged targets gone doubled, 7 to 14.[^7]

## It survives and is still losing

Reinforcement bought survival, not parity. Losses still outrun replacements — 4 → 2 — and the opponents went from 47 to 142 visible units over the same window.[^7]

## Churn, and the commitment that fixed it

Reinforcement exposed a defect that was invisible while the army died early: **743 attack orders across 48 attacking units against only 24 distinct targets** — about fifteen re-orders each.[^8] Nearest-to-centre was recomputed every sample, and the centre moves whenever a unit dies or a new one rolls out, so the whole army was re-tasked on a flip that could be a few world units wide.

The cause was architectural rather than tactical. The combat policy was a pure function of one sample with **no memory**, so it never committed to anything — while the engine's AI holds a target in a group object and refreshes on a timer ([[ai-opponent-strategy]]). Purity was not the problem; conflating *pure* with *stateless* was. The prior choice is now an argument, exactly as the build loop already passes its own progress in, and the function is still a value in and a value out.[^4]

Holding a target until it is gone cuts re-orders per unit **from 15.5 to between 2.2 and 5.9**.[^9]

## What two runs cannot tell you

The outcome did not follow the churn, and the two committed runs are why: identical code gave **army 4 → 0 at 1,013 samples** and **army 4 → 7 over the full 1,500** — the worst and the best results the bot has produced.[^9]

Outcome variance between runs is therefore larger than anything measured here. The opponents' unit mix is weighted-random by construction ([[ai-opponent-strategy]]), so no two runs are the same experiment, and two of them cannot separate a regression from noise. The churn reduction is safe to claim because it is a direct consequence of the code change and holds in both runs; nothing here says commitment helped or hurt the fight.

## The attrition, once there was a figure for it

Every scorecard until now reported the strongest rival's worth at the first and last observation only, and that pair cannot answer the question the whole fight exists to settle. An opponent that lost half its army and rebuilt reads identically at the last observation to one that was never touched.

Recording the largest fall from a running peak answered it in one run. Uncapped workers, tank goals, 1,500 samples, seed 12345:

    reinforced     63 produce orders
    army           0 -> 20
    best rival     4,700 -> 27,850  (peak 27,850, worst dip 700)

Sixty-three tanks ordered plus the four the opening plan builds, twenty alive at the end: about forty-five tanks lost, roughly 16,000 credits, against a 700-credit dent in the leader.

**That reading was an artefact of the measurement, and the correction is the finding.** The dip was taken against `_best_rival`, which is a `max` over hostile players — so it only ever answered "was the *leader* set back". Grinding one opponent down while another builds leaves the maximum tracking the second, and damage to anybody else is invisible to it. Every run measured that way reported roughly the same 700, which should have been suspicious on its own: an identical figure across arms that differ is usually a constant, not a measurement.

Measured **per opponent, against that opponent's own running peak**, the same strategy shows dips of **1,600 to 8,150** across nine matches — one of them 8,150 credits off a single player.[^11] The army was destroying things the whole time. What it never does is damage whoever is winning.

So the constraint is not that the bot cannot fight. It is that it cannot *finish*: nobody has been eliminated in any match yet, and the bot is ahead in all of them — our worth grows 11.6× from the opening position against the strongest rival's 6.9×.

Army size is not the lever either. A run that capped workers at four fielded **thirty-four** tanks rather than twenty and made no more impression than one fielding twenty.

## How much to mass before committing

`WAVE_SIZES` tops out at seven, and that number is the shipped AI's — copied from an opponent playing a different economy. Seven `c_tank` is 2,450 credits walking into turrets that out-range them by 1.27× to 3.54×.[^2]

The final rung is now an argument, so the question could be asked of a run. Three seeds against each of three ladders, everything else fixed:[^11]

| ladder | army | worth | rival | ratio | worst dip | attack orders |
|---|---|---|---|---|---|---|
| mass 7 (shipped) | 21.0 | 38,683 | 33,150 | 1.17 | 3,300 | 459 |
| mass 15 | 22.3 | 38,517 | 31,067 | 1.24 | 1,700 | 207 |
| mass 25 | 27.0 | 40,717 | 32,300 | 1.26 | 4,217 | 224 |

Massing more helps, mildly and consistently. The army survives larger and order churn halves, which is one fact stated twice: fewer, bigger commitments rather than a trickle. The worth ratio moves 1.17 → 1.26, which is inside the run-to-run spread and is not on its own a result.

Only the final rung moved. The early rungs govern the opening, when holding three units back is the difference between a first attack and none at all, and an experiment that moved both could not say which end mattered. Income, extractor count and worker count are flat across all nine matches, so the arms are isolated and what differs is how units were committed rather than how many were built.

## What a group id cannot be used for

Combat never checks whether a target can be walked to, and the obvious fix does not work. Entities carry a `group` — the connected component of their own movement layer — and the pool policy already reduces reachability to comparing two of them.[^10]

Applying the same comparison to targets fails on the data. A live capture:

| roster | flying | group | count |
|---|---|---|---|
| hostile | no | 1 | 21 |
| hostile | no | **-3** | 24 |
| ours | no | 1 | 3 |
| ours | no | -2 / -3 | 6 |

Negative ids mean "not on a movement grid", which is what every **structure** reports. More than half the visible hostiles are buildings, and a naive `attacker.group == target.group` test would refuse all of them — including the Command Centers that have to die for anyone to be eliminated. Pools work only because a pool carries `group_land`, a component id computed for the *tile* rather than for a unit. Doing this for targets needs the agent to emit the same thing at the target's position; it is not a policy-layer change.

[^10]: `src/rw_bot/policy/build_order.py` — `_can_walk_to`. See [[mechanics-movement-layers]].
[^11]: `wiki/sources/m24-wave-mass/wave-mass-ab.txt` — nine matches played four at a time on cloned game directories with the exchange locked to the tick, so CPU contention cannot shift when a sample is taken ([[harness-nodisplay]]).

[^1]: `agent/src/rwbot/agent/Orders.java` — `attack` adds the subject through the orderable setter and then calls the entity-taking setter, which is the `av.b` waypoint mode. The six candidate setters are listed in the decompiled command class; `av.c` is the build mode already pinned by [[issuing-orders]].
[^2]: `wiki/sources/m15-production/before-after.txt` [synthesis] — the probe is described in the session log for 2026-07-26; the target's fall from 1200 to 1183 was observed live against a Scout whose catalogue `direct_damage` is 17.
[^3]: `src/rw_bot/policy/campaign.py` — the `Battle` report names the field `killed` and documents it as targets engaged and no longer visible, explicitly not a kill count.
[^4]: `src/rw_bot/policy/combat.py` — `find_army`, `find_targets`, `choose_target` and `engagements`, all pure.
[^5]: `src/rw_bot/policy/production.py` — `sustain` and `idle_producers`, also pure.
[^6]: `src/rw_bot/policy/campaign.py` — `fight` reads samples, orders reinforcements, and dispatches attacks; production runs before the army check so a wiped wave still queues its replacements.
[^7]: `wiki/sources/m15-production/before-after.txt` — the two scorecards side by side, with the counts re-derived from the run log at the foot of the file.
[^9]: `wiki/sources/m15-production/target-churn.txt` — the three runs side by side, with the counts derived from each run log and the variance caveat stated in the file itself.
[^8]: `wiki/sources/m15-production/sustained-run.log` — 743 `channel: attack` lines and 49 `channel: produce` lines; the distinct target and attacker counts are the sorted-unique ids from those lines.
