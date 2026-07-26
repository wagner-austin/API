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
  - "wiki/sources/m13-expand/idle-after-plan.txt"
  - "agent/src/rwbot/agent/Orders.java"
  - "src/rw_bot/policy/combat.py"
  - "src/rw_bot/policy/production.py"
  - "src/rw_bot/policy/campaign.py"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-26"
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

And a new defect is now visible in the numbers. **743 attack orders were issued across 48 attacking units against only 24 distinct targets** — about fifteen re-orders per unit.[^8] Nearest-to-centre is recomputed every sample, and the centre moves whenever a unit dies or a new one rolls out, so the whole army is re-tasked on a flip that may be a few world units wide.

The engine's AI does not thrash like this: it holds a target and refreshes on an 800 ms timer rather than on change ([[ai-opponent-strategy]]). That is the same order volume spent on a stable choice instead of an unstable one, and it is the next thing to fix.

[^1]: `agent/src/rwbot/agent/Orders.java` — `attack` adds the subject through the orderable setter and then calls the entity-taking setter, which is the `av.b` waypoint mode. The six candidate setters are listed in the decompiled command class; `av.c` is the build mode already pinned by [[issuing-orders]].
[^2]: `wiki/sources/m15-production/before-after.txt` [synthesis] — the probe is described in the session log for 2026-07-26; the target's fall from 1200 to 1183 was observed live against a Scout whose catalogue `direct_damage` is 17.
[^3]: `src/rw_bot/policy/campaign.py` — the `Battle` report names the field `killed` and documents it as targets engaged and no longer visible, explicitly not a kill count.
[^4]: `src/rw_bot/policy/combat.py` — `find_army`, `find_targets`, `choose_target` and `engagements`, all pure.
[^5]: `src/rw_bot/policy/production.py` — `sustain` and `idle_producers`, also pure.
[^6]: `src/rw_bot/policy/campaign.py` — `fight` reads samples, orders reinforcements, and dispatches attacks; production runs before the army check so a wiped wave still queues its replacements.
[^7]: `wiki/sources/m15-production/before-after.txt` — the two scorecards side by side, with the counts re-derived from the run log at the foot of the file.
[^8]: `wiki/sources/m15-production/sustained-run.log` — 743 `channel: attack` lines and 49 `channel: produce` lines; the distinct target and attacker counts are the sorted-unique ids from those lines.
