---
title: Enemy Bot Behavior
tags: [combat, enemies, ai]
related: [[shot-range]], [[combat-chase-bug]]
sources: [see footnotes]
fact_checked: 2026-06-16
confidence: high
---

# Enemy Bot Behavior

## Movement patterns

- **Stand ground and fight**: enemy tank bots do not move while fighting. They hold position and return fire.[^1]
- **Flee at low HP**: once a damage threshold triggers (appears to be after taking several hits), bots begin moving away from the attacker. They move **every time they are hit** after this gate triggers.[^1]
- **No resource collection**: bots never collect fuel or equipment. They fight until destroyed or flee.[^1]
- **Never fight each other**: practice bots do not engage each other. Only we make corpses.[^2]

## Chase behavior

When a bot starts fleeing:
- It moves 1 tile per server tick (~2 seconds)
- It does not collect anything while fleeing
- It continues fleeing until destroyed or the pursuer disengages
- The right response is to chase them down and finish them, or use repeated homing shots[^1]

## Implications for combat strategy

- A stationary bot is a guaranteed kill if you can maintain adjacency
- A fleeing bot should be chased with **homing shots** (track off-viewport), not teleport hops[^4]
- Never abandon a target — shields and corpses both return positive hits, so a "miss" means the target moved, not that it's unkillable[^5]
- Disengaging forfeits all damage progress — tiers repair over time[^3]

[^1]: user (Austin), 2026-06-16 — "tank bots stand ground and fight, only move when low HP, then move every time hit; don't collect fuel or equipment; just run until you chase and finish them or use homing"
[^2]: user (Austin), 2026-06-11 — practice bots never fight each other; retracted a prior theory
[^3]: run 20260611-004505 — purple-3 healed 1→0→3 after bot disengaged; see [[tank-registry]]
[^4]: user (Austin), 2026-06-16 — "the bot is able to stay still and just fire homing shots at it. the homing shots go off viewport"
[^5]: user (Austin), 2026-06-16 — "shields don't return miss. they return a positive hit. a corpse returns a positive hit"
