---
title: Enemy Bot Behavior
tags: [combat, enemies, ai]
related:
  - "[[shot-range]]"
  - "[[combat-chase-bug]]"
  - "[[tournament-strategy]]"
source_paths:
  - see footnotes
fact_checked: "2026-07-06"
confidence: high
hubs: [combat]
---

# Enemy Bot Behavior

## Movement patterns

- **Stand ground and fight**: enemy tank bots do not move while fighting. They hold position and return fire.[^1]
- **Flee at low HP**: once a damage threshold triggers (appears to be after taking several hits), bots begin moving away from the attacker. They move **every time they are hit** after this gate triggers.[^1]
- **Roam viewport-to-viewport; no deliberate fuel-seeking, but accidental pickups happen** (user, 2026-07-19: "the bots teleport or walk to the next viewport usually. they dont seek fuel, but sometimes they may teleport away and happen to land or step on a fuel tank"). Landing on a container auto-picks it, so a fleeing bot can accidentally refuel — observed: orange-2 (id 528) fled our engagement at damage_state=1 (critical, 22:29:54), teleported away (0x58 at 22:30:00), and reappeared on the next map open at damage_state=2 (medium) — a REAL recovery (sync-vs-map damage encodings agree 17/17 on overlapping observations), explained by an accidental fuel landing. Fuel is the life pool; damage tiers recover when it refills.[^7]
- **Never fight each other**: practice bots do not engage each other. Only we make corpses.[^2]

## Chase behavior

When a bot starts fleeing:
- It moves 1 tile per server tick (~2 seconds)
- It does not collect anything while fleeing
- It continues fleeing until destroyed or the pursuer disengages
- The right response is to chase them down and finish them, or use repeated homing shots[^1]

## Return fire — bots use singles, not duals

Bots fire **single shots** back at attackers, not duals. Water-map humans can extract PPH from bots at reduced risk because damage-per-return-shot is 45 fuel (single-hit victim cost, see [[game-economy]]), not the 90 a dual would land.[^6] **Wire-confirmed 2026-07-19**: orange-2 (id 528) returned fire during our engagement with 0x53 ShootEvents carrying `weapon=0` (free singles) at our tile — capture run 22:29:56/:58, exactly the predicted shape.

## Shots to force a bot off-screen

Two independent counts from Sigma's guide (2015): the raw shot count and a shade-based shortcut using the bot's `damage_state 0-3` (see [[decode-coverage]] tank cache byte [6]).[^6]

| Bot rank | Total shots to force teleport | Shade shortcut |
|---|---|---|
| Recruit | 7 | last shade + 1 more |
| Private | 8 | last shade + 2 more |
| Corporal | 9 | last shade + 2 more |

"Last shade" = the darkest of the four `damage_state` tiers (state 3). The shade shortcut lets the bot fire freely for the first 6-7 shots without counting, then switch to shot-counting only once the target reaches state 3. Our own `damage_state` decoder already exposes the value we'd need to gate on.

**Verification status:** guide-sourced (2015 human observation), not yet wire-verified in this project. The next multi-tank capture with bot targets should count `You hit N/N` events against `damage_state` transitions to confirm.

## Same-color bots respond to chat commands

Same-color bots can be directed via chat with commands like **"use the radar"** and **"move out of the way"**, so a nearby friendly bot can be used as a supplemental radar source during equipment filling.[^6] Sigma credits sean/987 for the technique. Not currently exploited by our bot; if verified, this changes the practice-room fill economics because a same-color bot acts as a free extra-radar dispenser at zero fuel cost to us.

**Verification status:** guide-sourced, not yet wire-verified. Test on Practice Map by messaging a same-color bot and watching for a subsequent radar-response frame (0x4F) from that bot's tank_id.

## Implications for combat strategy

- A stationary bot is a guaranteed kill if you can maintain adjacency
- A fleeing bot should be chased with **homing shots** (track off-viewport), not teleport hops[^4]
- Never abandon a target — shields and corpses both return positive hits, so a "miss" means the target moved, not that it's unkillable[^5]
- Disengaging forfeits all damage progress — the target can refuel and recover tiers (fuel pickups are the ONLY repair mechanism; damage does NOT repair over time — user 2026-07-19)[^3]
- **Cost-of-engagement asymmetry:** we lose 45 fuel per bot hit (single), bot loses ~1/N of its rank's teleport threshold per our hit. Engaging bots to state 3 then finishing is cheaper than engaging to teleport-off from any earlier state.

[^1]: user (Austin), 2026-06-16 — "tank bots stand ground and fight, only move when low HP, then move every time hit; don't collect fuel or equipment; just run until you chase and finish them or use homing"
[^2]: user (Austin), 2026-06-11 — practice bots never fight each other; retracted a prior theory
[^3]: run 20260611-004505 — purple-3 healed 1→0→3 after bot disengaged; see [[tank-registry]]. Originally read as time-based repair; corrected 2026-07-19 by user: "they do not repair over time. only via fuel pickups" — that healing was a fuel pickup too.
[^4]: user (Austin), 2026-06-16 — "the bot is able to stay still and just fire homing shots at it. the homing shots go off viewport"
[^5]: user (Austin), 2026-06-16 — "shields don't return miss. they return a positive hit. a corpse returns a positive hit"
[^6]: Sigma's TankPit Tournament Guide v3.4, 16-Jan-2015 (`docs/sources/sigmas-tankpit-guide-v3.4.pdf`), §"Fill-fighting to Lieutenant" and Technical Note #1 (shot counts + shade shortcut); §"2 – How to maximize PPH" item 5 (bots return singles); §"Initial equipment fill" tip 1 (chat commands to same-color bots). 2015 human observation, not wire-verified in this project.
[^7]: user (Austin), 2026-07-19, plus wire forensics from run bot-20260719-222903: orange-2 damage 1 (0x2E sync 22:29:54) -> teleport departure (0x58 22:30:00) -> damage 2 (0x4C map entry 22:30:16). Cross-channel encoding check over the same run's captures: every near-in-time (0x2E sync, 0x4C map) damage pair agrees (1=1 x1, 2=2 x1, 3=3 x15), so the recovery is a real state change, not an encoding artifact.
