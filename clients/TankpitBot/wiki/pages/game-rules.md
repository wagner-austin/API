---
title: Official Game Rules (How To Play Screens)
tags: [game-mechanics, rules, official]
related:
  - "[[shot-range]]"
  - "[[radar-mechanics]]"
  - "[[equipment-system]]"
  - "[[teleport-mechanics]]"
  - "[[ferry-mechanics]]"
source_paths:
  - "docs/sources/sigmas-tankpit-guide-v3.4.pdf"
  - "tpclient.js"
source_git_blobs:
  "docs/sources/sigmas-tankpit-guide-v3.4.pdf": "6ec5665374ed38b2dfc8fda94aad35c4b99c1256"
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-06-16"
confidence: high
hubs: [game-mechanics]
---

# Official Game Rules

Transcribed from the five in-game "How To Play" screens.[^1]

## Movement and combat

- Click on a land tile and your tank will attempt to drive there
- Move to fuel to pick it up — don't let your tank run out of fuel
- **Click on an enemy tank to shoot it** — you target the enemy directly, not an adjacent tile
- Click on enemy mines to blow them up
- Click and hold to grab or drop an obstacle; tow it behind your tank to build a bridge or base
- Enemy tanks are blocked and damaged if they drive into your mines
- Click and hold to grab equipment
- Use a ferry to drive on water

## Map and teleport

- **Open the map, then click on it to teleport** your tank to a new location
- This confirms: you click the target position on the map (enemy, container, or tile) and the server places you

## Equipment types

| Type | Effect |
|------|--------|
| Armor shield | Protects from enemy fire and conserves fuel |
| Dual shot | Inflicts double damage |
| Missile shot | Fires OVER mountains, obstacles, or other tanks |
| Homing shot | Follows an enemy as they drive or teleport away |
| Extra radar | Scans the entire viewport instead of just the area around your tank |

## Equipment capacity

- Recruits hold **20** of each equipment type
- Each higher rank holds **5 more** of each type than the previous rank
- Equipment use can be enabled or disabled

## Ranks and promotion

| Rank | Points required | Additional requirement |
|------|----------------|----------------------|
| Recruit | 0 (starting) | — |
| Private | 500 | — |
| Corporal | 1,000 | — |
| Sergeant | 4,000 | — |
| Lieutenant | 10,000 | deactivate a corporal or higher |
| Captain | 20,000 | deactivate a sergeant or higher |
| Major | 30,000 | deactivate a lieutenant or higher |
| Colonel | 40,000 | deactivate a captain or higher |
| General | 50,000 | deactivate a colonel or higher |

**Higher rank tanks hold more fuel, equipment, and have a larger radar.** All three scalings are now quantified: fuel capacity = 1000 + 100·rank ([[game-economy]]), equipment = 20 + 5·rank (below), built-in radar radius = 2 + floor(rank/3) ([[radar-mechanics]]).[^2]

**Demotion:** if deactivated by an enemy, you lose one rank.

## Fuel

- Deactivation happens when DAMAGE takes your fuel to zero — fuel is
  the life pool and hits/mines drain it
- **You cannot deactivate yourself** (user contract 2026-07-20,
  verbatim: "you cant kill yourself in game its impossible... you
  cant die from walking, even at zero fuel it stops debiting. you can
  use radar. you cant teleport if theres insufficient fuel, but you
  wont die"). Self-spending clamps at zero: walking becomes free at 0
  fuel, radar stays usable, teleports refuse on insufficient fuel.
  The How-To-Play "run out of fuel = deactivated" line describes
  being drained BY ENEMY DAMAGE, not by your own spending

[^1]: in-game "How To Play" screens, transcribed 2026-06-16 from tankpit.com Practice room
[^2]: "Higher rank tanks... have a larger radar" — official text; resolved 2026-07-06 with exact formulas via client mining (tpclient.js Gc gauge draw) + user measurements at ranks 1/3/4/6/7 — see [[game-economy]] and [[radar-mechanics]]
