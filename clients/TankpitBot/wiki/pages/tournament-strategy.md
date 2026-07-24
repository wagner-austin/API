---
title: Tournament Strategy (Sigma v3.4, 2015)
tags: [combat, strategy, tournament, community, preservation]
related:
  - "[[game-modes]]"
  - "[[enemy-bot-behavior]]"
  - "[[gameplay-loop]]"
  - "[[equipment-refill-strategy]]"
source_paths:
  - docs/sources/sigmas-tankpit-guide-v3.4.pdf
source_git_blobs:
  "docs/sources/sigmas-tankpit-guide-v3.4.pdf": "6ec5665374ed38b2dfc8fda94aad35c4b99c1256"
fact_checked: "2026-07-06"
confidence: medium
hubs: [combat]
---

# Tournament Strategy (Sigma v3.4, 2015)

Preserved human tournament meta from Sigma's TankPit Tournament Guide v3.4 (16-Jan-2015). This is **not** wire-verified game mechanics — it is 2014-2015-era community wisdom about how to win 60-minute tournaments. The guide credits contributions from 987, Beaver, BlueGhost, Jay, Jordann, Kirby, PRIDE, Revelation, and ~15 others. This project's bot does not play tournaments; the page exists so the strategic knowledge is not lost when the DocDroid link rots.[^10]

Primary source: `docs/sources/sigmas-tankpit-guide-v3.4.pdf`. Section anchors below refer to headings in that PDF.[^10]

## Tournament arc — the 60-minute frame

Sigma frames the whole tournament around three concurrent goals: **kills** (needed for the promotion gates to captain / major), **PPH** (points-per-hour to place well among your rank), and **equipment refills** (the enabler for the other two). Cup awarded to top rank + top points within that rank. Top-3 get bronze / silver / gold decoration.[^1]

Typical decision arc:[^1]
- **0-10 min:** initial equipment fill as recruit
- **10-25 min:** fill-fight bots up to lieutenant; watch top players list for kill targets
- **20-25 min:** first realistic kill window (no-fill newbies making sergeant)
- **25-50 min:** PPH accumulation, opportunistic kills, top off radars
- **50-60 min:** shield-fighting phase for final point burst

## Initial equipment fill (recruit phase)

Fill AS a recruit before ranking up — abundance drops sharply after the first 15 minutes, and if you rank first you become a target for filled sergeants with dual/homing.[^2]

Method:
- Recruit built-in radar is 5x5 (tank center + 2 in each direction). See [[radar-mechanics]].
- Radar all nine positions of the current viewport (center + 4 corners + 4 edges) with no overlap, then walk to a new viewport
- On any extra-radar pickup: auto-scroll OFF → teleport / scope to a fresh viewport → sweep with extra → collect 3-5 boxes → resume regular fill when the 2-4 ER stack runs out
- Boxes under an obstacle or one tile into water: pick them up in the fill phase (guaranteed loot beats a 5x5 gamble); skip them in the polish phase (teleporting to a 3-4 box viewport is faster than bridging for 1)
- Once everything else is full and only extra radars remain: teleport-driven ER-only fill, only pick up easy boxes
- Same-color bots can be commanded to "use the radar" — see [[enemy-bot-behavior]] chat commands
- Water maps: click coast on entry to hunt ferries. If found, ferry all the way full, then hide the ferry 1-2 viewports away from your yellow-dot fuel cluster before ditching to land

**Target time:** 8-10 minutes. Under 8 is fast. Over 10 means switching to a kill-heavy or captain-farming approach mid-game to compensate.[^2]

## Fill-fighting to lieutenant

At 50+ radars and 60+ of everything else, enable only duals and extras (disable homing), teleport to the nearest bot, sweep on landing with an extra radar, smash to teleport-off, refuel + loot, repeat.[^3]

Bot shots-to-teleport-off values (7 recruit / 8 private / 9 corporal) plus the last-shade shortcut are on [[enemy-bot-behavior]].

Exit fill-fighting when either (a) you kill a corporal bot as sergeant and rank to lieutenant, or (b) you land on a viewport with a competent human PPH partner and switch to sessioning them.[^3]

**Kirby's radar rule (dry maps):** aim for `2 × minutes_remaining` extra radars before mid-game. So at 20 minutes in, 82 ERs is enough to finish. Wet maps (Desert, R&S) need only ~75. If early ER stockpiling isn't feasible, defer it until the second equipment spawn appears and radar count drops to 30-40.[^3]

## Kill target typing

Three categories:[^4]

- **Type 1 — no-fill newbies:** rank fast without filling. Spot them via the top-players list ('T' key) — corporals in the first 4 minutes are almost always no-fill. They become killable when they make sergeant (~15-20 min mark). Kill window: 15-20 min. Late joiners at corporal about to make sergeant are also Type 1; feed-them-into-sergeant technique: teleport to them at half fuel, shields on, fire singles, let them hit you until they promote, then plant mines / duals on / snag kill.
- **Type 2 — filled newbies:** carry shields, mis-manage equipment. Newbie award markers: silver tank, death medal, single or zero stars. PPH sits around #20 or worse. They eventually die but survive 1v1s. Kill them by joining a natural mob, or manufacture one via "Base is here" / "Meet me" chat with same-color competent players. In water maps (Deep Six, Iceland, Aquarium), if a low newbie drives away he burns shields fast — teleport half-to-one screen ahead of his heading so you land adjacent for point-blank duals.
- **Type 3 — veterans:** cupped before, don't die except to lag. Do not chase after a mob screen unless they took 40+ duals (= ~80+ shields lost, likely under 10 shields left).

**Marathoners:** target teleports every tick, you can only land 1-2 duals before he's gone and you're spending only homings. Stop chasing; let the endgame mob catch him.[^4]

**Support-callers:** every attack summons same-color newbies to cock-block, or he beelines to same-color bot clusters for firepower cover. Stop chasing; call your own backup or wait for endgame mobs.[^4]

**Self-refillers:** target has enough extras to always find fuel before insufficient. You'll run out of duals before he runs out of shields. Stop chasing.[^4]

## Kills vs. PPH balance

Cup requires rank + points. Kills unlock ranks (captain needs 1, major needs 2). PPH decides tie-breaks within a rank. Both matter.[^5]

Rules of thumb from the guide's community quotes:[^5]
- 10-15 player tournaments → top-PPH-1-kill can cup
- 20+ player tournaments → majors are likely, plan for kills
- Early kill (30-35 min mark) unlocks optionality: try for major, or stick captain with PPH
- 50-min mark still at lieutenant with no kill = shift entirely to PPH + endgame die-down attempt

## PPH mechanics

Homing OFF during PPH — duals only. Duals hit adjacent, cost less inventory to keep full, and don't waste inventory on likely misses. Homing only for kill runs.[^6]

- Prefer lieutenants+ over bots when captains and lieutenants are around: **3 shots on a private bot ≈ 2 shots on a human lieutenant** for points. Over 100+ duals in a session, that's 10,000+ points and the difference between #2 captain and #6 captain.
- Water-map exception: on Iceland / Aquarium / Deep Six, islands are too small for two humans to session; bots become the rational choice because bots return singles (see [[enemy-bot-behavior]]) and don't steal your equipment or fuel from the island.
- 1v1 partner reads: he doesn't shoot back after 2-3 of your hits on his full fuel = he's filling equipment, not PPH-ing. Move on.
- Don't let PPH partner get the last fuel on the viewport — you'll go insufficient first.
- Experienced partner stops shooting = he's mapping. Fire 2 more shots (he teleports next turn), then pivot to a bystander or a container pickup.
- Never fight 3+. In a 2v1, position so the first target blocks the second's line and shoot through them.

## Equipment management

Equipment discipline determines whether refills cannibalize combat time.[^7]

- **Fill as you fight:** if someone (you or someone else) radars mid-combat and reveals a container, grab it. Refill is inevitable; don't donate to the enemy.
- **Counter-cyclical filling:** fill immediately when a new spawn hits the map; avoid areas the mob just cleared. Target 4-5 boxes per extra-radar sweep. Full refill in 2-3 viewports (1-2 minutes).
- **Don't over-refill during PPH:** if partner sees you grabbing >5 boxes he'll leave. Rhythm: 2 boxes → wait for him to miss ground twice → resume duals → 7-8 shots → 2-3 more boxes.
- **Shortest-path pickup:** on a 5+ box viewport, take the traversal that minimizes drive distance. Auto-scroll OFF and drive clockwise / counter-clockwise for edge-heavy layouts. On spiral maps (Crazy Maze), bridging is sometimes shorter than routing.
- **Endgame fill target:** if going for a kill, fill lighter (dispensable duals). If shield-fighting for cup points, fill heavier (~100 duals with 8 minutes left).

Revelation on homing waste (2014-07-28): *"Don't purposely fire homings over rocks or obstacles unless you're going for a kill. It's a waste, just find a new angle or map."*[^7]

## Shield-fighting (endgame)

Reserved for the final 5 minutes if points, not kills, are the cup path.[^8]

- Fill shields + duals **before** the last spawn is used up
- Keep the second hand of your clock visible; count down last 60 seconds
- Press 'I' repeatedly while fighting to poll armor shield count
- Shields MUST last to zero seconds — running out with 10 seconds left = death + rank loss = catastrophe
- Done correctly: 10,000+ points in that final window, potentially multiple leaderboard passes

## Scenario templates

Three worked examples from the guide:[^9]

1. **#1 captain, 5 min left, 3-4 other captains, dry map:** survival > risk. Fighting bots to drop to #2 is acceptable if mobs are vicious.
2. **Early kill + captain by 35 min, high mob activity:** commit to major; high-PPH captain will lose to the likely 2+ majors.
3. **#5 PPH lieutenant, no kill, 10 min left:** PPH harder + endgame die-down kill; no time for major even with a kill now.

## Cross-references

- Mechanics (verified) → [[enemy-bot-behavior]] for shot counts, [[gameplay-loop]] for the bot's own cycle
- Encoding → [[game-modes]] for tournament flags, capacity ladder, elimination
- Economy → [[game-economy]] for the fuel/damage costs behind these rules of thumb

## Footnotes

All references are to `docs/sources/sigmas-tankpit-guide-v3.4.pdf`.[^10]

[^1]: §"Some Theory" (kills + PPH + refill triple) and §"Ethics" (top-3 cup structure)
[^2]: §"Initial equipment fill" — recruit-first fill, 5x5 recruit radar, ferry hunt on water maps, auto-scroll toggling with extras, obstacle-picking economics
[^3]: §"Fill-fighting to Lieutenant" — dual+ER only, Kirby's 2-ER-per-minute rule, second equipment spawn timing
[^4]: §"1 – Landing Kills" — Type 1/2/3 taxonomy, 15-20 min sergeant kill window, late-corporal feed-into-sergeant technique, marathoner / support-caller / self-refiller signs
[^5]: §"Scenario Analysis and Balancing Kills vs. Points" — community quotes from PRIDE, Beaver, Jordann on strategy adaptation
[^6]: §"2 – How to maximize PPH" — homing-off discipline, bot vs. lieutenant point economics (3 vs 2 shots), water-map bot exception
[^7]: §"3 – Equipment management" — counter-cyclical filling, fill-as-you-fight, PPH rhythm, shortest-path pickup, endgame fill sizing
[^8]: §"When to shield fight" — final 5-minute window, clock-hand + 'I' polling, catastrophe threshold
[^9]: §"Scenario Analysis and Balancing Kills vs. Points" — three worked scenarios

[^10]: the guide PDF itself, on disk and blob-pinned in frontmatter (`docs/sources/sigmas-tankpit-guide-v3.4.pdf`, hash `6ec5665...`): title page carries the v3.4 / 16-Jan-2015 date and the contributor credits; the DocDroid provenance note and the not-wire-verified caveat are this page's own framing.
