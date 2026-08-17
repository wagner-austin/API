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

## Combat rounds (turn structure, measured 2026-07-25)

Combat is ROUND-BASED, not real-time (user contract: "there are
turns... it's not instant. check the combat logs"). The server
resolves one action per tank per ~2 s tick in a single burst, and
within the burst actions resolve in **ascending tank id order** —
archive-wide: 1,820/1,825 multi-shooter bursts are perfectly
id-ordered, and all 5 exceptions are captures of our own simulator,
not the real server (bots 500-535 therefore always resolve before
players at ~1300+). A provoked fight's shape: tick 1 = your hit
lands alone; from tick 2 onward every round carries each provoked
responder's single AND your next shot, resolved id-first — engaging
a sighted cluster of three trades 1-for-3 every round from the
second tick ([[enemy-bot-behavior]] §Team aggro).[^4]

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

**Equipment inventory starts at zero and persists across logins** (user contract, verbatim, 2026-07-24: *"a new recruit starts with 0 all then they can stop up to 20 each item... if you rank up, there an increase in the equipment cap. and if you use them all. log out and log back in. teyre still empty"*). Archive: cross-session inventory carry-over is exact (120/260 consecutive pairs, with depletion chains like radar 8→7→6→5→4→3 spanning six logins) and 0/261 sessions exceed the 20+5·rank cap.[^3]

**Higher rank tanks hold more fuel, equipment, and have a larger radar.** All three scalings are now quantified: fuel capacity = 1000 + 100·rank ([[game-economy]]), equipment = 20 + 5·rank (below), built-in radar radius = 2 + floor(rank/3) ([[radar-mechanics]]).[^2]

**Demotion:** if deactivated by an enemy, you lose one rank.[^1]

## Tanks per account: one per color, per map

An account holds up to FOUR tanks per map — one per color (red=0,
purple=1, blue=2, orange=3) — with **separate inventories, fuel and
points per color, and shared awards** across them. Switching colors is
throttled: a **5-minute cooldown between logging out and picking
another color**.[^5]

The lobby wire carries this directly: a room's `+` entry states the
account's tank for that room in field 5 (`default_troop`), and an
account with **no tank yet on that room sends `-1`** — the state the
in-game UI answers with the color picker, and the room-enter request's
troop byte answers programmatically
(`browser/room_join.py::resolve_room_troop`, `TANKPIT_TROOP`). The `=`
frames beside it are per-room account records (creation date, name,
then four fields — plausibly the four color slots; unconfirmed).[^5]

**User corroboration (2026-07-24, verbatim):** *"usually when you die
you go down a rank. and usually a recruit would rank up during a
fight from earning enough promotion points from points per shot. a
kill gives you more points but isnt necessry to go from recruit to
private."* — matches the table (recruit→private has no kill
requirement) and the demotion rule; adds that promotion points
accrue per shot during fights, with kills as a bonus, not a gate.[^3]

## Fuel

- Deactivation happens when DAMAGE takes your fuel to zero — fuel is
  the life pool and hits/mines drain it ([[deactivation-format]],
  [[game-economy]])
- **You cannot deactivate yourself** (user contract 2026-07-20,
  verbatim: "you cant kill yourself in game its impossible... you
  cant die from walking, even at zero fuel it stops debiting. you can
  use radar. you cant teleport if theres insufficient fuel, but you
  wont die").[^3] Self-spending clamps at zero: walking becomes free at 0
  fuel, radar stays usable, teleports refuse on insufficient fuel.
  The How-To-Play "run out of fuel = deactivated" line describes
  being drained BY ENEMY DAMAGE, not by your own spending

[^1]: in-game "How To Play" screens, transcribed 2026-06-16 from the Practice room at https://tankpit.com — the official rules text, quoted inline above. Where a screen's wording was later resolved to an exact formula by client mining, that supersession is recorded in [^2].
[^3]: user (Austin), 2026-07-20 — self-deactivation-impossibility contract, quoted verbatim above. The consequence the bot depends on is that a strand is survivable rather than fatal, which is why the out-of-fuel path ends the session deliberately instead of treating it as a death: `SessionExitError` with reason `out_of_fuel` from the COLLECT owner `decide_collect_mode` at `src/tankpit_bot/bot/ai/collect_mode.py:199` — see [[fuel-system]] "Marooning hazard". Own-deactivation, when it does happen, is handled at `src/tankpit_bot/bot/tick_body.py:310`.
[^2]: "Higher rank tanks... have a larger radar" — official text; resolved 2026-07-06 with exact formulas via client mining (tpclient.js Gc gauge draw) + user measurements at ranks 1/3/4/6/7 — see [[game-economy]] and [[radar-mechanics]]
[^5]: user (Austin), 2026-08-13, verbatim: "you get like 4 accounts per map basically. 1 for each color. and theyre separate inventories, and fuel and points, but shared awards. and there is a 5 minute cd between logging out and picking another color." Wire receipts from the fresh Arterial account's lobby (run arterial 2026-08-13 21:23, captured by the room-discovery diagnostic): `+1|Practice|1|0,0,0,0,0,0,0|-1|p|field01.gif|2026` (no tank on Practice → troop -1) beside `+5|World (Desert)|...|3|n|...` (existing tank, troop 3), and the paired `=1|Jan. 08, 2013|Arterial|0|9|9|9|9` / `=5|Sep. 25, 2012|Arterial|1|9|9|9|9` records. The -1 handling landed the same night: `src/tankpit_bot/parser.py::is_room_info_text` accepts it and `browser/room_join.py` substitutes the `TANKPIT_TROOP` color on first entry.
[^4]: round-order sweep 2026-07-25: `analysis_scripts/mine_round_order.py` (0x53 bursts grouped at 100 ms, order vs sorted shooter ids) over every `runs/**/capture_session.json`; the worked example is the respawn-watch fight (rounds at 13.3-21.3 s: purple-2 510 -> blue-7 524 -> Artax 1301, identical all six rounds, 1 ms emission spacing).
