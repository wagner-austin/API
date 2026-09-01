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

**Kill-points floor (World, measured 2026-09-01): rank-0 recruits pay
no points; rank 1+ always pays — the floor is absolute, not relative
to the killer's rank.** Every kill banner on World carries a verdict
line: `You earned extra points` or `Enemy's rank was too low. / No
extra points are given`. Nineteen instrumented kills across four
fleet tanks (killer ranks 2, 4, 5, 6) split with zero contradictions:
all 9 rank-0 victims paid nothing to every killer rank, and all 10
rank-1+ victims (ranks 1-2 in the sample) paid extra points to every
killer rank — including rank-6 artax killing rank-1 red-4. Kills pay
inventory spoils ("N dual shots gained") regardless of the verdict,
so recruit kills are never worthless, just points-less. The threat
sort's pays-points component (`_threat_sort_key_for`,
[[bot-behavior-contract]]) prefers rank-1+ targets on this law.[^13]

[^13]: `kill_points_outcome` + `kill_registered` diagnostics, runs
    bot-20260901-03xxxx (artax/yuppler/arterial/despair, build
    b3c70bca), 19 verdict rows extracted 2026-09-01: victim ranks
    read from the wire registry (0x3D carries rank) at banner time,
    verdicts parsed from the DOM kill banner. Sample ceiling: no
    victim above rank 2 and no killer above rank 6 yet — a rank-0
    "unknown-rank default" in the registry is indistinguishable from
    a true recruit, so the sort component ORDERS rather than
    excludes.

## Tanks per account: one per color, per world

An account holds up to FOUR tanks per WORLD — one per color (red=0,
purple=1, blue=2, orange=3) — and the worlds are independent, so the
pool is four on the main world PLUS four on Practice, eight in all.
Each color carries **independent rank, inventory, fuel and points**;
**awards are shared** across all of them. Switching colors is
throttled per world: a **5-minute cooldown between exiting a world
and entering it again on a different color**. The cooldown is scoped
to the world you left, not to the account — leaving the main world
does not throttle a color choice on Practice.[^5][^12]

**A room's "Game start" date belongs to the room.** Field 2 of the
JOIN_CONFIRM is what the client's lobby panel prints as `Game start`,
and every account reads the SAME value for the same room: World shows
`Sep. 25, 2012` to Arterial and to Artax fifteen days apart, across a
room-id rotation (World was room 5, then 6), while Practice shows
`Jan. 08, 2013`. It survives map rotation as well, so it dates the
room itself and not the field currently loaded in it — which makes it
the one stable identifier a rotating room has.[^16]

**Every world carries 9 bots per color.** The lobby's JOIN_CONFIRM
ends with four ACTIVE FORCE counts — tanks playing each color in that
room, in the order the client's lobby panel prints them (orange,
purple, blue, red). A world with no humans in it reads `9,9,9,9`,
which is why `api/active_games` can report `playing=0` for the same
room at the same second: the API counts humans, the wire counts every
tank. The archived `9|10|10|9` sample is those same 36 bots plus one
human on two colors.[^15]

**The entry troop byte IS the team id.** Sending `TANKPIT_TROOP=3` on
a room-enter produced `Self: (128,128) team=3` on the wire, so the
color the operator picks and the team id every other decoder speaks
are one numbering space — `red=0, purple=1, blue=2, orange=3`
throughout. Worth stating because the client's lobby builds its color
picker in the opposite visual order (`pick-troop-orange` bound to 0,
`pick-troop-red` to 3); those are UI slots, not wire values, and
reading them as wire values would send orange when the operator asked
for red.[^14]

**The lobby's "default" color is just the last one you played.** Field
5 of a room's `+` entry (`default_troop`) names the color this account
played most recently on THAT room, and the client pre-selects it; the
operator is free to click another color and enter, and doing so is
exactly what arms the 5-minute cooldown. It is not a fixed account
preference — which is why a room the account has never entered has no
color to name and sends `-1`.[^13]

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
[^16]: Operator lobby panels, 2026-08-28, two accounts on World: `Game start: Sep. 25, 2012 / Name: Artax / Rank: major` and `Game start: Sep. 25, 2012 / Name: Yuppler / Rank: lieutenant`. The Artax panel is identifiably World because `major` is the rank the orange tank reported on entry that evening (`Self: team=3 rank=6`, [^14]). Practice pairs the same way: `Game start: Jan. 08, 2013 / Name: Artax / Rank: private` against the 2026-08-13 Arterial capture's `=1|Jan. 08, 2013|Arterial|...`, and Arterial's World row that day was `=5|Sep. 25, 2012|Arterial|...`. So both rooms are confirmed by two accounts each, and the two rooms disagree with each other — the value tracks the room and nothing else. The field was named `join_date` in `JoinConfirmDict` on no evidence; renamed `game_start` 2026-08-28 after the client's own label, since the label is observed and the semantics were not.
[^15]: user (Austin), 2026-08-28, verbatim: "its cuz there's always 9 bots of each color." Live receipt `runs/sniff/sniff-20260828-214942.log`: `JOIN_CONFIRM: room=6 tank=Artax major f5-8=9,9,9,9` while `tankpit.com/api/active_games` reported `World (Desert) playing=0` on both sides of the run. Corroborated by an in-session viewport census the same evening (`Tanks: 38 (allies=10, enemies=27)` — 36 bots plus humans). The fields were named `eq1..eq4` in `sniffer/decoders.py` and `equipment` in `JoinConfirmDict` on no evidence; renamed `active_forces` 2026-08-28. Field ORDER (orange first) comes from the client's own panel builder in `tpclient.js` and is NOT the wire team order — it has not been pinned by an asymmetric live sample yet.
[^14]: Live receipt `runs/sniff/sniff-20260828-214328.log` (2026-08-28, Artax, World/room 6): `Enter game: room=6 troop=3` answered by `Self: (128,128) team=3 rank=6 fuel=0` — troop byte in, same value out as the team id ((128,128) and fuel 0 are `ROOM_ENTRY_DEFAULT_X/Y` and the pre-sync placeholder, not a real position). Team 3 = orange is pinned independently by the mine-team encoding in [[deactivation-format]] (from JS `Pg.h`), the map-data team bits, and the practice-bot roster (`red-1` arrives team 0). The same run pins **independent rank per color on one world**: the lobby's JOIN_CONFIRM reported Artax **lieutenant** (rank 4) for its last-played red tank, while entering as orange reported **rank 6** (major) — two colors, one account, one world, two ranks. The client's `pick-troop-*` DOM order (orange 0 .. red 3) contradicts the wire and is not the troop byte.
[^13]: user (Austin), 2026-08-28, verbatim: "its cuz i was a different color before. so which ever color you were last is the default, but you can click another color and enter. that one was, the world one, is a red tank." Live receipt the same minute, `runs/sniff/sniff-20260828-213328.log`: Artax's World lobby row carried `default_troop` 0 (red — the color the bot had just been playing), `.env`'s `TANKPIT_TROOP=2` overrode it to blue, and the ENTER was refused inside the recolor window (`Enter response timeout: room=6 name=World`) while every SELECT still answered — the cooldown gates ENTRY, not the lobby query. The `-1` reading follows: never-played rooms have no last color, which is what [^5]'s Arterial Practice row showed.
[^12]: user (Austin), 2026-08-28, verbatim: "each account can have 4 tanks per world. so 4 on the main world and then 4 on the practice world. each tank color shares awards, but has independent rank, inventory. and there is a 5 min cd between exiting the world and selecting a different color tank on that same world." Refines [^5] on three points that the 2026-08-13 statement left open: the four-tank pool is per WORLD and the worlds are independent (eight tanks per account, not four); RANK is per color, not just inventory/fuel/points; and the recolor cooldown is scoped to the world just exited rather than to the account. Consumed by the fleet control page's color dropdown (`service/fleet_manager.py::FleetManager.troops`), which states the rule where the operator picks.
[^5]: user (Austin), 2026-08-13, verbatim: "you get like 4 accounts per map basically. 1 for each color. and theyre separate inventories, and fuel and points, but shared awards. and there is a 5 minute cd between logging out and picking another color." Wire receipts from the fresh Arterial account's lobby (run arterial 2026-08-13 21:23, captured by the room-discovery diagnostic): `+1|Practice|1|0,0,0,0,0,0,0|-1|p|field01.gif|2026` (no tank on Practice → troop -1) beside `+5|World (Desert)|...|3|n|...` (existing tank, troop 3), and the paired `=1|Jan. 08, 2013|Arterial|0|9|9|9|9` / `=5|Sep. 25, 2012|Arterial|1|9|9|9|9` records. The -1 handling landed the same night: `src/tankpit_bot/parser.py::is_room_info_text` accepts it and `browser/room_join.py` substitutes the `TANKPIT_TROOP` color on first entry.
[^4]: round-order sweep 2026-07-25: `analysis_scripts/mine_round_order.py` (0x53 bursts grouped at 100 ms, order vs sorted shooter ids) over every `runs/**/capture_session.json`; the worked example is the respawn-watch fight (rounds at 13.3-21.3 s: purple-2 510 -> blue-7 524 -> Artax 1301, identical all six rounds, 1 ms emission spacing).
