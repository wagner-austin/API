---
title: Enemy Bot Behavior
tags: [combat, enemies, ai]
related:
  - "[[shot-range]]"
  - "[[combat-chase-bug]]"
  - "[[tournament-strategy]]"
source_paths:
  - "runs/bot"
fact_checked: "2026-07-06"
confidence: high
hubs: [combat]
---

# Enemy Bot Behavior

## Movement patterns

- **Stand ground and fight**: enemy tank bots do not move while fighting. They hold position and return fire.[^1]
- **Flee at low HP**: once a damage threshold triggers (appears to be after taking several hits), bots begin moving away from the attacker. They move **every time they are hit** after this gate triggers.[^1]
- **Roam viewport-to-viewport; no deliberate fuel-seeking, but accidental pickups happen** (user, 2026-07-19: "the bots teleport or walk to the next viewport usually. they dont seek fuel, but sometimes they may teleport away and happen to land or step on a fuel tank"). Landing on a container auto-picks it, so a fleeing bot can accidentally refuel — observed: orange-2 (id 528) fled our engagement at damage_state=1 (critical, 22:29:54), teleported away (0x58 at 22:30:00), and reappeared on the next map open at damage_state=2 (medium) — a REAL recovery (sync-vs-map damage encodings agree 17/17 on overlapping observations), explained by an accidental fuel landing. Fuel is the life pool; damage tiers recover when it refills.[^7] (Corpus note 2026-07-24: on the wire, roaming is RARE — see §Corpus-mined policy below; the accidental-refuel mechanism stands but most observed bot time is stationary.)
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

"Last shade" = the darkest of the four `damage_state` tiers (state 3).[^6] The shade shortcut lets the bot fire freely for the first 6-7 shots without counting, then switch to shot-counting only once the target reaches state 3. Our own `damage_state` decoder already exposes the value we'd need to gate on ([[decode-coverage]]).

**Verification status (updated 2026-07-24):** the recruit and private rows are now wire-corroborated — the archive-wide sweep's modal hits-before-teleport values land exactly on 7 (recruit) and 8 (private); see §Corpus-mined policy.[^8] The corporal row remains guide-only: no corporal+ bot exists anywhere in the archive.

## Same-color bots respond to chat commands

Same-color bots can be directed via chat with commands like **"use the radar"** and **"move out of the way"**, so a nearby friendly bot can be used as a supplemental radar source during equipment filling.[^6] Sigma credits sean/987 for the technique. Not currently exploited by our bot; if verified, this changes the practice-room fill economics because a same-color bot acts as a free extra-radar dispenser at zero fuel cost to us.

**Verification status:** guide-sourced, not yet wire-verified.[^6] Test on Practice Map by messaging a same-color bot and watching for a subsequent radar-response frame (0x4F) from that bot's tank_id.

## Implications for combat strategy

- A stationary bot is a guaranteed kill if you can maintain adjacency
- A fleeing bot should be chased with **homing shots** (track off-viewport), not teleport hops[^4]
- Never abandon a target — shields and corpses both return positive hits, so a "miss" means the target moved, not that it's unkillable[^5]
- Disengaging forfeits all damage progress — the target can refuel and recover tiers (fuel pickups are the ONLY repair mechanism; damage does NOT repair over time — user 2026-07-19)[^3]
- **Cost-of-engagement asymmetry:** we lose 45 fuel per bot hit (single), bot loses ~1/N of its rank's teleport threshold per our hit. Engaging bots to state 3 then finishing is cheaper than engaging to teleport-off from any earlier state.

## Corpus-mined policy (2026-07-24 — all 246 sessions, 12.5 bot-hours)

User contract (verbatim, 2026-07-23): *"the bots have pretty simple
logic. rhey dont pickup fuel or equipment intwntionally."* The
archive-wide sweep bears it out and pins the policy numerically:[^8]

1. **Singles only, absolutely**: 2,247/2,247 bot 0x53 echoes carry
   `weapon=0`. The guide claim[^6] and the 2026-07-19 single-run
   confirmation are now corpus-absolute.
2. **Pure return fire on the next tick**: 96.2% of bot shots land
   within 3 s of the bot taking a hit, with the latency mass at
   1.5–2.5 s — exactly one 2 s global-queue tick after the hit.
   98.7% (2,144/2,173 with a known attacker) aim at the attacker's
   exact tile. Range mode is 1 (2,086 adjacent returns) but bots
   return fire at any observed range (up to 15 tiles).
3. **Near-stationary**: 79 walk echoes across the whole archive,
   ZERO unexplained 1–3-tile position drifts. Observed locomotion is
   dominated by the teleport-off escape (131 jumps). (Caveat: 0x47
   visibility for far-away tanks is an open instrumentation question,
   so "never walks off-viewport" is not claimed — but in-viewport,
   bots sit still unless fleeing.)
4. **Never mine, never kill**: zero 0x4B placements by bot ids; zero
   0x41 kills credited to a bot; 285 bot deaths.
5. **Teleport-off thresholds corroborated**: modal hits-before-jump
   is exactly **7 at recruit** (20/49 samples) and **8 at private**
   (37/82) — Sigma's table, on the wire. The spread around the modes
   is attribution noise (multi-attacker fights, stale positions).
6. **Rank ceiling in evidence**: 8,459 bot-session observations at
   recruit, 397 at private, none higher. The FAQ's
   "significantly smarter at sergeant" regime is completely
   uncaptured — any sim bot-policy is certified for ranks 0–1 only.

**Anomaly SOLVED (same day) — the "unexplained refuels" are
REACTIVATIONS**: drilling the 64 tier-up events decomposed them
completely. 56/64 land at exactly tier 3 (full fuel) — not a pickup
signature but a reset; 27 of them provably follow that same bot id's
own 0x41 death, with the death→tier-3 gap moded at **exactly 22 s**
(17/27; spread 21–38 s) — the corpse window ([[deactivation-format]]
§corpse window). The law: **practice bots reactivate in place with
the SAME id at full fuel when their corpse clears** (the fixed
36-slot roster reuses its ids — unlike human respawns, which join as
new ids). The remaining 8 partial jumps (0→1, 1→2, 0→2) are genuine
accidental pickups — the user's 2026-07-19 story, at its true low
rate. Viewport-restriction cross-check: of the 60 no-movement
tier-ups, 50 had stale positions (visibility gap as suspected), and
all 7 in-viewport cases were 0→3 reactivations of bots that died in
view.[^8]

**Correction (same day): reactivation is NOT in place.** The first
write-up of this law said bots reactivate "in place" — the user
corrected it (verbatim: *"dont the bots respawn in a different
location, not at their corpse, in game"*) and the archive agrees
emphatically: 102 death→next-seen pairs measured, **every one ≥ 24
tiles from the corpse (Chebyshev), 70/102 beyond 96 tiles** — bots
respawn far away, effectively anywhere on the map. Also confirmed:
the roster is exactly **36 fixed bots** (9 per team, red/purple/
blue/orange 1–9, all observed in the archive), each reusing its id.
The sim law (`sim/bot_policy.py::reactivate_practice_bot`, judged by
the `bot-reactivation` shadow law — first archive run 39 samples /
35 exact, PASS) respawns at a deterministic distant scatter point;
the real placement distribution stays a documented assumption.[^8]

[^8]: archive sweep 2026-07-24: `analysis_scripts/mine_bot_policy.py`
(production decode recipe — frame split → XOR → `decode_message`)
over every `runs/**/capture_session.json`; results snapshot
`analysis_scripts/bot_policy_sweep_2026-07-24.json` (246 sessions,
750 session-minutes, 0 decode errors). Re-run the script to re-derive
every number in this section. STANDING instrument since the same day:
the policy is executable in `sim/bot_policy.py` and the
`bot-return-fire` law of `make shadow` re-judges every archived and
future session against it (first full-archive run: 2,247 samples,
2,125 exact, PASS — [[physics-module-roadmap]] Bot policy as-built).

[^1]: user (Austin), 2026-06-16 — "tank bots stand ground and fight, only move when low HP, then move every time hit; don't collect fuel or equipment; just run until you chase and finish them or use homing"
[^2]: user (Austin), 2026-06-11 — practice bots never fight each other; retracted a prior theory
[^3]: run 20260611-004505 — purple-3 healed 1→0→3 after bot disengaged; see [[tank-registry]]. Originally read as time-based repair; corrected 2026-07-19 by user: "they do not repair over time. only via fuel pickups" — that healing was a fuel pickup too.
[^4]: user (Austin), 2026-06-16 — "the bot is able to stay still and just fire homing shots at it. the homing shots go off viewport"
[^5]: user (Austin), 2026-06-16 — "shields don't return miss. they return a positive hit. a corpse returns a positive hit"
[^6]: Sigma's TankPit Tournament Guide v3.4, 16-Jan-2015 (`docs/sources/sigmas-tankpit-guide-v3.4.pdf`), §"Fill-fighting to Lieutenant" and Technical Note #1 (shot counts + shade shortcut); §"2 – How to maximize PPH" item 5 (bots return singles); §"Initial equipment fill" tip 1 (chat commands to same-color bots). 2015 human observation, not wire-verified in this project.
[^7]: user (Austin), 2026-07-19, plus wire forensics from run bot-20260719-222903: orange-2 damage 1 (0x2E sync 22:29:54) -> teleport departure (0x58 22:30:00) -> damage 2 (0x4C map entry 22:30:16). Cross-channel encoding check over the same run's captures: every near-in-time (0x2E sync, 0x4C map) damage pair agrees (1=1 x1, 2=2 x1, 3=3 x15), so the recovery is a real state change, not an encoding artifact.
