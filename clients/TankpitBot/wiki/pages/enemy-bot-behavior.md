---
title: Enemy Bot Behavior
tags: [combat, enemies, ai]
related:
  - "[[shot-range]]"
  - "[[combat-chase-bug]]"
  - "[[tournament-strategy]]"
provenance:
  - "runs/bot -- gitignored runtime capture artifact (moved from source_paths 2026-09-06, code-paths contract)"
fact_checked: "2026-07-06"
confidence: high
hubs: [combat]
---

# Enemy Bot Behavior

## Movement patterns

- **Stand ground and fight**: enemy tank bots do not move while fighting. They hold position and return fire.[^1]
- **Flee at low HP**: once a damage threshold triggers (appears to be after taking several hits), bots begin moving away from the attacker. They move **every time they are hit** after this gate triggers.[^1]
- **Roam viewport-to-viewport; no deliberate fuel-seeking, but accidental pickups happen** (user, 2026-07-19: "the bots teleport or walk to the next viewport usually. they dont seek fuel, but sometimes they may teleport away and happen to land or step on a fuel tank"). Landing on a container auto-picks it, so a fleeing bot can accidentally refuel — observed: orange-2 (id 528) fled our engagement at damage_state=1 (critical, 22:29:54), teleported away (0x58 at 22:30:00), and reappeared on the next map open at damage_state=2 (medium) — a REAL recovery (sync-vs-map damage encodings agree 17/17 on overlapping observations), explained by an accidental fuel landing. Fuel is the life pool; damage tiers recover when it refills.[^7] (Corpus note 2026-07-24: on the wire, roaming is RARE — see §Corpus-mined policy below; the accidental-refuel mechanism stands but most observed bot time is stationary.)
- **Never fight each other unprovoked** — but a player's fight can ignite cross-team bot-vs-bot return-fire loops (81 archive samples, 0 same-team, live-witnessed 2026-07-24; see §Cross-team assist fire). Only we make corpses — zero archive kills are credited to bots.[^2][^11]

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
   recruit, 397 at private, none higher. User contract (2026-07-24,
   verbatim): *"there's no seargent bots in the map. i think they
   got rid of them"* — the site FAQ's "bots can be promoted up to
   the rank of sergeant" ([[rest-api]] FAQ facts) is likely stale;
   the archive is consistent with the user's reading (zero
   corporal+ bots in 8,856 rank observations). Any sim bot-policy
   is certified for ranks 0–1, which may simply be all that exists.
   Where bot ranks COME from (user, verbatim, 2026-07-24, hedged):
   *"they are allocated by the game config i think to have certain
   ranks, as far as i know"* — allocation, not promotion; consistent
   with bots never earning kills and with the recruit/private split
   being roughly stable across the archive.

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
35 exact, PASS) respawns at a deterministic distant scatter point.
Follow-up measurements (same day): the respawn placement is
**uniform across the map** — the 102 pairs cover all sixteen 64×64
quadrants (3–9 each) with the mean at the map center (127, 131) —
and the mid-fight **teleport-off displacement modes at 16–31 tiles**
(84/131 jumps), i.e. just past the viewport; the sim's escape band
now matches that measured mode.[^8]

**Reactivation LIVE-WITNESSED (2026-07-24, first `make respawn-watch`
session):** the probe killed purple-2 (id 510) via the classic
chase — duals while adjacent (tier 3→2→1), the bot **teleported off**
at t+23.3 s (the 0x58) after ~11 accumulated hits from two attackers,
and the probe's server-selected **homing** tracked it to (166,143)
and killed it there (0x41 victim 510, killer = the probe, promo
eligible) at t+25.3 s. Eleven consecutive 2 s map polls then show id
510 absent (corpses are not rendered in 0x4C map data), until
t+47.3 s: the SAME id reappears at (154, 216) — **death→respawn
22.0 s measured from the 0x41** (bounded 20.1–22.0 s by the poll
cadence), displacement from the true corpse (166,143) of 73 tiles
Chebyshev, then stationary in every subsequent poll. Every element of
the archive-mined law — same-id reuse, the 22 s corpse window,
≥24-tile displacement, post-respawn idleness — confirmed live in one
witnessed cycle.[^10]

**Team aggro — assist AND gang-up (user contracts + archive sweep,
2026-07-24):** two user contracts (verbatim): *"if you fight another
bot, like an orange one, and there is a blue bot that can see that
orange bot. it'll help you out"* and *"if you teleport into 3 orange
bots. and hit one, the other two orange bots will start hitting
you"*. The archive confirms both sides and measures the visibility
condition: classifying every bot-shooter 0x53 in the corpus gives
**2,115 personal return-fire shots, 48 GANG-UP shots** (the shooter
was never hit but a same-team bot was — enemy teammates joining
against the player) **and 81 ASSIST shots** (at an enemy-team bot,
the player's side), with only 3 unexplained. Shooter→target Chebyshev
distances: assist 3–8, gang-up 2–8 (one stale-position 12) — both
mechanisms operate within the ~8-tile viewport radius, i.e. exactly
"can see". Full model — SHOT-FOR-SHOT, no aggro state (user contract,
verbatim, 2026-07-24: *"bots are just 1:1. so you shoot them once,
they shoot back. if you stop they stop. they dont chase or keep
attacking. its shot for shot with the bots."*): every hit on a bot
draws at most one return that tick — from the victim, and from
same-team bots within sight in the gang-up case — and the response
stops the moment the hits stop. Archive verification: 3,031 hits on
bots drew 2,201 returns; the per-engagement fired/taken ratio NEVER
exceeds ~1 (mode 0.75–1.0 over 366 engagements — the deficit is the
one-shot-per-tick cap plus flee/death truncation); and in 99.2% of
397 engagements the bot's last shot lands within one tick of the
last hit it took — zero chasing, zero continued fire.
(MODELED 2026-07-25: `sim/bot_policy.py::note_hit_for_team_aggro`
implements the sight-gated per-hit reflex, and the `bot-return-fire`
shadow law now judges all three classes — archive exactness rose
from 94.6% to 97.6% (2,192/2,247) once the former "mismatches" were
recognized as lawful team aggro.)[^11][^12]

**The live witness of the assist side:** the fight
had a third combatant — **blue-7 (id 524, the probe's own team)**
opened fire on purple-2 exactly one bot-reaction tick after the
probe's first dual, and purple-2's return fire then switched to
blue-7's tile (every shot aimed at a real attacker's exact tile;
user contract (verbatim, 2026-07-24): *"you shoot the bot. we are
standing still. it hits us. the bots never miss."* — the
earlier "off-tile aim" reading of this capture was wrong). Archive
discrimination: **81 bot→bot shots exist corpus-wide** (vs 2,166 at
players), **zero ever target a same-team bot**, and 78/81 fall
within 10 s of a player shot — mutual return-fire loops between
enemy-team bots ignited by a player engagement. The "never fight
each other" law refines to: bots never SEEK each other unprovoked,
but a player's fight can ignite cross-team bot-vs-bot return-fire
loops; they still never produce corpses (zero archive kills credited
to bots). This mechanism also explains most of the corpus's 1.3%
"aimed off the attacker's tile" residual (attacker switching in
multi-combatant fights) and part of its hits-before-teleport spread
(purple-2 jumped after ~11 total hits, not 7 of ours).[^10][^11]

[^10]: live capture `respawn_watch_probe.capture_session.json`
(2026-07-24, `make respawn-watch`, tank "Artax" id 1301): probe duals
(weapon=1, −10 fuel each) at 11.3–21.3 s; purple-2 returns weapon=0
at the probe's tile (13.3 s, −45 fuel to the probe) then at blue-7's
tile (126,138) 15.3–21.3 s; blue-7 (id 524, team 2, persistent 25)
singles at purple-2 13.3–21.3 s; 0x58 (id 510) at 23.3 s; probe
homings (weapon=3) aimed (166,143) at 23.3/25.3 s; 0x41 victim=510
killer=1301 at 25.3 s; 0x4C entries for 510 absent 25.3–45.4 s,
present at (154,216) from 47.3 s onward.
[^12]: shot-for-shot sweep 2026-07-24:
`analysis_scripts/mine_shot_for_shot.py` — per-bot hits-taken vs
shots-fired per session (ratio buckets) and last-shot-minus-last-hit
gap distribution (0–2 s in 291/397, negative i.e. stopped-before-
last-hit in 103, >2 s in only 3). Re-run to re-derive.
[^11]: archive sweeps 2026-07-24:
`analysis_scripts/mine_bot_assist.py` (bot-shooter 0x53 frames vs
last-known tank positions: 2,166 at player tiles, 81 at enemy-team
bot tiles, 0 at same-team bot tiles, 78/81 within 10 s of a player
shot) and `analysis_scripts/mine_bot_aggro.py` (same frames
classified by whether the shooter or a same-team bot was hit within
10 s: 2,115 return-fire / 48 gang-up / 81 assist / 3 unexplained,
with the shooter→target distance tables). Re-run either script to
re-derive.

**First continuous undisturbed observation (2026-07-24, decisive
watch run): ten minutes adjacent to purple-2 — the bot emitted
NOTHING.** Zero 0x2E syncs, zero 0x47 movements, zero refuels, zero
shots across the full 617 s dwell, one tile away. This deepens the
near-stationary corpus row (which could only sample activity windows)
to continuous-observation depth: an unprovoked bot takes no actions at
all. Corollary from [[server-push-gating]]: since tanks appear in the
push stream only when they act, an idle bot's fuel cannot be read
passively — its state surfaces only via the join-time roster dump, map
snapshots, or its own next action. The dwell also confirmed **bots
block movement**: every walk targeting purple-2's tile drew the
`CANT_GO` supervisor rejection (154/154), so tank tiles are
impassable-occupied, not walk-through.[^9]

[^9]: decisive watch capture `bot_watch_probe.capture_session.json`
(2026-07-24, 1,198 messages, 617 s, landed (133,139) adjacent to
purple-2 at (132,139)): received t>60 s contains 0x2E only for self
(188/188 id 1301); supervisor codes 154× `CANT_GO` (code 1) on the
west-bound shuffle into the bot's tile, 45× `ALREADY_THERE` (code 6).
See [[server-push-gating]] for the run design.

[^8]: archive sweep 2026-07-24: `analysis_scripts/mine_bot_policy.py`
(production decode recipe — frame split → XOR → `decode_message`)
over every `runs/**/capture_session.json`; results snapshot
`wiki/sources/bot_policy_sweep_2026-07-24.json` (246 sessions,
750 session-minutes, 0 decode errors). Re-run the script to re-derive
every number in this section. STANDING instrument since the same day:
the policy is executable in `sim/bot_policy.py` and the
`bot-return-fire` law of `make shadow` re-judges every archived and
future session against it (first full-archive run: 2,247 samples,
2,125 exact, PASS — [[physics-module-roadmap]] Bot policy as-built).

[^1]: user (Austin), 2026-06-16 — "tank bots stand ground and fight, only move when low HP, then move every time hit; don't collect fuel or equipment; just run until you chase and finish them or use homing". The bot-side answer to "they run" is the stay-put contract rather than a chase: `close_target` at `src/tankpit_bot/bot/ai/combat_close.py:259`, whose branch 2 (`is_already_engaged`, `src/tankpit_bot/bot/ai/combat_target.py:64`) fires from the current tile instead of re-teleporting — see [[combat-chase-bug]].
[^2]: user (Austin), 2026-06-11 — practice bots never fight each other; retracted a prior theory. Independently swept from the archive by `analysis_scripts/mine_bot_policy.py` (see [^8]), which found no bot-on-bot engagement across the corpus.
[^3]: run 20260611-004505 — purple-3 healed 1→0→3 after bot disengaged; see [[tank-registry]]. Originally read as time-based repair; corrected 2026-07-19 by user: "they do not repair over time. only via fuel pickups" — that healing was a fuel pickup too.
[^4]: user (Austin), 2026-06-16 — "the bot is able to stay still and just fire homing shots at it. the homing shots go off viewport". Refined 2026-07-03: the seeker tracks off-viewport but the server refuses an AIM outside the viewport, so every dispatch is folded onto the visible bounds by `_clamp_aim_into_viewport` at `src/tankpit_bot/bot/ai/combat_strategy.py:47` while still carrying the target's `tank_id` — see [[combat-chase-bug]] and [[shot-range]].
[^5]: user (Austin), 2026-06-16 — "shields don't return miss. they return a positive hit. a corpse returns a positive hit". Because a positive hit therefore cannot distinguish a kill, deactivation is read from the wire rather than inferred from shot feedback: `src/tankpit_bot/combat.py:114` and `:119`. The DOM-scrape workaround this ruling originally motivated was deleted 2026-07-19 (`wiki/log.md:1198`) — see [[shoot-event-format]] [^2].
[^6]: Sigma's TankPit Tournament Guide v3.4, 16-Jan-2015 (`docs/sources/sigmas-tankpit-guide-v3.4.pdf`), §"Fill-fighting to Lieutenant" and Technical Note #1 (shot counts + shade shortcut); §"2 – How to maximize PPH" item 5 (bots return singles); §"Initial equipment fill" tip 1 (chat commands to same-color bots). 2015 human observation, not wire-verified in this project.
[^7]: user (Austin), 2026-07-19, plus wire forensics from run bot-20260719-222903: orange-2 damage 1 (0x2E sync 22:29:54) -> teleport departure (0x58 22:30:00) -> damage 2 (0x4C map entry 22:30:16). Cross-channel encoding check over the same run's captures: every near-in-time (0x2E sync, 0x4C map) damage pair agrees (1=1 x1, 2=2 x1, 3=3 x15), so the recovery is a real state change, not an encoding artifact.

## The map is a living-tanks list — and idle respawns are wire-silent (measured 2026-08-05)

Three corrections to this page's earlier reactivation account, all from
run bot-20260805-095935 (32 kills, the first session to farm past 20)
plus the 08-03 human fight:[^s32]

- **Idle tanks emit NOTHING.** 27 of 32 victims never appeared on the
  wire again after death — no 0x2E resync, no anything — for up to 35
  minutes. Even never-killed bots sent only 4-7 syncs in 36 minutes.
  The earlier "reactivation = same-id 0x2E at full fuel" signature only
  fires for respawns that MOVE (the five re-killed bots were revived by
  global 0x3D movement broadcasts from across the map, alive facing).
  A respawn that idles at its spawn point is invisible on the wire.
- **The 0x4C map is a strictly LIVING-tanks list.** Victims were absent
  from all 58 map snapshots taken during their corpse windows and
  present in all 204 taken after; the human Belton vanished from the
  map during each of his three corpse windows and returned at +24 s.
  Presence in map data IS aliveness — the server curates the dead out
  within the same second they die. Client liveness rule 4
  (`state/tank_mutations.py:156-164`, 2026-08-05) now consumes this: a deactivated
  tank listed in a map snapshot flips alive. Before that rule the
  registry filled with phantom corpses (positions faithfully tracking
  their living owners via map updates, liveness stuck dead) and
  sessions exited `no_viable_targets` in a full room.
- **Deactivated humans KEEP their wire id.** Belton was id 984 through
  all three deaths and respawns. The earlier "human respawns join as
  new ids" claim appears to describe leaving/rejoining the ROOM, not
  in-room deactivation.

CORRECTION (2026-08-05, same day): the first write-up of session
bot-20260805-173034 blamed "drained-map inheritance + ~1 dot/min
container regeneration" — citing a law [[game-economy]] had already
FALSIFIED on 2026-07-25 (dot appearances are our own radar exposure;
refills are discrete deposits, never regen). The real cause, mined
from the session's own events, is a client-side geometry trap:

- Spawn inventory was homing 23/25 → the hunt-only-when-full gate was
  closed by exactly 2 homings, with 27 live enemies in the viewport
  from tick 6 onward.
- The nearest equipment, (58,95), is enclosed on all sides by water
  and mines ((58,94) mine, (59,95) mine, (57,95) water; its southern
  neighbor (58,96) is another equipment container itself walled by
  water). The hop planner selected landing (59,95) — a KNOWN mine
  tile — so the server displaced every teleport (534
  `teleport_displacement` events; 1,068 of 1,130 `hop_selected`
  events targeted this one tile over 43 min). Nothing counts repeated
  displacement failures: the `unservable` release law only fires when
  NO landing candidate exists, `target_gone` never fires because the
  equipment stays visible, so the lock re-armed every tick.
- The loop consumed its own gate resource: each cycle re-scanned,
  burning radars 22 → 0. When weapons later hit cap via incidental
  pickups, radars ≥ cap−5 had become the failing condition instead.
- Fuel was never scarce: the tank hit capacity 19 times mid-loop
  (`tank_at_capacity` releases). Sessions 1-3 and 5 farmed the SAME
  room normally — room state was irrelevant; session 4 merely spawned
  two tiles from the trap.

Defects exposed and FIXED the same day (no escape hatches — plan-time
knowledge, not retry counters):[^s32]

1. **Landing attainability** (`find_attainable_landing_tile`,
   `bot/ai/reachability.py`): every pickup-serving teleport selector
   (equipment hop, fuel larder, desperation hop, locked-approach
   fallback) now requires the landing tile to be terrain-legal AND
   mine-free — a landing the tank will actually stand on. Legality
   alone remains the transport answer (combat aims, scouting,
   mine-flip escapes, where arriving near is fine).
2. **General clearance trigger** (`mine_clearance.py`): the free
   unlock shot fires on the general condition — a known hostile mine
   denies a worthwhile container's service access (covered tile OR no
   attainable landing) and the blast provably reopens it — replacing
   the two special cases (mined container tile; mined walk corridor)
   that both missed this pocket. In session 4's geometry the shot at
   the flank mine resolves the whole trap at tick 4.
3. **Lock verdict** (`_locked_target_is_unservable`): servable now
   means attainable landing OR shootable service mine OR pond ferry;
   a shootable denial HOLDS (the clearance step precedes the hops),
   an unshootable one releases `unservable`.

The radar burn (gate resource spent by the gate-satisfying loop) is
mooted by the fix — the loop no longer exists to burn it. Pinned by
the verbatim session-4 pocket in `tests/bot/ai/test_mine_clearance.py`
(`_session4_pocket`) and the loop-killer hop test.[^s32]
[^s32]: Run capture on disk: `runs/bot/bot-20260805-095935.capture_session.json` (32 kills, the first session to farm past 20) plus the 2026-08-03 human fight. The landing fix is `find_attainable_landing_tile` at `src/tankpit_bot/bot/ai/reachability.py:173`; the loop is pinned by the verbatim session-4 pocket `_session4_pocket` at `tests/bot/ai/test_mine_clearance.py:213`, used by the loop-killer hop test at `:265`. All paths verified present 2026-08-07.
