---
title: Flag Triage — bot-20260729-232252 (10 human flags)
tags: [bug, forage, hunt, mines, observability]
related:
  - "[[diagnostic-hud]]"
  - "[[mine-mechanics]]"
  - "[[bot-behavior-contract]]"
  - "[[game-economy]]"
source_paths:
  - "runs/bot/bot-20260729-232252.events.jsonl"
  - "src/tankpit_bot/bot/ai/resource_search.py"
  - "src/tankpit_bot/state/scan_coverage.py"
fact_checked: "2026-07-30"
confidence: high
hubs: [architecture, combat]
---

# Flag Triage — bot-20260729-232252

First live use of the flag channel: 10 flags, all captured with 8-tick
lead-up rings. Locate any flag: `grep '"human_flag"' <events.jsonl>` →
its `tick_n`/`flag_seq`, then filter the stream by `tick_n` (see
[[diagnostic-hud]] § Tracing a flag). Four root causes; fix status
table at the bottom.

## F1 — Pre-hunt top-off hop is direction-blind (flags 1, 2, 6)

The "extra teleport before engaging": at fuel 1083/1089/1082 the
hunt-only-when-full gate keeps COLLECT in charge for exactly one more
dot hop; the hop scorer (`dots * walkable / cost`) knows nothing about
the imminent acquisition. Flag 2's ring is the worst case: hop
(49,35)→(70,20) NE, top off, then the acquisition teleports straight
back SW to (48,43) — ~26 tiles out, ~30 back, two teleports where a
target-side dot would have cost one leg.[^1] Not a contract violation
(fuel-before-chasing held) — an economy bug in hop placement.

**Why the top-off hop exists at all** (flag-1 microscope, 23:24:35):
stock completed mid-viewport (dual 25/25, homing 25/25, radar 21 ≥
floor 20) with fuel at 1083 — 17 under the exact-capacity hunt floor
(`hunt_fuel_floor = fuel_capacity(rank)` = 1100). The only nearby
fuel was REFUSED by collect economics ("skip fuel at (21,78) vol=925:
clamped gain 17 not worth 10-tile walk"), so the only remaining way
to buy the last 17 points was a dot hop's landing auto-pickup. The
walk-worthiness heuristic and the exact-capacity floor disagree about
whether those clamped points matter, and the disagreement is settled
with a teleport. A direction-aware hop still pays this unnecessary
leg; a hunt-floor tolerance (or letting the walk heuristic accept
clamped gains that COMPLETE hunt readiness) removes it.[^5]

## F2 — Zero-yield hop churn: harvest memory conflated with scan TTL (flags 3, 5, 7, 9, 10)

Measured: **133 of 211 hops (63%) produced zero fuel/equipment gains**
before the next hop; longest dead streak 19 hops (ticks 380–409, the
flag-10 window).[^2] Two conspiring causes:

1. **The only "have I been here" memory is scan coverage with a 180 s
   TTL** (`FORAGE_COVERAGE_TTL_MS`). Ground harvested 3+ minutes ago
   reads *clean* and gets re-hopped, re-scanned, and yields nothing.
   The belief store carried 531 container entries with volumes and
   timestamps — known-empty viewports — and the hop scorer never
   consults it.
2. **The hop score measures fuel-dot density, not expected yield.** At
   full fuel (most SEARCH hops run at 1100/1100) a landing dot drinks
   nothing; the hop's real value is undiscovered equipment, which the
   score does not model.

Direction: split radar-coverage TTL (what's worth re-scanning) from
harvest memory (what we picked clean, belief-driven, much longer
lived), and score hops by expected yield: unknown ground + known
unharvested containers, with dots contributing only the fuel headroom
they can actually fill.

## F3 — Mine-covered equipment is unpickable; mine-shot clearance missing (flags 4, 8)

Twice the run saw equipment under enemy mine fields (purple at flag 4,
orange at flag 8) and had no counterplay. Wanted (user directive): a
single **lifted shot-clearance system** — straight-LOS test over
terrain + land movable obstacles — used by both combat fire and a new
mine-clearance behavior: one shot AT the covered equipment/fuel tile
destroys up to 9 overlapping mines at private+ ([[mine-mechanics]] §
rank-dependent blast), then teleport in and collect. No path clearing;
only the covering mines. Homing/missile arc over terrain applies to
TANK targets only — never mines.[^3]

## F4 — Ferry-rider acquisition cloak (Yuppler never targeted)

All 11 acquisition passes saw Yuppler (id 1229) at a frozen (128,102)
and rejected him: 7× `no_passable_adjacent`, 4× `stale_map_data` —
while picking practice bots every time.[^4] The root cause is NOT a
mine ring: (128,102) is **open water** on field01 (user 2026-07-29:
"yuppler was on a ferry. i mtelling you that"; terrain render
confirms water from x=126–131 on that row, nearest ground (125,102)
at distance 3). A water tile has no passable cardinal neighbor, so
the strictly-adjacent gate rejected a ferry rider on every pass. Both
prior fixes miss this path: human preemption (`0455d4ba`) fires only
on `unaffordable`; stand-off fire (`94fa85e9`) fires only after a
LOCKED target's failed landing — `no_passable_adjacent` rejects
before any lock exists. Mine rings cloak through the identical hole
(the mine-composed passability view marks the ring impassable), so
one fix covers both shapes. The
`stale_map_data`→forced-map-refresh rule (`d780103d`) also failed to
cure the 4 stale rejections — verify that path live.

Fix (2026-07-29): acquisition and in-viewport selection now gate on
`has_standoff_landing` — any passable tile within `SHOT_RANGE_TILES`
(8) of the target — instead of strict adjacency, and
`choose_combat_landing_tile` aims at the passable unoccupied tile
nearest the target (tie-broken toward self) when the target's own
tile is impassable, since the server refuses water landings. Water
never blocks shots ([[weapon-selection]]), so the shore tile is a
firing position; a target with NO ground inside the 8-diamond
(mid-ocean ferry) still rejects, now as `no_standoff_landing`.

**Static receipt for the landed gate** (independent verification,
2026-07-30): the field01 terrain map puts **86 passable tiles inside
the radius-8 shot diamond around (128,102)**, nearest at Manhattan 3
((125,102) and (128,105)) — the new gate accepts Yuppler's exact
ferry position, so next login the acquisition should lock and the
pursuit fire. Run-order note: the run's rejections read
`no_passable_adjacent` because the stand-off gate landed in the tree
at 23:49–23:51, AFTER the run ended at 23:47 — the run executed the
old strictly-adjacent gate, so the fix is untested live but not
contradicted by this run.

**Chat/HELLO trail** (the user's companion report "never said
hello"): the run has ZERO `chat_greeting`/`chat_sent`/`chat_received`
events — correct behavior given no lock ever formed. The greeting is
downstream of acquisition (`greeted_target_id` latch untouched all
run), so F4's fix is the HELLO fix too: first human lock → HELLO
rides the acquisition tick; delivery receipt is the `chat_received`
self-echo ([[chat-messages]]).

## F5 — Ferry as forage platform (flag 11)

User doctrine (2026-07-29, flag 11 narration): "ferries are actaully
the best way to get fuel and equipment, since you can use them to
access many equipment and fuel cannisters yu other wise couldnt. you
generally will need to teleport to the ferry since many times it will
be on its own area in the water. not touching land. but they are very
good to use." Feature, not a bug: the forage planner never considers
teleporting TO a ferry to harvest water-locked containers (the larder
scorer counts them as `no_landing`). Mechanics prerequisites already
proven in [[ferry-mechanics]] (free water movement, surface-gated
routing, water-container pickup while riding).

## Session-2 flags (bot-20260730-000030, first build with the F4 fix) + old-session flags 12–13

**F6 — Collect reachability ignores dynamic blockers (old flag 12,
23:57:00–02).** Equipment at (191,41) sat across a 2-tile water
channel from self at (194,41), and the only land route around was
choked by an enemy tank at (191,43); the collect planner's
passability view composes mines but NOT tanks, so it dispatched and
the server refused with `error_code=1` twice ((191,41) then
(190,37)), burning two actions and two failed-pickup marks. Same
lesson as the 2026-07-20 mine-veto loop recorded in `ferry.py`:
every dynamic blocker must compose into the ONE passability answer —
tanks and movable land blocks are still missing.

**F7 — Fuel locks are never re-validated (old flag 13, 23:59:24).**
After the red-4 kill at fuel 893, "continue locked fuel target at
(249,18) vol=84" drank a near-empty remainder left by an earlier
partial pickup (+84 → 965, not full), then paid a map-open plus a
teleport for the 462-volume container it had just skipped. A lock
that survives combat should be re-scored against the current deficit
and alternatives, not blindly continued.

**F8 — Acquire paid a teleport onto a 2-tile-distant enemy (new flag
1, 00:01:03).** purple-4 stood at (179,138) inside the freshly
scanned viewport with self ~2 tiles away, and the map-acquire path
teleported onto him before engaging — only `_combat_close` (which
requires an existing lock) asked the shot-range question. FIXED same
night: `_combat_teleport` now short-circuits to the shot when the
target is in-view within `SHOT_RANGE_TILES`, covering fresh, map,
and resume acquires.

**F9 — Escape/hop/shoot interleave under fire; deferred teleports
pre-empted every tick (new flag 7, 00:11:18–29).** The orange
minefield correctly walled off ALL 21 nearby containers
(`blocked_walk`, actionable=0 — the mine-composed passability doing
its job), so the escape needed the larder hop to (235,5). But each
tick COLLECT logged "deferring teleport: opening map first" and sent
`map_open`, and the NEXT tick the fresh map handed the decision to
HUNT, which shot orange-8 once ("break latch holding … continuing
escape" while actually standing still trading shots); then COLLECT
re-deferred the identical hop. Four cycles, 11 seconds, fuel 572→462
under fire before the teleport finally went out at 00:11:29. Same
family as the map-fire loop and the non-hysteretic-gate anti-pattern:
a decision that defers for a map open must RETAIN priority on the
following tick instead of re-entering open arbitration.

CORRECTION (cross-check 2026-07-30): the landing-vs-request receipt
**already existed** — the completions-side `teleport_displacement`
fired 7× in bot-20260730-000038, including a 4-in-17 s cluster
inside this same orange minefield at 00:07:27–44: (214,15)→(217,19),
(226,17)→(227,19), (220,9)→(220,2) (a 7-tile shove), and
(229,6)→(231,6); also (246,245)→(247,247) at 00:02:04,
(198,54)→(200,54) at 00:08:14, (207,7)→(207,2) at 00:09:30.
RESOLVED same day by unification, one emitter + one consumer: the
new wire-layer receipt (`_emit_teleport_displacement`, schema
`requested_*/landed_*/displacement`, zero-tolerance) is now the
SINGLE emitter of the kind — the completions-side emission (schema
`target_*/dist`, the source of those 7 historical receipts) was
removed, and completions keeps only the consumer:
`mark_move_target_failed` vetoes the bounced tile, with the 1-tile
enemy-bump tolerance. Historical analyzers reading the old `dist`
field must switch to `displacement` for runs after 2026-07-30.
Still open within F9's consumer story: displacement feeds only the
single-tile veto, not the mine belief or an area veto.

**F10 — Walk-blocked equipment gets no teleport service (new flag 4,
00:05:01).** The collect selector saw the equipment at (187,168) but
classed it `blocked_walk` → actionable=0 → generic search hop away;
only the fuel larder's hop machinery happened to land nearby later,
after which the same equipment picked up fine. Equipment cut off
from walking (water channels, mine walls) should be teleport-served
exactly like larder fuel — the user's shape: one teleport, then walk
the cluster ("we could do 1 teleport and walk to grab all 3").
Sharpening receipt (00:05:07–09): at hop-away time the belief store
showed "nearby=3 actionable=0 **blocked=3** nearest=(193,168)
blocked_walk" — all three of the user's containers were KNOWN, the
tank sat at 1087/1100 fuel needing only radars (17 vs floor 20), and
the planner still chose a blind fuel-dot hop to (204,162) over the
known cluster; the later return that claimed them proves they were
harvestable.

**Flag-by-flag map for session 2:** flag 1 → F8 (fixed; live receipt
next cycle). Flag 2 → healthy engage (user: "flag 2 is fine"). Flags
2's neighbors and 8/9's zero-delta hop streaks → F2 (user receipt:
the "Zoom in" game-log run of ~20 teleport+radar cycles for ~15
pieces of equipment around flag 9 — "poorest equipment recovery
phase ever"). Flag 3 → F5 (ferry at (252,200) in view, unused; flag
8 repeats it — "missed multiple ferries"). Flag 4 → F10. Flag 5 →
F8's shape against orange-2 at dist 7 (teleport paid where a shot —
or a 2 s walk to an open firing tile vs the ~4 s teleport, per the
user's timing law — was cheaper); the walk-to-clear-shot option is
F3's LOS module + close-range movement, still open. Flag 6 → F3
(mine shooting not yet implemented). Flag 7 → F9. Flag 10 → F2
again (post-orange-1-kill restock: ~12 dry "Zoom in" hops for ~15
pieces, user: "one of the worst ever viewport hops"). Flag 11 → F5
again (ferry at (56,20) in view, unused).

**Why known containers sometimes get a direct teleport and sometimes
a come-back-later** (user question at flag 10): a container is
picked NOW only if it is walk-reachable and passes the
walk-worthiness economics (clamped gain vs walk distance) at that
moment. An equipment hop with larder semantics has existed since
2026-07-27 (`_hop_toward_equipment`: teleport onto the container,
suppress the landing radar), but its candidate filter hid
IN-VIEWPORT walk-blocked equipment from both the walk step and the
hop step — that filter is F10's root and is being lifted so any
identified equipment that can't be walked to is teleport fair game
(one hop, then walk the cluster). F2's belief-driven hop scoring
closes the remaining inconsistency (dry search hops outranking known
containers). Flag s2-12 is the healthy contrast receipt: one radar,
four equipment containers, all collected in place.

## Session-3 flags (bot-20260730-004114, first build with F1/F2/F7/F8-tightened/F9/F10)

**s3-1 (00:42:22) and s3-3 (00:44:19) — "why did it open the map
there?" — answered, not a bug.** Both map flashes are the teleport
PRECONDITION, not searches: a wire teleport races the server's
map-open processing when sent in the same tick (run 20260610-024x
lost 4 of 15 same-tick attempts to 10 s stalls), so the executor
opens the map this tick and dispatches the teleport next tick
("deferring teleport to (65,33): opening map first" at s3-1;
identical line for the purple-4 acquire at s3-3). The server closes
the map on every shot, so the first teleport after any fight always
pays one visible map open. s3-1's ring also shows the F9 entry gate
working live: latch holding, zero shots interleaved with the escape
hop. s3-3's "then walked" was the locked fuel pickup at (229,146)
completing before the acquire teleport went out.

**s3-2 (00:43:28) — orange minefield over equipment ignored again** —
F3's third live receipt (after s2-flags 4/8). F3 is the last unbuilt
counterplay: shoot the equipment tile (private+ clears the 3×3),
teleport, collect.

**F12 — Wasted map opens: the hop's own lock pre-empts the deferred
teleport (s3 flags 4/6, plus the "shoot" case; user: "there's a bug
in our code somewhere").** Measured over session 3's first 50
minutes: 43 map opens, 36 correctly followed by their teleport, **6
followed by a fuel pickup and 1 by a shot** — those 7 map opens
bought nothing. By 00:53 the tally reached **13 wasted of 66 (~20%)
at a steady rate** (flags s3-4/6/9/11 all this shape) — a
deterministic mechanism, not a race. Companion receipt from flag
s3-10 (00:51:55): the walk-worthiness skip working CORRECTLY under
fire — a 212-fuel container 14 walking tiles away was refused
mid-escape at 27/tick incoming (~190 fuel of exposure to gain 212)
and the escape teleported instead, breaking the firing geometry;
user confirmed the question, receipts confirm the answer. Root: a larder/equipment hop decision LATCHES its
resource lock in the same tick the executor converts the teleport
into the mandatory map open ("fuel larder hop to (227,156) …
deferring teleport … opening map first", 00:45:06); the next tick's
collect cascade reaches the lock-CONTINUATION branch before the hop
branch and dispatches a walking pickup to the same container
("continue locked fuel target at (227,156)", 00:45:08) — the
teleport the map was opened for never flies, and for far containers
the walk is also slower than the paid-for hop. Same design-law
family as F9: a decision deferred for a map open must survive the
tick boundary. Fix direction: the lock-continuation branch must
re-issue the deferred teleport when the lock it is continuing was
created BY that hop and the landing is beyond walk-worthiness — or
the hop must not latch the lock until its teleport actually
dispatches.

Flag s3-9 (00:51:37→39) is the cleanest receipt and adds a second
sub-defect: the larder hop chose a cost-16 TELEPORT to a container
**4 tiles away** (self (77,201) → (79,203)) in the same breath as
refusing a 9-tile walk for 136 fuel — the larder hop never consults
walk-worthiness at all, so short trips that should be walks become
teleport+map_open pairs, and the lock-continuation flip then
"corrects" it at the price of the wasted map open. The larder step
needs the same walk-vs-teleport economics the pickup step has
(teleport ~4 s vs walk ~2 s, user timing law): inside walking range,
walk; the hop is for genuinely far remainders.

**F1 residual (s3 flag 7, 00:49:02–07; user: "we have that extra
teleport still… hit the equipment quota, teleport to one last
viewport, not scan, then teleport to the target").** The pre-acquire
top-off leg was a `fuel_larder` hop to (211,178) — larder hops
suppress the landing radar by design (the "not scan"), and the
landed F1 fix biases only DOT hops toward the prey; larder
candidates carry no hunt bias (`hop_selected` for fuel_larder has no
`hunt_biased` field) and the exact-capacity hunt floor still forces
the leg. Residual fix: extend the hunt-ready direction bias to
larder candidate selection, and settle the walk-heuristic vs
exact-capacity disagreement (accept clamped gains that COMPLETE hunt
readiness).

| # | Finding | Flags | Status |
|---|---|---|---|
| F1 | Top-off hop direction-blind before acquisition | 1, 2, 6 | FIX LANDED (2026-07-30: `_pick_fresh_dot_hop` scales dot scores by proximity to the nearest alive enemy when stocks are hunt-ready — `hunt_biased` field on `hop_selected`; live receipt pending) |
| F2 | Zero-yield hop churn (TTL-as-harvest-memory, yield-blind score) | 3, 5, 7, 9, 10 | FIX LANDED (2026-07-30: harvest-memory veto — a landing viewport whose believed containers are all drained within `HARVEST_MEMORY_TTL_MS` (10 min, unbracketed respawn assumption) is skipped with a `known_empty` tally even when the 180 s scan TTL reads clean; live receipt bot-20260730-004144: 36% zero-yield (8/22) vs 63-64% in sessions 1-2 — nearly halved, further gain expected as beliefs populate) |
| F3 | Shot-clearance system + mine-covered pickup counterplay | 4, 8, s2-6, s3-2, s3-14 | FIX LANDED + LIVE RECEIPT (s5-2: clearance shot → pickup → clearance shot → fuel hop onto the exposed container, 02:00:04) |
| F4 | Ferry-rider/mine-ring cloak at acquisition + stale-refresh rule not firing | Yuppler report | FIX LANDED + LIVE RECEIPT (s5-5, 02:04:17: lock on Yuppler + HELLO self-echo + lock-held fuel-chain approach; stale-refresh path still to verify) |
| F5 | Ferry-as-forage-platform: teleport to ferries, harvest water-locked containers | 11, s2-3, s2-8, s2-11, s3-15? | FIX LANDED for the fuel larder (2026-07-30: `ferry_landing.py` — a water-locked container with a fresh believed ferry within 12 tiles boards it as the landing, `ferry_served` tally; ride+pickup rides the existing lock machinery per the riding-pickup law; ferry beliefs gated at 60 s. Equipment-hop ferry service + exploration ferry hops remain OPEN) |
| F6 | Collect reachability ignores tanks/land blocks (server code=1 refusals) | 12 | OPEN |
| F7 | Fuel-lock continuation ignores dwindled volume | 13 | FIX LANDED (2026-07-30: `is_fuel_lock_release_warranted` — deliverable-score release path (min(volume, deficit) − distance, 2× hysteresis) alongside the distance rule; flag-13 shape pinned in `test_fuel_lock_value.py`; live receipt pending) |
| F8 | Acquire teleported onto an in-view enemy | s2-1, s2-5, s2-13 | FIX LANDED and tightened (`89459dd7` in-view shot short-circuit; `78b1483e` drops the 8-tile bound per flag s2-13 — in-view alone is the firing criterion at any range, which also covers the s2-13 dist-9 shape without a walk step and folds the mine-ring/ferry stand-off into the same rule; live receipt pending next cycle) |
| F9 | Deferred-teleport/map/shoot oscillation under fire + no teleport-displacement receipt | s2-7 | FIX LANDED, both halves (`4f63aa37` break latch gates every HUNT phase at entry; `fa93c47b`/`9acbe0be`/`a4c55419` requested-vs-landed `teleport_displacement` receipt); emitter UNIFIED 2026-07-30 — the pre-existing completions-side emission (schema `target_x/dist`, Chebyshev, the source of the run's 7 receipts) was removed so the wire-layer receipt is the single emitter of the kind; completions keeps ONLY the consumer (`mark_move_target_failed` tile veto + 1-tile enemy-bump tolerance); live receipts pending next cycle |
| F10 | Walk-blocked equipment never teleport-served (larder-style) | s2-4 | FIX LANDED (2026-07-30: `_hop_toward_equipment` external-only filter removed — the step runs after walk-pickup declines, so ALL tracked equipment including in-viewport walk-blocked is teleport fair game; live receipt pending) |
| F11 | Over-terrain homing spent where a reposition bought a clear-LOS dual (weapon economy) | s2-14 | OPEN (user ruling banked; needs F3's lifted LOS test + an ENGAGE reposition step) |
| F12 | Wasted map-open on deferred-then-replanned teleports (collect side) | s3-1, s3-3 | OPEN (receipt: tick 68 map_open produced no teleport — the fuel lock replanned to a pickup next tick; same defer-loses-priority family as F9) |
| F13 | Short-range teleports where walks win + dreg hops with near-zero net gain | s3-1, s3-3, s3-4, s3-9 | FIX LANDED for the larder (2026-07-30: `_WALK_DOMINANT_RANGE` Manhattan<=2 owns near containers, `_LARDER_MIN_GAIN` 100 floors dregs — waived when the gain completes the deficit; desperation stays with `_desperation_fuel_hop`, whose reserve gate already owns fuel<=threshold; new `dreg` tally on hop_declined; live receipt pending). Combat-close short teleports remain OPEN (hunt-side) |
| F15 | Quota met but free in-viewport equipment left behind at hunt handoff (user ruling s3-13) | s3-13 | OPEN (arbitration-order fix: drain walk-worthy in-viewport pickups before yielding to HUNT) |

Update this table as fixes land; close a row only with a live-run or
sim receipt cited next to it.

[^1]: flag rings in `bot-20260729-232252.events.jsonl`: flag 1 tick
    49 (hop (14,81)→(7,93), acquire red-9 →(3,105)); flag 2 tick 106
    ((49,35)→(70,20)→(48,43)); flag 6 tick 230 ((92,58)→(72,79)→(21,77)).
[^2]: measured over the same events file: 211 `hop_selected` (161 dot
    / 30 equipment / 25 fuel_larder), gains counted between
    consecutive hops; 133 windows had zero `equipment_gain` +
    `fuel_gain` events.
[^3]: user domain knowledge, 2026-07-30 flag narration (verbatim
    rules recorded in [[mine-mechanics]]); flag 4 tick 152 at (6,146),
    flag 8 tick 314 at (217,19).
[^4]: `acquisition_candidates` events ticks 428–661, filtered to
    tank_id 1229; picked column shows red-6/red-8/purple-8/red-2/
    purple-7/red-5/purple-5/purple-2 chosen instead.
[^5]: flag-1 lead-up in `bot-20260729-232252.events.jsonl`,
    23:24:31–23:24:41: pickups at (19,73)/(13,81) complete the stock,
    the 17-point fuel top-up at (21,78) is skipped as not worth the
    walk, hop to dot (7,93) dispatches at 23:24:37, landing
    auto-pickup reaches 1100, and the COLLECT→HUNT `mode_transition`
    fires at 23:24:39 with the red-9 acquisition.

**Flag s2-13 (00:31:37) → F8, closed by dropping the range bound.**
After a mid-fight pickup at (32,64) the bot teleported from (33,64)
back onto purple-9 at (38,60) — Manhattan 9, in view, one tile
outside the radius-8 short-circuit (user: "they could have shot from
where they were"). Resolution (`78b1483e`): the 8-tile bound never
existed in the user's law ("as long as theyre on the viewport…"), so
`_combat_teleport` now shoots any viewport-visible target at any
range — no walk step needed for in-view targets. The
walk-to-a-clear-shot move stays relevant only for F11's LOS case
(terrain in the shot line), not for range.

**Flag s2-14 (00:33:22) → F3's LOS module + a NEW weapon-economy rule
(user narration):** after the mid-fight equipment pickup the bot spent
a HOMING over terrain at red-8 ("you can see there was dual shots,
then 1 homing shot after picking up equipment. the bot should've
moved so it had a clear shot"). Ruling: over-terrain homings are a
last resort — when a short reposition buys a clear-LOS dual, move
first and shoot the cheap weapon. Needs the same lifted LOS test as
F3's mine clearance (terrain + land movables per shot line), plus a
reposition step in ENGAGE when LOS is dirty. The mid-fight
restock/refuel interleave itself (fuel 624→1090 on a 481 drink
between shots) drew no user objection — only the weapon choice did.

**Flag s2-15 (00:35:15) → explained, working as designed** (user:
"the bot going to the random map location after it had collected
equipment ... before going to fight an enemy"). The "random location"
was a `dot_relay` leg: red-9 sat ~117 tiles away, a direct approach
teleport would have broken the 650-fuel engagement reserve
(fuel-before-chasing, `22061123`), so the chain paid one fuel-dot leg
TOWARD the target — (238,142)→(175,131) closes the distance to 43 and
refuels on the landing — then the affordable CLOSE teleport engaged.
Not a defect; a legibility gap. The HUD shows `dot_relay` as the
reason but a watcher tracking the map can't tell a relay leg from a
wasted hop — candidate HUD tweak: render relay legs as
"RELAY→red-9 (leg 1)" instead of the bare reason string.

## Session-3 flags (bot-20260730-004144, first build with ALL fixes live)

**Flag s3-1 (00:42:27, tick 14), pending user narration:** between-kills
restock after the orange-5 kill — drop collected, then the fuel larder
teleported 3 tiles ((65,30)→(65,33), cost 18, `fuel_locked` vol 718)
where a 3-tile walk was available. If flagged for the short teleport,
this is the walk-vs-teleport family (F8/s2-13) in its larder form:
`_hop_toward_fuel_larder`'s `too_close` gate evidently permits dist-3
hops; the walk-vs-teleport economics (2 s/walk-tile vs ~4 s + 6/tile
fuel per teleport) belong in ONE shared movement-choice rule rather
than per-caller gates.

**Flag s3-2 (00:43:xx, tick 44), pending user narration + session-3
health readout (ticks 28-46):** the F10 equipment hop is live and
working — tracked-cluster harvesting at ticks 34-40 ((225,17) →
(220,10) → (228,6), each landing collected) where session 2 dot-hopped
away. Two `empty_container` rejections (ticks 33, 41) both dispatched
ONE TICK after a fresh radar: check whether radar 0x5A patches ever
ZERO a drained container or only assert present ones — if empties are
wire-invisible, in-viewport beliefs can only be corrected by the
failed pickup itself (which worked: neither tile was retried). Flag 2
clicked at a dot hop to (250,11) right after draining (235,3) to 4
remaining — awaiting the user's read.

**Flag s3-3 (00:44:22, tick 70), pending user narration:** ring shows
the s3-1 shape again — a ~5-tile `fuel_hop` teleport ((231,150)→
(230,146)) during restock, then a legitimate 20-tile CLOSE onto
purple-4. Short-range larder teleports where walks would do are now
the recurring session-3 theme (s3-1, s3-3): the walk-vs-teleport
economics belong in one shared movement-choice rule (2 s/walk-tile vs
~4 s + 6 fuel/tile teleport, break-even ~2 tiles) applied by larder
hops, fuel hops, AND combat closes alike.

**Flag s3-6 (tick 168) + the session-3 empty_container cluster → F14
candidate.** Six `empty_container` rejections in 168 ticks (t33, 41,
89, 108, 161, 163), every one a pickup dispatched on a belief 15-20
ticks (~30-40 s) past its last scan, two of them beside the user's
ferry ground ((127,102), (130,100)). Mechanism: a HUMAN sharing the
room drains containers and no wire signal reaches a bot that didn't
cause the pickup ([[server-push-gating]]) — so in shared rooms,
equipment beliefs rot in ~30 s. The failed-pickup blacklist self-heals
at one wasted tick each (no tile was retried). Fix direction: a
belief-age gate on pickup dispatch — beliefs older than ~30 s require
a cheap re-verify (the landing scan already in flight, or ordering by
freshness) before walking/teleporting to them; do NOT add retries.

**Flags s3-4, s3-5, s3-7 (pending user narration), ring shapes:**
s3-4 (00:46:01) — engagement break from purple-3 at fuel 869, then a
TWO-TILE `fuel_hop` teleport to a 94-volume dreg (F13's shape at its
worst: dist 2, walkable in 2 ticks, and the dreg's cheap cost is
exactly how a tiny container outbids walking); s3-5 (00:46:39) —
healthy-looking cluster harvest (three equipment pickups + forage
radar), then a 10-tile fuel_hop to a 936-volume container, awaiting
the user's objection; s3-7 (00:49:06) — the s2-15 relay shape again
(top-off → `dot_relay` (212,178)→(144,166) → CLOSE onto red-1 at
(114,186), ~100-tile pursuit): if the flag is "why that far a
target," the acquisition_candidates record at tick ~206 needs reading
against nearer roster options.

**Flag s3-8 (00:50:14, tick 241), pending user narration:** adjacent
duel with red-1 ends, then HUNT/REFRESH opens the MAP before
collecting the kill drops (pickups follow two ticks later, and the
tick-246 `empty_container` suggests one drop expired or was taken in
that gap). If the flag is "why the map mid-fight": REFRESH re-locates
the target roster after a kill, but its ordering ahead of
drop-collection costs the freshest pickups — candidate rule: drops
first, map second.

**Flag s3-9 (00:51:40, tick 283), pending user narration:** the
clearest dreg-chain receipt — 2-tile teleport to a 35-volume dreg
(net ~23 fuel after teleport cost), then a legitimate 31-tile 355-hop,
then another 2-tile hop to a 276. `min(vol, deficit)/cost` favors
close dregs structurally; F13's fix needs BOTH the walk-vs-teleport
rule and a net-gain floor on larder hops (deliverable must clear the
teleport cost by a real margin, not merely divide well by it).

**Flag s3-13 (00:59:05, tick 499) → NEW RULING, F15 candidate (user
verbatim): "we could have picked up two other containers and gotten
to full inventory even though we hit 20 radars and met the quota, but
they were right there on the viewport honestly. before we fought
orange-6."** The hunt-readiness floor (radars >= cap-5) is a FLOOR,
not a stop-collecting order: when quota completes with walk-worthy
equipment still in the CURRENT viewport, top off those freebies
before yielding to HUNT — the ring shows radar 20/25 at quota,
immediate acquisition teleport to orange-6, and the two in-viewport
containers left behind. Fix direction: the yield-to-hunt gesture
(collect exhausted / hunt_entry_permitted handoff) should first drain
in-viewport walk-worthy pickups — the walk economics already exist;
only the arbitration order changes. Complements F1's residual
(exact-capacity fuel floor) — both are "quota met, but free value in
reach" shapes.

**F13 — Hunt entry leaves easy in-viewport pickups behind (s3 flag
13, 00:59:05).** Radars reached exactly the cap−5 floor (20/25) with
weapons full and fuel 1100, and hunt entry fired the same tick —
teleporting to orange-6 while collectable equipment sat in the
current viewport ("we could ahve picked up two other cotainers and
gotte nto full ivnentory even though we hit 20 radars and met the
quota, but they were right there on the viewport"). The entry gate
is a hard threshold with no look at what one more WALK would buy.
Fix shape: when hunt-readiness first flips true, drain the current
viewport's actionable walk-reachable equipment before the acquire
tick — walk-only (zero teleports, so F1's extra-hop problem cannot
return), bounded to the viewport. Task #40.

**Flag s3-12 (00:57:08) — "stuck with extra map opens":** F12
compounding under fire — fuel bled 1064→653 across 00:56 while hop
decisions kept re-deriving across map-open tick boundaries,
including a cost-18 hop to a 52-volume container it had just refused
to walk 4 tiles for. Strongest receipt yet that F12 plus the shared
walk-vs-teleport movement rule is the top open economy fix.

**Page hygiene note:** two concurrent AI sessions triaged session 3
in parallel — the two "Session-3 flags" sections above overlap on
flags s3-1/3/9 (same walk-vs-teleport family read) and should be
merged into one section on the next quiet pass.

**Flag s3-15 — the F1-residual leg, caught mid-waste:** quota met →
COLLECT search hop teleports out → HUNT preempts BEFORE the landing
radar → acquire teleport to the enemy ("interrupt before radar and
teleport to the target. wasted teleport there"). The interrupt is
the proof that the hop's only value was the landing auto-pickup
buying the last fuel points to exact capacity — the leg exists
because of the exact-capacity floor and dies unused the moment it
has served it. Task #39's settlement (accept clamped gains that
complete hunt readiness / direction-aware final leg) closes it.

## F16 — The lethal-pressure dead zone: Artax died standing still (01:06:55)

The first bot death to a human this campaign, and the break math WAS
right: at fuel 216 under 72/tick incoming (Yuppler's real damage —
2.7x the 27/tick practice-bot rate the thresholds were tuned on), the
projection said fuel −704 at kill, and the deeper check said the
fight was unwinnable at ANY fuel (needs 1364 > 1100 capacity) →
blocked the target. The fatal gap: **blocking is not fleeing.** The
tick fell to COLLECT where fuel 216 sits in a structural dead zone —
every hop reserve-blocked (216 − cost < 200 for anything past ~2
tiles), desperation locked until fuel ≤ 200, walk-for-fuel likewise —
so the bot had no legal move and tanked four more hits to death at
(168,94). Companion defect: the final shots fired "toward last wire
position" (167,95) with over-terrain homings at half damage while
Yuppler stood adjacent.

USER RULING (2026-07-30, superseding the first fix direction): "the
bot can fight against a human and win... it should have collected
fuel and then kept fighting and collected as necessary. a dual shot
is just two single shots." The "unwinnable at any fuel" verdict is a
FALSE THEOREM twice over: (1) it models a static slugfest with
capacity as the fight budget, but fuel is replaceable MID-FIGHT
(receipt: the 481-volume drink between shots vs red-8, flag s2-14)
and tanks never heal except by pickups — against a human the fight is
a LOGISTICS WAR the tighter refuel loop wins; (2) the projection's
own output was halved by the over-terrain-homing defect (flag s3-16,
since fixed: occluded line → close for full-damage adjacent duals),
inflating hits-to-kill. Fix direction: DELETE the
unwinnable-at-any-fuel → block path — blocking is reserved for
physical unreachability (no standoff landing), never damage
arithmetic. The break verdict chains into the fight-refuel-refight
loop: disengage toward the nearest reachable fuel (the same
engagement-break machinery that already exists), top up, re-engage
while the lock holds. Flight/quit-to-lobby remains ONLY for the true
lethal case: death projected before any refuel pickup can land (no
reachable fuel inside the survival horizon) — and there the reserve
is void, since a dead tank holds no reserve.
Receipts: bot-20260730-004144 lines 17577–17607.

| # | Finding | Flags | Status |
|---|---|---|---|
| F16 | Death by standing: unwinnable-verdict blocks instead of refueling-and-refighting; reserve/desperation dead zone (fuel ~200–250) leaves no move under real human damage | Artax death 01:06:55 | OPEN — top priority; user ruling: delete arithmetic blocking, chain break → refuel → re-engage; flee/quit only when no refuel lands inside the survival horizon (reserve void there) |

## Session-4 flags (bot-20260730-012133, full fix stack: in-view law + larder dreg/walk/ferry/net-reserve)

**Flag s4-1 (01:23:24, tick 48), pending user narration:** the ring is
the F16 doctrine working — mid-fight refuel at 674 onto an 851-volume
container (→ 1100), teleport back onto orange-7, point-blank duals.
If the flag is the 6-tile teleport back after the refuel, that is the
walk-vs-teleport boundary case (beyond the 2-tile walk-dominant
range; a 6-tile walk costs 6 ticks vs the teleport's 2 + 36 fuel —
teleport arguably correct mid-fight where ticks are damage windows).

**Flag s4-2 (01:26:45, tick 146), pending user narration:** equipment-hop
chain through the orange minefield zone ((217,19)→(226,17)→(220,9)),
and the (220,9) hop landed at (220,2) — a 7-tile displacement in the
SAME field that shoved session 2 four times. The
`teleport_displacement` receipt fires but still has no consumer
(F9 remainder): nothing feeds the shove into mine beliefs or the
landing chooser, so the bot re-aims into the field. Same ground is
F3's target zone — mine-shot clearance (in build) + the displacement
consumer together retire this shape.

**Flag s4-3 (01:28:54, tick 211) → F6/F9 convergence, fix specified.**
Three consecutive `cant_go` walk rejections (ticks 208-210, targets
(160,98)/(158,94)/(162,94)) after a DISPLACED larder landing
((165,100), `landed_inexact`) in Yuppler's old fight zone: the field
shoved the landing — proof of unobserved mines — but larder hops
suppress the landing radar by design, so the bot walked blind and the
failed-target rotation burned three ticks. First concrete consumer
for the F9 displacement receipt: **a displaced harvest landing
un-suppresses the landing scan** — one radar reveals the field, the
mine-composed passability then vetoes the doomed walks pre-dispatch.
The 2026-07-27 "harvest hops never spend a radar" ruling holds for
CLEAN landings; a displacement is the server saying the ground is not
what you believed. FIX LANDED (2026-07-30 01:35): a suppressed
harvest landing standing >1 tile from its lock fires the landing
radar with the LOCK KEPT, and the landing-scan gate moved ahead of
lock continuation in the cascade (the 2026-07-03 "radar before any
pickup" policy, which the old order violated for held locks — s4-3's
blind walks came from exactly that). 737 bot-ai tests green.

**Flags s4-4 (01:31:41) and s4-5 (01:35:06) → narrated: the terminal
guaranteed miss.** User: "we did 7 homing shots, and 1 missed shot...
we have the ttl we can use. couldnt we avoid the missed shot
entirely?" (s4-5 "same scenario"). The pursuit fired homings at a
departed target past the ~12 s reroute/trace wall, booking one
guaranteed miss + wasted tick per pursuit. FIX LANDED (parallel
session): `pursuit_trace_is_live` (PURSUIT_TRACE_TTL_MS = 12 s on
last_viewport_observation_ms) gates the pursuit shot — past the wall
it skips straight to the map chase the miss would have triggered
anyway.

**Flag s4-3 user narration (received after the fix landed):** "we hit
mines that hadnt been detected. we were walking and ran into mines...
we could have used radar to detect them, we hit three mines that had
not been revealed" — the user independently named the fix's mechanism
(radar before walking mined ground). The walk-over hits (45 fuel
each) are the same blind-ground failure as the traced cant_go
rejections; the displaced-landing radar covers the landing case, and
the user also re-confirmed F12's law: "we should really only ever
have map open when we are teleporting."

**Flags s4-6 (01:36:34) and s4-7 (01:37:00), pending user narration:**
s4-6 — chained `dot_relay` legs at FULL fuel ((252,125)→(223,176)→
(239,242)) with a map_open between: the designed relay chain crossing
the map to a distant target; if flagged, it is the s2-15 legibility
gap again (relay legs read as random hops) or the between-leg map
churn. s4-7 — mid-pursuit refuel loop working (scan → 339-volume
fuel_hop, 724→1051 → CLOSE onto orange-2), with an `empty_container`
one tick earlier (F14's ninth+ occurrence this night).

**NUMBERING NOTE (audit, 2026-07-30 01:45):** F14 on this page =
shared-room container belief rot (s3-6 cluster). The parallel
session's "walks crossing unscanned ground should spend an available
radar first" finding (s4-3's three walk-over mine hits) is hereby
**F17** — it was announced as "F14" in that session's chat before
reaching this page. One number, one finding.

| # | Finding | Flags | Status |
|---|---|---|---|
| F17 | Walks across unscanned ground don't spend an available radar first (three undetected mine hits, 45 fuel each) | s4-3 | OPEN (parallel session's queue; complements the landed displaced-landing radar, which covers the landing case only) |

**Flag s4-10/11 (01:52, ticks 798-805) → CORRECTED to F6's decisive
receipt (user narration overturns the first read).** The six
consecutive `cant_go` rejections were NOT unscanned ground: "artax
could see the mines since he used radar, but the pathing system
didnt realize that he was locked in by yuppler and by the revealed
purple mines." The user (as Yuppler) mined Artax in, then stood on
the opposite side — a deliberate box: mines composed into
passability, but YUPPLER'S TANK does not compose (F6's exact missing
half), so the walk planner kept dispatching through the occupied
tile, six rejections while eating hits ("i couldve killed him but i
stopped shooting"). Two lessons: (1) tanks must join mines in the ONE
composed passability answer (F6, now top priority with this receipt);
(2) a fully-enclosed tank should recognize enclosure and act by the
SETTLED ESCAPE DOCTRINE (user Q&A 2026-07-30): **under fire or with
an adjacent enemy, TELEPORT to a fuel container ON THE CURRENT
VIEWPORT** (user refinement, verbatim: "a teleport to the fuel
container on that viewport would be best. since it takes the same
number of ticks as shooting mines and then walking (2 ticks)") — the
same 2 ticks buy escape + refuel + staying local for the re-engage,
where shoot-wall-then-walk exits into the enemy's kill zone at
walking speed. No in-viewport fuel → nearest affordable fuel dot
beyond it. **With no pressure, shoot the mine wall** (free shot) and
walk — saves the teleport fuel. One rule, keyed on incoming fire /
enemy adjacency; landing preference: in-viewport fuel first.

**Flag s4-12 (01:50:50, tick 849) → narrated: FRIENDLY FIRE → new
F18.** User: "flag 12 was a friendly fire shot. how did this occur
and howcome we didnt realize this." The tick-817-family
`friendly_fire` (0x52 code 3) rejection: the shot at orange-4's
last position was refused because an ALLY stood on/near the aim —
shot dispatch checks neither ally occupancy at the aim tile nor
allies on the line, and the rejection was absorbed silently (no
re-aim, no diagnostic beyond the command_error). Fix direction: the
aim selector vetoes ally-occupied aim tiles (registry has team data
in-viewport), and a friendly_fire rejection must mark the aim stale
and force a re-acquire tick, mirroring the cant_go → failed-move
marking.

| # | Finding | Flags | Status |
|---|---|---|---|
| F18 | Shot dispatch ignores ally occupancy (friendly_fire rejection absorbed silently, no re-aim) | s4-12 | OPEN (aim-tile ally veto + rejection consumer) |

**Flag s4-13 (tick 930), pending narration:** clicked one tick after
an `empty_container` rejection mid-COLLECT — presumed F14 (belief
rot) receipt; the count this session alone is ~12.

## Session-5 (first build with mine clearance + all fixes live)

**Milestone:** first `mine_clearance_shot` dispatched at tick 24 —
F3 executing live, three sessions after the first ask.

**Flag s5-1 (01:58:37, tick 72), pending narration:** ring shows the
new restock chain working — fuel lock drink to 1100, equipment
pickup, 21-tile equipment hop, landing pickup under the held lock.
Awaiting the user's objection if any.

**Flag s5-2 (02:00:04, tick 115) → F3's LIVE RECEIPT.** The ring is
the user's doctrine executing verbatim in the SAME mined tiles that
ate six cant_go rejections in session 4 ((158-162, 94-98)):
`mine_clearance_shot` at (160,98) → equipment collected at (162,97) →
second clearance shot at (162,94) → `fuel_hop` onto the newly exposed
453-volume container. Shoot the cover, collect the goods. F3's
fix-status row can take the live receipt pending the user's read.

**Flag s5-3 (02:01:45, tick 165) → the ORANGE FIELD receipt.** The
specific minefield flagged since s1-8 ("huge swathe of orange mines")
is being harvested: clearance shot at (225,17) → equipment collected
under it → fuel hop onto the exposed 1048-volume container at
(221,20) → full tank → straight into a HUNT acquisition. Four
sessions from first flag to farmland.

**Flag s5-4 (02:02:38, tick 192), pending narration:** another clean
chain — landing radar, equipment pickup, 14-tile fuel hop onto an
808-volume container, full tank, hunt close on orange-8. Nothing
anomalous in the ring; awaiting the user's read.

**Flag s5-5 (02:05:03, tick 266) → F4's LIVE PURSUIT RECEIPT + first
HELLO.** The ring holds lock `#1229` (Yuppler) throughout: acquisition
locked the human at 02:04:17, the greeting fired on the acquisition
tick (`chat(41)` → `chat_received` self-echo — the delivery receipt),
and the approach is fuel-chaining with the lock held (156 drink →
249 hop → 813 hop, fight-refuel-refight). The acquisition that
rejected Yuppler 11/11 times in session 1 now locks, greets, and
closes. F4's row takes the live receipt.

**Flag s5-6 (02:05:47, tick 288) → F16's LIVE COMBAT RECEIPT.** The
Artax-death scenario replayed against the same human and SURVIVED:
Yuppler's fire crashed fuel to 263 (the old dead zone killed at 216
with no legal move), and the net-of-gain reserve let the refuel chain
fire with the lock held — 400-vol hop → 543 → 898 → 302-vol hop →
1055, still hunting #1229 throughout. Fight-refuel-refight under real
human pressure, exactly as ruled after the death. F16's collect half
takes the live receipt; the hunt-side verdict rewire (delete
arithmetic blocking) remains the open half.

**Flag s5-7 (02:06:36, tick 312), pending narration:** between-fights
restock with the Yuppler lock HELD (#1229 threaded through every
tick) — equipment hops + pickups + a 567-volume drink to full, then
more equipment, rebuilding weapons for the re-engage. The
"finish the kill then the human is the next target" contract shape,
now in its restock phase.

**Flag s5-8 (02:10:50, tick 439), pending narration:** a 12-tile
mine-clearance shot into the purple field at (147,93) — the user's
own mines from the fight — then equipment hop, lock pickup, forage
radar. F3 clearing hostile player mines at range, as specified.

**Flag s5-9 (02:12:18, tick 483), pending narration:** mid-fight vs
orange-9 — a 189-volume refuel hop between shots (825→1010) then
resumed fire from range 5. If the objection is the non-adjacent
shooting economy (homings at range vs closing for duals), that is
F11/F8's open remainder; the refuel interleave itself is the ruled
doctrine.

**Session-5 narration batch (flags 4-9):**
- **s5-4 → answered from the ring:** all stocks 25/25 throughout; the
  "collect ran longer than it should" is the relay-leg oscillation —
  top to 1100 → `dot_relay` leg costs ~433 (lands at 667) → the
  exact-capacity hunt floor hands the tick back to COLLECT → rebuild
  to 1100 → next leg. Fix = the F1-residual's hunt-floor tolerance
  mid-relay (a leg's landing refuel should suffice to continue).
- **s5-5/6 → F17's true shape + the fuel race (user narration):**
  "i placed mines down AFTER artax did radar. artax ran into the
  unrevealed mines multiple times... and then i stole the fuel that
  artax was going to." Post-scan mine placement invalidates coverage
  (no re-scan under fire), and an approach target drained by the
  enemy mid-walk is the adversarial half of F14. Both are receipts
  for F17 (parallel session's queue).
- **s5-7 → two design items banked, no defect:** Yuppler at one-shot
  HP teleported onto a collecting Artax and survived because COLLECT
  held the tick. User's own read: Artax was at ~5 duals — "no kill
  potential without duals" — so NOT interrupting collect was probably
  right; the open ideas are (a) an opportunistic-kill exception when
  an adjacent enemy is at lethal-range HP and duals remain, (b) the
  bot NEVER USES ARMOR SHIELDS ("if i enabled shields i could've
  tanked the shot") — shields as a defensive capability is unbuilt
  entirely; candidate F19.
- **s5-8 → clearance needs a value gate:** "why did it shoot the
  purple mines on top of the 21 volume fuel container?" The clearance
  scorer counts exposed CONTAINERS, not exposed VALUE — a 21-volume
  dreg is not worth the shot tick. Fix: equipment always qualifies;
  fuel only above a floor (share `_LARDER_MIN_GAIN`'s bar). Candidate
  for whoever holds mine_clearance.py (parallel session, hot).
- **s5-9 → F12 receipt #7** (map open then walked to fuel on the same
  viewport).

| # | Finding | Flags | Status |
|---|---|---|---|
| F19 | Armor shields never used (defensive capability unbuilt) | s5-7 | OPEN (candidate — user musing, needs a ruling on when shields fire) |

**Session-5 flags 4–9 (bot-20260730-015x, the early-wake + clearance
build).** Flag 4 (long collect at full/near-full): investigation
pending, ring banked. Flags 5/6 (fight vs Yuppler): the user placed
mines AFTER Artax's radar and Artax walked into them repeatedly
(movement arrested each time) — key question for F14/F6: Yuppler's
0x4B placements inside our viewport SHOULD land in our mine layer
without a re-scan; if they did, the walk-dispatch legality ignored
them (F6's gap); if they didn't, viewport 0x4B decode has a hole.
Flag 7 doctrine seeds (user narration): shields exist and would have
saved Yuppler at 1 HP — the bot never shields (task opened); a
collect-mode bot under attack chose to keep collecting, which the
user leans toward keeping ("im not sure if we wanna interrupt
collecting"), and duals are the kill resource — Artax at ~5 duals
"is in a bad spot... no kill potential without duals" (duals-floor
task opened). Flag 8 FIXED same hour (`76f34832`): clearance shots
now require a deliverable (fuel ≥ 100 volume; equipment always).
Flag 9: another F12 map-open-then-walk receipt.

## Session-6

**Flag s6-1 (02:17:51, tick 13), pending narration:** orange-field
harvest chain — equipment hop, lock pickup, clearance shot at
(215,5), then a 2-tile equipment hop. Note the short hop is likely
CORRECT here: equipment hops fire only after walk-pickup declines, so
a 2-tile hop means the 2-tile walk was mine-blocked — teleporting
over the wall is the right move (unlike F13's fuel dregs, where the
walk was open).

**Flags s6-2 (02:18:39) and s6-3 (02:19:05), pending narration:**
s6-2 — chase cycle vs orange-6: ranged shot, map refresh, 16-tile
close, landing radar, adjacent dual; reads as the intended pursuit
with the new walk-close law only applying at short range.
s6-3 ring to follow in the record.

**Session-6 flags 1–6 (bot-20260730-021x).** Flag 1: a clearance
shot the LOS module called clear was visibly blocked by terrain —
the Bresenham raster steps around diagonal-corner cells the game's
ray evidently touches; supercover + empirical ray calibration task
opened (#44). Flag 2 ANSWERED from the wire: homings are not broken
— the window's homing (weapon=3, victim 532, ammo consumed) and
dual both hit; the misses were weapon-0 SINGLES resolving at tiles
the moving orange-6 had just vacated. The server serves point-aims
at in-view MOVING targets as ground singles, not homings — the
aim-staleness-against-movers defect (also seen in the Yuppler death
aims), now with a clean receipt. Flag 5 is the control: stationary
target, homings hit. Flag 4: repeated shots at red-2 DID follow,
but after a teleport the user judges unnecessary from (124,19) —
hypothesis: the post-landing viewport record lags, suppressing the
in-view short-circuit; folded into the calibration task. Flag 3
ring banked pending narration. Flag 6: third ferry-boarding-walk
receipt for F5 (ferry adjacent to land, equipment beyond).

**s6 narration batch (flags 1-6) + homing-range hypothesis:** s6-1 —
clearance shot blocked by terrain (user: "audit and review our clear
shot code"; parallel session investigating the LOS test). s6-2 vs
s6-5 — homings MISSED 5-in-a-row at Manhattan 30 ((231,98)→(220,117),
victim -1 every time) but HIT at Manhattan 23 ((182,27)→(167,35),
victim 532): candidate law REVISED with flag s6-7's data (hits at 18 AND
23; misses 4x at 24 and 5x at 30, every miss a repeated fixed-aim
shot at a departed target's stale coords): not a range bound — the
misses are pursuit fire whose homing reroute is not attaching to the
moved target. Question for the corpus: does the 0x53 reroute require
the target inside SOME bound (viewport? trace age?) that these
pursuit shots exceeded — and should pursuit fire stop after the FIRST
miss instead of repeating the dead aim (5x at (220,117)). s6-4 — "could hit red 2 from (124,19), why teleport
back": the walk-close/in-view boundary again (F8/F11 family). s6-6 —
ferry adjacent to LAND, equipment on water: F5's WALK-boarding case
(no teleport needed), second receipt after s5-1.

**Flag s6-8 (tick 243) + live cluster:** five `cant_go` rejections in
HUNT/ACQUIRE (ticks 236-245) — F6's shape again on the newest build
(the displaced-landing radar covers harvest landings only;
between-kills restock walks still dispatch blind into
tank/unobserved-mine blocks). F6's tank-composition build is now the
single most receipt-heavy open item; every session adds a cluster.

**AUDIT FINDING F20 (2026-07-30 02:27, cross-session handoff): the
walk-close candidates NEVER CONSULT TERRAIN.**
`combat_landing.combat_landing_candidates` filters only
`_is_dynamically_occupied` — correct for TELEPORT landings (the
server displaces off bad tiles) but `c5aaabf6` reused it for WALK
destinations, and a walk cannot be displaced: a short close with rock
(or any terrain block) on the near cardinals dispatches `move` into
the wall → `cant_go`. Proven by a withheld red test (all four
cardinals rock → the branch still returned `move` onto rock). Fix
shape: the WALK branch filters candidates through the composed
`ctx.terrain.is_passable` (which also brings hostile mines in — the
same hole flags s6-8/9's mine walks fell through); the teleport path
keeps the unfiltered list. Test spec ready: 3x3 rock box minus target
→ expect fall-through to the landing teleport, plus a mined-cardinal
variant expecting the same. Directly relevant to the parallel
session's live flags-8/9 investigation.

**Flag s6-10 (tick 286) → dropped-tick receipt for the early-wake
work:** the t285 teleport resolved in **3961 ms** (vs ~1850-2120 ms
for every neighbor) — a full extra server window burned right before
the dual pickup, exactly the phase drift the early-wake sleep
targets. Either the wake missed the landing confirm or the confirm
itself arrived late; the events carry the pair for the parallel
session's calibration.

## Session-7

**Flag s7-1 (02:33:59, tick 70), pending narration:** under fire from
red-3 (fuel 925→735 across the ring) the decisions THRASH between
restock destinations without executing — equipment_hop (46 tiles!) →
fuel_hop → pickup → different fuel_hop, position unchanged for three
of five ticks. Smells like the F12 defer-flip re-arbitrating each
tick under combat pressure; the parallel session's F12 fix remains
the named cure.

**Flags s7-2 (02:34:33) and s7-3 (02:34:40), pending narration:** the
second Yuppler fight of the night — the bot holds the fight-refuel
loop (900→1010 off a locked 808 mid-exchange) and map-refreshes to
re-find the kiting human; the shots land at Yuppler's VACATED tiles
((87,163) then (83,152)) — the aim-staleness-vs-movers defect the
parallel session just receipted, now against the human it matters
most for.

**Flags s7-3/4 (02:34:41-52) → F21 candidate: the mid-fight weapon
top-up detour.** With Yuppler locked and DUALS AT 22/25, the
equipment hop pulled the bot 42 tiles out of the fight (and 43 back)
to top three duals — ~500 fuel and two free windows gifted to the
human, because the exact-cap weapon floor (entry bar) is being
re-applied during a HELD pursuit. The entry-vs-held distinction
already exists for fuel (`should_exit_hunt` deliberately does not
re-check the entry bar); the equipment-restock path needs the same:
during a held human lock, top up only at a genuine weapon BREAK (the
resume floor), never the entry cap, and prefer near stock.

| # | Finding | Flags | Status |
|---|---|---|---|
| F21 | Entry-cap weapon floor re-applied mid-pursuit: 85-tile round trip at 22/25 duals during the Yuppler fight | s7-3, s7-4 | OPEN |

**F14 combat escalation (s7, ticks 134/138/139):** three empty-container
pickups inside the Yuppler fight — the human drains the fuel the
bot's beliefs point at, and each phantom pickup is a wasted tick
under fire. F14's cost concentrates exactly where ticks are most
expensive; the pending ruling (belief-age gate vs two-clocks) is now
a combat-effectiveness question, not just forage economy.

**Flag s7-5 (02:38:06) → F21 receipt #2:** the same mid-pursuit
top-up detour — duals 20/25 with Yuppler locked, a 50-tile
equipment hop out of the pursuit lane, then fuel hops. The entry-cap
floor is dominating the pursuit phase; F21 is now the top open item
for the human fights alongside aim-staleness.

**Flag s7-6 (02:38:41), pending narration:** pursuit-phase fuel churn
under fire — drank a 176, then LOCKED an 86-volume dreg via the
walk-pickup path (the larder's dreg floor does not govern walk
pickups), then a fuel hop, all while eating hits (916→826) with
Yuppler locked and duals at 20. Together with F21's detours, the
pursuit phase is spending its windows on grazing instead of closing;
the walk-pickup path may need the same deficit-aware value bar the
larder got (F7/F13's economics, applied to walk pickups mid-pursuit).

**Flag s7-7 (02:42:56), pending narration:** productive range-5 duel
with red-6 (duals consuming on hits 25→22) with one mid-fight
REFRESH map_open at fuel 989 — if flagged, the question is the map
window spent while the target was engaged; otherwise the exchange
reads healthy.

## Session-8 (bot-20260730-025337, first build with engagement-break projection + break latch + corridor guard live)

**Flag s8-1 (02:59:07, tick 164) — fight-refuel-refight WORKING:** mid
adjacent duel with red-6, the new engagement-break projection fired
("projected fuel 270 at kill < floor 354, incoming 27/tick over 3
hits, 9 hits to kill, fuel=693") and the escape was exactly the
ruled two ticks: map_open, then larder teleport onto a 417-volume
fuel container. Landed 630 → 1047, break latch released, re-engaged
red-6 at the same coordinates within the same second. First live
receipt for the full break→escape→refuel→re-engage loop. One
cosmetic residue: the deferral tick's plan said `equipment_hop
(138,54)` but the jump tick executed `fuel_hop (138,57)` — plan
re-derivation between the two escape ticks (benign here; same map
open served both, but it is the plan-continuity gap in miniature).

**Flag s8-2 (03:00:01, tick 188) — wasted tick at the escape landing:**
same break vs purple-5 (projected 277 < 354 at fuel 747), two-tick
escape via `equipment_hop` to (141,85). Defect after landing: the
next derivation re-selected `equipment_hop` TO THE TILE THE TANK WAS
STANDING ON — "deferring teleport to (141,85): opening map first"
while self was at (141,85) — burning a map_open tick before the
next tick's pickup (duals 21→25, then an 889 fuel drink). A plan
that survived the landing would have gone straight to the pickup.
Receipt for the committed-intent gap: post-landing re-derivation
does not know the hop's purpose was already served by arriving.

**Flag s8-3 (03:00:19, tick 197) — re-acquire overhead + mid-duel
map_open:** the return to purple-5 worked ("returning to locked
target — refreshing stale position via map" → re-teleport →
re-engage), with mid-combat fuel pickups keeping the tank at cap
during the duel (nice). Residue: at 03:00:13, one `find_target`
map_open fired BETWEEN shots at an engaged, in-view target — an
F12-family lost tick mid-duel.

**Flag s8-4 (03:01:14, tick 224) — the clean specimen:** break vs
purple-3 (projected 169 < 354 at fuel 639) → map_open → teleport
(185,33) onto a 419 container WITH pickup on landing → latch
released at 995 → re-engaged purple-3 the same second. Two ticks
under fire, zero wasted ticks at the landing. This is the contract
behavior end-to-end.

Session-8 residues to track: (a) s8-2's landing-tick self-teleport
re-derivation — new finding, needs a landing-serves-the-plan guard
or the intent layer; (b) s8-3's mid-duel find_target map_open (F12
family). The engagement-break projection numbers were consistent
across all three breaks (27/tick over 3 observed hits, floor 354).

**s8-2 FIXED (2026-07-30, committed-intent phase 1,
[[committed-intent]]):** the root fix is structural, not a point
latch. New `bot/ai/intent.py` owns collect-plan semantics over the
EXISTING lock fields (no new state): the under-fire escape now
finishes a held plan that completes within auto-pick reach (the
pickup IS the escape continuation), `_hop_toward_equipment` refuses
landings equal to the current position (`own_ground` tally — the
cost-0 candidate that structurally wins cost ranking is exactly how
the self-teleport got selected), and every plan release emits a
`plan_released` diagnostic with a closed-vocabulary reason
(`superior_candidate`, `not_executable`, `tank_at_capacity`,
`landing_scan_reset`, `walk_for_fuel_override`, `target_gone`,
`target_not_pursuable`, `kind_invalid`) so plan churn is measurable
per run instead of silent. `normalize_resource_target` lifted from
`context.py` into `intent.validate_collect_plan` with the same
pursuability predicate. Pins: `test_intent.py` (22),
`TestEscapePlanContinuity` (s8-2 scenario byte-for-byte: under fire,
lock on own tile → `pickup_equipment`/`equipment_locked`, never a
hop; far lock does NOT hijack the walk law), own-ground gate tests.
Gate: 5,481 tests, 100.00% statements+branches. Phase 2 (hunt +
clearance plans, supersede visibility) specified on
[[committed-intent]] — the s8-3 mid-duel map_open (F12) belongs to
the hunt-plan phase.

**F22 (found AND fixed same hour by the plan_released channel):**
the new events stream exposed three `not_executable` releases (run
bot-20260730-032x ticks 361/366/371) that all fired mid-approach
WITH the plan's own map_open in flight — and each released target
was re-locked and served 2-3 ticks later. Transient "no executable
route this tick" was being read as plan invalidity. Fix: the lock
continuations now HOLD the plan on a transient `walk_or_teleport`
None (yield the tick, keep the lock) and release with
`not_executable` only on the structural server-confirmed
move-failed mark (`is_move_target_failed`). Water-boxed plans now
survive for a later ferry/approach; the genuine release gates
(superior candidate, validity, at-capacity, move-failed) are
unchanged. Repinned: water-locked hold ×3, move-failed release ×2.
Gate: 5,484 tests, 100.00%.

**Same-run F14 quantification (ticks 360-448):** five phantom
pickups on distinct drained beliefs, each exactly one wasted tick
followed by a clean `target_gone` release — the feedback loop is
correct; the cost is purely the pending belief-age ruling.

**F23 (found by monitor cluster-trace, FIXED same hour): the
movement-dead escape loop.** Run bot-20260730-110x ticks 95-107:
mid-duel with purple-1 the under-fire escape dispatched a walk-pickup
every tick and the server rejected every one with `cant_go` — TWELVE
consecutive rejected ticks standing still in the firing line (fuel
972→663), each rejection burning that container (`failed_pickups` →
`target_not_pursuable`) and the next tick planning a walk to a
different container. The tank was movement-boxed (every direction
refused) and nothing recorded the shared fact "the server is
refusing this tank's movement" — collect-kind rejections only fed
the per-container marks, so the walk-first escape law kept assuming
walking works. The hop rung finally won at t112 (+437 refuel saved
the duel), but survival came from burn-through, not detection. Fix:
(1) `WorldService.movement_rejections` — every `cant_go` on a
move/collect/teleport dispatch records a timestamp
(`record_movement_rejection`), counted via
`recent_movement_rejections(now, window)` with in-place pruning;
(2) the under-fire escape checks the count against
`_MOVEMENT_DEAD_REJECTION_FLOOR = 2` inside the fire window — at the
floor the walk rungs are declared dead and the escape jumps straight
to the hop (a teleport needs no walk path and lands
displacement-safe). Pins: service record/count/prune, cant_go-on-
collect records + code-0 does NOT, movement-dead skips the walkable
in-viewport fuel for the larder hop, single rejection keeps the walk
law. Gate: 5,491 tests, 100.00%.

**F20 FIXED (2026-07-30, forced by a live 110-tick livelock):** run
bot-20260730-110x ticks 904-1017+: HUNT/CLOSE walked to (240,46)
adjacent to orange-6, the server rejected the move with `cant_go`,
the tile was marked failed — and the next tick dispatched the
IDENTICAL move, over 110 consecutive ticks, because the walk-close
branch takes `combat_landing_candidates[0]` which consulted neither
the composed terrain nor the failed-move marks (the teleport path
below it checks the mark, but the walk path returns first). Fix at
the candidate source: `combat_landing_candidates` now takes
`(terrain, now_ms)` and filters impassable composed tiles and
live-marked failed tiles; an unwalkable adjacency ring falls
through to the teleport path, whose failed-landing gate blocks and
replans. Pins: terrain-blocked candidate skipped, marked candidate
skipped, signature updated at all call sites. Gate: 5,493 tests,
100.00%. The stuck session was cycled via the stop file so the
relaunch runs the fix; the withheld red-test spec published earlier
under F20 is superseded by these landed pins.
