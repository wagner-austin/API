---
title: "Combat Chase Loop Bug (Diagnosed 2026-06-16)"
tags: [combat, bug, teleport]
related:
  - "[[shot-range]]"
  - "[[enemy-bot-behavior]]"
  - "[[teleport-mechanics]]"
  - "[[tank-freshness-model]]"
provenance:
  - "runs/bot -- gitignored runtime capture artifact (moved from source_paths 2026-09-06, code-paths contract)"
fact_checked: "2026-06-16"
confidence: high
hubs: [combat]
---

# Combat Chase Loop Bug

## Symptom

Bot gets stuck chasing a fleeing enemy for minutes, repeatedly teleporting 1 tile behind, never firing. Burns 6-18 fuel per hop, ~4 seconds per cycle (2s map open + 2s teleport).[^1]

## Root cause

`_combat_close` in `combat_strategy.py` requires `has_cardinal_combat_shot` (Manhattan distance == 1) before it will shoot. When the enemy moves 1 tile during the ~2s teleport, the bot lands at distance 2+ instead of 1, and re-teleports instead of shooting.[^2]

## Observed sequence (run 2026-06-16 18:26)

1. Bot at (9,128), target purple-9 at (5,128) dist=4 → teleports to (6,128)
2. Bot at (6,128), target now at (4,128) dist=2 → re-teleports to (5,128)
3. Bot at (5,128), target now at (4,128) dist=1 → finally adjacent, does scan_on_landing

Each failed attempt: target position drifts 1 tile between teleport dispatch and landing. The map open refreshes positions, confirming the enemy moved.[^2]

## Root cause

`combat_landing_candidates` was computing an adjacent tile client-side, then teleporting to THAT tile. When the map position was even 1 tile stale, the bot landed 2 tiles from reality. The correct behavior is to teleport directly to the enemy's coordinates and let the server place you adjacent — exactly how human players do it.[^5]

## Fix (implemented 2026-06-16)

- `choose_combat_landing_tile` now returns `(target["x"], target["y"])` directly — the enemy's exact position
- `find_teleport_landing_tile` (for containers) also returns `(goal_x, goal_y)` directly
- The server handles displacement: occupied tile → placed adjacent, mine → nearest open, terrain → nearest open
- Removed dead `CLOSE_WALK_RANGE_TILES` constant

This eliminates the chase loop entirely. Even if the map position is 1 tile stale, the server knows the real positions and places correctly.[^5]

## Follow-up fix (2026-06-26): stay put after engagement

The 2026-06-16 fix removed the multi-hop chase but left a related teleport-after-every-move pattern. Live run 2026-06-26 14:42: bot landed adjacent to purple-5 at (148,166), fired 9 dual shots (all hits) while the target sat at (147,166); on shot 10 the target teleported to (163,154) and the server auto-picked homing (one tracked hit). The planner's next decision saw distance 27, fell to `_combat_teleport`, and spent 114 fuel + a `map_open` + a `scan_on_landing` (~6s of wire time) to land adjacent to (163,153) — three shots' worth of wall-clock burned to position for one more dual instead of just firing another homing from the same tile.[^7]

**User-contract loop (2026-06-26):** open map → teleport adjacent → dual until they teleport away → stay put and fire homing until deactivated. Enemies don't move *within* the viewport; when they leave cardinal adjacency, they teleported, so chasing with another teleport is wasted fuel.[^7]

**Fix:** `close_target` (`combat_close.py:259`) now branches:[^8]

1. Cardinally adjacent → shoot (server picks dual at point-blank).
2. Already engaged (`last_shot_target_id == combat_target_id`) → shoot (server picks homing, which tracks).
3. Fresh acquire, not adjacent → teleport (the one-time initial close).

The engaged-vs-fresh predicate lives in `is_already_engaged` (`combat_target.py:64`) and replaces the duplicated inline expression that previously lived in `resume_locked_target_off_viewport` (`hunt_lock.py:96`). Both the on-viewport and off-viewport paths now consult the same predicate.[^8]

**Branch 1 has since widened (user ruling 2026-07-29).** It now fires whenever the shot is in range AND the line is clear — `has_cardinal_combat_shot(...) or (has_combat_shot(...) and has_clear_shot_line(...))` — not only at cardinal adjacency. The receipt is purple-8: after a break-driven pickup the bot stood at distance 2 and paid a teleport to regain adjacency instead of shooting. Branch 2 gained the mirror-image guard: an engaged stay-put shot with no clear line re-closes instead of firing, because a half-damage homing arcing over an occluder while the enemy duals back for 90 is the losing trade that killed Artax (flag s3-16).[^8]

## Follow-up fix (2026-06-26): unlimited homing restored by reverting the OUR_SHOT registry update

The "stay put" change above exposed a separate bug introduced earlier in the same branch. Live run 2026-06-26 15:13 caught the bot firing one homing shot at a teleporting target then dispatching `shoot(off_viewport_x, off_viewport_y, id=N)` repeatedly, each rejected by the server with `command_error` because shoot tiles must be inside the 18×18 viewport (see [[shot-range]]). Pre-2026-06-23 the bot reliably fired multiple homing shots in this scenario per the user-confirmed contract; the regression was traced to a two-line addition in `sniffer/world_state_dispatch.py:161-162` (commit `098d3d7`) that overwrote the locked target's registry x/y from `OUR_SHOT`'s homing-tracked landing tile every time the bot fired a homing or missile. The seeker's resolved tile is the target's current off-viewport position, so the registry update poisoned the next shoot dispatch.

**Fix:** delete the registry-update lines. Pre-098d3d7 the registry stayed at the last on-viewport coord (off-viewport tanks stop broadcasting `0x2E TankStatusSync`), so subsequent shoots dispatched at an in-viewport tile, the server accepted them, and the server's homing seeker tracked to the actual target. Unlimited homings until the kill.[^9]

## Follow-up fix (2026-07-03): viewport-clamped aim replaces the stale-registry accident

The 2026-06-26 fix worked by accident of that capture: `0x3D MovementResponse` in fact broadcasts every map tank's position ~every 2 s, so the pursuit registry DOES track the target's true off-viewport coordinates. Live run 2026-07-03 20:34: after one dual hit and one tracked homing (`weapon=3` aimed at the target's vacated in-viewport tile — the snipe working as designed), the registry followed orange-4 to (143,237), five rows below the viewport, and five pursuit dispatches at those raw coordinates drew 0x52 code-0 rejections ("You can't do this") — the server refuses any aim outside the viewport, and per the user's game knowledge specifically refuses homing at an enemy close enough that a viewport shift would reveal them. The rejections were also invisible to combat feedback (no ShootEvent, no ammo delta), so each one burned the full 4 s shot-feedback window before an identical redispatch.[^10]

Two-part fix: (1) `_clamp_aim_into_viewport` — every shoot dispatch aims at the registry coordinate clamped onto the visible viewport bounds, carrying the target's `tank_id`; the server picks homing and the seeker tracks (aim is a hint — the same wire pattern as the shot that killed). The clamp applies only when the viewport record contains the bot, so an unestablished viewport can't corrupt aims. (2) Shot-rejecting 0x52 codes (0/3/8) arriving while a shot is pending end the feedback wait immediately, classify as `rejected` (neither hit nor miss; scorecard counts them via `session_reject_count`), and block-and-replan the target.[^10]

The diagnosis that originally motivated the registry update (live run 2026-06-24 12:43, "4 homings missed because the registry kept purple-9 at (180,147) while she had moved off to (198,152)") was incorrect: homing aim is just a hint, the server tracks regardless, so a stale registry never caused those misses. The wire firing mechanism (`_combat_shoot` → `make_shoot_command` → `build_shoot_command`) is unchanged and has been since the first commit; only the registry-population side changed.[^10] The registry's true off-viewport tracking behaviour is documented with its own receipts in [[tank-freshness-model]] (Correction 2026-07-03).

**Test:** `tests/sniffer/test_world_state_dispatch_container.py::test_own_homing_does_not_overwrite_locked_target_position` -- inverted from the original `test_own_homing_refreshes_locked_target_position` to assert the registry stays at the last on-viewport tile after a homing dispatch.[^9]

## Caveat: server does NOT displace off equipment-container tiles

The "let server displace" rule was proven for combat targets (a tank occupies the tile) and ferry / water terrain. It is **not** symmetric for equipment containers. Live capture 2026-06-21 16:54:26: bot teleported to (253,141) with an equipment container there, server placed the bot **on** the container tile (not adjacent), then `pickup_equipment(253,141)` returned no `container_consumed` response.[^6] So:

- Combat teleport to enemy → server displaces adjacent → bot can shoot.
- Container teleport to empty-of-tanks tile with container → server places **on** container → distance-0 pickup never returns success.

The right behaviour for `pickup_equipment` is to dispatch it from inside the viewport (server handles the walk) rather than teleport directly onto the container tile. See [[equipment-system]] pickup mechanic.[^6]

[^6]: live capture 2026-06-21 16:54:26 — bot at (253,141) sent pickup_equipment two times, zero server response bytes; matched prior successful pickup at distance 3 from (252,136)

[^1]: run 2026-06-16 18:26 — bot chased purple-9 through 3 teleport hops at x=4-9, y=128, spending ~54 fuel over ~12s without firing
[^2]: run capture on disk: `runs/bot/bot-20260616-182446.events.jsonl` (the 18:26 session cited in [^1]) — diagnostic logs show the target position shifting between dispatch and check: target=(5,128) at dispatch → target=(4,128) at check
[^5]: user (Austin), 2026-06-16 — "I teleport to the same exact position as the enemy tank. so the game puts me adjacent"; the bot was computing adjacent tiles client-side instead of letting the server handle it. The ruling is code truth at `src/tankpit_bot/bot/ai/combat_landing.py:80`, `choose_combat_landing_tile`, whose docstring states it verbatim: "teleports directly to their coordinates: the server handles displacement … This is how human players teleport: click on the enemy, let the server place you." The same docstring records the corollary — the question asked is `is_landing_legal`, never `is_passable`, since an enemy always occupies its own tile and the walk question would reject every direct approach.
[^7]: run capture on disk: `runs/bot/bot-20260626-144149.*` (the 14:42 session); wiki-log entry "[2026-06-26] update | Combat stay-put: shoot when engaged at distance, don't re-teleport" records the run narrative and the user-contract loop
[^8]: code truth on disk, re-verified 2026-08-07 after the combat module was split: the branch order is `close_target` at `src/tankpit_bot/bot/ai/combat_close.py:259`, whose docstring enumerates the three branches at `:269-279`; the widened branch 1 is the `has_cardinal_combat_shot(...) or (has_combat_shot(...) and has_clear_shot_line(...))` test at `:290-292` with the purple-8 receipt in the comment at `:293-300`, and the branch-2 terrain guard is at `:303-310`. The predicate is `is_already_engaged` at `src/tankpit_bot/bot/ai/combat_target.py:64`. This footnote previously cited `_combat_close` at `combat_strategy.py:389` and `_resume_locked_target_off_viewport` at `hunt_mode.py:223`; neither symbol nor either path exists — the functions were renamed without the leading underscore and moved to `combat_close.py` and `hunt_lock.py:96`.
[^9]: wiki-log entry "[2026-06-26] fix | Unlimited homing shots restored by deleting the OUR_SHOT-driven registry update"; root-cause commit `098d3d7` (combat-rework, 2026-06-23) in git history; regression pinned by `tests/sniffer/test_world_state_dispatch_container.py::test_own_homing_does_not_overwrite_locked_target_position` (line 340, verified 2026-07-23)
[^10]: run capture on disk: `runs/bot/bot-20260703-203416.*` (the 20:34 session); wiki-log entry "[2026-07-03] fix | Pursuit rejection loop: viewport-clamped aim + rejected-shot feedback" records the five 0x52 code-0 rejections, the two-part fix, and the falsification of the 2026-06-26 stale-registry assumption
