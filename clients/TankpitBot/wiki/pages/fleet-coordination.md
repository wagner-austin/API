---
title: Fleet Coordination
tags: [fleet, architecture, coordination]
related:
  - "[[bot-behavior-contract]]"
  - "[[fleet-forage-allocation]]"
  - "[[bot-service-architecture]]"
  - "[[tank-freshness-model]]"
source_paths:
  - "src/tankpit_bot/fleetshare/__init__.py"
  - "src/tankpit_bot/fleetshare/codecs.py"
  - "src/tankpit_bot/fleetshare/merge.py"
  - "src/tankpit_bot/fleetshare/report.py"
  - "src/tankpit_bot/fleetshare/role.py"
  - "src/tankpit_bot/fleetshare/types.py"
  - "src/tankpit_bot/bot/tick_body.py"
  - "src/tankpit_bot/bot/ai/threat_primitives.py"
source_git_blobs:
  "src/tankpit_bot/fleetshare/__init__.py": e17b3322fd0ab35074c56cfff000a9328d245ae8
  "src/tankpit_bot/fleetshare/codecs.py": cfcd7ceac84cfdf2246fb215a9925b8e686620e1
  "src/tankpit_bot/fleetshare/merge.py": 3f0738090b0d27be86af0199f2a1a394adc55bc8
  "src/tankpit_bot/fleetshare/report.py": 02962fcb3cb27dc041e484ea404d53350ac20923
  "src/tankpit_bot/fleetshare/role.py": a82ac236569bc9444ea5bbb180a163dd443ccf5c
  "src/tankpit_bot/fleetshare/types.py": 368d8f5ced92ebe03bc183cc5e5d1f479aaeaec2
  "src/tankpit_bot/bot/tick_body.py": ccf7e6085214f06b2856774aa08db55b7f550910
  "src/tankpit_bot/bot/ai/threat_primitives.py": 01a0bb653c8186a41a3dc9aa7f9b122feb9e182b
fact_checked: "2026-08-19"
confidence: high
hubs: [architecture]
---

# Fleet coordination: the shared knowledge layer

*Established 2026-08-14 (fleet ruling: same-team allies; "a proper
lift that allows for single tank running, or multi tanks running,
with fighters and with a potential info gatherer").*

Same-team bots exchange beliefs through the run-directory filesystem —
the channel the fleet page already reads — so a **single tank runs
identically with zero siblings** and a **fleet coordinates with no
manager process required**. There is no network dependency and no
broker: presence of a fresh sibling file IS membership.

## The exchange

Every tick, after the HUD mirror, each bot:

1. **Publishes** its knowledge offer: `fleetshare.report.build_fleet_report`
   assembles fresh beliefs and `write_fleet_report` **atomically
   replaces** `knowledge.json` beside `hud.json` in the bot's run
   directory (`runs/bot/<instance>/`, or `runs/bot/` for the sole-bot
   namespace). Atomicity (temp file + `os.replace`, the
   `replace_text` hook) is what makes the reader's strict
   decode-and-raise sound: a torn read is impossible, so a malformed
   file is a genuine bug.
2. **Merges** siblings: `fleetshare.merge.read_team_reports` lists
   `runs/bot/*/knowledge.json` plus the sole-namespace file, skips its
   own, drops reports older than `FLEET_REPORT_TTL_MS` (10 s — a dead
   bot's last file ages out instead of steering the living) and other
   teams' reports (knowledge sharing is an alliance), then
   `merge_fleet_reports` applies the content.

The exchange starts with the first entered tick — before the session
has an established self there is nothing attributable to offer
(`build_fleet_report` returns `None`).

## The report (`FleetReportDict`)

| Field | Meaning |
|---|---|
| `instance`, `team`, `tank_id`, `role`, `x`, `y` | Reporter identity and position |
| `engaged_target_id` | The reporter's held combat lock (-1 none) — the focus-fire signal |
| `written_ms` | Write stamp; drives the reader's freshness TTL |
| `enemies` | Sightings with the reporter's own `last_position_update_ms` as `observed_ms` (within 30 s), excluding allies, corpses, unplaced and stale tanks |
| `containers` | The container atlas, excluding locally failed-pickup marks (a disproof is the reporter's own verdict, not knowledge) |
| `scanned` | The live scan map: tiles under radar coverage within the forage coverage TTL (180 s) — the worldview's negative space, so siblings stop paying radars for ground a teammate cleared |
| `removed` | The reporter's container tombstone map (bounded by the container share horizon, 60 s) — consumption propagation, so one bot's pickup stops the whole fleet chasing the ghost |

Codecs: `fleetshare.codecs` — full `require_*` validation on decode.

## Merge laws

- **Remote knowledge only adds or refreshes; own wire is the higher
  trust tier.** An enemy sighting applies only when FRESHER than the
  local registry entry; a container sighting rides
  `merge_container_sighting` with the same freshness law and never
  removes local beliefs. A local `failed_pickups` mark survives any
  remote refresh.
- **Merged sightings can never provoke a phantom shot.** They enter
  via `apply_tank_observation` with fact source `fleet_report`,
  `storage_source="world_state"`, `is_wire_sourced=False`: the
  viewport-observation gate that authorizes firing never advances,
  so merged enemies are acquirable (map-like) but not fireable until
  this bot's own viewport confirms them.
- **Tombstones (negative knowledge, 2026-08-14 first live fleet
  run):** every local container removal — code-4 disproof, emptied
  pickup, unreachable, radar-stated-empty — stamps
  `ws.container_disproofs[tile] = now`, and the merge admits a remote
  sighting only when OBSERVED AFTER the disproof. Without this,
  deletions don't propagate: a teammate that still believes in a dead
  container re-imports it every exchange, and the pickup loop never
  converges (run arterial 19:20: (102,85) disproved three times in
  five seconds, re-imported between each — the "Empty container /
  Nothing detected here" alternation the user watched live). A
  genuinely respawned container passes naturally: its fresh
  observation postdates the disproof.
- **Shared scan coverage (2026-08-14, user ruling "share the
  worldview"):** teammates' live coverage merges into
  `scanned_tiles` via `merge_scanned_coverage`, newest stamp per
  tile winning and own fresher coverage never regressed. The forage
  and sweep gates already read `scanned_tiles`, so scanner division
  of labor needs no behavioral surgery: a tile a sibling scanned is
  covered here too. Mines need no fleet row — reveals are
  team-scoped in the game itself ([[walk-mechanics]]), so teammates'
  radar reveals already arrive on each bot's own wire. User
  corroboration (verbatim, 2026-08-25): *"if another person on your
  team radars, you will see the revealed mines, fuel, and equipment.
  and if the person on your team collects any fuel or equipment or
  destroys enemy mines, you'll see that live, no re-radar
  necessary."* The wire carriers are the 0x43 `CacheUpdate`
  (fuel/equipment reveals and value-0 removals) and 0x40
  `OverlayUpdate` (mine reveals/clears), both ingested regardless of
  which team client caused them (`world_state_tiles.py`). The
  in-viewport live layer is therefore the SERVER's own; the fleet
  exchange adds only the cross-map dimension (container atlas,
  coverage, tombstones, enemy sightings) — and the s9-2 correction
  ([[radar-mechanics]]) is about exactly that seam: a cross-map
  coverage mark is not mine knowledge, because reveals only ever
  arrived on the wire of clients who could see the ground, and mines
  planted since a scan are invisible to the whole team until someone
  re-scans.
- **Coverage steps WALK, never teleport** (user free-radar doctrine:
  "scan, walk, walk, scan — to scan a whole viewport with the free
  radar"): forage dispatches `plan_viewport_walk`, a pure-walk
  movement primitive with no teleport fallback and no mine-flip — a
  free radar reveals ground for nothing, so no coverage step is
  worth a 45+ fuel hop (run arterial 19:30 paid two forage teleports
  on a zero-extras recruit). An unwalkable best position means the
  viewport is done for free-scan coverage and the search hop
  relocates.
- **Removal propagation (2026-08-14, user: "does it update the
  equipment for everyone when one of them takes the discovered
  equipment?"):** the report's `removed` ledger carries the
  reporter's tombstones; a receiver drops any local belief OBSERVED
  BEFORE a teammate's removal and inherits the tombstone — so
  consumption spreads transitively within one exchange. A local
  belief fresher than the removal survives as a possible respawn.
  Verified live in run pair bot-20260814-2047xx: each bot merged 4
  removals from its sibling.
- **Fleet kills are not OUR kills:** every 0x41 deactivation enters
  the dead-tank registry (`killed_tank_ids`, killer-agnostic — never
  target a corpse), but `session_kill_count` (scorecard +
  `session_kills` wind-down trigger) advances only when the 0x41
  names this tank as killer. The wire queue carries victim → killer
  for exactly this split. Solo sessions made the two numbers
  indistinguishable; the first fleet firefight falsified it
  (arterial banked artax's two kills on zero shots fired,
  bot-20260814-204751).
- **Focus fire:** teammates' `engaged_target_id`s land in
  `ws.fleet_engaged_target_ids` (REPLACED wholesale per merge — a
  disengaging or silent teammate stops steering within one exchange).
  The threat sort (`_threat_sort_key_for`) ranks fleet-engaged ids
  first *inside* a priority tier, so fighters converge on one enemy
  without outranking the human-priority doctrine.

## Roles (`FleetRole`, `TANKPIT_ROLE`, fleet spawn `role`)

- **fighter** (default): the full HUNT/COLLECT doctrine.
- **gatherer**: never hunts — the router's `_select_owner_mode`
  returns COLLECT unconditionally, `hunt_entry_permitted` is the
  doctrinal backstop (every yield-to-hunt gesture funnels through
  it), and an exhausted collect cascade returns a COLLECT-owned
  `gatherer_hold` no-op instead of the fighter's
  `no_productive_collect` exit — "cannot hunt" is the role, never
  "marooned". The gatherer lives in the collect cascade (scan, sweep,
  search hop, map-for-dots), roaming the map and publishing what it
  finds for the fighters of its color.

  First live gatherer run (bot-20260820-005115 pair): the contract
  held on the wire — zero shots, zero HUNT-owned ticks across 4m21s —
  but the run surfaced the **full-inventory equipment livelock**
  (FIXED same day): a gatherer at rank cap (recruit 20/20/20/20/20)
  kept the collect cascade's `equipment_locked` target while the
  dispatch was predictively suppressed every tick ("belief predicts
  0x52 code 7", 93 consecutive ticks on (133,129)); suppression never
  bumps `failed_pickups`, so the belief kept winning and the cascade
  never reached scan/sweep/frontier-walk (1 pickup in 4m21s; 512
  scanned tiles offered vs 2,734 received — the gatherer fed
  BACKWARD). Root cause was a role-shaped mask: the equipment HOP was
  barred only via `hunt_entry_permitted` (a full FIGHTER goes
  hunting), and the gatherer's unconditional False removed that bar;
  the LOCK continuation never had the capacity gate the fuel lock
  gained 2026-07-06. Both now apply the shared
  `equipment_pickup_refusal` law directly: the hop declines
  (`at_capacity` tally) and the lock releases `tank_at_capacity`, so
  a full gatherer falls through to coverage and roams. Belief
  deletion was rejected as a fix — the teammate's atlas re-feeds the
  tile within the share TTL. Pinned by
  `test_locked_equipment_released_when_every_slot_at_rank_cap` and
  `test_hop_toward_equipment_declines_when_every_slot_at_rank_cap`.

The fleet manager's spawn API carries the role per bot (2026-08-20:
`POST /bots` gained `"role"`, validated against `FLEET_ROLES`, empty
means fighter): the manager sets the child's `TANKPIT_ROLE`
explicitly on every spawn — never inherited, so a stray value in the
manager's own environment cannot silently re-role the fleet — and
restart carries the stored role ([[bot-service-architecture]]).

## Color assignment

`TANKPIT_TROOP`, when set, **overrides the account's lobby
`default_troop`** — accounts hold one tank per color per map and the
room-enter request's troop byte picks which one to play (the server
enforces the 5-minute recolor cooldown). Unset: the account default,
or blue on a first-time entry. The fleet plays blue
(`.env TANKPIT_TROOP=2`).

## Watch items (2026-08-19 skeptical review)

A full read of the module (all six files, at the pinned blobs) found no
defect — the gate was green (6,176 tests, 100% statement+branch
coverage) and the merge laws held up. Four documented hazards, none
actionable today, recorded so the next lift weighs them:

1. **A malformed FRESH sibling report crashes every teammate's tick,
   every tick** — deliberate (a live mixed-schema fleet is a genuine
   bug, and the no-fallbacks rule forbids swallowing it). Note the
   freshness gate reads `written_ms` BEFORE schema validation
   (`merge.read_team_reports`), so a future report schema that renames
   that one field turns a still-running old-build sibling into a
   fleet-wide crash source. The mitigation is operational: fleets run
   one build.
2. **Freshness-domain asymmetry in `_merge_enemy_sightings`:** the
   pre-gate compares local `timestamp_ms` (advanced by ANY
   observation, including damage-only syncs) against the remote's
   `observed_ms` (position freshness). A tank the local wire keeps
   damage-refreshing but hasn't positioned in a minute rejects a
   teammate's seconds-old position fix. Conservative direction — it
   only ever discards remote knowledge, and own wire is the higher
   trust tier by law — but the comparison crosses freshness domains.
3. **`ws.container_disproofs` never prunes.** The report builder
   filters by the share horizon but iterates the whole dict each
   tick; entries live for the session. Bounded by tiles disproved, so
   harmless at current session lengths — do not copy the pattern for
   anything larger.
4. **The exchange is O(fleet² × rows) per tick:** every bot parses
   every sibling's full report each ~2 s tick, and the scanned-tile
   ledger already ran 900–1,300 rows per exchange in a two-bot run.
   A non-issue at 2–3 bots; revisit the encoding (or share cadence)
   before running a large fleet.

## Deliberately deferred

Mine-reveal sharing, pre-engagement target brokering (a fleet-manager
endpoint designating one focus target before any shots exist), and
any push-style transport. The filesystem exchange is the foundation
they would build on.
