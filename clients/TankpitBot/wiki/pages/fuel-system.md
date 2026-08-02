---
title: Fuel System
tags: [fuel, containers, map-data]
related:
  - "[[teleport-mechanics]]"
  - "[[radar-mechanics]]"
  - "[[map-data-decode]]"
source_paths:
  - "runs/bot"
  - "src/tankpit_bot/physics"
source_git_blobs:
  "src/tankpit_bot/physics": "130c17d4a20d81886055bc97dc20140c9656f1c6"
fact_checked: "2026-08-01"
confidence: high
hubs: [game-mechanics]
---

# Fuel System

## Thresholds (current, re-verified against config 2026-07-19)

- `fuel_low_threshold`: 200 — single fuel threshold. Below this the
  bot enters `COLLECT`. Also the reserve a hop must leave behind
  (relay/dot hops require ``cost + 200 <= fuel``), and one of the two
  terms of the 650 engagement reserve (450 engagement budget + 200
  floor). This page previously said 300 (as of 2026-06-24); the
  config value has been 200 — trust `types.py:make_initial_ai_state`
  as the source of truth for the number.[^1]
- The COLLECT release / HUNT entry fuel bar is NOT a config value:
  since 2026-07-25 it is `fuel_capacity(rank)` (1000 recruit … 1800
  general) — the hunt-only-when-full contract derives every
  readiness bar from rank, and the old fixed `fuel_full_threshold`
  was deleted. A mid-session promotion leaves this bar stale-LOW
  until re-login ([[game-economy]] mid-session promotion law). See
  [[bot-behavior-contract]] §3.1.
- `hunt_min_fuel`: 100 — operating reserve for search/recovery
  teleport hops.
- The historical `fuel_critical_threshold` was collapsed into
  `fuel_low_threshold` 2026-06-22. The two-tier "polite low vs.
  emergency critical" distinction was dead because both thresholds
  had drifted to the same value (300). One threshold now governs
  COLLECT entry, combat-teleport reserve, and the in-cascade
  fuel-pickup predicate.[^5]
- COLLECT drains every viewport fuel container (`minimum_volume=1`).
  The old "skip volumes < 500" floor was dropped 2026-06-23 after a
  live run left 20-fuel partials in viewport while burning ~200 fuel
  per hop to find a fresh one.[^2]
- At the cap end the only pickup gate is the per-tile rate rule
  (2026-07-19): refuse when the clamped transfer
  `min(volume, headroom)` is under 25 fuel per Manhattan walk tile —
  adjacent pickups always taken, distant cap-slivers refused. The
  server clamps the transfer and answers code=5, which is handled
  cleanly (container kept, no blacklist). This replaced the
  2026-07-06 binary overfill gate, which refused ANY clamped pickup
  at walk >= 1 — at fuel 600 it walked past a 1000-volume container
  one tile away, forfeiting a 500-fuel transfer.

## Fuel data flow (single source of truth)

`self_state["fuel"]` is updated **only** from the wire's absolute-fuel
messages, in `sniffer/world_state_dispatch._dispatch_resource_update`:[^6]

- `0x2E TankStatusSync` — periodic fuel cadence.
- `0x44 FuelGain` — fuel pickup completion (and free-pickup events).
- `0x64 FuelDeposit` — depot returns.

All three flow through
`sniffer/world_state_containers.update_world_state_from_fuel_total`
→ `state.mutations.set_self_fuel`, which writes the absolute fuel and
emits the `"Fuel: A -> B (+delta)"` world-log line.[^6]

`state/container_mutations.pickup_container` is **strictly registry
maintenance** — it removes the picked-up container (or shrinks its
volume on partial pickup), but does NOT modify `self_state["fuel"]`.
A prior version computed `transferred = prior_volume -
remaining_volume` and added it to `self_state["fuel"]`; combined with
the wire's already-correct absolute fuel, that double-counted every
fuel pickup. Live observation 2026-06-23: a 438-volume container
yielded the correct 633 fuel in the worldstate log but the bot's
`ctx.fuel` reported 1071 (633 + 438) for the next decision tick. The
local fuel-delta branch was removed 2026-06-23; the wire is the
single source of truth.[^6]

## The fuel-pickup wire choreography (byte-mined 2026-08-01)

Every explicit ``pickup_fuel`` command answers with one of four
measured shapes (~1,600 archive windows, [[capture-differ]]; all
messages land in one server tick, and the self 0x2E sync carrying the
new absolute fuel LEADS the batch):

| Case | Shape |
|---|---|
| transfer, tank FILLS (clamp) | ``[0x47] + record x2 + 0x44 (gain form: is_free=True, flag=0, absolute fuel) + record x1 + 0x52 code 5, reset_action=0`` |
| transfer, container EMPTIES (drain) | ``[0x47] + record x2 (remaining 0) + 0x52 code 4, reset_action=1`` — no 0x44 |
| no transfer, walked (arrived to find it empty / full-tank walk-up) | ``0x47 + record x2 + close by stockedness`` |
| no transfer, no walk (own-tile / adjacent click) | ``0x44 (no-gain form: is_free=False, flag=43, unchanged fuel) + record x1 + close, reset_action=0`` |

Key laws inside it: the container records come in IDENTICAL
duplicates (twice, three times in the clamp case with the 0x44
between records 2 and 3) — not progressive drain steps; the same
duplicate-record law governs plain move and teleport-landing
auto-picks (``...pickup+pickup``, 129 move + 2,200+ teleport
windows), which carry NO 0x44 and NO 0x52 close. The 0x44 has two
byte-distinct forms (gain vs no-gain). **The walk executes even for
a KNOWN-drained container** — receipt bot-20260730-224244
@1785476734979: fuel 783 -> 783 across a 4-tile walk, then the
remaining-0 records and the code-4 close (``reset_action=1`` after a
walk; 0 without) — the sim's old walk-free pre-refusal was an
invention; only a click at a tile with NO container record at all
draws the moveless code-4 refusal. Code 5 remains the SUCCESS close
of a clamped transfer, code 4 the empty close — exactly the typed
0x52 vocabulary the production ledger already consumes. The whole
choreography is executable in the sim (`emit_fuel_pickup_close`,
pinned by `tests/sim/test_fuel_choreography.py`).

## Fuel recovery cascade

When `fuel <= fuel_low_threshold` the bot enters the unified `COLLECT`
mode (the historical `RECOVER_FUEL` / `RECOVER_EQUIPMENT` split was
collapsed 2026-06-24[^5]). The owner runs a single cascade per tick
(see [[bot-behavior-contract]] §3.4):

1. **Lock continuation** — continue a held equipment or fuel target
   from a previous tick when it is still executable and no markedly
   closer candidate beats it.
2. **Equipment pickup** — pick up the best walk-reachable equipment
   in the current viewport. A container only counts as actionable
   when a walk path to it exists inside the current viewport; the
   teleport-to-container fallback was removed 2026-06-26 because the
   server rejects pickups it cannot route to, leaving the container
   on the session-permanent blacklist after one failure.
3. **Fuel pickup** — pick up the best walk-reachable fuel in the
   current viewport when below the learned capacity (skipped at cap
   because "Tank full" is a wasted dispatch). Equipment ranks ahead
   of fuel per the user's gameplay loop.
4. **Sense** — fire a radar to reveal the current viewport when there
   are still unscanned tiles (paid radar covers the full viewport;
   free radar covers a 5×5 around the tank, clipped to viewport
   bounds). When radar is unaffordable, walk toward an unscanned
   tile so the next free radar covers fresh ground.
5. **Hop** — teleport to the **best-value fuel dot**. Candidates
   come from the 0x4C MapData fuel-dot atlas; hard gates are physics
   only (landing tile passable, teleport fuel-affordable, landing
   viewport not freshly scanned), and qualifiers are RANKED by
   ``dots_in_landing_viewport × walkable_fraction ÷ cost`` (user
   ruling 2026-07-18: "more dots, more walkable area. but not a 100%
   rule" — the original 2026-07-03 "100% clean viewport" contract was
   mis-implemented as a 100%-walkable hard filter that rejected 428
   of 622 dots and starved the cascade). The landing auto-pickup
   makes each hop partially self-funding — the server auto-picks a
   fuel container when a teleport lands ON it or CARDINALLY ADJACENT
   to it, no pickup command needed (user law 2026-07-27; corpus
   check the same day: 62 of 82 teleport landings across runs
   214102 + 211712 show the fuel gain arriving before any
   pickup_fuel was sent).
   With an empty atlas (no map open yet this session) the hop
   dispatches ``map_open`` first, guarded by ``map_open_cooldown_ms``
   so a dotless map cannot loop. When no dot qualifies the owner ends
   the session with exit reason ``out_of_fuel`` (``SessionExitError``,
   2026-07-02; previously an uncaught ``ValueError`` crash) rather
   than idle silently. (Dot hop replaced the blind 16-candidate
   compass-ring hop 2026-07-03; that in turn replaced the
   first-qualifying-direction hop 2026-07-01.)[^7]

The fuel-dot atlas was restored 2026-07-03 (stripped 2026-06-22):
`decode_map_data` materialises the skip-RLE dot coordinates and
`WorldService.map_fuel_dots` holds the session-cached copy. Dots are
~40% fresh and every wire-verified dot held high-volume fuel — see
[[map-data-decode]].[^3]

## Marooning hazard

A teleport can land the tank on a one-tile island in a lake (run
20260612-131003: dropped at 87 fuel with no walkable exit).
Marooning is a strand, NOT a death: you cannot deactivate yourself,
walking is free at zero fuel, and radar stays usable (user contract
2026-07-20) — a marooned tank survives indefinitely and can walk out
if any land route exists. Only a true island with no walkable exit is
a permanent strand — and the one famous "island" case was actually a
drivable ferry the model misread (see [[ferry-mechanics]]). A "marooned
escape" used to exist that bypassed the operating-reserve gate to
fire a fuel-dot teleport. The reserve-vetoed band it covered was
narrow -- fuel high enough for the teleport but below the hunt
reserve -- and the actual stranded case (fuel below any teleport
cost) was a permanent strand, not a death. The escape was removed with the rest
of the fuel-dot system 2026-06-22; when lock / pickup / sense / hop
all decline the COLLECT owner ends the session with exit reason
`out_of_fuel` (`SessionExitError`, 2026-07-02 — previously an uncaught
`ValueError` crash).[^4]

[^1]: AIConfigDict in bot/ai/types.py — thresholds lowered from 500→300 in Phase 3d (2026-06-14)
[^2]: user (Austin), 2026-06-11 — "only collect fuel containers with volume >= 500"
[^3]: Fuel-dot history: stripped 2026-06-22 (planner, state, protocol decoder all stopped surfacing dot coordinates), restored 2026-07-03 per user contract ("switch blind viewport hopping to yellow-dot hopping"; "use yellow dot teleporting while en route to the opponent"). Dot freshness ~40% and dot-held volumes >= 762 wire-verified 2026-06-11 (fuel dot probe, 6/6 dots held fuel).
[^4]: Run 131003 2026-06-12 — marooned at 87 fuel on one-tile island (actually a ferry; see [[ferry-mechanics]]). Reserve-bypass escape removed 2026-06-22.
[^5]: AIConfigDict 2026-06-22 — `fuel_critical_threshold` field removed; the COLLECT entry predicate (`should_enter_collect`) and the in-cascade fuel-pickup branch now consume `fuel_low_threshold` directly. The unused `try_collect_critical_fuel` / `try_collect_fuel` non-owner helpers and the `_plan_fuel_recovery` wrapper were deleted at the same time. The fuel-mode and equipment-mode owners themselves were then merged into one `decide_collect_mode` 2026-06-24.
[^6]: Live observation 2026-06-23 00:35:57 in `runs/bot/latest.log`: worldstate logged `Fuel: 195 -> 633 (+438)` from the 0x44 FuelGain, then the next AI decision read `ctx.fuel = 1071` (633 + 438). The 438 ghost was the container's volume being added a second time by `pickup_container`'s local fuel-delta branch on top of the wire's already-correct absolute fuel. Removed in `state/container_mutations.pickup_container` 2026-06-23; the function now only mutates the container registry. Wire-truth flow sites: `sniffer/world_state_dispatch.py` + `sniffer/world_state_containers.py` + `state/mutations.py`.
[^7]: cascade owner `decide_collect_mode` (merged 2026-06-24, see [^5]); step order + hop policy per user contracts 2026-06-26 (walk-only pickups) and 2026-07-03 (quoted in step 5); exit-instead-of-idle per `SessionExitError` change 2026-07-02.
