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
  "src/tankpit_bot/physics": "3790041eff16b20259291f5b60ed9bf184a35c45"
fact_checked: "2026-08-07"
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
  (2026-07-19; re-priced 2026-08-06): refuse when the clamped
  transfer `min(volume, headroom)` is under 3 fuel per Manhattan walk
  tile — the constant is derived from MEASURED walking speed (15
  tiles in 3.30 s, ~0.22 s/tile; diagonals two Manhattan steps) at
  the same ~12.5 fuel/s opportunity value the earlier 25 implied
  under its falsified one-tick-per-tile premise. Adjacent pickups
  always taken, only true dregs refused. The
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
→ `state/self_mutations.py:183::set_self_fuel`, which writes the absolute fuel and
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
pinned by `tests/sim/test_fuel_choreography.py`).[^8]

The close-by-stockedness law and the locally-provable refusals live
in ``physics/supervisor.py`` (added 2026-08-03 after the 20-kill soak
bot-20260802-205105 sent 48 provably-refusable pickups): the sim
emits with them, the bot predicts with them.[^8]

```json claims
{
  "claims": [
    {
      "id": "fuel-pickup-close-code",
      "code": "tankpit_bot.physics.supervisor:fuel_pickup_close_code",
      "formula": "5 (Tank full) while the container keeps any fuel, 4 (Empty container) once drained",
      "probes": [
        {"args": [231], "expect": 5},
        {"args": [1], "expect": 5},
        {"args": [0], "expect": 4}
      ]
    },
    {
      "id": "fuel-pickup-refusal",
      "code": "tankpit_bot.physics.supervisor:fuel_pickup_refusal",
      "law": "A fuel pickup transfers nothing when the container is drained (closes code 4; the walk still executes) or when the tank is at rank fuel capacity against a stocked container (closes code 5, the no-transfer branches). Both are pure functions of client-known state; every other case transfers and is not a refusal (48 live code-5 receipts at exactly-full fuel, bot-20260802-205105)."
    }
  ]
}
```

## The fuel-deposit wire choreography (byte-mined 2026-08-03)

Five manual deposits from the user-piloted session
sniff-20260620-190228 (2026-06-20, the capacity-verification
experiments), every window field-verified:[^9]

| # | Fuel before -> after | Container record | Amount |
|---|---|---|---|
| 1 | 1100 -> 300 | (174,47) remaining 800 | 800 |
| 2 | 294 -> 100 | (174,49) remaining 194 | 194 (max-deposit -> the floor) |
| 3 | (post) 686 | (178,54) remaining 400 | 400 |
| 4 | 1100 -> 800 | (177,51) remaining 300 | 300 |
| 5 | 1100 -> 900 | (178,50) remaining 200 | 200 |

The measured shape, single server tick: **self 0x2E (absolute
post-deposit fuel) + 0x64 FuelDeposit (the same absolute fuel) +
container record x1 with the container's new remaining volume.** Key
laws inside it:[^9]

* The container record comes ONCE — deposits do NOT double their
  record the way every pickup does. On the observer side that is the
  byte-level discriminator between a drink and a deposit.
* ``amount = fuel_before - fuel_after`` equals the record's new
  remaining exactly (fresh containers; a deposit CREATES the
  container record at the clicked tile).
* The max-deposit click leaves exactly ``DEPOSIT_FLOOR`` (window 2:
  294 -> 100) — the [[game-economy]]#deposit-floor law, now seen as
  bytes.
* Third-party deposits are wire-invisible (the 120-day atlas found
  zero cross-tank 0x64s and zero cross-tank refill records despite
  hundreds of inferred refills) — the 0x64 AND the record are
  client-only sends to the depositor.
* The sent command is type 0x07 with an XOR-encoded payload
  (coordinates + amount; encoding not yet cracked — only needed if
  the bot ever deposits).

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
[^2]: user (Austin), 2026-06-11 — "only collect fuel containers with volume >= 500". **This ruling is superseded and is cited here as history, not as current behaviour.** Code truth 2026-08-07: `src/tankpit_bot/bot/ai/collect_pickups.py:253` passes `minimum_volume=1` to `find_fuel_candidates`, so COLLECT drains every viewport container regardless of volume. The reversal is recorded in the same file's docstring at `:204-206`, verbatim: "falsified 2026-07-19; the 2026-06-23 minimum-volume lesson -- fuel is fuel -- applies at the cap end too." The only surviving pickup gate is the per-tile rate rule `pickup_not_worth_walk` at `:189-221`.
[^3]: Fuel-dot history: stripped 2026-06-22 (planner, state, protocol decoder all stopped surfacing dot coordinates), restored 2026-07-03 per user contract ("switch blind viewport hopping to yellow-dot hopping"; "use yellow dot teleporting while en route to the opponent"). Dot freshness ~40% and dot-held volumes >= 762 wire-verified 2026-06-11 (fuel dot probe, 6/6 dots held fuel). The restoration is code truth at `src/tankpit_bot/protocol/decoders/map_data.py:104`, `decode_map_data`, which materialises the skip-RLE dot coordinates; the session-cached copy is `map_fuel_dots` on the world service, read at `src/tankpit_bot/action_lab/density_probe.py:289` and `:337`. Anchors taken 2026-08-07.
[^4]: `runs/bot/bot-20260612-131003.log:7130-7160` (run 131003, 2026-06-12 13:19:08-13:19:10). The tank teleported toward a fuel dot for 81 fuel — `WORLD: Fuel: 168 -> 87 (-81)` — and the server placed it at **(132,180)**, whose rendered viewport row reads `W W W W @ W # #`: water both sides, rock below, the "one-tile island" that was actually a ferry (see [[ferry-mechanics]] [^3], which corrects the same incident's coordinates). Reserve-bypass escape removed 2026-06-22; re-verified 2026-08-06 that no `reserve_bypass` symbol remains anywhere in `src/`.
[^5]: `src/tankpit_bot/bot/ai/types.py:86` — `fuel_low_threshold: int` is the surviving single field, and the class docstring at `:23-29` records the collapse verbatim: "The historical ``fuel_critical_threshold`` was collapsed into this single value 2026-06-22; the two-tier 'polite low vs. emergency critical' distinction was dead because both thresholds had drifted to the same number." Re-verified 2026-08-06: `fuel_critical_threshold` survives ONLY in that historical note — there is no such field anywhere in `src/`. The COLLECT entry predicate is `should_enter_collect` at `src/tankpit_bot/bot/ai/mode_gates.py:16` (split out of `mode_controller.py` 2026-08-07). The three symbols this footnote says were deleted — `try_collect_critical_fuel`, `try_collect_fuel`, `_plan_fuel_recovery` — return zero matches across `src/`, confirming the deletion rather than asserting it.
[^6]: Live observation 2026-06-23 00:35:57 in `runs/bot/latest.log`: worldstate logged `Fuel: 195 -> 633 (+438)` from the 0x44 FuelGain, then the next AI decision read `ctx.fuel = 1071` (633 + 438). The 438 ghost was the container's volume being added a second time by `pickup_container`'s local fuel-delta branch on top of the wire's already-correct absolute fuel. Removed in `state/container_mutations.pickup_container` 2026-06-23; the function now only mutates the container registry. Wire-truth flow sites: `sniffer/world_state_dispatch.py` + `sniffer/world_state_containers.py` + `state/self_mutations.py` (the fuel field's owner since `state/mutations.py` was split by entity).
[^7]: Cascade owner `decide_collect_mode` at `src/tankpit_bot/bot/ai/collect_mode.py:42` — the single owner the fuel-mode and equipment-mode owners were merged into 2026-06-24 (see [^5]). Step order and hop policy follow user contracts of 2026-06-26 (walk-only pickups) and 2026-07-03 (quoted in step 5 above); neither conversation is recorded in the repo, but the policy they set is what the cascade implements. Exit-instead-of-idle is the `SessionExitError` change of 2026-07-02.

[^8]: Byte-mined 2026-08-01 over ~1,600 archive pickup windows; the choreography is executable in the sim as `emit_fuel_pickup_close` at `src/tankpit_bot/sim/emissions.py:213`, pinned by `tests/sim/test_fuel_choreography.py`. The locally-provable refusals are `fuel_pickup_close_code` at `src/tankpit_bot/physics/supervisor.py:52` and `fuel_pickup_refusal` at `:73`. The 48-refusable-pickup receipt is `runs/bot/bot-20260802-205105`. All paths verified present 2026-08-07.
[^9]: Five manual deposits from `runs/sniff/sniff-20260620-190228.capture_session.json` (2026-06-20, the user-piloted capacity-verification session), every window field-verified; the table above is the per-window reading. Capture verified present on disk 2026-08-07. **The deposit command's XOR payload encoding is not cracked** — only the observed response shape is claimed here.
