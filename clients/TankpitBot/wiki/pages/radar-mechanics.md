---
title: Radar Mechanics
tags: [radar, scanning, equipment]
related:
  - "[[viewport-frame]]"
  - "[[equipment-system]]"
  - "[[fuel-system]]"
source_paths:
  - "tpclient.js"
  - "runs/bot"
  - "runs/sniff"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-07-06"
confidence: high
hubs: [game-mechanics]
---

# Radar Mechanics

## Two scan types, one wire command

`CMD_RADAR` (0x66) is the only radar wire command. The server decides which type based on inventory:[^1]

- **Extra radar** (extras > 0, enabled): scans the **entire visible viewport** regardless of rank. Client tip text: "Extra radar scans the entire visible screen." Reveals at chebyshev distance 3-12.[^1]
- **Built-in radar** (extras = 0): chebyshev radius **2 + floor(rank/3)** — rank-scaled, NOT a fixed 5x5.[^2][^14]

| Ranks | Radius | Footprint | Verified |
|---|---|---|---|
| recruit(0), private(1), corporal(2) | 2 | 5x5 | private: ~120 built-in scans, zero reveals beyond distance 2 |
| sergeant(3), lieutenant(4), captain(5) | 3 | 7x7 | sergeant: (128,120)→(128,123); lieutenant: (111,129)→(111,126) |
| major(6), colonel(7), general(8) | 4 | 9x9 | major: (234,5)→(238,5); colonel: (165,125)→(165,129) |

The step boundaries are pinned by the sergeant and major measurements (a rival fit with steps at lieutenant/colonel predicted sergeant=2 and major=3; both falsified 2026-07-06).[^14] This resolves the official guide's "higher rank tanks have a larger radar" claim with exact numbers. **The code implication has shipped.** `REGULAR_RADAR_RADIUS = 2` is gone — no such symbol survives anywhere in `src/` — and the rank-derived form is `free_radar_radius(rank)` at `src/tankpit_bot/physics/capacity.py:86`, whose body is `return 2 + rank // 3` and whose docstring restates the three-step table above verbatim.[^15]

## Both scans are clamped to viewport bounds

Neither radar reveals tiles outside the visible viewport. At a viewport edge or corner the built-in radar reveals only the intersection of `(tank±radius)` with the viewport bounds -- a rank 0-2 tank pressed into the top-left of its viewport sees a ~3x3 instead of a 5x5.[^7] Coverage tracking that pretends a free scan always reveals 25 tiles over-claims and lets the forager skip ground that was never actually scanned.

The implication for the bot: track scanned tiles by the actual revealed region (intersection of scan footprint with viewport bounds), not by a fixed-size block centered on the tank.[^7]

## What radar reveals (and what it does NOT)

Radar reveals **fuel containers, equipment containers, and mines** -- entities that are hidden by default on spawn and become visible only after a scan.[^8] Both radar types reveal: extra (paid) covers the full viewport in one shot; built-in (free) reveals tiles within chebyshev radius 2+floor(rank/3) of the tank, clipped to viewport bounds.

Radar does **NOT** reveal enemies. Enemy tanks are always visible to the bot when they enter the viewport via the normal wire stream (0x3D MovementResponse, 0x28 TankEntry, etc.). Firing radar to "search for enemies" is a category error -- any radar dispatched while in HUNT mode is wasted unless it is acquiring nearby mines / containers around a combat tile.[^8]

This deletes a class of bot mistakes: HUNT mode must use map-open or viewport-edge walking to find enemies, never radar.[^8]

## The radar response is a delta sync (reveals + corrections, unchanged omitted)

The wire response to a radar scan (0x4F, JS handler `ch` -- a batch of per-tile cache/overlay writes) is a **delta sync of the scanned area**, not an append-only reveal list:[^12][^13]

- **Newly revealed hidden entities** arrive as cache entries (fuel volume / equipment) and overlay entries (mine team 0-3).
- **Corrections** arrive as explicit removals: a cache entry with value 0 means "this tile is now empty" (247 of 2093 cache entries across the 199-session corpus, scan 2026-07-03), and an overlay entry >= 8 (255 canonical) means "no mine here". The client applies every entry as a raw tile write; the rendered dot and the mouse-hover fuel value both read the same per-tile cache slot.
- **Unchanged already-visible entities are NOT re-sent** (live run 2026-07-01 20:20:10: the teleport landing's 0x5A registered 7 visible containers; the scan-on-landing extra radar's response listed only the 2 hidden ones).

Implication for state tracking: the radar response must never be treated as the complete container set for the viewport, but its explicit entries ARE authoritative -- including the removals (`update_container_from_radar` deletes on volume 0; overlay clears route to `remove_mine`).[^13] The omission-prune (`reconcile_radar_viewport_resources`) is scoped to radar-sourced registry entries only; visible-layer entries are owned by 0x5A/0x43. Before the 2026-07-01 fix the whole-envelope reconcile deleted every visible container on each scan -- the bot would land amid 7 containers, radar, and instantly forget 5 of them (the "picked up only 2 of 7" bug the user observed live). The earlier "lists ONLY newly revealed" wording (2026-07-01) was too strong -- it missed the correction entries the corpus proves the server sends.

## Walking does NOT reveal containers

**Walking is not a reveal action.** Stepping onto a tile that holds a hidden fuel or equipment container does NOT make the container appear -- the bot does not learn about a container by walking on it.[^10] Only radar reveals. This matters because the natural intuition ("explore the viewport on foot to discover what's there") is wrong: a tank can spend its entire fuel budget walking every viewport tile and never see a single container that wasn't already radar-revealed.

What walking IS useful for is **repositioning the tank so the next free radar covers fresh tiles**.[^10] When extras = 0, the free radar reveals (2·radius+1)² tiles at most (25 at ranks 0-2) (fewer at viewport edges -- see [Both scans are clamped to viewport bounds](#both-scans-are-clamped-to-viewport-bounds)). A second free radar from the same tile reveals nothing new. The bot's foraging loop when out of extras is:

1. Free radar (radius square around current position, clipped to viewport).
2. Walk roughly one footprint-width (2·radius+1 tiles) in some direction -- no overlap, no gaps.
3. Free radar from the new position.
4. Repeat until a viewport tile that may hold equipment surfaces -- pick it up, which (eventually) lifts an extra radar -- or until the viewport is fully covered, at which point teleport to a fresh sector.

Walking costs **1 fuel per tile**.[^11] So a single free-radar cycle (walk 5 + radar ~10) burns ~15 fuel for up to 25 fresh tiles. A paid extra radar (also ~10 fuel) covers the whole 16x16 viewport in one shot -- which is why the policy is "always use extra radar when you have one." See [Policy: always use extra radar](#policy-always-use-extra-radar).

## Viewport shifting

In the current game configuration viewport shifting is **OFF**. The viewport never moves when the tank walks; the only way to change which 16x16 region the bot can see is to teleport. Once the bot has scanned every tile inside the current viewport the only forward action is a teleport to a new viewport.[^9]

Each scan with extras available auto-consumes one (count decrements by 1). The bot cannot choose per-scan.[^3] **There is no starting stock — inventory PERSISTS across sessions** (user contract, verbatim, 2026-07-24: *"a new recruit starts with 0 all... if you use them all. log out and log back in. teyre still empty. you have to keep them stocked"*). The "25" our sessions usually open with is the PRIVATE-RANK CAP (20 + 5·rank, [[game-rules]]) carried over from earlier stocked play: consecutive archive sessions carry inventory exactly (radar count 8→7→6→5→4→3 across six logins, each last 0x49 equal to the next session's first; 120/260 consecutive pairs exact, the rest separated by play), 22 sessions open at 0 after exhaustion, and 0/261 sessions ever exceed the rank cap. **Fuel can never block a scan**: the debit clamps to min(10, remaining fuel) and scans keep working at fuel 0 ([[game-economy]] radar row, 2026-07-24).[^15] **Toggle state is wire-visible two ways** (2026-07-24): the 0x74 't' message pushes all five per-slot enabled flags (join + every toggle), and every 0x49 inventory response carries each slot's flag in bit 7 of its count byte — both already decoded (`EquipmentToggleDict.enabled`, `InventoryDict.enabled`); verified against the user's live panel (armor+missiles "(disabled)" ↔ `enabled [False, True, False, True, True]` in the same day's capture).

## Policy: always use extra radar

Never ration extras via the toggle. The viewport sweep is always worth one extra. Keep stock UP through reliable equipment collection. A proposed "radar floor" (disable extras when low) was explicitly rejected.[^4]

**2026-07-31 refinement — spend-gating, not toggle rationing:** after the 100-kill run twice drained extras to 0 (desync-rescan burn, ferry-orbit landing scans) the user ruled the death spiral itself unacceptable ("if the bot runs out of radar ever ... its like dead in the water cuz it takes so long to restock via free radar"). The shared economics rule (`radar_spend_worthwhile`, [[bot-behavior-contract]] §3.4) now escalates the reveal bar as stock falls: extras ≥ 2 → 32 uncovered tiles; the LAST extra → 128 tiles (half the viewport — the final paid sweep buys only a near-full reveal, never a sliver); extras 0 → any tile. The toggle stays enabled and every fired scan still auto-consumes the extra, so the 2026-06-12 rejection stands untouched — this gates *whether a discretionary scan fires*, not *which radar answers it*.

## Death spiral at 0 extras

At 0 extras: 25-tile reveal vs 324-tile. Equipment discovery collapses, refill stalls. Three consecutive runs at 0 gained duals/homings but zero radars.[^5]

## Radar for fuel is not waste

One viewport sweep reveals ~10 containers. Live data: 32 pickups vs 9 dot hops in one run. Dots are ~40% fresh and one-at-a-time. Spending a radar to surface ten containers is high-value.[^6]

## Equipment refill

Extra radars come from equipment containers — plus ONE measured
exception (corpus-cracked 2026-07-22): a kill scored while your
extra-radar count is ZERO grants a silent mercy bundle including
+1–2 radar (deterministic, 5/5 vs 0/254; see [[equipment-system]]).
The pre-sweep "no other source" claim was falsified by that bundle —
it held for every non-kill path.[^5]

## Machine-checked claims

Binding for the built-in radar radius formula ([[physics-module-roadmap]]
Phase 1; probes chosen at the measured step boundaries[^14]). Verified
by the `physics_claims` guard stage on every `make check`.

```json claims
{
  "claims": [
    {
      "id": "free-radar-radius",
      "code": "tankpit_bot.physics.capacity:free_radar_radius",
      "formula": "chebyshev radius 2 + rank // 3",
      "probes": [
        {"args": [0], "expect": 2},
        {"args": [2], "expect": 2},
        {"args": [3], "expect": 3},
        {"args": [5], "expect": 3},
        {"args": [6], "expect": 4},
        {"args": [8], "expect": 4}
      ]
    }
  ]
}
```

## The s9-2 correction: coverage is not mine knowledge (2026-08-21)

The radar-spend economics (flags s9-2/4/5) skip the landing radar when
the viewport sits in live scan coverage — correct for CONTAINER
knowledge, and stated too broadly for mines ("the mines the
un-suppression exists to reveal are known"). Mines are dynamic
(practice bots ring themselves continuously) and reveals are a
separate, decaying knowledge layer; fleet-shared coverage
([[fleet-coordination]]) marks ground live without the local reveals
ever having been ingested. In 7 of the 11 archived displacement-orbit
runs the skip sat inside the orbit window, suppressing exactly the
scan that would have repaired the mine beliefs. The law now: a fresh
displacement ([[teleport-mechanics]] § displacement evidence) forces
the landing repair radar regardless of coverage; only a
displacement-free landing in live coverage skips the spend.

[^1]: Originally a user (Austin) statement of 2026-06-12, corroborated at the time by the client's own tip text; the conversation itself has no transcript in the repo. The claim is now carried by the code and stated in its docstring: `src/tankpit_bot/physics/capacity.py:97-98` — "Only the extra radar sweeps the full viewport regardless of rank." That symbol is bound as claim `free-radar-radius` in this page's `json claims` block (`:112`), so `physics_claims` verifies its behaviour on every `make check`.
[^2]: ~120 built-in scans across the 2026-06-12 captures (`runs/bot/bot-20260612-*.capture_session.json`, 9 sessions) — zero reveals beyond chebyshev 2, hits at radius 1 and 2 only, replacing an unmeasured 7x7 assumption carried from April. Consistent with the shipped law rather than merely asserted alongside it: `free_radar_radius(rank) = 2 + rank // 3` (`src/tankpit_bot/physics/capacity.py:86-100`) returns 2 for ranks 0-2, and those sessions were flown at Private (rank 1) — so radius 2 is exactly what the formula predicts for that corpus, and the sweep cannot speak to the higher-rank steps. Those are pinned instead by the four axial measurements in [^14] and by the probe grid on claim `free-radar-radius`.
[^3]: run 20260611-062453 — extra count series 10→9→...→3, one consumed per scan
[^4]: user (Austin), 2026-06-12 — "ALWAYS use extra radar — never ration via toggle". The ruling is carried in code at `src/tankpit_bot/bot/ai/context.py:382-390`, whose docstring names it verbatim as the thing the current design is NOT: the reserve bar is "spend-gating inside the existing economics rule, NOT the extras-toggle rationing rejected 2026-06-12 -- the extras slot stays enabled and any scan that does fire uses the extra."
[^5]: three consecutive runs at extras=0 — gained duals/homings but zero radars; verified equipment is sole radar source. The consequence is encoded at `src/tankpit_bot/bot/ai/context.py:392-397`: the last extra is spent only above a 128-tile reveal floor because "once it is gone, discovery collapses to the built-in radius-2 scan and restock stalls". The built-in fallback is `free_radar_revealed_tiles` at `src/tankpit_bot/state/scan_coverage.py:120`.
[^6]: run 20260613-064xxx — single fuel scan led to ~10 "Picked up container" events; 32 pickups vs 9 dot hops all run
[^7]: user (Austin), 2026-06-21 — "it doesnt scan outside the viewport... only ever need to be able to accurate track the current tiles within the viewport"; viewport-edge intersection rule. Encoded as the clip in `free_radar_revealed_tiles` (`src/tankpit_bot/state/scan_coverage.py:120-145`), which returns only tiles inside the current viewport bounds; the bounds themselves are `VIEWPORT_SPAN = 16` at `src/tankpit_bot/sim/viewport_window.py:27` with the containment test at `:135`.
[^8]: User (Austin) statement of 2026-06-21, verbatim: "radar doesnt detect enemies. it is strictly and only for finding equipment and fuel and mines. those are hidden by default on spawn and revealed by radar". No transcript exists, but the wire format settles it structurally: the radar result `0x4F` decodes to `RadarScanResultDict` with exactly three payload fields — `containers`, `mines`, `mine_clears` (`src/tankpit_bot/protocol/decoders/radar.py:212-217`) — and **no tank or enemy field of any kind**. Enemy detection is a DIFFERENT message on a different command: `decode_enemy_detection` at `:53` produces `msg_type=0x48`, carrying an enemy's x/y/team/rank/tank_id, and is driven by `CMD_NEAREST_ENEMY = 104` (the 'e' key), machine-checked on [[client-commands]]. Confusing the two is easy from the wiki's prose alone; the decoders keep them entirely separate. Verified 2026-08-06.
[^9]: user (Austin), 2026-06-21 — "we have viewport shifting off. so the viewport will never move. the only way is to teleport". The shift path exists in the sim but is inert under this configuration: `apply_scope_shift` at `src/tankpit_bot/sim/viewport_window.py:92`, whose module docstring at `:4` records that the 0x5A patch is the ONLY message that sets the viewport and `:88` states the OFF condition. Later contextualised (2026-07-17) as a bot-side configuration rather than a game-wide law — see [[viewport-shift-protocol]].
[^10]: user (Austin), 2026-06-22 — "walking over contianers doesnt reveal them. only rsdar reveals euqipment snd fuel" (typos in the original). Radar being the sole reveal path is why coverage is tracked at all: `src/tankpit_bot/state/scan_coverage.py:120` is the only producer of revealed tiles, and `:232` records the extra-radar reveal footprint against the flag-triage receipts.
[^11]: User (Austin) statement of 2026-06-22, verbatim: "wlak does consume 1 fuel per tile btw". No transcript exists in the repo, but the number is not left resting on it — `WALK_COST_PER_TILE = 1` at `src/tankpit_bot/physics/costs.py:15`, bound as claim `walk-cost` on [[game-economy]] (`:310-312`) and compared by value on every `make check`.
[^12]: live run 2026-07-01 20:20:10 — landing 0x5A registered 7 visible containers, scan-on-landing radar listed only the 2 hidden ones; entity_alignment samples tick 5 vs tick 6
[^13]: corpus scan 2026-07-03 (199 sessions, 1817 0x4F bodies): 2093 cache entries — 247 removals (value 0), 1074 equipment (0xFFFF), 772 fuel; 545 overlay entries all team values 0/1/3; 0 top-level (untunneled) 0x4F. JS ch handler tpclient.pretty.js:4800-4813.
[^14]: Four manual axial measurements on own tanks, taken by the user (Austin) 2026-07-06: lieutenant (111,129)→(111,126)=3; colonel (165,125)→(165,129)=4; sergeant (128,120)→(128,123)=3; major (234,5)→(238,5)=4. Sergeant and major were chosen specifically to discriminate the two candidate step formulas. **Every one of the four fits the shipped formula exactly** — `free_radar_radius(rank) = 2 + rank // 3` at `src/tankpit_bot/physics/capacity.py:86-100`, against the ranks in `protocol/constants.py` (`SERGEANT = 3` → 3, `LIEUTENANT = 4` → 3, `MAJOR = 6` → 4, `COLONEL = 7` → 4). Re-checked 2026-08-06. The formula is probe-verified on every `make check` by claim `free-radar-radius` (`:112`), whose grid pins the two step boundaries at ranks 2/3 and 5/6 — the discriminating points these measurements were designed to hit.
[^15]: archive sweep 2026-07-24: `analysis_scripts/mine_radar_floor.py` — first-inventory histogram (46 sessions at exactly 25) and isolated-radar-window deltas by fuel bucket (fuel<50: 11 full debits, 2 clamped −6, 1 clamped −3, 14 zero-debit at fuel 0). Re-run to re-derive.
