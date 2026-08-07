---
title: Mine Mechanics
tags: [game, combat, mines, world-state]
related:
  - "[[shoot-event-format]]"
  - "[[decode-coverage]]"
  - "[[bot-behavior-contract]]"
  - "[[gameplay-loop]]"
source_paths:
  - "runs/sniff/sniff-20260620-150155.capture_session.json"
  - "tpclient.js"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-07-28"
confidence: high
verified: 2026-07-21 (cascade re-confirmed in manual capture sniff-20260721-212348; original 2026-06-20 real-combat capture matched the documented mechanic byte-for-byte)
hubs: [combat]
---

# Mine Mechanics

The complete server-side rules for the 3x3 mine placement primitive (`0x4B` MinePlacement / JS `Dg`) and the chain-detonation primitive (`0x45` MineDetonation / JS `dh`), grounded in the user's 2026-06-20 PvP capture and the user-supplied mechanic spec.[^1]

## Placement primitive

A player or bot issues one mine command. Mines are **not an inventory item** — nothing is consumed from the 0x49 equipment slots; the command just costs its flat 10 fuel ([[game-economy]]). The server attempts a **3x3 placement** centered on the placer's tile, **clipped to the visible viewport** (user contract 2026-07-21: standing on the viewport edge places 6 instead of 9), walking each of the remaining tiles in turn and classifying it:

| Tile contents at placement time | Server action |
|---|---|
| Outside the visible viewport | Skip silently. The 3x3 never extends past the viewport edge. |
| Water | Skip silently. Tile is unchanged. No wire signal. |
| Terrain block (rock, etc.) | Skip silently. |
| Another tank (any team) | Skip silently. |
| Enemy mine | Detonate that mine **1:1**. Tile is now **empty** -- the placement does NOT add a friendly mine here. Server includes the tile in the same-tick `0x45` MineDetonation payload. |
| Clear ground | Place your mine. Server includes the tile in the `0x4B` MinePlacement payload. |
| Equipment / fuel container | Place your mine. Containers can coexist with mines on the same tile. |

Total tiles covered by the placement = (placed via `0x4B`) + (detonated via same-tick `0x45`) = up to 9. Anything missing from both is water / terrain / tank.[^1]

Per the user's domain knowledge: **the detonated tiles are not re-filled by the original command** -- if the player wants to fill the gaps where enemy mines were destroyed, they must issue a second placement command at the same center.[^2]

### Wire layout

`0x4B` MinePlacement (tunneled inside 0x2E):[^3]

```
[subtype:1=0x4B] [mine_type:1] [tank_id:2 LE] [count:1] [positions: count*2]
```

Total length is `5 + count * 2` bytes; `count` varies from 1 (only one of the 9 tiles was clear) to 9 (entirely clear 3x3). Real-combat samples logged in `tests/container/test_mines.py`:[^3]

| Bytes | Count | Source |
|------|-------|--------|
| 7 B  | 1    | Synthesised in tests for the minimum-count case |
| 15 B | 5    | `MINE_PLACEMENT_15` fixture, solo practice room |
| 17 B | 6    | `unknown_container` from practice-vs-real-20260620-150138, Yuppler's placement |
| 19 B | 7    | `MINE_PLACEMENT_19`, Artax's placement at `(133, 124)` in the same PvP capture |
| 23 B | 9    | Theoretical maximum (clear 3x3) -- no corpus sample yet |

The prior decoder hardcoded `len == 15`; every other length silently dropped to `unknown_container`. Fixed 2026-06-20 via task #79.[^4]

## Walk-over detonation: single mine, movement stops

Stepping onto a mined tile detonates **only the mine occupying that
tile**, and the movement is stopped there — the walk does not
continue through the field, and the walk-over does NOT trigger the
adjacent-mine cascade below. Consequence: **a single movement can
never eat more than one mine hit** (45 fuel, [[game-economy]]
walking-into-a-mine row) — a tank crossing a dense field pays one
mine per movement command, not a chain.[^6]

## Teleport landings displace off ENEMY mines (live-proven 2026-07-28; team scope archive-proven 2026-08-06)

A teleport aimed AT an enemy-mined tile never lands on it: the server
displaces the tank to an adjacent tile, charges only the pure
teleport cost, and the mine survives. User law (verbatim: "it
displaces you"), wire-confirmed the same day by the mine-landing
probe (`make mine-landing-probe`, run `mine-landing-20260728-161432`):
3/3 teleports aimed dead-on at enemy mines landed exactly one tile
beside them, `extra_loss = 0` on every attempt, all three mines
intact in the registry afterward. Enemy-mine tiles join occupied
tiles in the displacement-excluded set.[^7]

**Team scope, archive-proven 2026-08-06**
(`analysis_scripts/mine_displacement_semantics.py`, 329 captures):
displacements off a known-mine tile split **1,227 enemy vs 2
friendly**, and 20 exact landings sat cleanly ON known friendly
mines — own-color mines do not displace, exactly mirroring their
walk-passability. The sim already encodes this
(`sim/actions.py::_tile_blocked_for_landing`). Corollary of the same
sweep: **an EXACT landing is a mine-clear receipt** — 88 exact
landings on live enemy-mine beliefs mean the belief was stale
(off-screen walk-over detonations never reach the wire); a live
enemy mine displaces deterministically (534/534 in
bot-20260805-173034), so landing on the tile proves the mine gone.[^disp]

Doctrine consequence ([[bot-behavior-contract]] §6, ring-2 item
RESOLVED): aiming an approach teleport at a mine-ringed enemy is
self-protecting -- the landing can never touch the ring -- so no
aim change is needed against miners. The ring's only remaining
threat is walking through it (one 45-fuel hit per movement,
walk-over law above).

## Cascade detonation

When a mine is hit by a NON-movement source (a shot, an adjacent placement, another mine's blast — walk-over is excluded, see above), the server detonates that mine **and every directly-adjacent mine in the same wire tick**. The cascade is broadcast as up to two `0x45` MineDetonation packets:[^5]

1. First packet: the directly-triggered mine.
2. Second packet (if any neighbours existed): every adjacent mine destroyed by the chain.

Real-combat sample (practice-vs-real-20260620-150138, t+62.15s, Artax shot `(134, 126)`):[^5]

```
0x53 Shoot tid=1301 src=(131,122) tgt=(134,126) weapon=0
0x45 MineDetonate positions=[(134, 126)]                          # directly hit
0x45 MineDetonate positions=[(135, 126), (134, 127), (133, 126),
                             (135, 127), (135, 125), (133, 127)]   # 6-mine cascade
```

Re-confirmed 2026-07-21 (manual capture sniff-20260721-212348, t+173.91): a
single shot at `(54,170)` produced `0x45 [(54,170)]` plus a same-tick second
packet `0x45 [(55,170),(54,171),(55,171)]` — 4 mines destroyed by one shot,
matching the user's on-screen count exactly. `0x45` IS the mine-removal wire
signal; there is no separate removal message.[^5]

Both packets arrive within the same WebSocket frame. The bot's world-state must apply each `0x45` independently -- iterating its positions and removing any mine at that tile regardless of team -- and tolerate consecutive packets without double-counting. This is the existing semantics of `_dispatch_mine_detonation`; the cascade tests in `tests/sniffer/test_world_state_dispatch_tank.py::test_mine_cascade_two_packet_chain_real_capture` lock it in.[^5]

**Rank-dependence (user domain knowledge, 2026-07-30):** the shot-triggered
cascade is rank-gated on the SHOOTER — a recruit's shot destroys only the
directly-hit mine (no cascade); private and above destroy the target mine
plus every cardinally or diagonally adjacent mine (full 3x3, up to 9).
Both corpus cascade samples above were fired at private, consistent with
this. Not yet wire-discriminated at recruit — needs a recruit-rank capture
to confirm the 1-mine case.[^8]

**Shot clearance (same ruling, refined 2026-07-30 flag s3-14):**
shooting a mine requires a CLEAR STRAIGHT shot — the shot line is
interrupted by mountains/rock terrain and by land movable obstacles,
and by NOTHING else: other mines never block the line ("we can shoot
over other mines of course. just not mountains or movable blocks on
land"), so one shot into a dense field still lands on the aimed tile
and its 3x3 clears — potentially exposing several containers at
once. Mine shots consume NO inventory (user law 2026-07-30:
"shooting a mine doesnt cost any inventory. you click and it shoots
a single shot, and destroys the mines on the ui. and it must also
tell the server too") — the clearance costs only the shot's tick. The homing/missile over-terrain arc
applies to TANK targets only, never to mines. Tactical consequence
(flags 4/8, [[flag-triage-20260729]]): equipment or fuel covered by an
enemy mine field needs no path clearing — one LOS shot at the
container tile detonates up to the full 3x3 of covering mines at
private+, then a teleport lands and collects.[^8]

## Mine-on-mine destruction

A special case of cascade: when your placement lands a mine **adjacent to an enemy mine**, the enemy mine detonates without any shot. The server emits the same-tick pairing of `0x4B` + `0x45`:[^2]

```
0x4B MinePlacement type=2 tid=1301 positions=[7 of 9 clear tiles]
0x45 MineDetonation positions=[2 enemy-mine tiles destroyed]
```

The detonation removes the enemy mines; the placement does not re-add friendly mines at those tiles. Locked in by `tests/sniffer/test_world_state_dispatch_tank.py::test_mine_on_mine_destruction_real_capture`.[^2]

[^1]: captures on disk: `runs/bot/practice-vs-real-20260620-150138.capture_session.json` (the 2026-06-20 PvP session; also the fixture source for six test modules) and `runs/sniff/sniff-20260620-150155.capture_session.json` (frontmatter-pinned); user mechanic spec delivered 2026-06-20 alongside the capture — every spec claim in the placement table is wire-checked against these files.
[^2]: user (Austin) 2026-06-20 mechanic spec (the "user-supplied mechanic spec" this page is grounded in); the non-refill is wire-visible in the same-tick `0x4B`+`0x45` pair — detonated tiles are absent from the placement payload — locked by `tests/sniffer/test_world_state_dispatch_tank.py::test_mine_on_mine_destruction_real_capture` (line 264, verified 2026-07-23).
[^3]: decoder truth on disk: `src/tankpit_bot/container/decoders/mines.py` (`decode_mine_placement`); byte-length fixtures `MINE_PLACEMENT_15`/`MINE_PLACEMENT_19` exercised in `tests/container/test_mines.py:37/:56`; JS sender `Dg` in `tpclient.js` (blob-pinned in frontmatter).
[^4]: commit `59a097e1` (2026-06-20, "Multi-record ContainerPickup + dispatch-layer pickup dedup") introduced the variable-length `MINE_PLACEMENT_19` fixture and the count-driven decode — the `len == 15` hardcode and its removal are both visible in that commit's diff in git history. "Task #79" was the session-internal tracker id, kept as a historical label only.
[^7]: probe artifacts on disk: `runs/probe/mine-landing-20260728-161432.json` (three attempts: aim (131,124)→land (132,124), aim (146,93)→land (147,93), aim (145,93)→land (145,92); each `extra_loss: 0`, `mine_survived: true`) with paired `.log` + `.capture_session.json`; probe source `src/tankpit_bot/action_lab/mine_landing_probe.py`; user law (Austin, 2026-07-28, verbatim): "it displaces you".
[^6]: user (Austin), 2026-07-28, verbatim: "mines explode on walk over and its only the mine occupying the tike you walk on that explodes and your kovement is stopped. so you cant get hit hy more than one mine at a single time" (typos in the original) — executable in the sim at `src/tankpit_bot/sim/movement.py:188` (`_unrevealed_enemy_mine_at`, the per-tile test that stops the mover) and `src/tankpit_bot/sim/actions.py:300` (`process_mine_press`), which together enforce one-mine-per-step rather than a cascade. Domain law, correcting the assistant's cascade-on-walk misreading of the shot-triggered chain samples; consistent with the wire-verified 45-fuel walk-into-mine sample ([[game-economy]] t+373.35s row) and the 50-kill run's mine deaths booking single hits.
[^8]: user (Austin), 2026-07-30 flag narration during bot-20260729-232252, verbatim: "shots at mines dont go over terrain so it has to be a clear shot... one shot for recruits blows up one mine. for any rank above recruit you blow up the target mine, plus each adjacent mine cardinally or diagonally adjacent... a single shot at the equipment or fuel location will destroy 9 mines as a private and above"; homing/missile over-terrain arc for tanks only, same narration.
[^5]: cascade wire samples on disk: `runs/bot/practice-vs-real-20260620-150138.capture_session.json` t+62.15s (1+6 two-packet chain) and `runs/sniff/sniff-20260721-212348.capture_session.json` t+173.91 (1+3 chain, user-counted on screen — the frontmatter `verified:` field records this re-confirmation); dispatch semantics locked by `tests/sniffer/test_world_state_dispatch_tank.py::test_mine_cascade_two_packet_chain_real_capture` (line 350, verified 2026-07-23).

## Strategic implications for the bot

These follow from the mechanics above; see [[bot-behavior-contract]] §3 for the contract entries that will encode them:

- **Coverage maximization**: a placement against terrain or water yields fewer mines for the same command cost. The bot should prefer mining in open areas when the goal is dense coverage.
- **Mine clearing**: laying a single mine adjacent to a known enemy mine cluster destroys all adjacent enemies for the cost of one own mine -- a 1-for-N exchange.
- **Cascade exploits**: shooting a single mine in a cluster destroys all adjacent mines. Useful for clearing a path through a minefield with one shot.
- **Re-placement to fill gaps**: after mine-on-mine destruction, the original placement leaves the cleared tiles empty. A follow-up placement at the same center will now succeed on those tiles (no enemy mines to detonate).
- **Resource overlap**: mines and containers (equipment / fuel) coexist on the same tile. The bot can deny enemies a container by mining its tile without losing the container.

## What this section does NOT cover yet

- Mine team-decode in `0x4B`: the `mine_type` byte (`data[1]`) and the relationship between `mine_type` and team are observed but not yet cracked from JS. See [[v-table-complete]].
- ~~Teleport landing on a mined tile~~ **ANSWERED 2026-07-28**: see §Teleport landings displace off mines above.
- Damage delivered by mine vs shot. The shoot-event format covers shots; mines drop straight to `0x41` Deactivation as `is_mine_kill=true`.
- Detection range and effectiveness of mines vs different vehicle classes. Out of scope -- bot doesn't expose vehicle class.

## Shot-clearance claims (physics binding)

The lifted line-of-sight primitives implement the shot-clearance law
recorded in this page (mine shots need a clear straight line; rock
and land movable blocks interrupt; water and intermediate mines never
do; homing/missile over-terrain arcs are tank-targets-only). Neither
symbol is int-probe-able — one takes a terrain protocol view, the
other returns the Bresenham raster — so both bind as ``law`` claims
(schema extension 2026-07-30, [[physics-module-roadmap]]).

```json claims
{
  "claims": [
  {
    "id": "shot-line-tiles",
    "code": "tankpit_bot.state.line_of_sight:shot_line_tiles",
    "law": "The shot line is the Bresenham raster between shooter and target with BOTH endpoints excluded - the shooter's own tile and the target tile never occlude their own shot (user law 2026-07-30, flag s3-14 narration)."
  },
  {
    "id": "shot-line-clear",
    "code": "tankpit_bot.state.line_of_sight:is_shot_line_clear",
    "law": "A straight shot is interrupted ONLY by rock/mountain terrain and land movable blocks (bridge/land/stacked/ferry-rock wire types); water never blocks, and mines, containers, and tanks on intermediate tiles never block (user law, verbatim 2026-07-30: 'we can shoot over other mines of course. just not mountains or mobable blocks on land')."
  }
  ]
}
```
[^disp]: `analysis_scripts/mine_displacement_semantics.py` — the 2026-08-06 archive sweep behind the team-scope claim; its artifact is `runs/analysis/displacement_semantics.json`. Both verified present 2026-08-07.
