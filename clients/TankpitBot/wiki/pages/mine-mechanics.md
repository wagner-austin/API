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
fact_checked: "2026-06-20"
confidence: high
verified: 2026-07-21 (cascade re-confirmed in manual capture sniff-20260721-212348; original 2026-06-20 real-combat capture matched the documented mechanic byte-for-byte)
hubs: [combat]
---

# Mine Mechanics

The complete server-side rules for the 3x3 mine placement primitive (`0x4B` MinePlacement / JS `Dg`) and the chain-detonation primitive (`0x45` MineDetonation / JS `dh`), grounded in the user's 2026-06-20 PvP capture and the user-supplied mechanic spec.

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

Total tiles covered by the placement = (placed via `0x4B`) + (detonated via same-tick `0x45`) = up to 9. Anything missing from both is water / terrain / tank.

Per the user's domain knowledge: **the detonated tiles are not re-filled by the original command** -- if the player wants to fill the gaps where enemy mines were destroyed, they must issue a second placement command at the same center.

### Wire layout

`0x4B` MinePlacement (tunneled inside 0x2E):

```
[subtype:1=0x4B] [mine_type:1] [tank_id:2 LE] [count:1] [positions: count*2]
```

Total length is `5 + count * 2` bytes; `count` varies from 1 (only one of the 9 tiles was clear) to 9 (entirely clear 3x3). Real-combat samples logged in `tests/container/test_mines.py`:

| Bytes | Count | Source |
|------|-------|--------|
| 7 B  | 1    | Synthesised in tests for the minimum-count case |
| 15 B | 5    | `MINE_PLACEMENT_15` fixture, solo practice room |
| 17 B | 6    | `unknown_container` from practice-vs-real-20260620-150138, Yuppler's placement |
| 19 B | 7    | `MINE_PLACEMENT_19`, Artax's placement at `(133, 124)` in the same PvP capture |
| 23 B | 9    | Theoretical maximum (clear 3x3) -- no corpus sample yet |

The prior decoder hardcoded `len == 15`; every other length silently dropped to `unknown_container`. Fixed 2026-06-20 via task #79.

## Cascade detonation

When a mine is hit by any source (a shot, an adjacent placement, another mine's blast), the server detonates that mine **and every directly-adjacent mine in the same wire tick**. The cascade is broadcast as up to two `0x45` MineDetonation packets:

1. First packet: the directly-triggered mine.
2. Second packet (if any neighbours existed): every adjacent mine destroyed by the chain.

Real-combat sample (practice-vs-real-20260620-150138, t+62.15s, Artax shot `(134, 126)`):

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
signal; there is no separate removal message.

Both packets arrive within the same WebSocket frame. The bot's world-state must apply each `0x45` independently -- iterating its positions and removing any mine at that tile regardless of team -- and tolerate consecutive packets without double-counting. This is the existing semantics of `_dispatch_mine_detonation`; the cascade tests in `tests/sniffer/test_world_state_dispatch_tank.py::test_mine_cascade_two_packet_chain_real_capture` lock it in.

## Mine-on-mine destruction

A special case of cascade: when your placement lands a mine **adjacent to an enemy mine**, the enemy mine detonates without any shot. The server emits the same-tick pairing of `0x4B` + `0x45`:

```
0x4B MinePlacement type=2 tid=1301 positions=[7 of 9 clear tiles]
0x45 MineDetonation positions=[2 enemy-mine tiles destroyed]
```

The detonation removes the enemy mines; the placement does not re-add friendly mines at those tiles. Locked in by `tests/sniffer/test_world_state_dispatch_tank.py::test_mine_on_mine_destruction_real_capture`.

## Strategic implications for the bot

These follow from the mechanics above; see [[bot-behavior-contract]] §3 for the contract entries that will encode them:

- **Coverage maximization**: a placement against terrain or water yields fewer mines for the same command cost. The bot should prefer mining in open areas when the goal is dense coverage.
- **Mine clearing**: laying a single mine adjacent to a known enemy mine cluster destroys all adjacent enemies for the cost of one own mine -- a 1-for-N exchange.
- **Cascade exploits**: shooting a single mine in a cluster destroys all adjacent mines. Useful for clearing a path through a minefield with one shot.
- **Re-placement to fill gaps**: after mine-on-mine destruction, the original placement leaves the cleared tiles empty. A follow-up placement at the same center will now succeed on those tiles (no enemy mines to detonate).
- **Resource overlap**: mines and containers (equipment / fuel) coexist on the same tile. The bot can deny enemies a container by mining its tile without losing the container.

## What this section does NOT cover yet

- Mine team-decode in `0x4B`: the `mine_type` byte (`data[1]`) and the relationship between `mine_type` and team are observed but not yet cracked from JS. See [[v-table-complete]].
- Damage delivered by mine vs shot. The shoot-event format covers shots; mines drop straight to `0x41` Deactivation as `is_mine_kill=true`.
- Detection range and effectiveness of mines vs different vehicle classes. Out of scope -- bot doesn't expose vehicle class.
