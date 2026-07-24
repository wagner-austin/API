---
title: Client State Machine
tags: [js-client, state-machine, game-loop]
related:
  - "[[js-source-map]]"
  - "[[client-commands]]"
  - "[[client-constants]]"
source_paths:
  - "tpclient.js:49"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (every state transition traced through JS)
hubs: [js-client]
---

# Client State Machine

The game client uses a numeric state field (`this.s`) to track what the player is currently doing. This controls which commands can be sent and how input is processed.

## States

| Value | Name | Description | Next States |
|-------|------|-------------|-------------|
| -1 | UNINITIALIZED | Before game load | → 0 |
| 0 | PROCESSING | Command sent, waiting for server | → 1 when response arrives |
| 1 | **IDLE** | Ready for input | → 2-13 on action |
| 2 | MOVE_PENDING | Move command queued | → 0 on send |
| 3 | FIRE_PENDING | Fire command queued | → 0 on send |
| 4 | RADAR_PENDING | Radar scan queued | → 0 on send (tick=50ms) |
| 5 | FUEL_PICKUP | Fuel container pickup queued | → 0 on send |
| 6 | EQUIP_PICKUP | Equipment container pickup queued | → 0 on send |
| 7 | OBSTACLE_ACTION | Build/pickup/drop obstacle queued | → 0 on send |
| 8 | MAP_OPEN | Map open command queued | → 0 on send |
| 9 | ENEMY_DETECT | Nearest enemy detect queued | → 0 on send |
| 10 | FUEL_DEPOSIT | Fuel deposit command queued | → 0 on send |
| 12 | MINE_PLACE | Mine placement queued | → 0 on send |
| 13 | SCOPE_CHANGE | Viewport scope change queued | → 0 on send |

Note: State 11 is skipped in the code.

## State Transitions

### From IDLE (1)

The IDLE state is where most input is processed. Entry points:

- **Click on tile** → state 2 (MOVE) or state 3 (FIRE) depending on what's there
- **Double-click** → state 3 (FIRE) always
- **Click+hold** → state 5 (FUEL), 6 (EQUIP), 7 (OBSTACLE), or 10 (DEPOSIT) based on tile content
- **Keyboard shortcut** → state 4 (RADAR), 8 (MAP), 9 (ENEMY), 12 (MINE), 13 (SCOPE)
- **Chat command** → fires through Bb controller (separate from state machine)

### Move vs. Fire Decision (state 2)

When a tile is clicked from IDLE, the code checks (line 66-67):

1. If tile has enemy tank (different team) OR has mine from another team:
   → FIRE (state 3)
2. If tank is carrying an obstacle and target is water with no obstacle:
   → BUILD BRIDGE (state 7 via `b` command)
3. If carrying and adjacent to movement direction:
   → DROP OBSTACLE (state 7)
4. If fuel > 100 and tile has a rock/obstacle:
   → PICK UP OBSTACLE (state 7)
5. Otherwise:
   → MOVE (state 2 via `p` command)

### Fuel Check (line 71)

Before certain actions, `ce(a)` checks if fuel > 100. If not, logs "Insufficient fuel" and returns to IDLE. Applies to: ENEMY_DETECT (9), MINE (12), FUEL_DEPOSIT (10).

## Tick Loop (pb function, lines 49-51)

The main game loop runs at variable tick rate:

```
1. If animations playing:
   - Process action queue (draw, advance, cleanup)
   - Tick rate = animation-specific (3-33ms)

2. If queued server command (Qa):
   - Execute it
   - Clear Qa
   
3. If server message queue has entries:
   - Process next message
   - Tick rate = ACTION_PENDING (0ms = immediate)
   
4. Else:
   - Tick rate = IDLE (200ms)
   - Run keep-alive check (30s)
   - Run tip display (60s)
```

## Action Queue (vb class)

The action queue (`this.h.j`) holds animation objects (Re for drive, yf for shoot, uf for radar, lf for explosion). Each animation:

1. `.start()` — initialize (play sound, set up state)
2. `.Na()` — advance one frame
3. `.sa()` — draw current frame
4. `.ra()` — erase previous frame (dirty-rect)
5. `.Ma()` — cleanup on complete (spawn sub-animations)
6. `.j` flag — true when complete → removed from queue

## Command Processing (Ha=true gate)

The `Ha` flag gates whether the state machine processes the next queued action:

```javascript
// Set to true when:
0 < this.h.j.actions.length  // animations done
|| 1 === this.s              // idle
|| 0 === this.s              // processing done
|| -1 === this.s             // uninitialized
|| this.map.h                // map open
|| this.ga                   // scope mode

// Each state case processes its command, sends to server, resets to state 0
```

## Input Handling During States

- **State 1 (IDLE)**: All input accepted
- **State 0 (PROCESSING)**: Input blocked (waiting for server)
- **State -1 (UNINIT)**: Input blocked
- **During animations**: Input queued (max 2 commands in Bb chat queue)
- **Map open**: Only map clicks (teleport), scope, and close-map accepted
- **Scope mode (ga=true)**: Mouse movement adjusts viewport offset, click confirms

## Deactivation Flow

When deactivated (ob function, line 61):
1. State → 0
2. Play death sound
3. Close map if open
4. Cancel scope mode
5. Reset fuel bar to 0
6. Show deactivation overlay (Yd class: yellow text on red background)
7. Set tick rate to 100ms (Mb)
8. Enter repair wait (timer bar fills over 20 seconds)
9. On reactivate: reset state, reload equipment, resume IDLE

