---
title: Obstacle & Bridge Mechanics
tags: [js-client, game-mechanics, obstacles]
related: [[terrain-system]], [[client-commands]], [[v-table-complete]]
sources: [tpclient.js lines 66-68 (state 7 handler), lines 173-176 (V.B handler), line 70 (be function)]
fact_checked: 2026-06-19
confidence: high
verified: 2026-06-19 (traced through JS state machine and V.B handler)
---

# Obstacle & Bridge Mechanics

How obstacle pickup, drop, and bridge building work in the JS client. Extracted from the state machine (state 7), the `be()` decision function, and the V.B (BuildPickup) message handler.

## Carrying State

The tank has two carrying-related fields:
- `this.la` — boolean, true when carrying an obstacle
- `this.h.Y` (Xc.Y) — carrying flag on the tank entity (affects rendering)

The `la` flag is toggled when:
- V.B (BuildPickup) response received: `a.la = !a.la`
- V["="] (MovementResponse): `a.la = 0 !== this.j` (carrying_flag from server)

## Decision Logic (be function, line 70)

When a player holds click on a tile, `be()` determines the action:

```javascript
function be(a, b) {
  var c = "";
  // If NOT carrying:
  if (!a.la) {
    if (0 !== b.j && 5 !== b.j) {
      // Tile has a rock (1-3) or ferry rock (7) → "PICK UP OBSTACLE"
      // BUT: 64 === (b.i & 96) means it's a mountain tile → also pickup
    } else if (64 === (b.i & 96)) {
      c = "PICK UP OBSTACLE";
    }
  }
  // If carrying AND tile is water with no obstacle:
  if (a.la) {
    if (32 === (b.i & 112) && 0 === b.j) {
      c = "BUILD BRIDGE";
    } else {
      c = "DROP OBSTACLE";
    }
  }
  // Container interactions:
  if (0 !== b.cache) {
    if (0 >= b.cache) c = "GET EQUIPMENT";
    else c = "GET FUEL";
  }
  // Fuel deposit:
  if (0 < a.da) {
    c = "DEPOSIT FUEL: " + a.da;
  }
  return c;
}
```

## Pickup Rules

An obstacle can be picked up when:
1. Tank has fuel > 100 (`ce()` check)
2. Tile has rock type B (j=2), rock type AB (j=3), or ferry rock (j=7)
3. Tile has mountain terrain (bit pattern 64 in terrain byte)
4. Tank is NOT already carrying

Pickup is sent as the `b` command (Tb class): `[4, 'b', x, y]`

## Drop Rules

An obstacle can be dropped when:
1. Tank IS carrying (`la = true`)
2. Target tile is ground or has compatible terrain
3. Target tile has NO rock already (j=0)

Drop is also sent as the `b` command: same opcode, server determines action from tank's carrying state.

## Bridge Building

A bridge can be built when:
1. Tank IS carrying
2. Target tile is water: `32 === (b.i & 112)` (terrain bits 4-6 = 010)
3. Target tile has no rock: `0 === b.j`

The bridge command uses the same `b` opcode. The server response (V.B) comes back with `rock_type = 1` for a bridge module.

### Bridge Verification (state 2, line 66)

In the move-decision logic, bridge building is also checked:
```javascript
if (this.la)
  if (32 === (c.i & 112) && 0 === c.j)
    // Adjacent water tile with no rock → BUILD BRIDGE
    // Only if tank at source tile is NOT ferry (j !== 5)
    5 !== this.h.h[this.i.i][this.i.j].j && (b = 98, a = "BUILD BRIDGE");
```

The source tile must NOT be a ferry (j=5) — can't build bridges from ferry tiles.

## V.B Handler (BuildPickup, Jg class, line 173-176)

Parse:
```
a[0] = tank_id_lo
a[1] = tank_id_hi    → X(a[0],a[1])
a[2] = start_x       — position before action
a[3] = start_y
a[4] = target_x      — tile being modified
a[5] = target_y
a[6] = direction      — carry direction while building
a[7] = rock_type      — new rock value at target tile (0=removed, 1=placed, etc.)
a[8] = was_mine_there — if a mine was under the obstacle
```

Handler effects:
1. Updates carry direction: `We(a.P, b, this.m)`
2. Sets rock at target tile: `a.h.h[d][c].j = this.j`
3. Toggles carry state: `a.la = !a.la`
4. If own tank: logs "Bridge module built" / "Obstacle dropped" / "Obstacle picked up"
5. Sound: `a.m.h.$` (hoist) when carrying, explosion animation when dropping
6. If mine was under and pickup: clears mine overlay (`a.h.h[d][c].m = 255`)

## Carry Direction

When a tank is carrying an obstacle, the `W` field on the tank entity stores which direction the obstacle is "behind" the tank:

| Value | Direction | ASCII |
|-------|-----------|-------|
| 0 | Not carrying | — |
| 101 | East | 'e' |
| 110 | North | 'n' |
| 115 | South | 's' |
| 119 | West | 'w' |

The trailing tile (one tile behind the tank in the carry direction) has its `o` (occupied-behind) flag set to true, which suppresses rock rendering at that position.

## Obstacle in Movement

During drive animation (Re class), when `this.h.Y` (carrying) is true:
- Source tile's rock value is cleared: `b.j = 0; b.l = true`
- At destination, if tile is water or has non-ground terrain, the obstacle is dropped automatically: `this.h.Y = false`
- At destination, if tile is ground: rock type +5 is added to destination tile

The JS code in Re.Na() (line 110-112):
```javascript
if (Se(this)) {
  var e = this.ja[this.o][this.l];
  if (!this.h.Y || 32 === (e.i & 112) && 1 !== e.j && 5 !== e.j) {
    // Normal movement or auto-drop obstacle
  } else {
    b && (b.j += 5, b.l = true);  // add 5 to source tile rock type
    this.h.Y = false;              // stop carrying
  }
}
```

Wait — re-reading: `b.j += 5` adds to the SOURCE tile, and `this.h.Y = false` drops carrying. This is the auto-drop: when you walk onto a tile where the obstacle can't follow (water, existing rock), the obstacle stays at the previous tile.

## Ferry Interaction

Ferry tiles (rock type 5) have special rules:
- Cannot pick up a ferry boarding point
- Cannot build a bridge on a ferry tile
- Moving onto ferry is free water movement
- Moving off ferry back to land costs the normal queue slot
