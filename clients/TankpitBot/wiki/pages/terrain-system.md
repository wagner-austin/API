---
title: Terrain System
tags: [js-client, terrain, rendering]
related:
  - "[[js-source-map]]"
  - "[[rendering-pipeline]]"
  - "[[viewport-frame]]"
source_paths:
  - "tpclient.js:145"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (terrain byte encoding fully traced through JS)
hubs: [js-client]
---

# Terrain System

How TankPit encodes terrain types, determines tile adjacency, and renders the ground. Extracted from the tile engine in tpclient.js.[^1]

## Terrain Classification

The game has exactly 3 base terrain types (dg array, line 145):[^1]

| Value | Type | Walkable | Description |
|-------|------|----------|-------------|
| 0 | Ground | Yes | Open land, grass |
| 1 | Rock/Mountain | No | Impassable terrain |
| 2 | Water | No* | Water tiles (*walkable with ferry) |

Terrain is determined by pixel-sampling the map image at the viewport position (sg function, line 152). The function reads `getImageData()` from the map canvas and classifies each pixel's blue channel:[^1]

```javascript
function e(z) {
  return z === k.l ? 2          // water blue → water
       : 120 === z ? 2          // also water
       : z === k.m ? 0          // dark green → ground
       : z === k.o ? 1          // medium green → rock
       : 0 === z ? (h=true, 0)  // black (border) → ground + edge flag
       : -1;                    // unknown → retry with ±1
}
```

The three reference colors (`k.m`, `k.l`, `k.o`) are calibrated per-map from the most common blue-channel values in the image.[^1]

**Coast tiles are art, not a terrain class.** The shoreline "water edge" look is generated per-tile by the adjacency renderer (`ug`, line 155) from the 3-class neighborhood — there is no fourth walkable-coast type anywhere in the client. Server-verified live 2026-07-26 (movement probe, capture `runs/probe/coast_test.movement_probe.json`): from shore tile (130,124), single-step moves onto the adjacent blue tiles (129,124) and (130,125) both rejected with `0x52 err=1` (`cant_go`), zero fuel spent, tank unmoved — while the walk TO the shore tile was accepted normally. Blue is water; water is impassable on foot.[^2]

**Containers spawn ON water tiles near shores and are pickable from adjacent land.** 19 server-confirmed pickups across runs bot-20260726-094309/-145124 consumed containers sitting on water-classified tiles, in every case with the tank on a land tile exactly 1 cardinal step away (pickup reach = 1). Water containers ≥2 tiles from the nearest land are unreachable — the equipment-hop atlas accumulates them and `find_teleport_landing_tile` correctly declines them (`no_landing`), which is the dominant, correct source of `hop_declined` spam. A `0x5A` patch enumerating such a tile carries `terrain_type=0`, which means "no dynamic block/ferry feature", NOT "ground" — static walkability is never overridden by a type-0 patch entry.[^2]

## Terrain Byte Encoding

Each tile has a terrain byte (`cg.i` field) that encodes both the base type AND adjacency information.[^1]

### Bit Layout

```
Bits 0-3: Adjacency flags (which diagonal neighbors share the same terrain)
  bit 0 = NE neighbor is same type
  bit 1 = SE neighbor is same type
  bit 2 = SW neighbor is same type
  bit 3 = NW neighbor is same type

Bits 4-6: Base terrain sprite index:
  000 (0)   = ground variants (tile 0-15 based on adjacency)
  001 (16)  = grass border variant
  010 (32)  = water tiles (tile 32-47 based on adjacency)
  011 (48)  = water corner overlay
  100 (64)  = mountain/obstacle (tile 64-79 based on adjacency)

Bit 7: Sub-variant (for Z-order deterministic pseudo-random)
```

### Adjacency Lookup (ug function, line 155)

The `ug` function builds a 5-element adjacency signature `[center, N, E, S, W]` for each tile, then matches it against the 32-entry `gg` table (line 145) to determine the exact sprite tile index.[^1]

```javascript
// s[0..4] = [center_type, N_same?, E_same?, S_same?, W_same?]
// where "same" means neighbor has same terrain type as center
for (c = 0; c < gg.length; c++)
  if (Fb(a.s, gg[c])) { b = hg[c]; break; }
```

The `hg` array maps each adjacency pattern to a sprite tile index:[^1]
```javascript
hg = [32,33,34,35,...,79]  // water tiles and mountain tiles
```

### Pseudo-Random Variant (ng function, line 148)

For visual variety, some tiles get a pseudo-random variant based on their world coordinates:[^1]

```javascript
function ng(a, b) {
  // Morton code interleave of x,y coordinates
  for (var c = 0, d = 0; 16 > d; d++)
    c |= (a & 1) << 2*d, a >>= 1,
    c |= (b & 1) << 2*d + 1, b >>= 1;
  // CRC-8 of the 4 bytes of the Morton code
  for (var b = 0; 4 > b; b++)
    mg[b] = a >> 8*b & 255;
  for (c = b = 0; c < 4; c++)
    b = lg[b ^ mg[c]];
  return b;
}
```

This generates a deterministic "random" byte from tile (x,y) world coordinates. Used for:[^1]
- Water tiles: bit 0 → add 16 to sprite index (alternate water pattern)
- Ground tiles: `a % 8` → select from 8 ground variants (`jg = [0,1,2,3,4,5,6,7]`)
- Rock tiles: `(a >> 2) % 4` → select from 4 rock variants (`ig = [71,79,87,95]`)

## Edge/Border Tiles

When `sg()` encounters black (0) pixels at the viewport edge, it sets an edge flag and draws special border tiles:[^1]

```
Tile 16: horizontal border (top/bottom edge)
Tile 17: NE corner
Tile 18: NW corner  
Tile 19: SE corner
Tile 20: horizontal border variant (every 3rd column)
Tile 21: horizontal border variant (every 3rd column alternate)
Tile 22: vertical border (left/right edge)
Tile 23: vertical border variant (every 3rd row)
```

## Overlay Tiles

Drawn on top of base terrain:[^1]

| Index | Overlay Type |
|-------|-------------|
| 8 | Ferry rock |
| 15 | Equipment container marker |
| 24 | Mine (red team) |
| 25 | Mine (purple team) |
| 26 | Mine (blue team) |
| 27 | Mine (orange team) |
| 29 | Fuel container marker |
| 30 | Rock type A |
| 31 | Rock type B |

## Water Corner Overlays

When a water tile (32) is adjacent to non-water:[^1]

```javascript
if (32 === (t & 112)) {
  // Check each diagonal neighbor
  if (3 === (t & 3) && 32 !== (neighbor_NE.i & 112))
    draw overlay tile 48;  // NE corner
  if (6 === (t & 6) && 32 !== (neighbor_SE.i & 112))
    draw overlay tile 49;  // SE corner
  if (12 === (t & 12) && 32 !== (neighbor_SW.i & 112))
    draw overlay tile 50;  // SW corner
  if (9 === (t & 9) && 32 !== (neighbor_NW.i & 112))
    draw overlay tile 51;  // NW corner
}
```

Mountain corners work the same way (tiles 56-63), with an extra variant layer (tiles 60-63 for non-pure-mountain neighbors).[^1]

## Rock Types

Rocks are separate from terrain — they're entities placed ON terrain tiles:[^1]

| cg.j Value | Meaning |
|------------|---------|
| 0 | No rock |
| 1 | Rock type A (natural obstacle) |
| 2 | Rock type B (player-placed obstacle) |
| 3 | Both types (A + B stacked) |
| 5 | Ferry boarding point |
| 7 | Ferry + rock combined |

Rock type B (j=2) can be picked up/placed by players. Rock type A (j=1) is permanent terrain. When `cg.o` (occupied-behind) is true, the rock sprite is NOT drawn (tank is driving away from this tile and the trailing overlay covers it).[^1]

## Tile Sheet Layout

The Tiles sprite sheet (Ie, 192×192) is organized as an 8×12 grid of 24×16 pixel tiles:[^1]

```
Row 0 (y=0):   ground tiles 0-7 (8 variants)
Row 1 (y=16):  ground tiles 8-15 (adjacency variants)  
Row 2 (y=32):  border tiles 16-23
Row 3 (y=48):  mine overlays 24-27, fuel/equip 28-31
Row 4 (y=64):  water tiles 32-39
Row 5 (y=80):  water tiles 40-47
Row 6 (y=96):  water corners 48-55
Row 7 (y=112): mountain corners 56-63
Row 8 (y=128): mountain tiles 64-71
Row 9 (y=144): mountain tiles 72-79
Row 10 (y=160): obstacle sprites 80-87 (rocks, bridges, ferries)
Row 11 (y=176): tank trailing-tile overlays 88-95
```

Each tile: `sprite_x = (index % 8) * 24, sprite_y = floor(index / 8) * 16`[^1]

The ViewportUpdate (V.Z) sends terrain as the low 4 bits of each entity's packed 24-bit value.[^1]

[^1]: JS truth: `tpclient.js` on disk (frontmatter-pinned `tpclient.js:145`) — `dg`/`gg`/`hg` tables (line 145), classifier `sg` (line 152), adjacency `ug` (line 155), variant `ng` (line 148), all quoted verbatim in the fences above; terrain byte encoding fully traced 2026-06-19 (frontmatter `verified:` field), re-checkable by grep. The rock-family walkability semantics are wire-confirmed in [[movable-blocks]] and composed into the bot's terrain in [[terrain-composition]].
[^2]: Live measurement 2026-07-26: movement probe `runs/probe/coast_test.movement_probe.json` + its capture (walk-onto-blue rejections, `0x52 err=1`, settled position unchanged, fuel unchanged); containers-on-water from runs bot-20260726-094309/-145124 event logs (19 consumed/clamped pickups at water-classified `target_x/y`, tank `landed_x/y` on land 1 cardinal tile away in every case, cross-checked against the tracked-equipment atlas rebuilt from bot-20260726-145124's capture: 39/53 tracked equipment on water, dist-to-land 1-7 tiles). The type-0-patch-is-not-ground reading: 13-capture terrain mine, 2,729 tiles, zero cross-capture disagreements — type-0 entries appear over deep-lake container tiles the probe proved unwalkable.
