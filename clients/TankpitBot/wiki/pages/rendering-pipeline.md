---
title: Rendering Pipeline
tags: [js-client, rendering, canvas]
related:
  - "[[js-source-map]]"
  - "[[client-constants]]"
source_paths:
  - tpclient.js lines 4-5 (canvas)
  - 108-127 (animations)
  - 142-155 (tile grid)
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (rendering system traced through JS)
hubs: [js-client]
---

# Rendering Pipeline

How the game client renders everything: canvas layer stack, sprite sheets, tile engine, animation system, and dirty-rect invalidation.

## Canvas Layer Stack

6 named canvas layers, composited via CSS z-index (line 237):

| Z | Layer | Class | Size | Purpose |
|---|-------|-------|------|---------|
| 0 | Background | sa("Background", 256) | 384×256 | Terrain tiles |
| 1 | Tanks | sa("Tanks", 256) | 384×256 | Tank sprites |
| 2 | Action | sa("Action", 256) | 384×256 | Projectiles, explosions, radar sweeps |
| 3 | Map | sa("Map", 256) | 384×256 | Mini-map overlay (shown/hidden) |
| 4 | Overlay | sa("Overlay", 256) | 384×256 | Deactivation screen, repair bar, scope rectangle |
| 5 | Menu | sa("Menu", 48) | 384×48 | Toolbar buttons (below game area, y=256-304) |

Each layer has a ScaledContext (`qa` class) wrapper that multiplies coordinates by the DPI scale factor.

## DPI Scaling (qa class, line 4)

```javascript
qa.prototype.h = scale_factor;  // 1, 1.5, 2, etc.
qa.prototype.m = drawImage (normal)
qa.prototype.s = drawImage (ceil-based for anti-aliasing)
qa.prototype.i = clearRect (for dirty-rect erase)
```

Two draw modes: `.m` (exact pixel) for scale multiples of 25%, `.s` (ceil-rounded with 1.03× oversize) for non-integer scales. Selected at layer creation time based on scale factor.

## Tile Engine (og class, lines 148-155)

The game world viewport is an 18×18 grid of `cg` tile objects.

### Tile Structure (cg, line 145)

```
i: terrain type byte (complex bitfield)
m: overlay (mine team 0-3, or 255=none)
cache: fuel/equipment value (>0=fuel, <0=equipment, 0=empty)
j: rock type (0-7)
h: tank reference (Xc instance or null)
o: occupied-behind flag (trailing tile of moving tank)
l: dirty flag → triggers redraw
```

### Terrain Byte Encoding

```
Bits 0-3: adjacency flags (which neighbors share this terrain type)
  bit 0 = NE, bit 1 = SE, bit 2 = SW, bit 3 = NW
Bits 4-6: base terrain
  000 = ground/grass
  010 = water (32)
  100 = mountain/obstacle (64)
Bit 7: variant flag
```

Terrain type 0 = ground, water corners (48-51) are overlays on water (32), obstacles (64+) have their own corner overlays (56-63).

### Tile Drawing (gd function, line 149-151)

For each dirty tile:
1. Draw base terrain sprite from Ie (Tiles) sheet: `tile_index % 8 * 24, floor(tile_index / 8) * 16`
2. If water: draw corner overlays for adjacent non-water tiles
3. If mountain: draw corner overlays for adjacent non-mountain tiles
4. If has cache: draw fuel dot (sprite 29) or equipment dot (sprite 15)
5. If has rock: draw rock sprite (30=type_a, 31=type_b, 8=ferry_rock)
6. If has mine overlay: draw team-colored mine sprite (24+team)
7. Clear dirty flag

### Viewport Scroll (ViewportUpdate handler, line 189)

On viewport change:
- If shift < 16 tiles: blit existing canvas with offset, redraw only edge tiles
- If shift >= 16 tiles: full redraw of all 16×16 tiles
- Terrain colors read from map image via getImageData() pixel sampling

## Sprite Sheets

### Tank Sprites (Ee, 112×680)

```
Width: 28px per team (4 teams × 28 = 112)
Height: 20px per direction (34 rows × 20 = 680)
  Rows 0-15: 16 facing directions (alive)
  Rows 16-31: 16 "walking from" directions (trailing tile)
  Row 32: corpse (even death)
  Row 33: corpse (odd death)

Draw: Ee.h(ctx, team*28, direction*20, 28, 20, x, y, 28, 20)
```

### Explosion Sprites (De, 42×764)

Variable-size frames. Frame offsets stored in parallel arrays:
- `Cf[]` = frame heights
- `Df[]` = frame widths
- `zf[]` = X draw offsets
- `Af[]` = Y draw offsets
- `pf[] = [9,17,26,32,42,45,48]` — cumulative frame starts per explosion type

### Dust/Splash Sprites (Ae 22×470, Ce 29×486)

Similar variable-frame structure. Dust used for ground movement particles, splash for water.

### Radar Sprites (He, 68×619)

17-frame sweep animation. Frame offsets in `qf[]` (widths) and `rf[]` (heights).

## Dirty-Rect System (Hc class, line 36)

Each animated object tracks a bounding rectangle (`Hc` instance):
```
x, y, w, h: current dirty region
j: active flag
```

`Ic(a, x, y, w, h)` — expand dirty rect to include new draw region (union).
`Jc(a)` — reset after erase.

Erase sequence: `ra()` → `clearRect(dirty_rect)` → redraw overlapping tanks.

## Animation Pipeline

Each tick in the action queue:

1. **Check `.s` (started)**: If not started, call `.start()` (play sound, init state)
2. **Check `.j` (done)**: If not done, call `.Na()` (advance frame)
3. **If done**: call `.ra()` (erase), `.Ma()` (cleanup/spawn sub-animations), remove from queue
4. **If not done**: call `.sa()` (draw), `.ra()` (erase previous frame position)
5. **Set tick rate**: to animation-specific value (`.u` field)

Multiple animations can run simultaneously in the queue. The tick rate is set to the LAST animation's preferred rate.

## Map Rendering (Tc class, lines 100-106)

Three canvas buffers:
- `o` (384×256): Final composited map for display
- `ia` (256×256): Base map terrain (from field image)
- `U` (256×256): Color-adjusted map (team colors applied)

Map coordinates: `j` and `i` are scroll offsets (0, 64, or 128 in each axis).

Tank dots: 3×2 pixels per tank, color from team `Y[]` array.
Fuel dots: 1×1 pixels, stored in `aa[]` (x coords) and `la[]` (y coords).

Own position: Flashing cursor (4 colors cycling at 200ms) at 3×2 pixel size.

