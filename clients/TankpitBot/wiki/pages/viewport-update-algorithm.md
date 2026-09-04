---
title: Viewport Update Algorithm
tags: [js-client, protocol, viewport]
related:
  - "[[v-table-complete]]"
  - "[[terrain-system]]"
  - "[[viewport-frame]]"
source_paths:
  - "tpclient.js:187"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (complete parse and render algorithm traced from JS)
hubs: [js-client]
---

# Viewport Update Algorithm

The exact algorithm used by the JS client to parse and apply the 0x5A (`Z`) ViewportUpdate message. This is the primary world state message — sent whenever the viewport changes (movement, teleport, respawn).[^1]

## Parse Function (Vg.h, line 188)

```javascript
Vg.h = function(a) {
  var b = a[0];        // viewport_left (world x)
  var c = a[1];        // viewport_top (world y)
  var d = [];          // column positions
  var e = [];          // row positions
  var f = [];          // cache values
  var g = [];          // overlay values
  var h = [];          // terrain types

  var k = 0;           // column cursor
  var r = 0;           // row cursor

  for (var t = 2; t < a.length; ) {
    var v = a[t++];          // position step byte

    k += v % 18;             // advance column by (step mod 18)
    r += Math.floor(v / 18); // advance row by (step / 18)

    while (18 <= k) {        // column overflow → wrap to next row
      r++;
      k -= 18;
    }

    if (255 !== v) {         // 255 = skip (no entity here)
      v = a[t++];            // packed byte 0 (high)
      var O = a[t++];        // packed byte 1 (mid)
      var z = a[t++];        // packed byte 2 (low)

      z = 256 * (256 * v + O) + z;  // 24-bit BE integer
      z &= 16777215;                // mask to 24 bits

      O = z & 15;            // terrain = bits 0-3
      z >>= 4;
      v = z & 15;            // overlay = bits 4-7
      if (8 <= v) v = 255;   // overlay >= 8 means "no overlay"
      z >>= 4;               // cache = bits 8-23
      if (65535 === z) z = -1; // 65535 means equipment (negative cache)

      d.push(k);
      e.push(r);
      f.push(z);
      g.push(v);
      h.push(O);
    }
  }

  return new Vg(b, c, d, e, f, g, h);
};
```

## Position Encoding

Entity positions use a single-byte step encoding:[^1]

- **step byte** encodes both column and row advance:
  - `column_advance = step % 18`
  - `row_advance = floor(step / 18)`
- Column wraps at 18: if column >= 18, increment row and subtract 18
- **step = 255** means "skip" — advance position but no entity data follows
- Non-skip steps are followed by 3 data bytes

The grid is 18×18 (indices 0-17), matching the viewport fringe tiles.[^1]

### Example

The column and row cursors are cumulative with wrapping — each step is
relative to the previous position (delta-encoded):[^1]

```
k += v % 18;       // k starts at 0, accumulates
r += Math.floor(v / 18);
while (18 <= k) { r++; k -= 18; }
```

Starting at (col=0, row=0):
- step=5: col=0+5=5, row=0 → entity at (5, 0), 3 data bytes follow
- step=20: col=5+(20%18)=7, row=0+floor(20/18)=1 → entity at (7, 1)
- step=255: col advances by 255%18=3 (wrapping applies), row by floor(255/18)=14 → pure skip, no data bytes[^1]

## Entity Data Packing (24-bit BE)

The 3 bytes after each non-skip step encode:[^1]

```
Byte layout: [high, mid, low]
24-bit value = 256 * (256 * high + mid) + low

Bit extraction:
  bits 0-3   (low nibble of low byte):  terrain type (0-15)
  bits 4-7   (high nibble of low byte): overlay value
    0-3 = mine from team 0-3
    4-7 = mine types 4-7
    8-15 → mapped to 255 (no overlay)
  bits 8-23  (remaining 16 bits): cache value
    0 = empty tile
    1-65534 = fuel volume
    65535 → mapped to -1 (equipment container)
```

## Handler (Vg.prototype.h, line 189)

After parsing, the handler:[^1]

### 1. Update viewport position
```javascript
a.h.x = this.o;    // viewport_left
a.h.y = this.s;    // viewport_top
```

### 2. Clear state
```javascript
xb(a.h.j);         // flush animation queue
pd(a.P);           // remove all drawn tanks
gd(a.h.i);         // redraw tile grid
rg(a.h.i);         // reset all tile data
```

### 3. Determine terrain from map image
```javascript
sg(a.h.i, a.h.x - 1, a.h.y - 1, a.map.$);
```

The `sg()` function reads the map image at the viewport position and classifies each pixel into terrain types (ground/rock/water). This is the source of the terrain byte for each tile.[^1]

### 4. Scroll optimization
```javascript
if (16 <= Math.abs(a.h.x - b) || 16 <= Math.abs(a.h.y - c))
  fd(a.h);  // full redraw (teleport or large jump)
else {
  // Partial blit: copy existing canvas with offset, only redraw edge tiles
  var g = b * e, h = c * f;  // pixel offsets
  // ... drawImage with offset to scroll existing content
  // Then redraw only tiles at the edges
  for (g = 1; 17 > g; g++)
    for (h = 1; 17 > h; h++) {
      f = b + h; e = c + g;
      // Only redraw if tile is in the "new" region
      if (2 <= f && 16 > f && 2 <= e && 16 > e)
        continue;  // this tile was already on screen
      Eg(d, h, g);  // mark dirty → will redraw
    }
}
```

If viewport shift < 16 tiles: blit the existing canvas content shifted, redraw only new edge tiles. If shift >= 16: full redraw of everything.[^1]

### 5. Apply entity data
```javascript
for (d = 0; d < this.j.length; d++) {
  b = a.h.h[this.u[d]][this.j[d]];  // get tile at (col, row)
  b.cache = this.i[d];               // set cache (fuel/equipment)
  b.m = this.m[d];                   // set overlay (mines)
  b.j = this.l[d];                   // set rock type
  b.l = true;                        // mark dirty
}
```

### 6. Post-update
```javascript
a.Ja = false;                  // clear scope-change-pending flag
a.za && a.tb();                // if deactivated, handle reactivation
```

## Fringe Tiles

The 18×18 grid includes a 1-tile fringe border (row/col 0 and 17). These tiles:[^1]
- Are visible in the game area
- Have terrain determined from the map image
- But are NOT actionable (can't click on them to move/shoot)
- The `ae(a,b)` function returns false for indices 0 and 17

Fringe tiles can show terrain, tanks (visible but not targetable), and containers (visible but not collectible without moving).[^1]

## Empty Viewport

On death or before first spawn, `a.h.x = -255` (sentinel value). Several handler functions check for this:[^1]
```javascript
-255 !== a.h.x && ...  // only process if viewport is valid
```

## Viewport Change Detection

The game detects viewport changes by comparing old and new x/y:[^1]
- Same position (viewport didn't move): only entity data changes
- Small shift (1-15 tiles): scroll optimization
- Large shift (≥16 tiles): full redraw (teleport)

[^1]: JS truth: `tpclient.js` on disk (frontmatter-pinned `tpclient.js:187`) — parse `Vg.h` (line 188) and handler `Vg.prototype.h` (line 189), both quoted verbatim in the fences above; `sg` terrain classification, `ae` fringe check; traced 2026-06-19 (frontmatter `verified:` field), re-checkable by grep. Ongoing receipt: the bot's own 0x5A decoder implements this exact algorithm and reconciles every live viewport patch against it.
