---
title: MAP_DATA Algorithm
tags: [js-client, protocol, map-data]
related:
  - "[[v-table-complete]]"
  - "[[map-mechanics]]"
  - "[[fuel-system]]"
source_paths:
  - "tpclient.js:171"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (complete parse algorithm traced line by line from JS)
hubs: [js-client]
---

# MAP_DATA Algorithm

The exact algorithm used by the JS client to parse the 0x4C (`L`) MAP_DATA message. This is the full fuel dot atlas + tank positions blob.[^1]

## Parse Function (Ig.h, line 172)

```javascript
Ig.h = function(a) {
  // --- Fuel dot section (skip-RLE) ---
  var b = X(a[0], a[1]);   // dot_byte_count = LE u16
  var c = 2;                // read offset
  var d = 1;                // x cursor (starts at 1, NOT 0)
  var e = 1;                // y cursor (starts at 1, NOT 0)
  var f = [];               // x coordinates of fuel dots
  var g = [];               // y coordinates of fuel dots

  while (c < 2 + b) {
    var h = a[c++];         // step byte
    d += h;                 // advance x by step
    if (255 < d) {          // x overflow → wrap to next row
      e++;
      d %= 256;             // d = d mod 256 (NOT d -= 256)
    }
    if (255 !== h) {        // 255 = pure skip (no dot here)
      f.push(d);            // record fuel dot x
      g.push(e);            // record fuel dot y
    }
  }

  // --- Tank section (remaining bytes) ---
  var b_tanks = [];   // x positions
  var d_tanks = [];   // y positions
  var e_tanks = [];   // tank IDs
  var h_tanks = [];   // ranks
  var k_tanks = [];   // rank categories
  var r_tanks = [];   // teams

  while (c < a.length) {
    b_tanks.push(a[c++]);           // x
    d_tanks.push(a[c++]);           // y
    e_tanks.push(X(a[c++], a[c++])); // tank_id (LE u16)
    var t = a[c];
    var v = a[c] & 3;               // team = bits 0-1
    var z = a[c++] >> 2 & 3;        // rank_category = bits 2-3
    t = t >> 4 & 15;                // rank = bits 4-7
    h_tanks.push(t);
    r_tanks.push(v);
    k_tanks.push(z);
  }

  return new Ig(f, g, b_tanks, d_tanks, e_tanks, h_tanks, k_tanks, r_tanks);
};
```

## Fuel Dot Skip-RLE Encoding

The fuel dot section uses a skip-RLE (run-length encoding) where each byte represents a horizontal skip distance:[^1]

1. Start at position (1, 1)
2. Read byte `h`
3. Add `h` to x: `x += h`
4. If `x > 255`: increment y, set `x = x % 256`
5. If `h != 255`: this position has a fuel dot — record it
6. If `h == 255`: this is a pure skip byte — no dot, just advance
7. Repeat until `dot_byte_count` bytes consumed

### Key Details

- **Coordinate space**: (1, 1) to (255, 255) — NOT zero-indexed
- **x wrapping**: Uses modulo 256, NOT subtraction of 256. This means `x=256` becomes `x=0` on the next row, not `x=1`
- **255 is skip-only**: When `h=255`, no dot is recorded, x advances by 255 and may wrap
- **Dots are delta-encoded**: Each byte is relative to the previous position, not absolute
- **Row-major order**: Dots are encoded left-to-right, top-to-bottom across the 256×256 world

### Example

Bytes: `[3, 10, 255, 2]`[^1]

Starting at (1, 1):
1. h=3: x=1+3=4, y=1, h≠255 → dot at (4, 1)
2. h=10: x=4+10=14, y=1, h≠255 → dot at (14, 1)
3. h=255: x=14+255=269, 269>255 → y=2, x=269%256=13, h=255 → skip (no dot)
4. h=2: x=13+2=15, y=2, h≠255 → dot at (15, 2)[^1]

## Tank Entry Format

After the fuel dot section, each tank is encoded as 5 bytes:[^1]

```
[0] = x position (0-255)
[1] = y position (0-255)
[2] = tank_id low byte
[3] = tank_id high byte
[4] = packed byte:
      bits 0-1 = team (0=red, 1=purple, 2=blue, 3=orange)
      bits 2-3 = rank_category (0-3)
      bits 4-7 = rank (0-8)
```

## Handler (Ig.prototype.h, line 173)

After parsing, the handler:[^1]

1. Logs "Zoom in" message
2. If active game session ($d): resets state to 0, clears teleport flag
3. Clears "waiting for" status
4. Registers each tank in the map's position hash: `a.map.m[(x << 8) + y] = tank`
5. Sets map viewport to current tank position
6. Sets map state flags: `h=true` (open), `u=true` (needs redraw)
7. Calls `Ne(b, true)` to center and render the map

The map position hash uses `(x << 8) + y` as the key — packing both coordinates into a single 16-bit integer.[^1]

## Map Canvas Rendering

The map renders at 3:2 pixel ratio (3 pixels per tile-x, 2 pixels per tile-y):[^1]

- Fuel dots: 1×1 pixel in the map's fuel color (`pa` field)
- Tank dots: 3×2 pixels in the tank's team color (`Y[team]`)
- Own position: 3×2 pixel flashing cursor cycling through 4 colors at 200ms intervals

Map scrolling uses 64-tile increments across a 128-pixel viewport window. Total map: 256×256 tiles = 768×512 pixels at 3:2, but only a 128×128 section visible at once (384×256 when scaled to canvas).[^1]

[^1]: JS truth: `tpclient.js` on disk (frontmatter-pinned `tpclient.js:171`) — parse function `Ig.h` (line 172, quoted verbatim in the code fence above) and handler `Ig.prototype.h` (line 173); traced line-by-line 2026-06-19 (frontmatter `verified:` field), re-checkable by grep; the worked example is arithmetic derived from the quoted parse loop, and the bot's own 0x4C decoder mirrors it.
