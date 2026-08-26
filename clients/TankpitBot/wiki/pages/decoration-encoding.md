---
title: Decoration Encoding
tags: [js-client, protocol, decorations]
related:
  - "[[v-table-complete]]"
  - "[[client-constants]]"
source_paths:
  - "tpclient.js:204"
  - "tpclient.js:128"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (encoding and rendering traced from JS)
hubs: [js-client]
---

# Decoration Encoding

How tank decorations (awards) are packed into 4 bytes and rendered. Extracted from the `yg()` decode function and the decoration rendering in `ed()`.[^1]

## Wire Format

Decorations are transmitted as 4 bytes in TankInfo (0x21), TankEntry (0x28), and TankStatusFull (0x3E) messages:[^1]

```
a[3], a[4], a[5], a[6]  — 4 decoration bytes
```

**The packed state is live-verified (2026-08-26)** and implemented in
`protocol/decorations.py` (`unpack_decoration_state`, the yg law):
Arterial's live `04000000` unpacks to slot 1 level 1 — the BRONZE
TANK AWARD earned an hour earlier — and Artax's `1e000000` to
(2, 3, 1): DOUBLE STAR, GOLDEN TANK AWARD, COMBAT HONOR MEDAL, three
medals the account had carried unread while the bytes were mislabeled
"cosmetic skin" in the dispatch comment. Every tank announces this
state in its 0x21 TankInfo at identification — **no check request
exists and none is needed; the server pushes the full profile
unprompted**. The `tank_identity` diagnostic now carries the decoded
`awards=` list beside the raw hex.

## Decode Function (yg, line 204)

```javascript
function yg(a, b, c, d) {
  var e = new Uint8Array([0,0,0,0,0,0,0,0,0]);
  a = a | b << 8 | c << 16 | d << 24;  // combine into 32-bit integer
  for (b = 0; 9 > b; b++)
    e[b] = a & 3,   // extract 2 bits per slot
    a >>= 2;        // shift right 2
  return e;
}
```

This unpacks a 32-bit integer into 9 decoration slots, each 2 bits wide:[^1]

```
Bits 0-1:   slot 0 (Stars: Single/Double/Triple)
Bits 2-3:   slot 1 (Tank Award: Bronze/Silver/Golden)
Bits 4-5:   slot 2 (Honor Medal: Combat/Battle/Heroic)
Bits 6-7:   slot 3 (Sword: Shining/Battered/Rusty)
Bits 8-9:   slot 4 (Shield: Bronze/Silver/Defender)
Bits 10-11: slot 5 (Cup: Bronze/Silver/Golden)
Bits 12-13: slot 6 (Purple Heart: 1/2/3)
Bits 14-15: slot 7 (War Correspondent: 1/2/3)
Bits 16-17: slot 8 (Lightbulb: 1/2/3)
```

Each slot value:
- 0 = no award in this category
- 1 = bronze / level 1
- 2 = silver / level 2
- 3 = gold / level 3[^1]

The remaining bits (18-31) are unused (14 bits). Since `9 × 2 = 18 bits`, only 18 of 32 bits are used.[^1]

## Storage

Decorations are stored on the tank entity (Xc class) as a Uint8Array(9):[^1]

```javascript
// Line 127:
this.v = new Uint8Array([0,0,0,0,0,0,0,0,0]);
```

Updated when TankInfo, TankEntry, or Decoration messages arrive:[^1]
```javascript
// V["!"] handler (line 157):
a.v = this.j;  // set full decoration state

// V.N handler (line 163):
b.v[this.i] = this.j;  // set single slot: i=slot, j=level
```

## Decoration Names

Names are indexed as `nb[3 * slot + level - 1]` from the nb array:[^1]

| Slot | Level 1 | Level 2 | Level 3 |
|------|---------|---------|---------|
| 0 | SINGLE STAR | DOUBLE STAR | TRIPLE STAR |
| 1 | BRONZE TANK AWARD | SILVER TANK AWARD | GOLDEN TANK AWARD |
| 2 | COMBAT HONOR MEDAL | BATTLE HONOR MEDAL | HEROIC HONOR MEDAL |
| 3 | SHINING SWORD | BATTERED SWORD | RUSTY SWORD |
| 4 | BRONZE SHIELD | SILVER SHIELD | DEFENDER OF THE TRUTH |
| 5 | BRONZE CUP | SILVER CUP | GOLDEN CUP |
| 6 | PURPLE HEART | PURPLE HEART 2 | PURPLE HEART 3 |
| 7 | WAR CORRESPONDENT | WAR CORRESPONDENT 2 | WAR CORRESPONDENT 3 |
| 8 | LIGHTBULB AWARD | LIGHTBULB 2 | LIGHTBULB 3 |

## Award Rendering (ed function, line 128)

```javascript
function ed(a) {
  var b = a.ba;    // off-screen canvas for decoration bar
  a = a.v;         // Uint8Array(9) decoration state
  // Calculate total width
  for (var c = 0, d = 0; 10 > d; d++)
    a[d] && (c += yb[d]);
  // Resize canvas
  b.width = c;
  b.height = 16;
  b = b.getContext("2d");
  // Draw each earned award
  for (var e = d = c = 0; 10 > e; e++)
    0 < a[e] && (
      b.drawImage(zb,          // awards sprite sheet
        d,                     // source x = cumulative width of previous slots
        16 * (a[e] - 1),       // source y = 16px per level (0-indexed)
        yb[e],                 // source width from yb array
        16,                    // source height (fixed 16px)
        c, 0,                  // destination
        yb[e], 16),
      c += yb[e]
    ),
    d += yb[e];  // advance source x for next slot regardless
}
```

Award sprite widths per slot (yb array): `[15, 31, 11, 11, 13, 15, 11, 12, 16, 9]`[^1]

Note: The loop runs to `10 > d` and `10 > e`, but there are only 9 decoration slots (0-8). The 10th position (index 9) exists in yb but v[] only has 9 elements. This means slot 9 (width=9) is an unused/phantom slot that's never drawn because `a[9]` is always 0 (Uint8Array initialized with 9 elements, indexed 0-8).[^1]

## Display Function (Ff, line 128)

Generates a text description of a tank's decorations:[^1]

```javascript
function Ff(a) {
  for (var b = "", c = 0; 9 > c; c++) {
    var d = a.v[c];
    0 < d && (b = b + "\n " + nb[3 * c + d - 1]);
  }
  return b;
}
```

Returns empty string if no decorations, otherwise newline-separated award names.[^1]

## V.N — Decoration Event (0x4E, Sf class, line 163)

**Live-confirmed 2026-08-26 05:11:16** — the first 0x4E ever received
in the capture corpus: `tank_id=602 (Arterial), slot=1, level=1` →
BRONZE TANK AWARD (100 career deactivations), decoded identically
from both fleet bots' captures, byte-for-byte per the Sf trace below.
The event also exposed two bot defects, both fixed the same hour: the
`tank_decoration` diagnostic used a field named `level` (a reserved
JSONL record key — the collision crashed BOTH bots simultaneously;
field renamed `decoration_level`, and reserved-name validation moved
to the emit call so any covered emit proves its names), and awards
now decode to their names via `protocol/decorations.py` (this page's
nb table, in code — unknown slots render as raw numbers, never
crash).

When a tank earns a new decoration:[^1]

```javascript
Sf.h = function(a) {
  return new Sf(
    X(a[0], a[1]),  // tank_id (LE u16)
    a[2],            // award_slot (0-8)
    a[3]             // award_level (1-3)
  );
};
```

Handler:
1. Only shows message if new level > current level (no demotion messages)
2. Formats: "You have been decorated with the {NAME}" or "{tank} has been decorated..."
3. Updates `b.v[slot] = level`
4. Re-renders decoration bar (`ed(b)`)
5. If map is open, defers the redraw to after map closes[^1]

## How Awards Are Earned (from guide, line 262)

| Slot | Category | Level 1 | Level 2 | Level 3 |
|------|----------|---------|---------|---------|
| 0 | Stars | Achieve Major | Achieve Colonel | Achieve General |
| 1 | Tank | Deactivate 100 | Deactivate 200 | Deactivate 500 |
| 2 | Honor | Get deactivated 20× | 50× | 100× |
| 3 | Sword | 100 hours played | 200 hours | 500 hours |
| 4 | Shield | Report bugs | (escalating) | Defender of the Truth |
| 5 | Cup | Tournament 3rd | 2nd | 1st |
| 6 | Purple Heart | Create quality content | (PH contests) | — |
| 7 | War Correspondent | Promote TankPit | — | — |
| 8 | Lightbulb | Brilliant ideas | — | — |

[^1]: JS truth: `tpclient.js` on disk (frontmatter-pinned `tpclient.js:204` and `:128`) — decode `yg` (line 204), storage on `Xc` (line 127), render `ed` and display `Ff` (line 128), decoration event `Sf` (line 163), award criteria from the in-client guide (line 262), all quoted verbatim in the fences above; traced 2026-06-19 (frontmatter `verified:` field), re-checkable by grep.
