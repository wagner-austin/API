---
title: Input Handling
tags: [js-client, input, ui]
related:
  - "[[client-state-machine]]"
  - "[[toolbar-layout]]"
source_paths:
  - "tpclient.js:79"
  - "tpclient.js:107"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (all input paths traced from JS event handlers)
hubs: [js-client]
---

# Input Handling

How the JS client processes mouse, keyboard, and touch input. Extracted from the event handlers in the $d (game session) and Rc (game canvas) classes.[^1]

## Mouse Events

Three mouse handlers on the game canvas (`Rc` class, line 233):[^1]

### mousemove → Sh (line 233)
Updates the Pc (mouse state) object and calls Kd() for hover effects:
```javascript
function Sh(a) {
  a = ne(a.clientX, a.clientY, this.l, this.v.h);  // scale to game coords
  me(b.u, a[0], a[1]);   // update mouse position
  b.qa = b.u;            // set as active pointer
  Kd(b);                 // process hover (status bar, cursor, tooltip)
}
```

### mousedown → Th (line 233)
Records click start position:
```javascript
function Th(a) {
  var a = ne(a.clientX, a.clientY, ...);
  this.i.ab(a[0], a[1]);  // delegates to $d.ab
}
```

$d.ab (line 79): Records start time, position, and calculates tile (col, row):[^1]
```javascript
d.v = Math.floor(a / 24) + 1;   // start tile col (1-indexed)
d.P = Math.floor(b / 16) + 1;   // start tile row (1-indexed)
```

### mouseup → Uh (line 233)
Triggers the actual game action:
```javascript
function Uh(a) {
  this.i.bb(a[0], a[1], a.button);  // delegates to $d.bb
}
```

## Mouse Action Decision ($d.bb, lines 80-84)

The mouseup handler determines what to do based on:
1. Where the click started vs. where it ended
2. How long the click was held
3. Whether it was a single, double, or long click
4. What's at the target tile[^1]

### Game Area Clicks (Ld = true)

```
Same start and end tile:
  - Long press (>300ms):
    If tile has fuel:        → GET FUEL (state 5)
    If tile has equipment:   → GET EQUIPMENT (state 6)
    If tile has obstacle:    → PICK UP OBSTACLE (state 7)
    If carrying + fuel:      → DEPOSIT FUEL (state 10)
    Else:                    → set as "double-click ready" (ea flag)
  
  - Double-click (ea flag set from previous click):
    → FIRE (state 3)
  
  - Normal click:
    → set ea flag, then MOVE (state 2)

Different start and end tile:
  → MOVE to end tile (state 2), play click sound
```

### Map Area Clicks (when map is open)

```
Click on map → TELEPORT to clicked map coordinates
  Ca = floor(a/3) + map.j    // world x
  Da = floor(b/2) + map.i    // world y
  Send: Ob(Ca, Da)           // teleport command
  Set ba=true (teleport pending)
```

### Toolbar Clicks

```
Click on toolbar region → action per toolbar-layout mapping
  - Index 0: map open/close
  - Index 1: radar
  - Index 2: mine
  - Index 3-11: scope direction
  - Index 12-16: equipment toggle
  - Index 17: show promotion info
```

## Keyboard Events

### keydown → Vh (line 233)
```javascript
function Vh(a) {
  a.preventDefault();
  a.stopImmediatePropagation();
  this.i.$a(a.code);  // delegates to $d.$a
  return false;
}
```

### keyup → Wh (line 233)
```javascript
function Wh(a) {
  a.preventDefault();
  a.stopImmediatePropagation();
  return false;  // consumed, no action
}
```

Keyboard processing is in $d.$a (lines 73-78). Each key code is mapped through `this.l.j` (hotkey map) to an action number, then dispatched via a switch statement.[^1]

## Touch Events

Touch handling uses a multi-touch tracking system with the `Bd` class.[^1]

### Touch State (Bd class, line 107)

```javascript
function Bd(a, b, c, d) {
  this.id = a;            // touch identifier
  this.v = b;             // start timestamp
  this.$ = c;             // start x (pixel)
  this.aa = d;            // start y (pixel)
  this.m = c; this.l = d; // current x, y
  this.o = 0; this.s = 0; // cumulative movement (abs x, abs y)
  this.u = true;          // still at start tile
  this.U = Math.floor(c / 24) + 1;  // start tile col
  this.W = Math.floor(d / 16) + 1;  // start tile row
  this.h = this.U;        // current tile col
  this.j = this.W;        // current tile row
  this.P = false;         // ended flag
  this.Y = false;         // gesture recognized flag
  this.i = null;          // fire-at-enemy flag (null=undecided, true=enemy, false=no)
}
```

### Gesture Recognition (pe function, line 108)

```javascript
function pe(a, b) {
  // Reject if total movement < 50px in either axis
  if (a.Y || 50 > a.o && 50 > a.s) return -1;
  // Reject if held too long (>300ms)
  if (300 < b - a.v) return a.Y = true, -1;
  
  // Check if movement is roughly linear (not erratic)
  var b_dx = a.m - a.$;
  var c_dy = a.l - a.aa;
  if (Math.abs(Math.abs(b_dx) - a.o) > 0.1 * a.o ||
      Math.abs(Math.abs(c_dy) - a.s) > 0.1 * a.s)
    return a.Y = true, -1;
  
  // Determine swipe direction
  var d = a.o / a.s;  // horizontal/vertical ratio
  if (5 > d && 0.2 < d) {
    // Diagonal swipe
    return 0 > c_dy
      ? (0 < b_dx ? 1 : 7)   // NE or NW
      : (0 < b_dx ? 3 : 5);  // SE or SW
  }
  // Cardinal swipe
  return a.o > a.s
    ? (0 < b_dx ? 2 : 6)     // E or W
    : (0 > c_dy ? 0 : 4);    // N or S
}
```

Returns direction 0-8 or -1 (no gesture). Mapped to scope directions via `qe()`.[^1]

### Touch Handlers

- **touchstart** ($d.eb, line 84): Creates new Bd for each changedTouch
- **touchmove** ($d.cb, line 85): Updates Bd position, checks for swipe gestures
- **touchend** ($d.La, line 86): Determines action based on hold duration and movement:
  - Short tap (<100ms): fire intent
  - Medium tap (100-250ms): normal click → move
  - Long tap (>250ms): long-press action (pickup, deposit)
  - Swipe: scope direction change

### Multi-Touch

The `touches` dictionary tracks all active touches by identifier. The "most recent" touch (`qa`) gets hover feedback. The "active" touch (`U`) determines pointer state.[^1]

When a new touch starts at the same tile as an existing active touch, the old touch is force-ended:[^1]
```javascript
null !== this.U && this.U.h === d.h && this.U.j === d.j && this.U.end(true);
```

## Coordinate Conversion

All pixel coordinates are converted from screen space to game space:[^1]

```javascript
function ne(a, b, c, d) {
  return [
    Math.floor((a - Math.floor(c.left)) / d),
    Math.floor((b - Math.floor(c.top)) / d)
  ];
}
```

Where `c` is `getBoundingClientRect()` and `d` is the game scale factor.[^1]

Game coordinates to tile (1-indexed):[^1]
```javascript
col = Math.floor(x / 24) + 1;  // x pixels → column (1-16 actionable)
row = Math.floor(y / 16) + 1;  // y pixels → row (1-16 actionable)
```

Game coordinates to world:[^1]
```javascript
world_x = col + viewport_left - 1;
world_y = row + viewport_top - 1;
```

[^1]: JS truth: `tpclient.js` on disk (frontmatter-pinned `tpclient.js:79` and `:107`) — canvas handlers on `Rc` (line 233), click dispatch `ab`/`bb` (lines 79-84), keyboard `$d.$a` (lines 73-78), touch `Bd`/`pe` (lines 107-108), conversion `ne`, all quoted verbatim in the fences above; every input path traced 2026-06-19 (frontmatter `verified:` field), re-checkable by grep.
