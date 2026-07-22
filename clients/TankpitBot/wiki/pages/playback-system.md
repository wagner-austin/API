---
title: Playback System
tags: [js-client, recording, playback]
related:
  - "[[js-source-map]]"
  - "[[v-table-complete]]"
source_paths:
  - tpclient.js lines 92-94 (re/Kf classes)
  - lines 132-134 (If recording class)
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (recording format and playback logic traced from JS)
hubs: [js-client]
---

# Playback System

The game has a built-in recording/playback system. Recordings can be loaded from files or URLs for replay. Extracted from the `If` (recording), `Kf` (playback controller), and `re` (playback session) classes.

## Recording Format (If class, line 132)

Recordings are binary blobs with this structure:

### Header
```
For each segment:
  [0-1] = LE u16 length of segment (including these 2 bytes + 2 timestamp bytes)
  [2-3] = LE u16 timestamp offset
  [4..length+3] = payload bytes
```

### First Segment — Metadata
The first segment is a JSON string:
```javascript
if (e.hasOwnProperty("version"))
  this.version = e.version;
```

If no "version" key, it's the game info:
```javascript
var c = JSON.parse(d);
var d_field = c[0];    // field_id
var e_name = c[1];     // game name
var f_fieldId = c[2];  // field number
var g_flags = c[3];    // flags (comma-separated string → int array)
var ja_team = c[4];    // team
var ua_mode = c[5];    // mode
var year = 8 <= c.length ? c[7] : 2018;  // year (default 2018)
```

### Subsequent Segments — Game Messages
Each remaining segment is a raw server message (the same binary format as live WebSocket messages).

### Timestamp Encoding
```javascript
Lf = d[2] + 256 * d[3];  // LE u16 timestamp
if (this.U.version > 0)
  Lf /= 1000;             // version > 0: timestamp in milliseconds/1000
```

Version 0 timestamps are raw; version 1+ divide by 1000 to get seconds.

## Playback Controller (Kf class, line 133)

### Initialization
```javascript
function Kf(a, b, c) {
  this.U = a;              // If recording instance
  this.v = c;              // message queue (Qc)
  this.m = [];             // parsed message objects
  this.P = [];             // timestamp offsets (seconds)

  for (c = 0; c < a.h.length; c++) {
    var d = a.h[c];
    Lf = d[2] + 256 * d[3];
    if (this.U.version > 0) Lf /= 1000;
    var e = Lf;
    if (d = Mf(new Uint8Array(d.buffer, 4)))  // parse via V table
      this.m.push(d),
      this.P.push(e);
  }

  // Build timeline UI
  Of(this.s, this.m, this.W.bind(this));
  Pf(this.s, 0);

  this.i = 0;              // current message index
  this.l = null;            // current message to execute
  this.o = this.j = false;  // j=playing, o=finished
  this.h = 1;               // playback speed multiplier
}
```

### Playback Loop
```javascript
Kf.prototype.u = function() {
  if (this.j) {
    if (null !== this.l) {
      ve(this.v, this.l);     // dispatch current message
      Pf(this.s, this.i);     // update timeline cursor
      this.i++;
    }
    if (this.i >= this.m.length) {
      this.l = null;
      this.o = true;           // finished
    } else {
      this.l = this.m[this.i];
      var a = this.P[this.i];  // time to next message (seconds)
      if (1 !== this.h) a *= this.h;  // apply speed multiplier
      setTimeout(this.u.bind(this), 1000 * a);
    }
  }
};
```

### Speed Control
```javascript
// Arrow keys adjust speed (re.$a handler, line 93):
case "ArrowLeft":
  if (20 >= c.h) c.h *= 1.25;  // slow down (max 20x slower)
  break;
case "ArrowRight":
  if (0.1 <= c.h) c.h *= 0.8;  // speed up (min 0.1x = 10x faster)
  break;
```

Speed also scales the L timing constants:
```javascript
for (c in L)
  L.hasOwnProperty(c) && (L[c] = oc[c] * this.W);
```

### Pause/Resume
```javascript
// Space or Q key:
case 2:   // Space
case 11:  // Q
  this.s.j ? (ue(this.s), "PAUSE") : (start, "RESUME");
```

### Restart
```javascript
// R key:
case 4:
  this.Ya();    // reset display
  sd(this);     // reset game state
  se(this.s);   // restart playback
```

## Timeline UI (Nf class, line 257)

The timeline is a horizontal strip of colored spans, one per message:

```javascript
function Of(a, b, c) {
  for (var f = 0; f < b.length; f++) {
    var g = b[f];
    var h = document.createElement("span");
    h.className = "entry";

    if (g instanceof Qf) h.classList.add("start"), e = g.o;   // TankStatusFull → game start
    else if (g instanceof Vg) {
      h.classList.add("zone");                                  // ViewportUpdate → clickable
      h.dataset.index = f.toString();
      w(h, "click", d);                                        // click to jump
    }
    else if (g instanceof Pg && g.i === e) h.classList.add("death");   // own death
    else if (g instanceof Pg && g.j === e) h.classList.add("kill");    // own kill
    else if (g instanceof Rf) h.classList.add("promote");              // promotion

    a.h.appendChild(h);
  }
}
```

Timeline entries are color-coded:
- **start**: game start marker (TankStatusFull)
- **zone**: viewport change (clickable — jumps to that point)
- **death**: own deactivation
- **kill**: own kill of another tank
- **promote**: rank promotion

## Playback Session (re class, line 92)

The playback session (`re`) inherits from `Oc` (base session) and overrides:

- `ya()` — starts playback instead of joining game
- `qb()` — checks if playback is finished
- `$a()` — custom key handling (speed, pause, restart)
- `ab/bb/eb/cb/La/Oa/sb` — all input handlers disabled (no-op)
- `Za/ub` — error handlers throw instead of reporting to server

## Loading Recordings

Recordings can be loaded from:

1. **File upload** — `<input type="file">` in the playback selector
2. **URL** — `tankpit.playback_url` global → XHR download

Both paths read the blob as `ArrayBuffer`, convert to `Uint8Array`, and pass to `O(l)` which creates the `If` instance and starts playback.

## Message Filtering

During fast-forward (seeking via timeline click), the playback pre-dispatches certain message types:

```javascript
// Kf.W callback (line 133):
for (var b = this.i; b < a; b++) {
  var c = this.m[b];
  if (c instanceof Qf || c instanceof Rf || c instanceof Sf ||
      c instanceof Tf || c instanceof Uf || c instanceof Vf ||
      c instanceof Wf || c instanceof Xf || c instanceof Yf)
    ve(this.v, c);
}
```

These are state-critical messages that must be processed during seek:
- Qf: TankStatusFull (own tank identity)
- Rf: Promotion
- Sf: Decoration
- Tf: TankInfo
- Uf: TankEntry
- Vf: TankExit
- Wf: EquipmentGain
- Xf: Inventory
- Yf: EquipmentToggle
