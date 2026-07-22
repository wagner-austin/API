---
title: Toolbar Layout
tags: [js-client, ui, toolbar]
related:
  - "[[js-source-map]]"
  - "[[client-commands]]"
source_paths:
  - tpclient.js lines 33-34 (pc/qc/rc/sc arrays, xc function)
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (all 18 hitbox regions traced from JS arrays)
hubs: [js-client]
---

# Toolbar Layout

The toolbar is a 384×48 pixel strip at y=256-304 (below the game area). It contains 18 clickable regions defined by position/size arrays.

## Hitbox Arrays

```javascript
// X positions (pc):  [10,53,97,151,175,197,151,166,206,151,175,197,233,263,282,304,328,362]
// Y positions (qc):  [2,2,2,2,2,2,16,16,16,31,31,31,8,8,8,8,8,8]
// Widths (rc):       [43,44,43,24,22,24,15,40,15,24,22,24,30,19,22,24,31,20]
// Heights (sc):      [44,44,44,14,14,14,15,15,15,15,15,15,26,26,26,26,26,30]
```

Note: Y positions are offset by +3 in the xc() function: `b += 3` before comparison.

## Region Map

| Index | X | Y(+3) | W×H | Action | Key |
|-------|---|-------|-----|--------|-----|
| 0 | 10 | 5 | 43×44 | Open Map | F |
| 1 | 53 | 5 | 44×44 | Radar Scan | S |
| 2 | 97 | 5 | 43×44 | Place Mine | D |
| 3 | 151 | 5 | 24×14 | Scope NW | Home |
| 4 | 175 | 5 | 22×14 | Scope N | ↑ |
| 5 | 197 | 5 | 24×14 | Scope NE | PgUp |
| 6 | 151 | 19 | 15×15 | Scope W | ← |
| 7 | 166 | 19 | 40×15 | Scope Center | F8 |
| 8 | 206 | 19 | 15×15 | Scope E | → |
| 9 | 151 | 34 | 24×15 | Scope SW | End |
| 10 | 175 | 34 | 22×15 | Scope S | ↓ |
| 11 | 197 | 34 | 24×15 | Scope SE | PgDn |
| 12 | 233 | 11 | 30×26 | Equipment: Armor Shield | 1 |
| 13 | 263 | 11 | 19×26 | Equipment: Dual Shot | 2 |
| 14 | 282 | 11 | 22×26 | Equipment: Missile Shot | 3 |
| 15 | 304 | 11 | 24×26 | Equipment: Homing Shot | 4 |
| 16 | 328 | 11 | 31×26 | Equipment: Extra Radar | 5 |
| 17 | 362 | 11 | 20×30 | Experience/Promotion Bar | — |

## Hitbox Detection (xc function, line 33)

```javascript
function xc(a, b) {
  b += 3;  // offset Y by 3 pixels
  var c;
  for (c = 0; 18 > c && !(
    a >= pc[c] && a < pc[c] + rc[c] &&
    b >= qc[c] && b < qc[c] + sc[c]
  ); c++);
  18 === c && (c = -1);  // no hit
  return c;
}
```

Returns -1 if click is outside all regions, otherwise the region index 0-17.

## Click Handler Mapping

When a toolbar region is clicked (from bb/La handlers):

| Index | State Machine Action |
|-------|---------------------|
| 0 | Open map (state 8) or close map if open |
| 1 | Radar scan (state 4) |
| 2 | Place mine (state 12) |
| 3 | Scope NW direction |
| 4 | Scope N direction |
| 5 | Scope NE direction |
| 6 | Scope W direction |
| 7 | Scope center / toggle autoscroll (on long-press) |
| 8 | Scope E direction |
| 9 | Scope SW direction |
| 10 | Scope S direction |
| 11 | Scope SE direction |
| 12-16 | Toggle equipment slot (sends `r` command with key codes 17-21) |
| 17 | Show promotion requirements (Vd function) |

## Scope Direction Mapping

The scope buttons use a direction remapping (qe function, line 204):

```javascript
function qe(a) {
  switch(a) {
    case 0: return 4;   // button N → direction 4
    case 1: return 5;   // button NE → direction 5
    case 2: return 6;   // button E → direction 6
    case 3: return 7;   // button SE → direction 7
    case 4: return 0;   // button S → direction 0
    case 5: return 1;   // button SW → direction 1
    case 6: return 2;   // button W → direction 2
    case 7: return 3;   // button NW → direction 3
    default: return a;
  }
}
```

And the map scroll directions in le() (line 103):
- 0=N: y-=64
- 1=NE: y-=64, x+=64
- 2=E: x+=64
- 3=SE: y+=64, x+=64
- 4=S: y+=64
- 5=SW: x-=64, y+=64
- 6=W: x-=64
- 7=NW: x-=64, y-=64
- 8=center: reset to viewport center

## Equipment Count Display

Equipment slots (indices 12-16) display the current count. On hover (Kd function, line 56):

```javascript
case 12: case 13: case 14: case 15: case 16:
  b -= 12;                          // slot index 0-4
  0 > a.oa[b] && (a.oa[b] = 0);    // clamp negative to 0
  Qd(a.o, b, a.oa[b]);             // show "equipment_name ( count )"
```

The `oa[]` array tracks local equipment counts, decremented on shoot (when weapon type matches) and updated from server messages.

## Menu Sprite System

The toolbar uses 4 sprite variants:
- `tc` — normal state
- `uc` — pressed/highlighted state
- `vc` — autoscroll normal
- `wc` — autoscroll pressed

Loaded from `/images/menu/0.png`, `0_down.png`, `0_autoscroll.png`, `0_autoscroll_down.png`.

The active button is highlighted by redrawing its region with the pressed sprite variant (Ec function, line 35).

## Fuel Bar

The experience/health bar at index 17 is a composite:

1. Background bar (gray): 127×2 pixels at position (233, 40)
2. Damage bar (black): overlaid proportional to tank level
3. Fuel bar (team color): `Math.floor(7 * fuel / 100)` pixels wide

The fuel bar value comes from `this.v.j` (fuel percentage 0-10000 mapped to 0-700 pixels).

## Promo State Bar

Below the fuel bar, a small vertical bar at position (366, 10):
- Height: `22 - 2 * promo_level`
- Green (`rgb(188, 207, 32)`) when promo-eligible (`this.o` flag)
- Dark red (`rgb(128, 0, 0)`) when not eligible
