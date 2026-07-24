---
title: JS Source Map
tags: [js-client, reverse-engineering, source-map]
related:
  - "[[v-table-complete]]"
  - "[[client-commands]]"
  - "[[client-constants]]"
source_paths:
  - "tpclient.js"
  - "tpclient.pretty.js"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (complete manual walk of all 329 lines)
hubs: [js-client]
---

# JS Source Map

Complete annotated structure of `tpclient.js` (329 lines, ~82k tokens of minified Closure Compiler output). Every class, function, and data structure identified and mapped.[^1]

## File Structure Overview

| Lines | Section | Contents |
|-------|---------|----------|
| 1-3 | Polyfills | Object.defineProperties, Object.assign, Object.create, setPrototypeOf |
| 4 | Inheritance helper | `n(a,b)` — Closure's prototype chain setup, used everywhere |
| 4-5 | **qa** — ScaledContext | Canvas drawing wrapper with DPI scaling (.h=scale factor) |
| 5 | **sa** — CanvasLayer | Named canvas element factory (384×h pixels, used for Background/Tanks/Action/Map/Overlay/Menu layers) |
| 6-10 | **va** — BaseCommand (and subclasses) | Client→server command base class: wa(AUTH), xa(GameSelect), ya(JoinGame), Aa(BinaryCmd), Ba(Quit), Ca(Error), Ea(FatalError), Ha(Volume), Ia(Autoscroll), Ja(OverallSeries), Ka(Chat), La(Scale), Ma(Sprites), Na(Hotkeys), Oa(ChatMessages) |
| 11-14 | **Pa/ab** — WebSocket transport | Connection lifecycle: open→message→close→error, binary framing (2-byte LE length prefix), XOR decode pipeline |
| 14-15 | **C** — RGB Color | `new C(r,g,b)`, toString()→"rgb(r,g,b)", cb()→"#rrggbb", db()→equality, eb()→parse hex |
| 15-16 | **Global constants** | fb(isHTTPS), gb(isEmbed), hb(connectServer), ib(wsURL), D(imagesDir), jb(teamNames), kb(yellow), lb(black), mb(deactivatedMessages), nb(awardNames 27 entries), Za(errorCode=1) |
| 16-17 | **XOR cipher** | pb=tankpit.magic, qb[1000]=XOR table derived from hardcoded string^magic, za(a,b)=XOR encode/decode, q(str→Uint8Array), p(Uint8Array→str) |
| 17-21 | **sb** — Fingerprint | Browser fingerprinting (MurmurHash3): userAgent, language, colorDepth, screen, timezone, storage, canvas, plugins |
| 22 | **vb** — ActionQueue | Animation action list, wb()=add, xb()=flush |
| 22 | **yb** — Award sprite widths | `[15,31,11,11,13,15,11,12,16,9]` — pixel widths per award slot |
| 22-24 | **Bb** — ChatController | Chat message queue with 2400ms cooldown between sends, team/zone filtering |
| 24-30 | **K** — BaseClientCommand (and subclasses) | Binary command constructors with .h() serializers: |

### Client Command Classes (lines 25-31)

| Class | Code | Bytes | Purpose |
|-------|------|-------|---------|
| Jb | `?` | 2 | Heartbeat/ping |
| Kb | `.` | 2 | Ping request |
| Lb | `s` | 6 | **Shoot**: x, y, target_id(LE u16) |
| Mb | `f` | 2 | Radar scan — RE-TRACED 2026-07-24 end to end: `function Mb(){this.code="f"}`; keymap `nh()` `KeyS:26` → input case 26 → `P(this,4)` → action case 4 `new Mb`. The **S key** fires radar |
| Nb | `l` | 2 | Open map — RE-TRACED 2026-07-24 end to end: `function Nb(){this.code="l"}`; keymap `KeyF:14` → input case 14 → `P(this,8)` → action case 8 `new Nb` (zoom-out when the map is already open). The **F key** opens the map |
| Ob | `t` | 4 | **Teleport**: x, y |
| Pb | `p` | 4 | **Move**: x, y |
| Qb | `k` | 2 | Deactivate/exit |
| Rb | `Z` | 3 | **Scope/view extend**: direction byte |
| Sb | `z` | 4 | **Scope move**: x, y |
| Tb | `b` | 4 | **Build/pickup obstacle**: x, y |
| Ub | `j` | 4 | **Pick up item**: x, y |
| Vb | `d` | 4 | **Drop/deposit**: x, y |
| Wb | `D` | 6 | **Deposit fuel**: x, y, amount(LE u16) |
| Xb | `h` | 2 | Detect enemy (nearest enemy scan) |
| Hb | `m` | 4-6 | **Fire**: action_type, use_special, x, y |
| Yb | `i` | 2 | Request inventory |
| Zb | `*` | 2 | Active forces request |
| $b | `1` | 3 | Top 10 request: team_filter byte |
| ac | `/` | 2 | Active players request |
| bc | `v` | 2 | Statistics request |
| cc | `r` | 3 | Hotkey action: key byte |
| dc | `!` | 2 | Keep-alive heartbeat |

### Default keymap `nh()` — keys are NOT the wire chars (re-trace 2026-07-24)

The June trace's Mb/Nb swap came from assuming letter-key ↔ command-char
identity. The real chain has an arbitrary keymap in between: `nh()`
returns `{Space:2, KeyT:3, KeyR:4, KeyP:5, KeyB:6, KeyO:7, Slash:8,
KeyC:9, KeyI:10, KeyQ:11, KeyH:12, KeyE:13, KeyF:14, KeyX:15, KeyL:16,
Digit1-5/Numpad1-5:17-21, KeyM:22, KeyZ:23, KeyN:24, KeyA:25, KeyS:26,
KeyD:27, Arrows:29-32, ...}` (keydown handler passes `event.code` into
`m.$a`, which looks the index up in this map). Verified chains: **S**
→ 26 → `P(this,4)` → `new Mb` → wire `'f'` (radar); **F** → 14 →
`P(this,8)` → `new Nb` → wire `'l'` (map open / zoom-out); **E** → 13
→ `P(this,9)` → `new Xb` → wire `'h'` (nearest enemy). **KeyL is the
sound toggle** (case 16) and never reaches the wire; **KeyM in this
build is the tips toggle** (case 22) — the user-contract map-close 'm'
behavior lives in the map component's own handler, not this dispatcher.
The bot codebase's key comments in `protocol/commands.py` ('s' radar,
'f' map, 'e' nearest-enemy) were correct all along.[^1]

Two more chains traced 2026-07-24 (user contract, verbatim: *"r uses
the radar, "5" enables and disables it. 1 enables and disables
armor shields, 2 dual shots, 3 missiles, 4 homing shots."*):
**Digit1–5 → input cases 17–21 → `b-17+49` → `new cc(49..53)`** — the
wire `'r'` hotkey command carrying the ASCII digit; slots toggle in
inventory order (1 armor, 2 dual, 3 missile, 4 homing, 5 radar),
confirming the user contract exactly. **KeyR → input case 4 →
`fe(this,0)` → `new $b(0)`** — in THIS build's default map R sends
the red-team Top-10 request (R/P/B/O → `$b(0..3)`), not a radar
scan; the user's "r uses the radar" matches the classic binding
(and `nh()` is only the default table — the live keymap `this.l.j`
may be rebound), while the pinned client's default radar key is
S.[^1]

### Game Constants (lines 31-33)

```
ec = ["recruit","private","corporal","sergeant","lieutenant","captain","major","colonel","general"]
fc = ["rec","pri","cor","ser","lie","cap","maj","col","gen"]  — abbreviated rank names
gc = ["armor shield","dual shot","missile shot","homing shot","extra radar"]
hc = ["rgb(224, 0, 0)","rgb(224, 0, 224)","rgb(0, 179, 255)","rgb(255, 144, 0)"]  — team colors
ic = [-2,5,9,14,16,14,10,6,-2,-8,-13,-17,-19,-18,-15,-9]  — projectile X offsets (16 directions)
jc = [-15,-14,-13,-9,-4,2,7,9,10,8,6,2,-4,-7,-11,-14]  — projectile Y offsets (16 directions)
kc = [44,44,28,28,28,12,12,12,12]  — tank tile flags per rank slot
```

### Timing Constants — L object (line 33)

```javascript
L = {
  wb: 30,     // LOADING tick rate (ms)
  Vb: 10,     // SHOOT animation tick
  Pb: 10,     // PROJECTILE_VISIBLE tick
  Qb: 4,      // PROJECTILE_CLOSE tick (aimed at viewport target)
  Rb: 3,      // PROJECTILE_FAR tick (outside viewport)
  xb: 0,      // JOIN tick rate
  IDLE: 200,  // IDLE tick rate — THE MAIN GAME TICK (5 ticks/sec)
  vb: 0,      // ACTION_PENDING tick
  Ub: 0,      // ??? (unused?)
  Nb: 33,     // DRIVE_MOVING tick (in-viewport, ~30fps)
  Ob: 7,      // DRIVE_DESTINATION tick (at destination)
  Mb: 100,    // DEACTIVATED tick
  Wb: 6,      // RADAR_SWEEP tick
  tc: 0,      // ??? (unused?)
  Xb: 250,    // RADAR_OBJECT_BLINK tick
}
```

**Key finding**: IDLE is 200ms (5 Hz). The game ticks at 5 Hz when nothing is animating. Drive animation runs at ~30 Hz (33ms).[^1]

**Dual-purpose field**: The tank entity field `b.u` (Xc.u) is initialized as `rank_category` (from packed byte bits 2-3 in TankEntry/sd()), but during gameplay it is overwritten with `damage_state` by Movement, MovementResponse, and TankStatusSync. Capture data confirms gameplay values track damage progression (0→3→2→1). See [[rank-category-bug]] for the full analysis.

### Promotion Point Thresholds (line 205, function Wd)

```
Rank 0 (Recruit→Private):    500 points
Rank 1 (Private→Corporal):   1,000 points
Rank 2 (Corporal→Sergeant):  4,000 points
Rank 3 (Sergeant→Lieutenant): 10,000 points
Rank 4 (Lieutenant→Captain):  20,000 points
Rank 5 (Captain→Major):       30,000 points
Rank 6 (Major→Colonel):       40,000 points
Rank 7 (Colonel→General):     50,000 points
```

Promotion from Sergeant+ also requires deactivating an enemy of the previous rank or higher (line 60, Vd function).[^1]

## Lines 34-71: Game Session ($d class)

The main game session class. Inherits from Oc (base session).[^1]

### State Machine (s field)

| State | Meaning | Transition |
|-------|---------|------------|
| -1 | Uninitialized | → 0 on load |
| 0 | Processing | → 1 when done |
| 1 | **IDLE/READY** | → 2-13 on user action |
| 2 | Move pending | → 0 after send |
| 3 | Fire pending | → 0 after send |
| 4 | Radar scan pending | → 0 after send |
| 5 | Fuel pickup pending | → 0 after send |
| 6 | Equipment pickup pending | → 0 after send |
| 7 | Build/obstacle pending | → 0 after send |
| 8 | Open map pending | → 0 after send |
| 9 | Detect enemy pending | → 0 after send |
| 10 | Deposit fuel pending | → 0 after send |
| 12 | Mine placement pending | → 0 after send |
| 13 | Scope change pending | → 0 after send |

### Keep-alive (line 71)

```javascript
m.Fb = function() {
  var a = this.va;
  3E4 < y() - a.j && I(this.W, new dc)  // send heartbeat every 30 seconds
};
```

30-second keep-alive interval. Our bot should match this.[^1]

### Action Priority in pb() tick (lines 49-51)

1. If action animations playing → process animations (set tick rate to animation speed)
2. If queued command (Qa) → execute it, clear
3. If server messages in queue → process next, set ACTION_PENDING tick
4. Else → IDLE (200ms), run Fb() keep-alive check

## Lines 95-98: Asset Loading (Yc class)

Sprite sheet loader. 7 sprite categories loaded per game:[^1]
- `[0]` Tiles
- `[1]` Dust
- `[2]` Splash
- `[3]` Explosions
- `[4]` Radar
- `[5]` Tanks
- `[6]` Awards

Custom sprite support via localStorage `custom_sprites` key.[^1]

## Lines 100-107: Map System (Tc class)

- Map canvas: 384×256 (3x world pixel = 1 map pixel for X, 2x for Y)
- Map offset: j=x, i=y (0-128 in 64-step increments for scrolling)
- Tank positions stored in `m[(x<<8)+y]` hash
- Fuel dots rendered as single 1×1 pixels on map
- `le()` handles 9-direction scroll (0=N, 1=NE, 2=E, ..., 8=center)

## Lines 108-127: Animation System

### Re — Drive animation
- Waypoint-based path following
- 4-pixel X / 4-pixel Y per step increments
- Direction encoding: 101=east, 115=south, 119=west, 110=north (ASCII codes for e/s/w/n)
- Obstacle carrying detection via `.Y` flag
- Ferry (terrain 5) auto-detection and sprite handling

### yf — Shoot animation
- Bresenham-like projectile path from source to target
- 12-pixel X / 8-pixel Y step sizes
- Weapon type determines explosion: 0=normal(2), 1=small(1), 2=large(3), 3=large(3)
- Direction calculated from angle: `Math.atan2()` → 16-direction index

### uf — Radar sweep animation
- 17-frame animation cycle
- On completion: marks containers (fuel=wf, equipment=vf) at discovered tiles

### lf — Explosion animation
- Types: 0=splash(ca), 1=normal(i), 2=default(kf=2), 3=big, 4=build(Y), 5=splash(ca)
- pf=[9,17,26,32,42,45,48] — cumulative frame starts per type

## Lines 127-130: Tank Entity (Xc class)

```
id, name, team(h), rank(l), direction, damage tier(u)
j/i = viewport col/row (-8 = off screen)
v = Uint8Array(9) — decoration/award state
W = carry direction (0=none, 101=E, 115=S, 119=W, 110=N)
Y = carrying obstacle flag
s = leaderboard score (1E5 = 100000 initial)
aa = persistent tank_id (for profile links, ≥500 means registered)
```

## Lines 142-144: Tile Grid (Wc/cg classes)

### cg — Tile data structure
```
i = terrain type byte (complex bitfield: bits 0-3=adjacency, bits 4-6=base type)
m = overlay value (255=none, 0-3=mine team)
cache = fuel/equipment value (>0=fuel volume, <0=equipment, 0=empty)
j = rock type (0=none, 1=type_a, 2=type_b, 3=both, 5=ferry_rock, 7=ferry_with_rock)
h = tank reference (null or Xc instance)
o = occupied-behind flag (tank driving away)
l = dirty flag (needs redraw)
```

### Terrain byte encoding (lines 145-155)

```
bits 0-3: adjacency flags (N,E,S,W connections to same terrain)
bits 4-6: base terrain type:
  000 (0)  = ground
  001 (16) = grass variant
  010 (32) = water
  011 (48) = water corner overlay
  100 (64) = mountain/obstacle
dg = [0,1,2] — terrain classification (ground, rock, water)
```

Edge tiles (0/17) are border fringe — not actionable.[^1]

## Lines 155-203: V Table — Message Handlers

The complete server→client message dispatch table. See [[v-table-complete]] for field-by-field layouts.

| Key | Char | JS Class | Handler |
|-----|------|----------|---------|
| `~` | 0x7E | xe | ConnectionLost |
| `` ` `` | 0x60 | we | PingResponse |
| `?` | 0x3F | vg | Sync (resets action state) |
| `<` | 0x3C | wg | SupervisorText (free text message) |
| `R` | 0x52 | xg | **CommandResult** (error_code, close_map, reset_action) |
| `!` | 0x21 | Tf | TankInfo (name, team, decorations, score) |
| `(` | 0x28 | Uf | TankEntry (team, rank_category, tank_id, position) |
| `)` | 0x29 | Vf | TankExit (was_silent, was_eliminated) |
| `>` | 0x3E | Qf | **TankStatusFull** (own tank: team, rank, decorations, score, name) |
| `+` | 0x2B | Rf | **Promotion** (new_rank, was_promoted flag) |
| `N` | 0x4E | Sf | **Decoration** (tank_id, award_slot, award_level) |
| `K` | 0x4B | Dg | MinePlacement (type, tank_id, positions) |
| `F` | 0x46 | Fg | RadarResult (detection_type, found flag) |
| `S` | 0x53 | Gg | **ShootEvent** (10 bytes: flags, shooter, target_xy, proj_xy, fuel, weapon, ammo) |
| `I` | 0x49 | Xf | Inventory (show flag, 5 counts + 5 enabled booleans) |
| `g` | 0x67 | Wf | EquipmentGain (show_message, 5 gained counts) |
| `t` | 0x74 | Yf | EquipmentToggle (5 enabled booleans) |
| `L` | 0x4C | Ig | **MapData** (fuel dots via skip-RLE, tank entries with team/rank/position) |
| `B` | 0x42 | Jg | BuildPickup (obstacle build/drop/pickup result) |
| `T` | 0x54 | Kg | ActionDone (bare completion ping) |
| `G` | 0x47 | Lg | Movement (tank_id, start_xy, direction, flag, lb_score, rank, ferry_flag, waypoints) |
| `=` | 0x3D | Mg | **MovementResponse** (team, tank_id, xy, direction, rank, lb_score, carrying_flag) |
| `.` | 0x2E | Og | **TankStatusSync** (team, tank_id, damage, rank, lb_score, promo_state, has_fuel_bar, fuel) |
| `A` | 0x41 | Pg | **Deactivation** (status, victim_id, promo_eligible, killer_id, is_mine_kill) |
| `M` | 0x4D | Qg | Chat (sender_id, message_type, x, y) |
| `D` | 0x44 | Rg | FuelGain (absolute fuel, is_free flag) |
| `d` | 0x64 | Sg | FuelDeposit (absolute fuel) |
| `H` | 0x48 | Tg | EnemyDetect (x, y, rank, team, tank_id) |
| `X` | 0x58 | Ug | TankRemove (tank_id) |
| `Z` | 0x5A | Vg | **ViewportUpdate** (viewport_x, viewport_y, entity tiles with cache/overlay/terrain) |
| `V` | 0x56 | Wg | Statistics (playtime, destroyed, deactivated, score) |
| `*` | 0x2A | Xg | ActiveForces (4 team counts) |
| `/` | 0x2F | Yg | ActivePlayers (tank_id + rank list) |
| `1` | 0x31 | Zg | Top10 (team_filter, rank, entries with name/score) |
| `C` | 0x43 | $g | CacheUpdate (x, y, cache_value patches) |
| `@` | 0x40 | ah | OverlayUpdate (x, y, overlay patches) |
| `J` | 0x4A | bh | TerrainUpdate (x, y, terrain patches) |
| `O` | 0x4F | ch | CombinedTileUpdate (cache patches + overlay patches) |
| `E` | 0x45 | dh | MineDetonate (x, y positions, triggers explosion animation) |

## Lines 203-213: Utility Functions

- `S(a,b)` — in-viewport check: 0≤a<18 && 0≤b<18
- `ae(a,b)` — in-actionable check: 1≤a<17 && 1≤b<17
- `Ld(a,b)` — in-game-area: 0≤a<384 && 0≤b<256
- `Td(a,b)` — in-toolbar-area: 0≤a<384 && 256≤b<304
- `X(a,b)` — LE u16: (a&255)+256*(b&255)
- `Xe(a)` — direction char→sprite index: 101→20, 115→8, 119→28, 0→32
- `Wd(a)` — rank→promotion point threshold (see constants above)
- `zg(a)` — format tank name (link if aa≥500)
- `Ed()` — smooth-spin tank sprite toward target angle

## Lines 214-224: Main Application Bootstrap

IIFE that initializes everything:[^1]
1. Creates WebSocket connection (`ab` instance)
2. Creates settings (`mh` → Ga)
3. Creates UI (`zh` → A) with lobby, settings, guide panels
4. On `+` message: adds game to lobby select list (Jf instances in la[])
5. On `-` message: removes game from lobby
6. On `$` message: game join accepted → create $d session → load assets → start
7. On `.` message (byte 46 = 0x2E): XOR decode → dispatch to V table handler

## Lines 224-329: UI Classes

- **zh** — Main app container (fullscreen, resize, microphone/speech recognition)
- **Rc** — Game canvas + input bindings (mouse, touch, keyboard)
- **Ch** — Lobby (game list, troop picker, field preview)
- **Dh** — Settings dialog (size, sounds, graphics, hotkeys, leaderboard, colors)
- **Eh** — How To Play guide (6 pages: basics, controls, equipment, ranks, awards, tournaments)
- **gi** — Log/history panel
- **hi** — Status bar (tank info, coordinates, hover tooltips)
- **ii** — Chat message selector (61 of the 65 predefined messages; 4 hidden — [[chat-messages]])
- **Nf** — Playback timeline for recorded games
- **Oh** — HSV color picker for map customization

## Key Architectural Insights

1. **All commands are 2-byte length-prefixed** before XOR. The first byte after length prefix is the command character.

2. **The V table is the single dispatch point** for all server messages. `Mf(a)` at line 203 calls `V[String.fromCharCode(a[0])].h(a.subarray(1))`.

3. **Direction encoding uses 16 values (0-15)** plus high bits for dead/corpse state. Bits 4-7 carry the "walking from" direction for the trailing tile overlay.

4. **The game has exactly 65 chat messages** (E[0] through E[64], no gaps — the earlier "63, some gaps" figure here was a miscount; all 65 are traced in [[chat-messages]]). 61 appear in the selector (4 are hidden: ids 5, 29, 63, 64). Most have voice recognition keywords for microphone input.

5. **Tank IDs ≥ 500 are registered accounts** — shown as profile links. Below 500 are bots or unregistered.

6. **The toolbar at y=256-304** has 18 clickable regions (xc function, line 33) for map/radar/mine/scope/equipment/experience.

[^1]: JS truth: `tpclient.js` on disk (blob-pinned in frontmatter) with `tpclient.pretty.js` as the readable companion — every table row and claim above carries its (class, line) locator inline, from the complete 329-line manual walk of 2026-06-19 (frontmatter `verified:` field); re-checkable by grep against the pinned file.
