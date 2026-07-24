---
title: Client Constants
tags: [js-client, constants, game-mechanics]
related:
  - "[[js-source-map]]"
  - "[[game-rules]]"
  - "[[equipment-system]]"
source_paths:
  - "tpclient.js:15"
  - "tpclient.js:31"
  - "tpclient.js:205"
  - "tpclient.js:261"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (extracted directly from JS source)
hubs: [js-client]
---

# Client Constants

Every hardcoded constant in tpclient.js, organized by category.

## Team & Rank

```javascript
// Team names (jb array, line 15)
["red", "purple", "blue", "orange"]

// Team colors (hc array, line 31)
["rgb(224, 0, 0)", "rgb(224, 0, 224)", "rgb(0, 179, 255)", "rgb(255, 144, 0)"]

// Rank names (ec array, line 31)
["recruit", "private", "corporal", "sergeant", "lieutenant", "captain", "major", "colonel", "general"]

// Abbreviated rank names (fc array, line 31)
["rec", "pri", "cor", "ser", "lie", "cap", "maj", "col", "gen"]

// Promotion point thresholds by rank (Wd function, line 205)
rank 0→1:   500
rank 1→2:  1,000
rank 2→3:  4,000
rank 3→4: 10,000
rank 4→5: 20,000
rank 5→6: 30,000
rank 6→7: 40,000
rank 7→8: 50,000

// Promotion from rank 3+ also requires deactivating an enemy of (rank-1) or higher
// (line 60, Vd function: b-- for ranks > 2, then check ec[b] name)
```

## Equipment

```javascript
// Equipment names (gc array, line 31)
["armor shield", "dual shot", "missile shot", "homing shot", "extra radar"]

// Equipment capacity: Recruits hold 20 each, +5 per rank (from guide text, line 263)
// So: Recruit=20, Private=25, Corporal=30, ... General=60

// Tournament exception: Recruits hold 60 each in tournaments (line 265)
```

## Fuel (mined 2026-07-06)

```javascript
// Fuel gauge draw (Gc function): capacity is rank-derived, never on the wire.
// fill width = 7*fuel/100 px; capacity region = 7*(10+rank) px
// => FUEL CAPACITY = 100*(10+rank) = 1000 + 100*rank
// Verified vs user deposits at ranks 1/3/6/7 -- see game-economy.

// Fuel setter (Cc function): display sanity clamp only
function Cc(a,b){if(0>b||1E4<b)b=0;a.j=b;a.P=!0}

// Fuel gate (ce function): actions blocked at fuel <= 100 with local
// "Insufficient fuel" log line (never reaches the wire):
function ce(a){return 100>=a.v.j?(F(a.j,"Insufficient fuel\n",H),Q(a),!1):!0}
// Gated actions: targeted shot (aim tile holds a tank), mine drop 'k',
// nearest-enemy 'h', fuel deposit 'D', obstacle pickup.
// NOT gated: untargeted shot, radar 'f', map open 'l', move, teleport.
```

## Action dispatch facts (mined 2026-07-06)

```javascript
// Shoot (Lb, code 's', 6 bytes): x, y, u16 LE target_id.
// target_id = id of the tank on the aimed tile, else 0 -- clicking a
// tank's tile IS the homing shot; there is no separate homing mode.
// Client-side pre-checks (never reach the wire): aim at own tile ->
// silent abort; aim at teammate -> local "Friendly fire!" line.
// A wire 0x52 code-3 therefore only occurs on races (tank moved onto
// the tile after dispatch).

// Fuel deposit (Wb, code 'D' = 0x44, 6 bytes): x, y, u16 LE amount.
// Amount accumulates during long-press, clamps to current fuel.
// Server enforces the 100-fuel floor (max deposit = capacity - 100).

// Radar (Mb, code 'f'): no client fuel gate; sets a client-side
// cooldown counter (this.ca = 50) on dispatch. Units unconfirmed.

// Long-press action resolver (be function): tile cache > 0 -> "GET FUEL",
// cache < 0 -> "GET EQUIPMENT", empty tile + accumulated amount ->
// "DEPOSIT FUEL: N". Confirms the tile-cache sign convention.

// Obstacle carrying exists (unused by bot): "PICK UP OBSTACLE" needs
// fuel > 100 and tile flag bit 2; "DROP OBSTACLE" / "BUILD BRIDGE"
// (obstacle onto water, overlay bits 32===(i&112)) via Tb, code 'b'.
```

## Error Strings

```javascript
// Supervisor error messages (Gb array, line 31)
[
  "You can't do this",          // 0
  "You can't go there!",        // 1
  "Uncontrollable tank",        // 2
  "Friendly fire!",             // 3
  "Empty container",            // 4
  "Tank full",                  // 5
  "You are already there!",     // 6
  "Inventory full",             // 7
  "Insufficient fuel",          // 8
  "No enemies found",           // 9
  "Congratulations!"            // 10
]
```

## Deactivation Messages

```javascript
// (mb array, line 15)
["Your tank has been deactivated.", "Your tank is being repaired. Please wait."]
```

## Award Names

```javascript
// (nb array, line 15-16) — 27 entries, 3 per category × 9 categories
[
  "SINGLE STAR", "DOUBLE STAR", "TRIPLE STAR",          // 0: rank-based (major/colonel/general)
  "BRONZE TANK AWARD", "SILVER TANK AWARD", "GOLDEN TANK AWARD",  // 1: kills (100/200/500)
  "COMBAT HONOR MEDAL", "BATTLE HONOR MEDAL", "HEROIC HONOR MEDAL",  // 2: deaths (20/50/100)
  "SHINING SWORD", "BATTERED SWORD", "RUSTY SWORD",     // 3: hours (100/200/500)
  "BRONZE SHIELD", "SILVER SHIELD", "DEFENDER OF THE TRUTH",  // 4: bug reports
  "BRONZE CUP", "SILVER CUP", "GOLDEN CUP",             // 5: tournament (3rd/2nd/1st)
  "PURPLE HEART", "PURPLE HEART 2", "PURPLE HEART 3",   // 6: content creation
  "WAR CORRESPONDENT", "WAR CORRESPONDENT 2", "WAR CORRESPONDENT 3",  // 7: promotion
  "LIGHTBULB AWARD", "LIGHTBULB 2", "LIGHTBULB 3"       // 8: ideas
]
```

## Award Sprite Widths

```javascript
// (yb array, line 22) — pixel widths per award slot in decoration bar
[15, 31, 11, 11, 13, 15, 11, 12, 16, 9]
```

## Timing Constants

```javascript
// (L object, line 33) — tick rates in milliseconds
L = {
  wb: 30,     // Loading screen tick
  Vb: 10,     // Shoot animation frame
  Pb: 10,     // Projectile visible (in viewport, not aimed)
  Qb: 4,      // Projectile close (aimed at viewport target)
  Rb: 3,      // Projectile far (outside viewport)
  xb: 0,      // Joining game tick (immediate)
  IDLE: 200,  // ★ Main game idle tick — 5 Hz
  vb: 0,      // Action pending (immediate processing)
  Ub: 0,      // Unused
  Nb: 33,     // Drive animation (~30 fps)
  Ob: 7,      // Drive destination (faster for final approach)
  Mb: 100,    // Deactivated state (slow tick, 10 Hz)
  Wb: 6,      // Radar sweep animation
  tc: 0,      // Unused
  Xb: 250,    // Radar object blink interval
}

// Chat cooldown: 2400ms between messages (Bb class, line 24)
// Keep-alive: 30,000ms idle heartbeat (dc command, line 71)
// Touch double-tap: 300ms window (line 63)
// Tip display: 60,000ms interval (line 62, Dd function)
```

## Direction Encoding

```javascript
// ASCII-based direction codes (movement waypoints)
110 = 'n' = north (y--)
101 = 'e' = east  (x++)
115 = 's' = south (y++)
119 = 'w' = west  (x--)

// Direction→sprite mapping (Xe function, line 204)
101(e) → 20
115(s) → 8
119(w) → 28
  0    → 32  (dead/stationary)

// 16-direction sprite system: 0-15 for facing direction
// High nibble (bits 4-7): "walking from" direction for trailing tile
// Corpse directions: 32 = even death, 33 = odd death
```

## Projectile Offsets

```javascript
// X offsets per 16 directions (ic array, line 31)
[-2, 5, 9, 14, 16, 14, 10, 6, -2, -8, -13, -17, -19, -18, -15, -9]

// Y offsets per 16 directions (jc array, line 31)
[-15, -14, -13, -9, -4, 2, 7, 9, 10, 8, 6, 2, -4, -7, -11, -14]
```

## Map Colors

```javascript
// Default colors (line 207)
fh = C(255, 79, 79)    // Red team
gh = C(255, 79, 255)   // Purple team
hh = C(0, 0, 255)      // Blue team
ih = C(255, 143, 0)    // Orange team
Ke = C(60, 129, 85)    // Land (default green)
Le = C(76, 161, 105)   // Rock (lighter green)
Oe = C(89, 180, 120)   // Rock variant
Me = C(1, 10, 78)      // Water (dark blue)
jh = C(255, 255, 0)    // Fuel dot (yellow)
kh = C(0, 0, 0)        // Flash color 1
lh = C(255, 255, 255)  // Flash color 2
```

## Canvas Dimensions

```javascript
// Game area: 384 × 256 pixels (base, before scaling)
// Toolbar: 384 × 48 pixels (y=256 to y=304)
// Total: 384 × 304 pixels
// Tile size: 24 × 16 pixels
// Grid: 16×16 actionable + 1-tile fringe = 18×18 total

// Map canvas: 384 × 256 (3 pixels/tile-x, 2 pixels/tile-y)
// Map world range: 0-255 for both axes (256×256 tile world)

// Scale factor: container 570×330 at scale=1
```

## Sprite Sheets

```javascript
// Sprite categories (uj array, line 305)
["Tiles", "Dust", "Splash", "Explosion", "Radar", "Tanks", "Awards"]

// Sprite sheet dimensions (line 142)
Tiles:      192 × 192 pixels (8×12 tiles of 24×16)
Tanks:      112 × 680 pixels (4 teams × 28px wide, 34 directions × 20px high)
Explosions:  42 × 764 pixels
Radar:       68 × 619 pixels
Dust:        22 × 470 pixels
Splash:      29 × 486 pixels

// Theme presets (tj array, line 305)
["Default", "Grass", "Desert", "Ice", "Lava", "Toxic", "Namek", "Boardwalk", "Arid", "Dusk", "Space", "Custom"]
```

## Hotkey Defaults

```javascript
// Default key bindings (nh function, line 212)
Space: Shoot (2)
KeyT: Top 10 (3)
KeyR: Top 10 Red (4)        KeyP: Top 10 Purple (5)
KeyB: Top 10 Blue (6)       KeyO: Top 10 Orange (7)
Slash: Active Players (8)   KeyC: Statistics (9)
KeyI: Inventory (10)        KeyQ: Quit (11)
KeyH: Help (12)             KeyE: Nearest Enemy (13)
KeyF: Open Map (14)         KeyX: Active Forces (15)
KeyL: Toggle Sound (16)     Digit1: Toggle Armor (17)
Digit2: Toggle Dual (18)    Digit3: Toggle Missile (19)
Digit4: Toggle Homing (20)  Digit5: Toggle Radar (21)
KeyM: Show/Hide Tips (22)   KeyZ: Show/Hide Chat (23)
KeyN: Next Tip (24)         KeyA: Toggle Autoscroll (25)
KeyS: Radar (26)            KeyD: Mine (27)
ArrowLeft: Scope W (29)     ArrowRight: Scope E (30)
ArrowDown: Scope S (31)     ArrowUp: Scope N (32)
PageUp: Scope NE (33)       PageDown: Scope SE (34)
End: Scope SW (35)          Home: Scope NW (36)
F6: Ping (37)               F8: Scope Mode (38)
Equal: Volume Up (39)       Minus: Volume Down (40)
KeyK: Toggle Microphone (41)
```

## Chat Messages

63 predefined messages (E[0] through E[64], with gaps). Each has:
- `id`: message number
- `text`: display text
- `h`: team filter (0=same team only, 1=allies+team, 2=zone, 3=all)
- `l`: includes position flag
- `m`: visible in message list
- `i`: has voice recognition
- `j`: voice recognition keyword arrays

Key messages for bot: 4="HELP - Enemy!", 6="HELP - Fuel low!", 8="Fuel detected here", 9="Equipment detected here", 53="I need equipment!", 54="I need fuel!"
