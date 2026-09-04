---
title: V Table Complete
tags: [js-client, protocol, v-table]
related:
  - "[[js-source-map]]"
  - "[[decode-coverage]]"
  - "[[client-commands]]"
source_paths:
  - "tpclient.js:155"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (every V[x].h() parser traced line by line)
hubs: [js-client]
---

# V Table Complete

Every server→client message handler from tpclient.js, with exact byte-level parse logic extracted from the `.h()` static methods. This is the authoritative JS-side truth for how every message is parsed.[^1]

All byte indices are AFTER the message type byte has been stripped (the V table dispatcher does `a.subarray(1)`).[^1]

## Combat Messages

### V.S — ShootEvent (0x53, Gg)

```
a[0]  = flags byte (NOT shooter_id — our decoder had this wrong)
a[1]  = shooter_id low byte
a[2]  = shooter_id high byte     → X(a[1],a[2])
a[3]  = target_x
a[4]  = target_y
a[5]  = projectile_start_x
a[6]  = projectile_start_y
a[7]  = fuel (single byte)       — NOT 3-byte; our decoder read 3 here
a[8]  = weapon_type              — equipment slot (0=armor,1=dual,2=missile,3=homing,4=radar)
a[9]  = ammo_count               — remaining ammo after this shot
```

Handler logic: Determines visibility context (shooter in view? target in view? is self?) to format log messages. Triggers shoot animation with projectile path. Resets own action state if we were the shooter.[^1]

### V.A — Deactivation (0x41, Pg)

```
a[0]  = status byte              — always present
a[1]  = victim_id low byte
a[2]  = victim_id high byte      → X(a[1],a[2])
a[3]  = promo_eligible           — 1=earned extra points, 0=rank too low
a[4]  = killer_id low byte
a[5]  = killer_id high byte      → X(a[4],a[5])

If killer_id >= 65530:
  killer_id -= 65530             — mine kill (is_mine=true)
  actual mine team = killer_id
```

Handler: Sets victim's direction to 32/33 (corpse sprite). Stops victim's drive direction. Triggers deactivation sound + log message. If victim is self: enters deactivated state.[^1]

### V.K — MinePlacement (0x4B, Dg)

```
a[0]  = mine_type
a[1]  = tank_id_lo
a[2]  = tank_id_hi               → X(a[1],a[2])
a[3]  = count
a[4..] = pairs of (x, y) for each mine position
```

### V.E — MineDetonate (0x45, dh)

```
pairs of (x, y) — all mine positions that detonated
```

Handler: Sets overlay to 255 (clear mine) at each position, spawns explosion animation.[^1]

## Movement Messages

### V.G — Movement (0x47, Lg)

```
a[0]  = tank_id_lo
a[1]  = tank_id_hi               → X(a[0],a[1])
a[2]  = start_x
a[3]  = start_y
a[4]  = direction                — direction byte (see encoding below)
a[5]  = damage_state              — sets b.u (dual-purpose: rank_category on init, damage during gameplay)
a[6]  = lb_score byte 0 (high)
a[7]  = lb_score byte 1
a[8]  = lb_score byte 2 (low)    → 256*(256*a[6]+a[7])+a[8]  (24-bit BE)
a[9]  = rank
a[10] = ferry_flag               — unknown
a[11] = is_carrying              — 1=carrying obstacle
a[12..] = waypoint bytes         — direction codes: 110(n), 101(e), 115(s), 119(w)
```

### V["="] — MovementResponse (0x3D, Mg)

```
a[0]  = team                     — bits 0-1 of first byte
a[1]  = tank_id_lo
a[2]  = tank_id_hi               → X(a[1],a[2])
a[3]  = x
a[4]  = y
a[5]  = direction
a[6]  = damage_state              — sets b.u (dual-purpose: rank_category on init, damage during gameplay)
a[7]  = rank                      — sets b.l
a[8]  = lb_score byte 0 (high)
a[9]  = lb_score byte 1
a[10] = lb_score byte 2 (low)    → 256*(256*a[8]+a[9])+a[10]  (24-bit BE)
a[11] = carrying_flag            — `a.la = 0 !== this.j` (0=not carrying, else carrying)
```

## Tank Identity Messages

### V["!"] — TankInfo (0x21, Tf)

```
a[0]  = team                     — (a[0] & 255) → bits 4-7=rank_category
a[1]  = tank_id_lo
a[2]  = tank_id_hi               → X(a[1],a[2])
a[3]  = dec_byte_0
a[4]  = dec_byte_1
a[5]  = dec_byte_2
a[6]  = dec_byte_3               → yg(a[3],a[4],a[5],a[6]) = 9 x 2-bit decoration values
a[7]  = persistent_tank_id byte 0
a[8]  = persistent_tank_id byte 1
a[9]  = persistent_tank_id byte 2 → 256*(256*a[7]+a[8])+a[9]  (24-bit BE, sets a.aa)
a[10..] = name                   → p(a.subarray(10))
```

### V["("] — TankEntry (0x28, Uf)

```
a[0]  = flags                    — 255=known tank (boolean check)
a[1]  = tank_id_lo
a[2]  = tank_id_hi               → X(a[1],a[2])
a[3]  = packed byte:
        bits 0-1 = team          — a[3] & 3
        bits 2-3 = damage_state  — (a[3] >> 2) & 3 (dual-purpose b.u field)
        bits 4-7 = rank          — (a[3] >> 4) & 15
a[4]  = lb_score byte 0
a[5]  = lb_score byte 1
a[6]  = lb_score byte 2          → 256*(256*a[4]+a[5])+a[6]  (24-bit BE, sets b.s)
a[7]  = x position
a[8]  = y position
```

### V[")"] — TankExit (0x29, Vf)

```
a[0]  = team
a[1]  = tank_id_lo
a[2]  = tank_id_hi               → X(a[1],a[2])
a[3]  = was_silent               — 1=don't show message
a[4]  = was_eliminated           — 1=eliminated (tournament), 0=left voluntarily
```

### V[">"] — TankStatusFull (0x3E, Qf)

```
a[0]  = packed byte:
        bits 0-1 = team
        bits 2-3 = rank_category
        bits 4-7 = rank
a[1]  = tank_id_lo
a[2]  = tank_id_hi               → X(a[1],a[2])
a[3]  = dec_byte_0
a[4]  = dec_byte_1
a[5]  = dec_byte_2
a[6]  = dec_byte_3               → yg() decoration decode (9 slots × 2 bits)
a[7]  = lb_score byte 0
a[8]  = lb_score byte 1
a[9]  = lb_score byte 2          → 256*(256*a[7]+a[8])+a[9]  (24-bit BE, sets b.s)
a[10] = persistent_tank_id byte 0
a[11] = persistent_tank_id byte 1
a[12] = persistent_tank_id byte 2 → 256*(256*a[10]+a[11])+a[12]  (sets b.aa, ≥500 = registered)
a[13..] = name                   → p(a.subarray(13))
```

This is the **own tank** identity message. Sent once on game join. Sets tank.id, rank, team, decorations, name.[^1]

### V["."] — TankStatusSync (0x2E, Og)

```
a[0]  = team
a[1]  = tank_id_lo
a[2]  = tank_id_hi               → X(a[1],a[2])
a[3]  = damage_state              — 0-3, sets b.u (dual-purpose: rank_category on init, damage during gameplay)
a[4]  = rank                     — 0-8, sets b.l
a[5]  = lb_score byte 0
a[6]  = lb_score byte 1
a[7]  = lb_score byte 2          → 256*(256*a[5]+a[6])+a[7]  (24-bit BE)

If length > 8:
  a[8]  = promo_state            — bar position (Dc handler: 11=suppress)
  a[9]  = has_fuel_bar           — 1=show fuel bar
  a[10] = fuel_lo
  a[11] = fuel_hi                → X(a[10],a[11]) = LE u16 fuel
```

Short form (8 bytes) = status update for other tanks. Long form (12 bytes) = self status with promo + fuel.[^1]

## Resource Messages

### V.D — FuelGain (0x44, Rg)

```
a[0]  = fuel_lo
a[1]  = fuel_hi                  → X(a[0],a[1]) = absolute fuel total
a[2]  = is_free                  — 0=free (no message), 1=gained
```

### V.d — FuelDeposit (0x64, Sg)

```
a[0]  = fuel_lo
a[1]  = fuel_hi                  → X(a[0],a[1]) = absolute fuel total
```

### V.g — EquipmentGain (0x67, Wf)

```
a[0]  = show_message             — 1=show "X gained" log
a[1]  = armor_count
a[2]  = dual_count
a[3]  = missile_count
a[4]  = homing_count
a[5]  = radar_count
```

### V.I — Inventory (0x49, Xf)

```
a[0]  = display_mode             — 1=show full, 2=alternate
a[1..5] = equipment bytes:       — each: count = (byte & 127), enabled = ((byte & 128) === 0)
```

### V.t — EquipmentToggle (0x74, Yf)

```
a[0..4] = enabled flags          — 1=enabled for each slot
```

## Scan Messages

### V.F — RadarResult (0x46, Fg)

```
a[0]  = detection_type           — scan type
a[1]  = found                    — 1=something found, 0=nothing detected
```

### V.H — EnemyDetect (0x48, Tg)

```
a[0]  = x
a[1]  = y
a[2]  = team                     — this.j, used as color index ("color"+this.j+3)
a[3]  = rank                     — this.m, used with ec[] rank names (ec[this.m])
a[4]  = tank_id_lo
a[5]  = tank_id_hi               → X(a[4],a[5])
```

Handler: Calculates compass direction (N/NE/E/SE/S/SW/W/NW) from own position using ratio comparison.[^1]

### V.L — MapData (0x4C, Ig)

```
a[0]  = dot_count_lo
a[1]  = dot_count_hi             → X(a[0],a[1]) = fuel dot byte count

Fuel dot section (skip-RLE):
  offset = 2; x = 1; y = 1
  while offset < 2+dot_count:
    step = a[offset++]
    x += step; if x > 255: y++, x %= 256
    if step != 255: record (x, y) as fuel dot

Tank section (remaining bytes):
  while offset < length:
    x = a[offset++]
    y = a[offset++]
    tank_id = X(a[offset++], a[offset++])
    packed = a[offset++]:
      team = bits 0-1
      rank_category = bits 2-3
      rank = bits 4-7
```

## World Messages

### V.Z — ViewportUpdate (0x5A, Vg)

```
a[0]  = viewport_left
a[1]  = viewport_top

Entity section:
  offset = 2; col = 0; row = 0
  while offset < length:
    step = a[offset++]
    col += step % 18
    row += floor(step / 18)
    while col >= 18: row++, col -= 18
    if step != 255:
      byte0 = a[offset++]
      byte1 = a[offset++]
      byte2 = a[offset++]
      packed = 256*(256*byte0 + byte1) + byte2  (24-bit BE)
      terrain = packed & 15
      packed >>= 4
      overlay = packed & 15; if overlay >= 8: overlay = 255
      packed >>= 4
      cache = packed; if cache == 65535: cache = -1
      record (col, row, cache, overlay, terrain)
```

### V["+"] — Promotion (0x2B, Rf)

```
a[0]  = new_rank                 — 0-8
a[1]  = was_promoted             — 1=actual promotion, 0=rank sync
```

Handler: Updates own rank, resets promo bar, shows promotion message.[^1]

### V.N — Decoration (0x4E, Sf)

```
a[0]  = tank_id_lo
a[1]  = tank_id_hi               → X(a[0],a[1])
a[2]  = award_slot               — 0-8 (which decoration category)
a[3]  = award_level              — 1-3 (bronze/silver/gold)
```

### V.R — CommandResult / Supervisor (0x52, xg)

```
a[0]  = reset_action             — 1=reset to idle
a[1]  = close_map                — 1=close map view
a[2]  = error_code:
         0-10 = index into Gb[] error strings
         128+ = custom text in remaining bytes → p(a.subarray(3))
```

### V["<"] — SupervisorText (0x3C, wg)

```
entire payload = text message    → p(a) = free-form server text
```

### V["?"] — Sync (0x3F, vg)

```
a[0]  = sync_flag                — 1=reset action state (Q(a) called)
```

### V.B — BuildPickup (0x42, Jg)

```
a[0]  = tank_id_lo
a[1]  = tank_id_hi               → X(a[0],a[1])
a[2]  = start_x
a[3]  = start_y
a[4]  = target_x
a[5]  = target_y
a[6]  = direction_while_building — carry direction
a[7]  = rock_type                — 0=none, 1=type_a, 2=type_b, etc.
a[8]  = was_mine_there           — affects explosion type on pickup
```

Handler: Updates obstacle at target tile, toggles tank carrying state, plays build/hoist sound.[^1]

## Scoring Messages

### V.V — Statistics (0x56, Wg)

Trace-verified from beautified JS (line 4617-4630). Two formats based on `a.length > 12`:[^1]

Constructor: `this.i=hours, this.l=minutes, this.s=seconds, this.m=destroyed, this.j=deactivated, this.o=promo_points`[^1]

Execute confirms via log output: `"Play time: "+this.i+":"+this.l+":"+this.s`, `"Destroyed enemies: "+this.m`, `"Deactivated: "+this.j`, `"Promotion points: "+this.o`[^1]

Long format (a.length > 12):[^1]
```
a[0:2]  = playtime_hours          → X(a[0],a[1]) = LE u16 → this.i
a[2]    = minutes                  → this.l
a[3]    = seconds                  → this.s
a[4:8]  = destroyed                → 256*(256*(256*a[4]+a[5])+a[6])+a[7] = 32-bit BE → this.m
a[8:10] = deactivated              → X(a[8],a[9]) = LE u16 → this.j
a[10:14] = promo_points            → 256*(256*(256*a[10]+a[11])+a[12])+a[13] = 32-bit BE → this.o
```

Short format (a.length ≤ 12):[^1]
```
a[0:2]  = playtime_hours          → X(a[0],a[1]) = LE u16
a[2]    = minutes
a[3]    = seconds
a[4:6]  = destroyed                → X(a[4],a[5]) = LE u16
a[6:8]  = deactivated              → X(a[6],a[7]) = LE u16
a[8:12] = promo_points             → 256*(256*(256*a[8]+a[9])+a[10])+a[11] = 32-bit BE
```

### V["*"] — ActiveForces (0x2A, Xg)

```
a[0]  = red_count
a[1]  = purple_count
a[2]  = blue_count
a[3]  = orange_count
```

### V["/"] — ActivePlayers (0x2F, Yg)

```
repeating:
  tank_id = X(a[i], a[i+1])
  rank = a[i+2]
```

### V["1"] — Top10 (0x31, Zg)

```
a[0]  = team_filter              — 255=all teams
a[1]  = own_position byte 0
a[2]  = own_position byte 1
a[3]  = own_position byte 2      → 256*(256*a[1]+a[2])+a[3]  (24-bit BE)
a[4]  = own_rank

entries (variable length):
  position = a[i++]
  score = 256*(256*a[i]+a[i+1])+a[i+2]; i+=3  (24-bit BE)
  rank = a[i++]
  team = a[i++]
  name_len = a[i++]
  name = p(a.subarray(i, i+name_len)); i += name_len
```

## Tile Update Messages

### V.C — CacheUpdate (0x43, $g)

```
repeating:
  x = a[i++]
  y = a[i++]
  cache = X(a[i++], a[i++])     — if 65535 → -1 (equipment)
```

### V["@"] — OverlayUpdate (0x40, ah)

```
repeating:
  x = a[i++]
  y = a[i++]
  overlay = a[i++]
```

### V.J — TerrainUpdate (0x4A, bh)

```
repeating:
  x = a[i++]
  y = a[i++]
  terrain = a[i++]
```

### V.O — CombinedTileUpdate (0x4F, ch)

```
header:
  cache_count = X(a[0], a[1])

cache section (cache_count entries):
  x = a[i++]
  y = a[i++]
  cache = X(a[i++], a[i++])     — 65535 → -1

overlay section (remaining bytes):
  x = a[i++]
  y = a[i++]
  overlay = a[i++]
```

## Connection Messages

### V["~"] — ConnectionLost (0x7E, xe)
No payload. Triggers disconnection handler.

### V["`"] — PingResponse (0x60, we)
No payload. Heartbeat acknowledgment.

### V.T — ActionDone (0x54, Kg)
No payload. Resets map state and action state.

### V.X — TankRemove (0x58, Ug)
```
a[0]  = tank_id_lo
a[1]  = tank_id_hi               → X(a[0],a[1])
```

Handler: Removes tank from tile grid and drawing list.[^1]

### V.M — Chat (0x4D, Qg)
```
a[0]  = sender_id_lo
a[1]  = sender_id_hi             → X(a[0],a[1])
a[2]  = message_type             — index into E[] chat message table
a[3]  = x (optional)
a[4]  = y (optional)
```

[^1]: JS truth: `tpclient.js` on disk (frontmatter-pinned `tpclient.js:155`) — the V dispatch table and every `.h()` parser (lines 155-203; Statistics from `tpclient.pretty.js` lines 4617-4630); every field walk above is the transcription of its handler body, traced line by line 2026-06-19 (frontmatter `verified:` field), re-checkable by grep. Standing receipt: the bot's decoders mirror these layouts and decode every live capture in `runs/` — layout drift would surface as decode garbage in [[decode-coverage]].
