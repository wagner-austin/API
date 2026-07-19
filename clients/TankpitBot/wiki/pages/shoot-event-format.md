---
title: ShootEvent (0x53) Wire Format
tags: [protocol, combat, wire]
related: [[deactivation-format]], [[weapon-log-markers]], [[shot-range]]
sources: [tpclient.js Gg.h / V.S, runs/bot/bot-20260619-050303 capture t+25.47s, see footnotes]
fact_checked: 2026-06-19
confidence: high
---

# ShootEvent (0x53) Wire Format

ShootEvent is sent by the server for **every shot** — hits, misses, corpses, mines, ground. It carries the shooter, both endpoints (source + target tile), and the weapon type, but it cannot by itself distinguish hit from miss.[^1]

## Wire layout

JS handler `Gg.h` (V.S), verified against three independent witnesses on 2026-06-19: enemy source tracking, homing target tile, and damage transitions on the target.

```
[0]    team (flags byte, bits 0-1)
[1:3]  shooter_id (LE u16)
[3]    source_x — shooter position when the shot was fired
[4]    source_y
[5]    target_x — impact tile (where the projectile resolves)
[6]    target_y
[7]    aim_x — aim tile (where the gun is pointed, == target for straight shots)
[8]    aim_y
[9]    weapon (0=single, 1=dual, 2=missile, 3=homing)
```

Top-level 0x53 message: body is 10 bytes (above) after the message-type byte. Tunneled inside 0x2E: outer subtype is `0x53`, inner is 10 bytes — same layout.

**History:** before 2026-06-19, the decoder had wrong field names — it reported `source_x` as `target_x` and treated `target_x` as `projectile_x`. The byte offsets were correct; the semantic labels were swapped. Three independent verifications (own-tank firing position, homing-shot impact tile, enemy damage-state transitions on target tiles) forced the rename.

**2026-06-20:** `a[7]`/`a[8]` promoted from `unk1`/`unk2` to `aim_x`/`aim_y` after JS proof: `Gg.h` passes them to the projectile-animation constructor `yf` as `z` and `O`; inside `yf`, `this.qa = 24 * z + 12` and `this.ta = 16 * O + 8` compute the PIXEL CENTRE of the aim tile, and `yf.start()` uses `atan2(this.h - this.qa, this.ta - this.i)` to set the tank's facing direction. For straight shots aim == target; for missile/homing weapons the aim is the initial barrel direction and the target is the impact tile.

## Hit vs miss

ShootEvent fires regardless of outcome. To detect hit vs miss, correlate with:

- **TankStatusSync (0x2E) damage_state transitions** on the target tank — a `damage_state` decrement that occurs within ~200ms of a ShootEvent at the target's tile is a confirmed hit.
- **Deactivation (0x41)** for ALL kills including the bot's own — arrives 0x2E-tunneled; the earlier "does not fire for own kills" claim was a decoder blind spot falsified 2026-07-19 (see [[deactivation-format]]).

## Hit behavior by target state

- **Live tank**: positive hit, damage applied.
- **Shielded tank**: positive hit (shields absorb, no `damage_state` change).
- **Corpse / deactivated tank**: positive hit (no effect, but the wire reports a hit).
- **Miss**: the shot literally missed — target moved off the tile between dispatch and resolution.

Shields and corpses do **NOT** return miss. A miss on a stationary target at range is impossible — it means the target moved. Deactivation detection comes from the 0x41 wire message (0x2E-tunneled, fires for own kills too — falsified-claim history in [[deactivation-format]]), not from per-shot hit/miss feedback.[^2]

## Damage tiers (from correlated TankStatusSync / MovementResponse)

`damage_state` counts **down** toward deactivation:

- `0` = full / unsynced
- `3` = light
- `2` = medium
- `1` = critical

Every observed kill died from tier 1.[^3]

The original assumption (1=light, 3=critical) was inverted — the bot preferred the **healthiest** equal-distance enemy for finish-off. Fixed in `_finish_priority` in `bot/ai/threats.py`.[^3]

Damage tiers **repair over time**. Purple-3 healed `1→0→3` after disengagement. Finish damaged targets; disengaging forfeits progress.[^4]

## What NOT to use ShootEvent for

Don't use ShootEvent's `target_x` / `target_y` alone to decide hit/miss. Use TankStatusSync `damage_state` transitions or DOM scraping. ShootEvent tells you where the shot went and what weapon was used — not whether it landed.

[^1]: protocol analysis with make sniff, 2026-06-10 — server emits 0x53 for every shot fired; final byte is weapon type, not hit/miss
[^2]: user (Austin), 2026-06-16 — "shields don't return miss, they return a positive hit. a corpse returns a positive hit. that's why we have to use the DOM scrape to determine when a bot is deactivated"
[^3]: run 20260610-231x — every fight ran 0→3→2→1; all 5 kills died from tier 1
[^4]: run 20260611-004505 — purple-3 healed 1→0→3 after bot disengaged; 19/19 damage_state changes matched registry `u` field
