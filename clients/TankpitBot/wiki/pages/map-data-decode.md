---
title: MAP_DATA (0x14) Decode
tags: [protocol, map, fuel]
related:
  - "[[map-mechanics]]"
  - "[[fuel-system]]"
  - "[[tank-registry]]"
  - "[[map-data-algorithm]]"
source_paths:
  - "tpclient.js"
  - "runs/bot"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-07-25"
confidence: high
hubs: [protocol]
---

# MAP_DATA (0x14) Decode

The 0x14 MAP_DATA blob's first section (historically misnamed "terrain deltas") is the map's **yellow fuel-pixel layer** — a skip-RLE fuel-container atlas.[^1]

**Decode status:** the dot coordinates are materialised by `decode_map_data` (`MapDataDict.fuel_dots`) and stored on `WorldService.map_fuel_dots`, overwritten on every map open. Restored 2026-07-03 for dot-hop restocking and dot-relay travel; between 2026-06-22 and then the decoder skipped the RLE region for length validation only.[^6]

## Decode algorithm

Lifted from client JS `Ig.h` in `tpclient-b45bd1ebc9c0c668.js`:[^1]

```
cursor = (x=1, y=1)
for each byte b:
    x += b
    if x > 255: y += 1; x %= 256
    if b != 255: drop a dot at (x, y)
byte 255 = pure skip (no dot)
```

~650 dots map-wide on field01. Sum of bytes is invariant (64993).[^2]

## What dots represent

Client draws dots as 1x1 pixels in theme color index 7 (yellow). A dot marks a fuel container **with volume ≥ 500 that has been EXPOSED** (user model 2026-07-25, archive-proven: 605/605 within-session dot appearances were preceded by our own 0x4F/0x5A reveal of a ≥500 container at that coordinate; 0/163 sub-500 reveals ever joined; 0/1,400 equipment reveals ever dotted — `analysis_scripts/mine_map_dot_semantics.py` + `mine_dot_appearances.py`). Most large fuel is hidden: only ~7% of ≥500 reveals were already dotted. The old spot checks fit (every verified dot held ≥ 762; off-dot fuel 34/57).[^3]

## Cache semantics

The layer is server-persisted **exposure memory**, not a live feed and not a spawn record. It GROWS within a session when you expose a new ≥500 container (the 2026-06-11 "byte-identical across 32 opens" run simply exposed none); it persists across sessions (569–656 dots standing census — per the user, shared with the team; solo captures cannot discriminate team- from account-scope); dot disappearances track consumption. Dots outlive their volume: a container drained below 500 keeps its dot, which is why only ~40% of dots still hold fuel when visited.[^2]

## Tank entries

MAP_DATA also carries tank position entries. Full format (from JS Ig.h, verified 2026-06-19):[^1]

```
Per tank (5 bytes):
  [0] = x position
  [1] = y position
  [2] = tank_id low byte
  [3] = tank_id high byte   → LE u16 tank_id
  [4] = packed byte:
        bits 0-1 = team (0=red, 1=purple, 2=blue, 3=orange)
        bits 2-3 = rank_category → stored as tank.u
        bits 4-7 = rank (0-8) → stored as tank.l
```

The previously-undecoded `(packed>>2)&3` field is **rank_category** — stored as `.u` on the tank object, also used in TankEntry (0x28) and TankStatusFull (0x3E) messages. See [[map-data-algorithm]] for the complete parse function.[^4]

## Viewport entities (separate from MAP_DATA)

Viewport entity IDs are NOT tank IDs. They use a separate system: `entity_id == -1` (0xFFFF) = equipment container, `entity_id > 0` = fuel container (entity_id ~ fuel volume). Tanks are tracked via separate protocol messages, not viewport entities.[^5]

[^1]: tpclient-b45bd1ebc9c0c668.js, function Ig.h — exact skip-RLE decode loop; "terrain deltas" was a misnomer
[^2]: 15 runs 2026-06-11 — pickup correlation 33-71% on dots by gain bucket; bytes identical across 32 map opens in 240s
[^3]: fuel dot probe 2026-06-11 — 6/6 nearest dots held fuel, volumes 762/807/880/1042/1189
[^4]: client JS stores `(packed>>2)&3` as tank `.u` on the map object; identified as rank_category (2026-06-19 JS walk)
[^5]: viewport entity decode — entity_id field mapping; tanks tracked separately via 0x4F and other messages
[^6]: `protocol/decoders/map_data.py` (`decode_map_data`) + `sniffer/world_service.py` (`map_fuel_dots`) — decode/storage sites; restoration recorded in wiki log 2026-07-03
