---
title: Radar Mechanics
tags: [radar, scanning, equipment]
related: [[viewport-frame]], [[equipment-system]], [[fuel-system]]
sources: [see footnotes]
fact_checked: 2026-06-12
confidence: high
---

# Radar Mechanics

## Two scan types, one wire command

`CMD_RADAR` (0x66) is the only radar wire command. The server decides which type based on inventory:[^1]

- **Extra radar** (extras > 0, enabled): scans the **entire visible viewport**. Client tip text: "Extra radar scans the entire visible screen." Reveals at chebyshev distance 3-12.[^1]
- **Built-in radar** (extras = 0): **5x5 only** — chebyshev radius 2. Zero reveals beyond distance 2 across ~120 built-in scans. `REGULAR_RADAR_RADIUS = 2`.[^2]

Each scan with extras available auto-consumes one (count decrements by 1). The bot cannot choose per-scan.[^3]

## Policy: always use extra radar

Never ration extras via the toggle. The viewport sweep is always worth one extra. Keep stock UP through reliable equipment collection. A proposed "radar floor" (disable extras when low) was explicitly rejected.[^4]

## Death spiral at 0 extras

At 0 extras: 25-tile reveal vs 324-tile. Equipment discovery collapses, refill stalls. Three consecutive runs at 0 gained duals/homings but zero radars.[^5]

## Radar for fuel is not waste

One viewport sweep reveals ~10 containers. Live data: 32 pickups vs 9 dot hops in one run. Dots are ~40% fresh and one-at-a-time. Spending a radar to surface ten containers is high-value.[^6]

## Equipment refill

Extra radars come ONLY from equipment containers. No other source. See [[equipment-system]].[^5]

[^1]: user (Austin), 2026-06-12 — extra radar = full viewport; client tip text confirms
[^2]: ~120 built-in scans across captures 2026-06-12 — zero reveals beyond chebyshev 2; hits at 1 and 2 only; was an unmeasured 7x7 assumption from April
[^3]: run 20260611-062453 — extra count series 10→9→...→3, one consumed per scan
[^4]: user (Austin), 2026-06-12 — "ALWAYS use extra radar — never ration via toggle"
[^5]: three consecutive runs at extras=0 — gained duals/homings but zero radars; verified equipment is sole radar source
[^6]: run 20260613-064xxx — single fuel scan led to ~10 "Picked up container" events; 32 pickups vs 9 dot hops all run
