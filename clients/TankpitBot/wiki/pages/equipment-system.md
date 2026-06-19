---
title: Equipment System
tags: [equipment, containers, inventory]
related: [[radar-mechanics]], [[fuel-system]]
sources: [see footnotes]
fact_checked: 2026-06-11
confidence: high
---

# Equipment System

## Priority

- Any inventory item < 10 = must collect equipment ASAP[^1]
- Equipment is CRITICAL — 0 dual shots or 0 radars means severely handicapped[^1]
- Equipment takes priority over fuel top-ups when items are depleted[^1]
- Don't constantly top up fuel; only collect when actually needed[^2]

## Equipment types

Containers hold dual shots, homing shots, and extra radars. In viewport entities: `entity_id == -1` (0xFFFF) = equipment container; `entity_id > 0` = fuel container (entity_id ~ fuel volume).[^3]

## Radar refill

Extra radars come ONLY from equipment containers. No other source. This makes equipment collection the lifeline — at 0 extras the bot is nearly blind. See [[radar-mechanics]].[^4]

## Container blacklisting

When `find_teleport_landing_tile()` returns None for a container, it goes on a per-session permanent blacklist. Cleared on death/respawn only. The old 30-second TTL caused retry loops — bot tried container (91,65) THREE times in one session.[^5]

[^1]: user (Austin), 2026-06-11 — "any inventory item < 10 = must collect equipment ASAP"; user was frustrated bot kept prioritizing fuel over equipment at 0 duals/0 radars
[^2]: user (Austin), 2026-06-11 — "don't constantly top up fuel — only collect when actually needed"
[^3]: viewport entity decode in state/viewport_entities.py — entity_id field mapping
[^4]: three consecutive runs at extras=0 — gained duals/homings but zero radars; see [[radar-mechanics]]
[^5]: run 20260610 — container (91,65) attempted 3 times with "no passable landing tile" each time; 30s TTL was the cause
