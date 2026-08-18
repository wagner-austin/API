---
title: "Interception — Mobile Defence at the Engine's Own Radius"
tags: [policy, combat, defence, interception, measured]
related:
  - "[[policy-holding-ground]]"
  - "[[engine-ai-zones]]"
  - "[[policy-combat]]"
source_paths:
  - "src/rw_bot/policy/guard.py"
  - "src/rw_bot/policy/dispatch.py"
  - "runs/decompiled/com/corrodinggames/rts/game/a/a.java:1189"
source_git_blobs:
  "src/rw_bot/policy/guard.py": "3dacb9802da4a42ad8c8593bc97b26a5aaedcdbb"
  "src/rw_bot/policy/dispatch.py": "00de7ee6898666e0459d7bf2d822d368b61dba7d"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [bot-architecture]
---

# Interception — Mobile Defence at the Engine's Own Radius

A hostile standing inside **360 world units** of any of our structures is an
intruder, and the reserve turns on the deepest one it can shoot.[^1] The
radius is the engine's own: its AI creates every resource-outpost zone at
exactly that figure and defends what is inside, so we call "intrusion"
precisely what the opponent calls "territory" ([[engine-ai-zones]]).

## Why mobile and not turrets

Four turret arms were measured and refuted before this existed — ahead of
income, from surplus, aimed at the base, aimed at what dies
([[policy-holding-ground]]). The diagnosis they converged on: extractors die
out where the pools are while the army rallies at the base, and static cover
is always somewhere else. The engine's AI answers raids with *mobile*
defensive groups stationed at its zones; interception is the one-rally-point
bot's minimal equivalent.

## The gate exception, precisely

The wave gate exists to stop units trickling into *defended* ground. It was
never an argument for watching a raider kill the extractor beside the rally
point: inside our own radius there are no enemy turrets and the reserve has
local numbers, so intrusion bypasses the gate — and only intrusion.[^2]
Guards forget their rally when the raid ends, so they re-gather instead of
standing where the fight finished.

## Measured, twice

Same twelve seeds, one field (`intercept`) from control, in-batch controls:

| rung | wins | extractor drops | note |
|---|---|---|---|
| Hard | 9/12 → 10/12 | 35 → 20 | zero-extractor endings eliminated |
| Very Hard | 3/12 → 4/12 | 35 → 20 | non-wins end at 46-70/s income, alive |

The exchange rate did not move (0.45–0.46 drawdown-credits per credit lost,
priced from the traces): interception does not fight better, it fights
**where the income is**, and extractor survival is the whole difference.
Wins also arrive faster — 2,144 mean samples against 2,485 at Hard. The cost
case is real: one match logged 870 intercepts and never massed an attack, so
answer-with-everything versus a capped detachment remained an open one-field
question.[^3]

## The cap question, closed

`guard_cap 3` — the three nearest engageable units race the intruder, the
rest keep gathering — was measured against an in-batch control on the same
twelve seeds: **wins 5 → 2, drops 18 → 34, intercept engagements 3,886 →
1,693.**[^4] Answer-with-everything *is* the mechanism: the deepest intruder
dies to local numbers before the extractor does, and a three-unit detachment
loses that race often enough to pay in both the detachment and the pool. The
870-intercept match was the price of the thing that works, not waste to
reclaim. `guard_cap 0` stands in the champion.

[^1]: `src/rw_bot/policy/guard.py` — `OUTPOST_RADIUS`, `deepest_intruder` with the engageability and tie-break rules; `dispatch.py` `WaveController._guard`.
[^2]: `src/rw_bot/policy/dispatch.py` — the `_guard` docstring carries the argument; only `externallyArmed` intrusion paths bypass `muster`.
[^3]: `runs/sweeps/guard-ab-hard`, `runs/sweeps/guard-ab-veryhard`; `wiki/log.md:842`, 2026-07-29.
[^4]: `runs/sweeps/cap-raid5-veryhard`; `wiki/log.md:1020`, "cap refuted with its mechanism attached", 2026-07-30.
