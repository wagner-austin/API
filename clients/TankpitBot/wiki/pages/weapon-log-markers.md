---
title: Weapon-Specific Game Log Markers
tags: [combat, weapons, game-log]
related:
  - "[[shoot-event-format]]"
  - "[[shot-range]]"
source_paths:
  - "runs/bot"
fact_checked: "2026-06-10"
confidence: high
hubs: [combat]
---

# Weapon-Specific Game Log Markers

## Marker mapping (exact 1:1 counts verified)

| Weapon | Log marker | Notes |
|--------|-----------|-------|
| Dual shot | `You hit <name>` | Named hit line with target name |
| Homing shot | `You fire` | Launch marker only, NO target name, no hit line ever |
| Single shot | (none observed) | Single shots from missed duals show no log line |

## Verification

Run 20260610-223x: 53 dual wire-shots ↔ 53 "You hit" lines; 27 homing wire-shots ↔ 27 "You fire" lines. Exact 1:1 correspondence.[^1]

## Common mistake

"You hit"-count ÷ shots-sent looks like a hit rate, but is actually the **dual-shot share** of total shots. Shots in range don't miss — there are no miss lines in the game log for in-range shots. Bucket by weapon first before computing any stats.[^1]

## Homing effectiveness

Homing effectiveness is **invisible** in the game log. Needs enemy-health truth from the client tank registry (`u` field, see [[tank-registry]]).[^2]

[^1]: run 20260610-223x — 53 dual wire-shots = 53 "You hit" lines; 27 homing = 27 "You fire"; zero exceptions
[^2]: There is no game-log signal for homing hit/miss; the only observable is a `damage_state` change under the client's live `activeGame.P.j`. **That path is a RUNTIME object, not a source location** — `activeGame` resolves to `window.__tankpitActiveGame`, reached by the injected structure probe at `src/tankpit_bot/action_lab/client_structure.py:53-57`, and greps to zero occurrences in the pinned `tpclient.js` because the minified source never names it. So this claim is inspectable only during a live session via that probe; it cannot be re-verified from any committed file, and the `P.j` sub-path in particular is a minified accessor that a client update could rename without notice. Checked 2026-08-06.
