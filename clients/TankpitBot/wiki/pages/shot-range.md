---
title: Shot Range
tags: [combat, shooting, geometry]
related:
  - "[[shoot-event-format]]"
  - "[[tank-registry]]"
  - "[[combat-chase-bug]]"
provenance:
  - "runs/bot -- gitignored runtime capture artifact (moved from source_paths 2026-09-06, code-paths contract)"
fact_checked: "2026-07-03"
confidence: high
hubs: [combat]
---

# Shot Range

## Measured data (350 shots, 2026-06-11)

| Manhattan distance | Hits | Misses | Rate |
|-------------------|------|--------|------|
| 1 | 255 | 0 | 100% |
| 2 | 1 | 0 | 100% (single sample) |
| 4+ | 0 | 80+ | 0% |

Distance-15 "hits" in the raw data were homing shots (which track the target). Excluded from range measurement.[^1]

## Rules

- **Shots in range never miss** — the range is cardinal adjacency (Manhattan distance 1)[^1]
- Distance 2 has only 1 sample — not enough to trust as reliable range[^2]
- `SHOT_RANGE_TILES = 8` in `combat_landing.py` — the landing-diamond
  radius — while `has_cardinal_combat_shot` still requires Manhattan
  distance exactly 1 for the *proven* point-blank shot[^2]
- Must be within the 18x18 viewport to hit; larger ranges miss because they're outside the viewport (`COMBAT_RANGE=8` is the awareness range, not shot range)[^3]

## Cardinal adjacency

The bot requires **Manhattan distance exactly 1** (same row or column, 1 tile apart) before firing. This is the only proven reliable geometry for a guaranteed hit.[^1]

## Aim legality: the shoot command must target a tile inside the viewport

The server rejects any `shoot` whose aim tile is outside the visible viewport with 0x52 code 0 ("You can't do this") — no ShootEvent, no ammo delta, just the refusal.[^4] The user's game knowledge sharpens the rule: when the fleeing enemy is close enough that shifting the viewport (scope) would bring them into view, the game refuses a lobbed homing at their raw coordinates — you are expected to shift and take a legal shot.[^5] The bot's answer (2026-07-03) is the viewport-clamped aim: dispatch at the registry coordinate clamped onto the visible bounds, carrying the target's `tank_id`; the server picks homing and the seeker tracks regardless (aim is a hint, wire-proven: the same run's `weapon=3` hit was aimed at the target's vacated tile).

## Homing shots

Homing shots track the target regardless of distance. They bypass normal range rules. See [[weapon-log-markers]] for detection.[^1]

[^1]: 350 shots analyzed from run 20260611 — Manhattan 1 = 255/255, 4+ = ~0%, distance-15 were homing
[^2]: **Corrected 2026-08-06 — this footnote was wrong twice over.** It said "SHOT_RANGE_TILES=2 in combat_strategy.py". The constant is `SHOT_RANGE_TILES = 8` and it lives in `src/tankpit_bot/bot/ai/combat_landing.py:29`, not `combat_strategy.py`; the comment above it at `:25-28` explains the placement ("Lives here (not combat_strategy) because landing choice and acquisition viability both key off it"). It is the radius of the landing diamond, consumed at `:123`. The 2 was presumably right when written and was never revised after the range bound was reworked — see [[flag-triage-20260729]] F8, where the radius-8 short-circuit is discussed and the conclusion is that no tile bound existed in the user's law at all. The second half of the claim still holds exactly: `has_cardinal_combat_shot` (`src/tankpit_bot/bot/ai/combat_strategy.py:101`) gates on Manhattan distance exactly 1, its docstring calling that "the geometry required for a guaranteed hit at point-blank range". The distance-2 sample remains a single observation.
[^3]: user (Austin), 2026-04-20 — "must be within 18x18 viewport to hit". The 18x18 is the observable patch at `src/tankpit_bot/state/viewport_geometry.py:11-12` (`VIEWPORT_PATCH_WIDTH = VISIBLE_VIEWPORT_WIDTH + RADAR_ENVELOPE_MARGIN * 2` over the 16x16 actionable window at `:8-9`); the shot-side consequence is `SHOT_RANGE_TILES = 8` at `src/tankpit_bot/bot/ai/combat_landing.py:29`.
[^4]: live run 2026-07-03 20:34 — five `shoot(143,237,id=530)` dispatches with viewport (129,217)-(144,232) each drew 0x52 error_code=0; game log showed five "You can't do this" lines; zero 0x53 echoes, zero ammo deltas.
[^5]: user (Austin), 2026-07-03 — "the enemy was close enough that if we shifted the viewport down we could have seen them which makes the game prevent subsequent homing shots". The bot's answer to that rule is `_clamp_aim_into_viewport` at `src/tankpit_bot/bot/ai/combat_strategy.py:47`, which folds every shoot dispatch onto the visible bounds and carries the target's `tank_id` so the server's seeker still tracks — see [[combat-chase-bug]] (2026-07-03 fix).
