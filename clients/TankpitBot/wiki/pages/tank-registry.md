---
title: Client Tank Registry (activeGame.P.j)
tags: [protocol, registry, client-js]
related:
  - "[[shoot-event-format]]"
  - "[[shot-range]]"
source_paths:
  - see footnotes
fact_checked: "2026-06-11"
confidence: high
hubs: [protocol]
---

# Client Tank Registry

`activeGame.P.j` holds BOTH static roster slots (ids 1-36 players, 500-523 bots, all-defaults) AND live session tanks (e.g. own tank id 1301). Key on live entries.[^1]

## Verified fields

| Field | Meaning | Verification |
|-------|---------|-------------|
| `u` | **damage tier** | matched wire tier 5/5 at every transition across 3 enemies (run 004505)[^1] |
| `h` | **team** | red=0, purple=1, blue=2, orange=3; self blue=2 matches[^1] |
| `j` | drawn viewport col | motion correlation: belief (131,126)→(133,125) moved j 9→11[^1] |
| `i` | drawn viewport row | same motion; i 9→8; self renders at col 9, row 9[^1] |
| `s` | **rank points** | panel rank_points=151 at 00:45:20 matched s=151 at 00:45:18[^2] |

## Rank points behavior

- Persistent across sessions AND across deaths (purple-3 died at s=559, respawned still 559)[^2]
- Counts DOWN as tank earns promotion points[^2]
- 100000 = roster default before a tank is ever seen live[^2]
- Bots decrement ~1 per hit they land; bots that only TAKE hits stay frozen[^2]

## Damage tier for self

19/19 wire `damage_state` changes matched registry `u` in run 004505. Tiers REPAIR over time — purple-3 healed 1→0→3 after disengagement.[^1]

## Retracted theories

- **`P`/`U` as presence flag**: RETRACTED 2026-06-11. 478/478 known-dead had P=-8, BUT live tanks mid-firefight also carry P=-8. A P-based filter skipped live targets (run 110445: 59 skips, 0 kills). P is a render-frame artifact.[^3]
- **`l` as live-link flag**: REFUTED — orange-7/purple-8 l=0 with s updating. Semantics still unverified.[^3]
- **"Practice bots fight each other"**: RETRACTED — user states they never do; only we make corpses.[^3]

## Stale entries

Dead or departed tanks keep their last drawn state for minutes. Not distinguishable by any captured field. Working defense: shot-response check (miss on stationary target at range → block on kill cooldown; miss on mover → re-aim). Open crack: wire-traffic presence (live tanks generate wire messages; stale entries don't).[^3]

## Unverified fields

`o.x/o.y/o.w/o.h` (viewport bounds?), `v.0..v.8`, `aa` (self constant 62913), `$`/`active` (flip on live self), `direction` (values 0-32; facing?), `W`/`Y`/`m` (inert). Wire-score path (0x3E) is CLOSED — never sent by practice server.[^3]

[^1]: run 20260611-004505 — full registry field verification; damage tier matched 5/5 transitions; viewport position matched motion
[^2]: run 20260611-004505 — panel rank_points exact match; persistence across death verified on purple-3
[^3]: run 20260611-110445 + 013801 + 003415 — P/U, l, practice-bot theories tested and retracted
