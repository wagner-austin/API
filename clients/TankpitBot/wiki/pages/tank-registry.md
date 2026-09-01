---
title: Client Tank Registry (activeGame.P.j)
tags: [protocol, registry, client-js]
related:
  - "[[shoot-event-format]]"
  - "[[shot-range]]"
source_paths:
  - "src/tankpit_bot/state"
  - "runs/bot"
source_git_blobs:
  "src/tankpit_bot/state": "9fbcf07602990d544c6e69b0060f4aba1ae2a863"
fact_checked: "2026-08-07"
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
| `s` | **rank number** | panel rank number 151 at 00:45:20 matched s=151 at 00:45:18[^2] |

## Leaderboard position behavior

`s` is the tank's PLACE in the room's standings, shown in parentheses
after the rank name on the stats panel ("Rank: private (18)"). 1 is the
top. Not a points total — the points are the separate
`promotion_points` line. It descends as the tank accumulates ("im
currently rank 26. as we get more kills ill move down to 25, 24, ...,
and eventually 1"); own-tank trace across the archive: 160 (Jun 10) →
151 (Jun 11) → 27 → 26 (Aug 5, one 20-kill session apart). Startup
scrape lands in `session_account_stats.leaderboard_position` and in the
canonical runtime account model `SelfAccountDict`
(`state/types/self_account.py`, read via
`sniffer.world_state.get_self_account()`) -- the plug-in point for
rank-aware features.

**Read as a promotion COUNTDOWN from 2026-08-05 to 2026-09-01.** The
archive settled it: two tanks at 0 kills and 0 promotion points read
**28946** and **28952**, seventeen seconds apart. A countdown to the
next rank is a function of rank and points alone, so identical state
MUST yield an identical number; a place in a standings table must not.
Confirmed from the other end — one tank went 148 kills → 12055 and
then 149 kills → **12060**, the number worsening as it scored, which
positional drift explains and a countdown cannot. It is per TANK, not
per account: one account's four colours read 18, 4562 and 28946 on the
same day.[^leaderboard]

The three older observations all fit the corrected reading, and two of
them only fit it:

- Persistent across sessions AND across deaths (purple-3 died at s=559, respawned still 559)[^2]
- Descends as the tank earns promotion points[^2]
- 100000 = roster default before a tank is ever seen live[^2] — a
  sentinel for "unplaced", which a countdown has no need of
- Bots decrement ~1 per hit they land; bots that only TAKE hits stay
  frozen[^2] — landing hits moves you UP a table; taking them does not

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
[^leaderboard]: Measured 2026-09-01 from `runs/bot/*/*.events.jsonl` `session_account_stats` records across seven instances. The identical-state pair is artax recruit 0 kills / 0 promo → 28946 (20:30:33) and arterial recruit 0 kills / 0 promo → 28952 (20:30:50), both 2026-08-28. The worsening-with-kills pair is arterial sergeant 148 kills → 12055 (2026-08-28) and 149 kills → 12060 (2026-09-01). Per-tank spread on one account, same day: artax private 1933 kills → 18, artax captain 691 kills → 4562, artax recruit 0 kills → 28946. Supersedes the 2026-08-05 countdown reading this page carried; the operator named it a leaderboard the same day the measurement was taken. The registry writer that consumes `s` is `apply_tank_observation` at `src/tankpit_bot/state/tank_mutations.py:28`. Field renamed `rank_number` → `leaderboard_position` across the code the same day.
