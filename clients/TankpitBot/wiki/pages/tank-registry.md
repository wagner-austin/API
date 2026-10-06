---
title: Client Tank Registry (activeGame.P.j)
tags: [protocol, registry, client-js]
related:
  - "[[shoot-event-format]]"
  - "[[shot-range]]"
source_paths:
  - "src/tankpit_bot/state"
provenance:
  - "runs/bot -- gitignored runtime capture artifact (moved from source_paths 2026-09-06, code-paths contract)"
  - "runs/probe -- gitignored probe capture artifacts (the page snapshots behind the bookkeeping-field table)"
  - "tpclient.js -- gitignored copy of the game client, build saved 2026-09-02 (the bookkeeping-field readings)"
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
(`state/types/self_account.py`, held as `WorldService.self_account`
in `sniffer/world_service.py`) -- the plug-in point for rank-aware
features. The bot never reads the registry's `s`: its leaderboard value
comes from the stats panel alone.[^leaderboard]

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

## Client bookkeeping fields

The rest of an entry's fields belong to the client's tank class `Xc`. Each
reading below comes from the code that writes the field, and each was then
checked against 6,358 registry entries in 86 page snapshots from three probe
runs.[^fields]

| Field | Meaning | Evidence |
|-------|---------|----------|
| `aa` | **persistent tank id**, the website's profile id; it outlives the per-session tank id | `zg` links a name to `/tanks/profile?tank_id=` + `aa` only when `aa` ≥ 500, and TankInfo (0x21) and TankStatusFull (0x3E) set it from a 24-bit id (see [[v-table-complete]]). Captured: Artax is 62913 under session ids 601 and 1301, Arterial 63008, another player 104156, practice bots 1-36 by colour slot, and no id ever carried two values[^fields] |
| `v.0`..`v.8` | **decoration slots**, one award level 0-3 each | TankInfo's handler (`Tf.prototype.h`) sets it from `yg`, which unpacks four packed wire bytes into nine 2-bit levels (see [[decoration-encoding]]); `ed` draws the ribbons and `Ff` names slot c at level d as `nb[3c+d-1]`. Captured values: only 0, 1, 2 and 3[^fields] |
| `o.x`, `o.y`, `o.w`, `o.h` (and `o.j`) | **erase rectangle**: the canvas pixels the tank was last drawn into, -1 with `o.j` false when nothing is drawn. Not viewport bounds | `$e` and `ef` merge each sprite into it through `Ic`, with a 1 px margin; `Xc.prototype.ra` clears exactly that rectangle and resets it through `Jc`. Captured: 159 of 161 drawn rectangles were exactly 30x22 at (24(j-1)-3, 16(i-1)-5), the 28x20 sprite plus that margin; the other 2 were larger merged rectangles (30x29)[^fields] |
| `$` | **drawn on the canvas** | set by the tank's draw (`Xc.prototype.sa`, and `Re.prototype.sa` mid-move), cleared by the erase, whose error text prints it ("Bad erase (d ..."). Captured: equal to `o.j` in all 6,358 entries[^fields] |
| `active` | **a movement animation is running** for this tank | `Re.start` sets it and `Re.Ma` clears it. The `G` handler (`Lg.prototype.h`) creates an `Re` per move for any tank, and the redraw pass `qd` refuses an active tank ("Tried to draw active tank."). Captured: true in 2 of 6,358 entries, both drawn[^fields] |
| `m` | **needs a static redraw** | set when the facing changes and on deactivation; `qd` erases, redraws and clears it in the same frame. Captured: false in all 6,358 entries, as a flag cleared within its own frame would read[^fields] |
| `direction` | **facing**: low nibble is a 16-point compass index; 32 or 33 is the corpse sprite; -1 is unset | the move animation writes `(direction & 240) + (heading & 15)`, and the deactivation handler (`V.A`, `Pg.prototype.h`) writes 33 when the high nibble was set, else 32. Captured: -1, 3, 4, 5 and 8, all with high nibble 0[^fields] |
| `W` | **direction of the carried obstacle**: 0 for none, or ASCII 110/101/115/119 for n/e/s/w | `We` sets it and flags the neighbouring tile in that direction; deactivation zeroes it. Captured: 0 in all 6,358 entries[^fields] |
| `Y` | **carrying an obstacle** | set from byte 11 of the `G` move message (`1 === a[11]`). Captured: false in all 6,358 entries[^fields] |

The captures confirm the rectangle, the drawn flag, the animation flag and the
persistent id. No captured tank was carrying, dead, or caught between a
facing change and its redraw. So `W`, `Y`, `m` and the corpse values 32/33 of
`direction` were never seen set, and those four readings rest on the client
code alone.[^fields]

The wire-score path (0x3E) is closed: the practice server never sends it.[^3]

[^1]: run 20260611-004505 — full registry field verification; damage tier matched 5/5 transitions; viewport position matched motion
[^2]: run 20260611-004505 — panel rank_points exact match; persistence across death verified on purple-3
[^3]: run 20260611-110445 + 013801 + 003415 — P/U, l, practice-bot theories tested and retracted
[^fields]: Decoded 2026-09-29 for board task 8b8e5725 from `tpclient.js` (the client build saved 2026-09-02; beautified, its 7,664 lines are identical to the 2026-06-19 copy `tpclient.pretty.js`). Functions: `Xc` (constructor), `Hc`/`Ic`/`Jc` (rectangle), `Xc.prototype.ra` (erase), `Xc.prototype.sa` (draw at 24(j-1)-2, 16(i-1)-4), `$e`/`ef` (sprite helpers), `qd` (redraw pass), `Re.start`/`Re.Ma` (movement animation), `Lg.h`/`Lg.prototype.h` (`V.G`), `Pg.prototype.h` (`V.A`, deactivation), `We` (carry), `ed`/`Ff` (decorations), `zg` (profile link). Measured on every `world_collections["P.j"]` entry of `runs/bot/enemy_teleport_probe.json` (40 snapshots, 2026-06-13), `runs/probe/burst-probe2-20260826.json` (40, 2026-08-26) and `runs/probe/coast_test.movement_probe.json` (6, 2026-07-26): 6,358 entries across 112 tank ids.
[^leaderboard]: Measured 2026-09-01 from `runs/bot/*/*.events.jsonl` `session_account_stats` records across seven instances. The identical-state pair is artax recruit 0 kills / 0 promo → 28946 (20:30:33) and arterial recruit 0 kills / 0 promo → 28952 (20:30:50), both 2026-08-28. The worsening-with-kills pair is arterial sergeant 148 kills → 12055 (2026-08-28) and 149 kills → 12060 (2026-09-01). Per-tank spread on one account, same day: artax private 1933 kills → 18, artax captain 691 kills → 4562, artax recruit 0 kills → 28946. Supersedes the 2026-08-05 countdown reading this page carried; the operator named it a leaderboard the same day the measurement was taken. No code reads the registry's `s` (re-read 2026-09-29): `apply_tank_observation` (`src/tankpit_bot/state/tank_mutations.py:29`) takes the wire rank, not `s`, and the leaderboard value is parsed from the stats panel at `src/tankpit_bot/diagnostics/account_stats.py:152`, then stored in `SelfAccountDict.leaderboard_position` (`src/tankpit_bot/state/types/self_account.py:25`) and in the per-colour record (`src/tankpit_bot/bot/tank_registry.py:109`). Field renamed `rank_number` → `leaderboard_position` across the code the same day.
