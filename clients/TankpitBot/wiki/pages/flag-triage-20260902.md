---
title: Flag Triage — bot-20260901-210631 (Practice, 8 operator flags)
tags: [bug, forage, hunt, mines, radar, architecture]
related:
  - "[[flag-triage-20260729]]"
  - "[[quad-sweep-doctrine]]"
  - "[[committed-intent]]"
  - "[[bot-behavior-contract]]"
  - "[[game-economy]]"
  - "[[enemy-bot-behavior]]"
source_paths:
  - "src/tankpit_bot/bot/ai/block_harvest.py"
  - "src/tankpit_bot/bot/ai/quad_sweep.py"
  - "src/tankpit_bot/bot/ai/combat_break.py"
  - "src/tankpit_bot/bot/ai/mine_pin.py"
  - "src/tankpit_bot/bot/ai/scope_scout.py"
  - "src/tankpit_bot/state/scan_coverage.py"
  - "src/tankpit_bot/sniffer/world_service.py"
source_git_blobs:
  "src/tankpit_bot/bot/ai/block_harvest.py": "36583d8cfa1c9ebfa43a3ebe48b44ddd3f0caad0"
  "src/tankpit_bot/bot/ai/quad_sweep.py": "99f6254cb5dc91fa22c561996e420b4be1528373"
  "src/tankpit_bot/bot/ai/combat_break.py": "295d58704a009d1f30fa2f09f1f6d0984c2bb6a7"
  "src/tankpit_bot/bot/ai/mine_pin.py": "fd87a23c17de213fd7b7a9782dfd374226b8833b"
  "src/tankpit_bot/bot/ai/scope_scout.py": "7f8bf25f88e840e9491e9d829f2eb3dd621ba326"
  "src/tankpit_bot/state/scan_coverage.py": "143210fd78dd06544d635b734ca459b6b4de348f"
  "src/tankpit_bot/sniffer/world_service.py": "9a2ba3beff7e69e172442c19122da8a2c88d629e"
fact_checked: "2026-09-02"
confidence: high
hubs: [architecture, combat]
---

# Flag Triage — bot-20260901-210631 (Practice)

Eight operator flags raised while watching **Artax in the PRACTICE room**
(room 7, `field01.gif`) during a 4-bot fleet run. The other three bots
(arterial, despair, malignant) were in World and are not implicated.

> **Release under test: `v0.1.0-fa1c1ae7`**, which was 24 commits behind
> `HEAD` when these artifacts were produced. In particular the per-room
> rank floor (`bf5ec9bd`), the container claim mutex (`b5c89f6c`) and the
> denial-memory fix (`648fa151`) are NOT in it. Blob pins above track
> HEAD as fixes land (originally they were `fa1c1ae7`'s); the release
> tree named below is where the as-flagged sources live.

Evidence: `runs/bot/artax/latest.events.jsonl` inside the release tree
(`tankpit-releases/v0.1.0-fa1c1ae7/clients/TankpitBot/`), ~46 min,
1,174 planner decisions, 16 kills.

Four root causes. Two of them are the same defect the wiki has already
recorded once, recurring one layer up.

## Root cause 1 — the collect cascade livelocks (flags 1, 4, 6)

**Measured: 21:10:12 → 21:19:17, nine minutes five seconds, 273 viewport
shifts, tank frozen at (18,121), zero progress — 20% of the session.**

Two subsystems alternated every 2 s:

```
quad sweep shift    dir=7 toward window (3,106)      → window origin (18,106)
harvest frame shift dir=3 toward fuel at (22,131)    → window origin (18,121)
   ... 136 pairs ...
```

The engine is not the window. It is the **resource lock**:

1. `plan_block_harvest_leg` latches the fuel lock on (22,131) and frame-shifts.
2. Next tick the lock-continuation reports `locked fuel target at (22,131)
   not executable this tick - holding plan` and yields no command. Harvest
   then declines *by design* — `if resource_target_kind != "": return None`,
   a held lock owns the pursuit ([[committed-intent]]).
3. Nothing else serves, so `plan_quad_sweep` wins the tick. Its
   `_anchored_state` calls **`clear_resource_target(base_state)`** — it
   deletes the lock — and shifts the window away.
4. Free slate → harvest re-latches the same tile and shifts back. → 2.

**This defeats the fix recorded in `block_harvest.py` for the 2026-08-13
"flag-4 two-shift oscillator".** That fix cured harvest oscillating against
*itself* by introducing the lock. Nothing protects the lock from a peer,
and nothing bounds how long a lock may be held without dispatching.

Distinguishing the two:

| | v1 (fixed 2026-08-13) | v2 (this run) |
|---|---|---|
| Targets | **two** containers, opposite sides | **one**, (22,131), all 136 times |
| Alternation | harvest ↔ harvest | harvest ↔ quad_sweep |
| Cause | no committed target | committed target **erased by a peer** |

**Why the lock was never executable:** the tank had teleported into a
two-tile rock pocket — `(18,121)` and `(19,121)` open, every other
neighbour `#`. No walk route. The target was visible in-window 10 tiles
away and *teleport*-reachable, which the larder proved at 21:19:27 by
hopping onto it for 126 fuel. `plan_block_harvest_leg` gates
`walk_or_teleport` behind `frame_direction(...) is None`, so the movement
branch was unreachable for the whole nine minutes.

**How it ended:** not by resolution. `plan_forage_frontier_hop` sits above
the sweep but is gated on stale coverage; as blocks aged past
`FORAGE_COVERAGE_TTL_MS`, `stale_blocks` grew until it qualified at
21:19:21 and teleported out. **The bot escaped by waiting for its own scan
coverage to rot.**

Artax: 308 frame/sweep shifts. arterial 24, despair 26, malignant 39 — a
state entered, not a systemic rate.

### Are viewport shifts worth it?

Yes, outside the livelock — and the answer differs sharply per issuer:

| issuer | shifts | acted on |
|---|---|---|
| `harvest_frame_shift` (outside stall) | 31 | **16 (52%)** |
| `ferry_scope_scout` | 31 | **1 (3%)** |
| both, inside the stall | 274 | 1 |

Adjacent-reversal rate: **53% overall, 2% excluding the stall.** The
mechanism is sound; the livelock is what makes it look pathological. The
ferry scout is separately weak (flag 2).

## Root cause 2 — clocks standing in for facts (flags 3, 4, 8)

`scan_coverage.py` ages swept ground out after
**`FORAGE_COVERAGE_TTL_MS = 180000`** (3 min) so that "equipment that
respawns later is eventually re-discovered". The neighbouring
`HARVEST_MEMORY_TTL_MS` docstring concedes the basis: *"Container respawn
cadence is unmeasured — this bound is a working assumption, not a
wire-derived law."*

[[game-economy]] has since falsified that assumption:

- **"The 2026-07-22 'container respawn law' (~1 dot/min) is FALSIFIED"** —
  605/605 within-session atlas additions were our own exposure events.
- **"Containers GAIN fuel — discrete deposits, NOT regeneration"** —
  corr(Δv, Δt) = −0.13 over 169 events.

And [[enemy-bot-behavior]]: practice bots have *"no deliberate
fuel-seeking"*, and *"on the wire, roaming is RARE — most observed bot
time is stationary."*

**So in a Practice room with no humans, the container layer is effectively
static and swept ground cannot go stale except by our own consumption.**
The 3-minute clock re-scans a world that cannot have changed. None of the
coverage clocks are room-aware; Practice and World run identical values.

Measured consequences:

- **260 frontier-hop decisions across 65 unique blocks** — block (120,24)
  targeted 12 times.
- **49% of all radar dispatches fired within 7 tiles of an earlier radar
  this session.**
- **143 pickup → teleport → pickup sequences where the teleport never left
  the 16×16 viewport** (operator flag 8), several returning to the tile
  they started on:

```
21:07:30  pickup(120,130) → TELEPORT land(125,134) d=5 → pickup(120,130)
21:08:06  pickup(97,147)  → TELEPORT land(104,152) d=7 → pickup(97,147)
21:23:25  pickup(115,22)  → TELEPORT land(120,24)  d=5 → pickup(115,22)
21:09:32  pickup(32,159)  → TELEPORT land(32,159)  d=0 → pickup(32,165)
```

This is the supply drain, and the mechanism is one step removed from where
it looks: **`scan_on_landing` is 167 of 179 radar dispatches (93%)** — every
teleport landing fires a radar — so radar burn is a function of *teleport
count*, and the unnecessary teleports come from re-visiting ground the
coverage clock forgot. The quad sweep, often suspected, spent only **8
radars (4%)**.

This is the same failure `flag-triage-20260729` F2 recorded ("re-hopped
picked-clean viewports the moment the 180 s coverage expired, 63%
zero-yield hops"). The patch then was a longer clock
(`HARVEST_MEMORY_TTL_MS = 600000`). Still a clock, still finite.

## Root cause 3 — absolute fuel thresholds on a rank-variable capacity (flag 1)

`combat_break.py` projects:

```
projected     = fuel − hits_to_kill × (20 + incoming_rate)
escape_floor  = fuel_low_threshold + hunt_min_fuel + 2 × incoming_rate
break if projected < escape_floor
```

`hits_to_kill` and `fuel_capacity` are properly derived. The floor is
not: `fuel_low_threshold = 200` and `hunt_min_fuel = 100` are flat
constants, and `make_default_ai_config`'s own docstring says they are
*"suitable for lieutenant rank"* (capacity 1400). **Artax is a private —
capacity 1100.**

Solving for a full-health same-rank target (`hits_to_kill` = 13):

> the break is unavoidable at **every** fuel level once
> `560 + 15 × rate > 1100`, i.e. **rate > 36**

`rate` = incoming fuel ÷ 5 ticks at 45 fuel/hit, so **four confirmed hits
in ten seconds makes a healthy same-rank enemy permanently unkillable** —
reachable by one attacker, trivial for a pair. Live proof at a literally
full tank:

```
21:43:44  fuel=1100 (cap)  rate=54  htk=10  → projected 360 < floor 408  → BREAK
```

The 16 kills the session did land were all against already-damaged targets
(`hits_to_kill` ≤ 10) at rate ≤ 45.

**Flag 1 in full:** 21:42:20 → 21:44:19, five teleports between two red
pairs at (209,57) and (198,42), four engagements, zero kills. On break,
`release_at >= capacity` routes to `block_combat_target_and_replan`, whose
TTL is `kill_cooldown_ms` = **30 s** — so the block on pair A expired just
as pair B broke, and the shuttle period matched the TTL.

## Root cause 4 — single-slot latches (mine flag)

`mine_pin.py` documents *"a re-engage of the same target (resume, pursuit
return) never pays a second press."* **That is false whenever another
target intervenes**, because `mine_pin_target_id` is one scalar. Flag 1's
shuttle is exactly A→B→A→B:

```
21:42:23  pin red-1 @(210,57)
21:43:06  pin red-5 @(199,42)     ← latch overwritten
21:43:34  pin red-1 @(210,57)     ← same tile re-mined
21:44:04  pin red-5 @(199,42)     ← same tile re-mined
```

Four presses, two unique tiles, identical 3×3 patterns. 20 fuel and two
combat ticks bought nothing.

## The clock inventory

Auditing all time constants found **15 across 12 modules**, in three
populations that want different treatment:

**Dead forks — `world_service.py` declares three constants nothing reads**
(the live ones are in the split-out modules; a module split forked rather
than lifted, and the originals were never deleted):

```
_FAILED_MOVE_TTL_MS = 30000            → real: world_service_movement.py:19
_FAILED_SCAN_VIEWPORT_TTL_MS = 30000   → real: world_service_movement.py:21
_RADAR_CACHE_REFRESH_WINDOW_MS = 2000  → real: world_service_radar.py:17
```

**Wire-derived laws** (describe the server; belong beside their physics):
`REROUTE_TTL_MS` 12_920, `_PENDING_INCOMING_TTL_MS` 4000,
`INCOMING_RATE_WINDOW_MS` 10_000, `WIRE_PRESENCE_TTL_MS` 7000,
`POSITION_FRESHNESS_TTL_MS` 7000, `PURSUIT_TRACE_TTL_MS` 12_000,
`VIEWPORT_PRESENCE_TTL_MS` 5000. **Six of seventeen carry no docstring at
all**, making their derivation unauditable.

**Policy knobs frozen as literals** — `FORAGE_COVERAGE_TTL_MS` (no
docstring), `HARVEST_MEMORY_TTL_MS`, `FRONTIER_VISIT_TTL_MS`,
`SCOPE_SCOUT_COOLDOWN_MS`. These are indistinguishable in kind from
`kill_cooldown_ms`, `scan_cooldown_ms`, `map_open_cooldown_ms` and
`map_intel_horizon_ms`, which **already live in `AIConfigDict`** with
encode/decode, `require_int` validation and env override. Half the
population never moved into the home built for it.

For the record, since the operator asked: **no 30-minute clock exists.**
The world-map read (`map_intel_horizon_ms`) is **12 s**; the longest clock
in the bot is the 10-minute harvest veto.

## Fix status

| # | Finding | Status |
|---|---|---|
| 1 | Cascade livelock v2 (lock cleared by peer) | **fixed 2026-09-02** — quad sweep lock-gated and its raw clear deleted; forage coverage decisions preserve held locks; the search hop releases enumerated (`relocated`); the raw `clear_resource_target` is guard-restricted to `intent.py` (`restricted-symbols`); and `RESOURCE_LOCK_HOLD_BOUND_TICKS` releases any future hold-forever shape as `progress_stalled` |
| 2 | `walk_or_teleport` unreachable behind `frame_direction` | **fixed 2026-09-02** (as its deeper statement) — the walk-territory RESPONSIBILITY GAP: larder and equipment-hop deferrals now ask the pickup dispatch's own reachability predicate, so walk-blocked near stock is teleport fair game; the exact pocket shape resolves in one tick (pinned in `tests/bot/ai/test_collect_pocket_serving.py`) |
| 3 | Coverage staleness is clock-based, not event-based | **fixed 2026-09-03** — the SETTLED-KNOWLEDGE LAW: a scan stamp is valid while recent OR while it postdates the last FOREIGN HUMAN sighting (`state/knowledge_floors.py`, `ws.knowledge_floor_ms`, watermark swept from the tank registry with self + fleet siblings excluded via the merge's new `fleet_sibling_tank_ids`). No foreign human ever seen → knowledge is permanent; one present → exactly the old TTL; one departed → pre-departure scans age out once. Fact-based, room-agnostic, zero config plumbing |
| 4 | Landing radar fires on already-covered ground | **fixed 2026-09-03** — structural: the landing radar was already gated by `radar_spend_worthwhile`, whose uncovered-count now reads the settled floor, so landings on ever-scanned settled ground skip the spend. Combat-landing scans deliberately untouched (they reveal MINES, which are dynamic) |
| 5 | Intra-viewport teleports (143) | **fixed 2026-09-03** (as row 3's symptom) — lane attribution showed 139 of the measured teleports were `forage_frontier_hop` chasing clock-rotted blocks; with block coverage and visit tombstones on the settled floor the churn source is gone. The hop lanes' own contribution (~14) was already closed by the 2026-09-02 walk-territory law |
| 6 | Break floor absolute, not rank-relative | **fixed 2026-09-03** — the three fuel reserves are now the RANK-4 REFERENCE tuning, read rank-scaled through `DecideCtx.fuel_low_floor` / `hunt_reserve_floor` / `engagement_budget` (`physics.capacity.rank_scaled_reserve`, integer-exact at lieutenant, claim-bound below). The measured 21:43:44 full-tank break FLIPS: floor 408→343 vs projection 360, the private holds and finishes; the identical fight at the reference rank still breaks, exactly as tuned. RESIDUAL, now row 11 |
| 7 | Mine pin single-slot latch | **fixed 2026-09-03** — `mine_pin_target_id` scalar replaced by the per-target `mine_pin_presses` map ({str(id): "x,y" placer tile}), the same scalar→map cure `greeted_tank_ids` got on 2026-07-31 for the identical ping-pong shape. The A→B→A→B shuttle now buys exactly one press per target, and the recorded placer tile skips any press that would re-lay an identical 3×3 from already-pressed ground — the measured incident's two re-mined tiles are barred by BOTH rules. Regression pinned in `tests/bot/ai/test_mine_pin.py::test_an_intervening_target_does_not_rearm_the_press` |
| 8 | Ferry scout has no negative memory (1/31) | **fixed 2026-09-03** — the pan now IS the negative belief: every pan records its goal in `scope_scout_looks` ({"x,y": timestamp}), and a goal whose look postdates the settled-knowledge floor (`DecideCtx.scout_floor_ms`, TTL `FERRY_LOOK_TTL_MS` = 30 s under human presence, permanent settled) never draws a second pan — ferries move only when ridden, so a no-ferry look is a fact until a foreign human appears. `last_scope_scout_ms` stays as the cross-goal rate limit. The 2026-09-03 validation-run shape (pickup → tiny NE shift → teleport away) is this fix's live signature: that pan can now happen at most once per goal per settlement epoch |
| 9 | Three dead forked constants in `world_service.py` | **fixed 2026-09-03** — deleted; grep confirms the only `_FAILED_MOVE_TTL_MS` / `_FAILED_SCAN_VIEWPORT_TTL_MS` / `_RADAR_CACHE_REFRESH_WINDOW_MS` symbols left are the live ones in `world_service_movement.py` / `world_service_radar.py` |
| 10 | Six undocumented `*_MS` constants | **fixed 2026-09-03** — three were row 9's dead forks (deleted); the three live bare ones (`_FAILED_MOVE_TTL_MS`, `_FAILED_SCAN_VIEWPORT_TTL_MS`, `_RADAR_CACHE_REFRESH_WINDOW_MS`) now carry derivation docstrings tying the 30 s family to the ~2 s replan cycle with early knowledge-based release, and the 2 s pairing window to its single-consume lifetime. Every `*_MS` constant in `src/` is now documented |
| 11 | Incoming rate is not attributed per attacker | **fixed 2026-09-03** — PRESENCE-SCOPED rate: a shooter's windowed hits count only while they can still fire. Three classes: registry-deactivated excluded (the 2026-07-31 arena-soak law, kept); registry-alive but wire-silent past `WIRE_PRESENCE_TTL_MS` excluded (NEW — the disengaged pair-mate no longer prices the duel for the window's tail; a shooter actively hitting us refreshes their own presence through their 0x53s); registry-unknown always counts (a gap can never under-report). The book stays policy-free (`excluded_shooter_ids`); the service owns the law; the projection formula is untouched. Pinned in `tests/sniffer/test_incoming_rate_presence.py` including the flip-back guarantee |
| 12 | Harvest frame-shift toward stock the hop lane teleports onto (2026-09-03 validation run, 20:03:50) | **refuted 2026-09-03** — measured across the whole validation run: 11 `harvest_frame_shift` decisions, 8 led directly to a served pickup at the shifted-toward target, 1 released for a `superior_candidate`, 1 held-then-served, and exactly 1 died to `landing_scan_reset` — an IN-FLIGHT frontier teleport (dispatched 20:03:56, landed ~20:04:14) resetting the plan mid-frame, a landing race, not a walk-lane failure. The hypothesized shape cannot occur: `_harvest_candidates` already gates every candidate on `is_collection_reachable_within_bounds` over the leg corridor. Residual: the in-flight-landing race cost one tick in 16 minutes; an in-flight-teleport planning interlock is not worth its complexity at that rate |
| 13 | Browser-teardown wedge: `make run` exits before its scorecard (operator flag off the 2026-09-03 validation run) | **fixed 2026-09-03** — teardown escalation ladder in `browser/lifecycle.py`: close → engine kill at 15 s (driver spared to resolve the close) → forced exit at 60 s carrying the SESSION's outcome (0/75), not a blanket 75; separate CLI-only 30 s post-session exit deadline for the post-close wedge shape (run 010551); `make run` now prints the scorecard for every exit code and exits with the bot's code. Corpus: 15 hung / 251 clean archived runs, both wedge shapes represented; the 30 s watchdog sat inside the host's measured tens-of-seconds slow-teardown band |

## Machine-checked claims (row 6)

The rank-scaled reserve law, bound by the `physics_claims` guard the
same way every wire law is — the scaler is exact at the reference
lieutenant and proportional elsewhere:

```json claims
{
  "claims": [
    {
      "id": "reserve-reference-rank",
      "code": "tankpit_bot.physics.capacity:RESERVE_REFERENCE_RANK",
      "value": 4
    },
    {
      "id": "rank-scaled-reserve",
      "code": "tankpit_bot.physics.capacity:rank_scaled_reserve",
      "formula": "reference * fuel_capacity(rank) // fuel_capacity(4)",
      "probes": [
        {"args": [200, 4], "expect": 200},
        {"args": [100, 4], "expect": 100},
        {"args": [450, 4], "expect": 450},
        {"args": [200, 1], "expect": 157},
        {"args": [100, 1], "expect": 78},
        {"args": [450, 1], "expect": 353},
        {"args": [200, 0], "expect": 142},
        {"args": [450, 8], "expect": 578}
      ]
    }
  ]
}
```

## Live validation (2026-09-03, rows 1-6)

One bounded 16-minute Practice session from HEAD (`9993aa4e`), same
account, same room class as the flagged run, measured with one
instrument over both artifacts (`validate_run.py`, proven against the
baseline first). The room stayed SETTLED throughout — zero foreign
humans observed — so the settled-knowledge law ran in its
permanent-knowledge regime the whole session.

| instrument | baseline (74 min) | validation (16 min) |
|---|---|---|
| radars within 7 tiles of an earlier radar | 56% (132/235) | **8% (4/45)** |
| frontier journeys → unique blocks | ~260 → 65, one block targeted 12× | **28 → 28, ZERO re-targets**[^v1] |
| intra-viewport pickup→teleport→pickup | 173 | 23[^v2] |
| adjacent scope-shift reversals | 204 | 8 |
| livelock signature | one 9-minute stall | none |
| `progress_stalled` releases | n/a | **0** |
| engagement breaks at ≥1000 fuel | 8 | **0** (11 breaks, all mid-fuel) |
| kills per minute | 0.28 | **0.56** |

The kill rate DOUBLED while radar spend per minute fell — the ticks
the clock churn used to eat went to fighting (`shoot_target` is the
top decision reason, 106 of 420; in the baseline it was
`forage_frontier_hop` at 435 of 1,954).

One teardown observation, pre-existing and not from these fixes: the
session ended cleanly (`quit_game`, artifacts saved) and then browser
teardown exceeded its 30 s watchdog, forcing a nonzero exit that
stops `make run` before its scorecard step. Flagged on the board;
the artifacts are complete regardless.

[^v1]: Journey-level, by TARGET block with the map-open-defer
      duplicate collapsed. Nine blocks were LANDED in twice — landing
      displacement across block edges, not re-targets.
[^v2]: By lane: 11 `forage_frontier_hop` (adjacent-block exploration
      geometry), 7 `equipment_hop` (walk-blocked near stock served by
      design under the 2026-09-02 walk-territory law), 3 combat
      approaches, 2 fuel. The pathological class — clock-rot
      re-visits — is gone.

## Live validation (2026-09-03, rows 7-8-11-13)

One bounded 7-minute Practice session from HEAD (`27fa6225`), same
account and room class, same instrument. Every fix under test showed
its live signature:

| fix | live signature |
|---|---|
| Row 7 (mine pin map) | 1 press, 1 target, 1 tile — no re-press |
| Row 8 (scout look memory) | **9 pans, 9 UNIQUE goals, zero repeats** (baseline: 31 pans re-hammering the same water) |
| Row 11 (presence-scoped rate) | 1 break, mid-fuel, priced by a live measured rate (27/tick over 3 hits, projection 285 vs floor 289) — a marginal call made on real numbers |
| Row 13 (teardown ladder) | `Teardown: browser closed` in-line, scorecard printed, **`make run` exit 0** — the first flagged run shape, gone |

Rows 1-6 held: radar redundancy 6% (baseline 49%), adjacent scope
reversals 4/16, `progress_stalled` 0, zero full-tank breaks, frontier
journeys 27 → 27 unique blocks with zero re-targets (the instrument's
raw "revisited" count is the map-open-defer tick pattern — the same
block on consecutive ticks of ONE journey, never a return).

## The pattern worth carrying forward

Four independent flags, one shape: **a clock or a scalar standing in for a
fact the bot could observe.** The break floor stands in for real capacity;
the block TTL for real threat state; the coverage TTL for real
consumption; the mine latch for real tile history. Each was individually
defensible when written and each fails the same way — the proxy drifts
from the fact, and no component owns the invariant that the bot must make
progress.

The 2026-08-13 oscillator fix and the 2026-07-29 F2 fix were both *the
next clock out*. The livelock and the re-scan waste are what that strategy
costs on the third iteration.
