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
  "src/tankpit_bot/bot/ai/block_harvest.py": "c57b3272341a541aeffb0441d5156febf3e8ca94"
  "src/tankpit_bot/bot/ai/quad_sweep.py": "b94b237d3e51fbb291d87e84e8740aae9da2997f"
  "src/tankpit_bot/bot/ai/combat_break.py": "0c1454184769c23e315e87c9d68b249c55b84adb"
  "src/tankpit_bot/bot/ai/mine_pin.py": "73685b65d26e43cda9f9ef0f308c66060bb3e3ae"
  "src/tankpit_bot/bot/ai/scope_scout.py": "ac49a8329539e1f4b27b896bce05a5423ab3191d"
  "src/tankpit_bot/state/scan_coverage.py": "7255804c3e14c0c869a0f8fe85691b779f9ade0c"
  "src/tankpit_bot/sniffer/world_service.py": "4814a16be10db6906ff175dcda07ecd8eca2eee7"
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
> denial-memory fix (`648fa151`) are NOT in it. Blob pins above are
> `fa1c1ae7`'s, not HEAD's.

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
| 6 | Break floor absolute, not rank-relative | **open** |
| 7 | Mine pin single-slot latch | **open** |
| 8 | Ferry scout has no negative memory (1/31) | **open** — NOT fixed by the settled law (an earlier board note overclaimed this): the scout's precheck compares against the current window only, and per-goal look history is a separate design |
| 9 | Three dead forked constants in `world_service.py` | **open** — deletion |
| 10 | Six undocumented `*_MS` constants | **open** |

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
