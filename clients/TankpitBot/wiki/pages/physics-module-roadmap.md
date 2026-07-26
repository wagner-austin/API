---
title: Physics Module Roadmap — Wiki as Executable Truth
tags: [architecture, roadmap, physics, wiki, enforcement]
related:
  - "[[terrain-composition]]"
  - "[[game-economy]]"
  - "[[executor-rejection-loops]]"
  - "[[self-observing-architecture]]"
  - "[[coding-standards]]"
  - "[[movable-blocks]]"
  - "[[walk-mechanics]]"
  - "[[weapon-selection]]"
  - "[[mine-mechanics]]"
  - "[[teleport-mechanics]]"
source_paths:
  - "src/tankpit_bot/physics"
  - "src/tankpit_bot/sim"
  - "src/tankpit_bot/validate"
source_git_blobs:
  "src/tankpit_bot/physics": "9822fa76c696d3e4e0f2722bb63614d79659119f"
  "src/tankpit_bot/sim": "2896ceac9ae7b10eb87b5a02442c1e7d8df70e8d"
  "src/tankpit_bot/validate": "0f9542f68581853517832736b3a063b0bab9f8c2"
fact_checked: "2026-07-20"
confidence: high
hubs: [architecture]
---

# Physics Module Roadmap — Wiki as Executable Truth

**Status: Phase 4 COMPLETE (2026-07-22), all eight laws implemented.
Phases 1–3 and the executor track IMPLEMENTED 2026-07-20/21 (make
audit 11/11 claims, both live books at zero divergences, executor
pure dispatch). Phase 4 steps (a)–(e) IMPLEMENTED 2026-07-21/22:
encoders byte-identical on the full corpus, laws 1–8 (law 4 landed
last, with the viewport model it required), the live CDP seam, the
divergence-zero soak with a proven-teeth negative control, and the
audit cross-check — `make audit`'s validators price sim-generated
wire at real-archive exactness. Fidelity statement in the step-(e)
as-built below.**
Written for a future session (human or AI) to execute end-to-end
without access to the 2026-07-20 design conversation. Read
[[terrain-composition]] first — it is the completed seed of this plan.

## The goal in one paragraph

This project reverse-engineers a game's physics by falsification: the
wiki records the rules, the code acts on them, the wire grades them.
Today the truth lives twice (wiki prose + scattered code) and drifts
(the teleport-cost row said "unknown" for weeks while
`teleport_cost.py` had the exact validated formula). The end state:
ONE executable physics module mirroring the wiki rule-for-rule,
machine-checked bindings between wiki claims and code symbols,
re-runnable validators against the capture archive, and live runs
that count physics divergences. Drift becomes a red `make check`, not
an archaeology project.[^2]

## What already exists (do not rebuild)

- **Composed decision terrain** (`bot/ai/ferry.py`,
  `compose_decision_terrain`): the single walkability owner — static
  map + ferries + hostile mines + movable blocks. Pattern and
  invariants in [[terrain-composition]].
- **A complete economy** ([[game-economy]], "What's still open:
  Nothing"): walk=1/tile, single=6 free, dual/missile/homing=10 (+1
  round per landed shot), radar=10, mine press=10 flat, block ops
  free, teleport=floor(6×euclid to ACTUAL landing), enemy hits 45/90,
  capacity=1000+100×rank.
- **Scattered code homes for those facts**:
  `bot/ai/teleport_cost.py::compute_teleport_fuel_cost` (the only
  clean physics module today), capacity/rank math, walk/radar costs
  implicit in planner arithmetic, surface rules in `ferry.py`,
  reroute TTL only in the wiki (since corpus-swept 2026-07-22 to
  12 920 ms, boundary [12.91, 12.93] s).
- **Ledger** (`ledger/`): typed per-action outcomes with
  decision↔outcome correlation.
- **Validation precedent**: the teleport validator (2,538 hops,
  wiki log entry 2026-07-20) and the firing-cost isolation sweep
  (204 captures) were both one-off scratchpad scripts. Their METHOD
  is the template for Phase 2; the scripts themselves are gone —
  rewrite from the log entries' method descriptions.
- **Capture archive**: `runs/bot/*.capture_session.json` (200+) and
  `runs/sniff/*.capture_session.json`, decodable via
  `protocol.decode_message` + `sniffer.xor.build_global_xor_table`
  (frame split: 2-byte LE length prefix; see the decode recipes in
  log entries 2026-07-20).

## Phase 1 — `physics/` module + claim binding (do this first)

**Deliverable**: `src/tankpit_bot/physics/` — pure functions and
constants, no I/O, no state. Every symbol carries a wiki anchor in
its docstring (`Wiki: [[game-economy]]#<claim-id>`). Selectors and
planners import physics; game knowledge becomes unwritable elsewhere
(guard rule).

Contents (consolidate, don't invent):
- `costs.py`: WALK_COST_PER_TILE=1, SINGLE_SHOT_COST=6,
  DUAL_SHOT_COST=10, MISSILE_SHOT_COST=10, HOMING_SHOT_COST=10,
  RADAR_COST=10, MINE_PRESS_COST=10, BLOCK_OP_COST=0,
  `teleport_cost(dx, dy)` (move `compute_teleport_fuel_cost` here;
  note: charged on ACTUAL landing, planner estimates on target).
- `damage.py`: SINGLE_HIT_VICTIM_COST=45, DUAL_HIT_VICTIM_COST=90,
  MINE_DETONATION_COST=45.
- `capacity.py`: `fuel_capacity(rank) = 1000 + 100 * rank`; deposit
  floor 100.
- `combat.py`: reroute TTL constant (12_920 ms, corpus-swept
  boundary [12.91, 12.93] s — 2026-07-22), consumption-equals-hit
  rule helpers.
- Terrain vocabulary stays in `state/types/constants.py` (already
  truth-named after 2026-07-20); physics may re-export.[^2]

**Claim binding**: each physics wiki page (game-economy,
teleport-mechanics, ferry-mechanics, weapon-selection,
movable-blocks) gets a fenced machine-readable claim block:[^2]
```yaml
claims:
  - id: teleport-cost
    value: "floor(6 * sqrt(dx^2 + dy^2))"
    code: "tankpit_bot.physics.costs:teleport_cost"
  - id: walk-cost
    value: 1
    code: "tankpit_bot.physics.costs:WALK_COST_PER_TILE"
```
A new guard stage (`make check`) parses claim blocks, imports each
`code` address, and verifies computationally (constants compared
directly; formulas evaluated on a probe grid). Reverse direction:
every public symbol in `physics/` must be referenced by exactly one
claim. Drift in either direction = red gate.[^2]

**Definition of done**: all economy/damage/capacity constants live in
`physics/`, zero magic game-numbers left in `bot/ai/` (guard-greppable),
claim checker wired into `make check`, 100 % coverage held.[^2]

### Phase 1 as-built (2026-07-20)

Implemented exactly as designed with these recorded deviations:[^2]

- **Claim blocks are fenced JSON, not YAML** (` ```json claims ` …
  ` ``` ` in the page source). No YAML parser exists in the
  dependency tree; the repo's established strict-typing idiom is
  `platform_core.json_utils` (`load_json_str` + isinstance narrowing
  over `JSONValue`), which the checker uses with zero new
  dependencies. Formula claims carry an explicit `probes` grid
  (`{"args": [...], "expect": N}`) instead of a formula evaluator —
  computational, no expression parsing.
- **Checker**: `scripts/physics_claims.py::run_physics_claim_rules`,
  wired into `scripts/guard.py::main` beside `contract_rules` (runs
  in the guard stage of `make lint` / `make check`). Both directions
  enforced: every claim verified against its imported symbol; every
  `__all__` symbol of every `physics/` submodule must be bound by
  exactly one claim. Dynamic imports use the annotate-at-assignment
  Protocol pattern (`fn: _ProbeFn = getattr(module, name)`) — no
  `Any`, no `object` annotations, no casts.
- **Module homes**: `physics/costs.py` (8 constants +
  `teleport_cost`, moved from the deleted `bot/ai/teleport_cost.py`
  and renamed), `physics/damage.py` (45/90/45 — first time these
  existed as code), `physics/capacity.py` (absorbed
  `state/rank_formulas.py`, deleted; + `DEPOSIT_FLOOR`),
  `physics/combat.py` (`REROUTE_TTL_MS` estimate). The
  consumption-equals-hit rule stays prose (it is a bookkeeping
  identity, not a constant). `combat_radar_min` was POLICY, not
  physics — it moved to its only consumer,
  `bot/ai/mode_controller.py`, with a docstring saying exactly that.
- **Claim locations**: [[game-economy]] (15 claims: 8 costs +
  teleport probes, 3 damage, capacity/inventory probes,
  deposit-floor), [[radar-mechanics]] (`free-radar-radius` probed at
  the measured step boundaries), [[shoot-event-format]]
  (`reroute-ttl-ms`).
- **Verified**: `make check` green — 4,551 tests, 100 % statement +
  branch coverage, guard clean; DoD grep confirms no game-number
  formula or damage constant survives outside `physics/`
  (`tests/scripts/test_physics_claims.py::test_real_repo_binding_is_green`
  pins the live binding forever). Behavior-neutrality soak
  bot-20260720-233953: 314 ticks, 2 kills, 25/26 hits, 32/32
  teleports landed, 0 rejections/discards/error-6 — only analyzer
  flag is a map open cut off by the session clock.

## Phase 2 — validators vs the archive (`make audit`) — IMPLEMENTED 2026-07-21

Promote the one-off analysis method to `tools/validate/` scripts, one
per claim family, each re-deriving its claim from
`runs/**/capture_session.json`:
- `validate_teleport_cost.py` (method: pair every teleport dispatch
  with its wire fuel delta; pre-hop position fix, contamination
  exclusion — full method in wiki log 2026-07-20 "teleport cost
  systematically validated")
- `validate_firing_costs.py` (method: fuel-sync windows containing
  only own 0x53 echoes of one weapon — log entry "per-weapon firing
  costs closed")
- `validate_walk_cost.py`, `validate_capacity.py`,
  `validate_hit_damage.py` (same window technique)[^2]

Claim blocks gain `validator:` and `evidence:` fields. `make audit`
runs every validator, reports per-claim sample counts, and rewrites
each page's `fact_checked:` from validator output — computed, not
hand-typed. A claim whose validator fails or whose evidence files
vanished is flagged like a stale hash (mcps wiki-check analogy).[^2]

### Phase 2 as-built (2026-07-21)

Implemented as `src/tankpit_bot/validate/` (not `tools/validate/` —
`src` is where mypy/ruff/coverage/guard already apply; follows the
`diagnostics/` precedent, CLI `tankpit-audit`, `make audit` target):[^2]

- **`wire_timeline.py`** — one typed extraction per capture session
  using the production decode recipe (frame split → XOR →
  `protocol.decode_message`): self identity (first 0x21), absolute
  fuel readings (own long-form 0x2E sync + tunneled 0x44 FuelGain),
  own/enemy 0x53 echoes, own 0x47 Manhattan tile counts, sent client
  commands (`!`-prefix, command byte + coords), 0x43 pickups, 0x49
  count snapshots.
- **`archive.py`** — the isolation-window method as re-runnable code:
  `build_fuel_windows` slices between consecutive fuel readings
  (end-inclusive by timestamp — the wire delivers cause and closing
  sync in the same millisecond; 738/738 lone-hit windows pin it);
  validators for the four firing costs (one-shot clean windows;
  homing −5 split counted as sample, not mismatch), WALK EPISODES
  (as-designed rationale said "a walk drains across many windows" —
  SUPERSEDED 2026-07-21: movement is instant and the full cost lands
  in the echo window ([[walk-mechanics]]); the episode structure is
  retained because it also absorbs boundary-straddling debits;
  SINGLE-ECHO only, because a second 0x47 can be a route that never
  executed — the 2026-07-21 probes showed every multi-echo mismatch
  overcounts tiles, never fuel, and zero mid-path echoes), hit damage
  (lone-enemy-shot windows; zero-delta = shot at someone else, not a
  sample), and fuel capacity (every reading vs the rank bound).
- **`events_validators.py`** — the events.jsonl teleport pairing
  (dispatch → landed_x/y → belief-fuel fixes either side,
  contamination-excluded by action kind, predicted on the ACTUAL
  landing). Post-2026-06-24 runs only — the pre-fix era's bot-side
  fuel corruption is a bot artifact, not physics. Capture-based
  validators sweep ALL eras (their windows exclude pickups, where the
  old bug lived, and wire fuel values were never corrupted).
- **Verdict = exactness share, floor 0.85** (`EXACTNESS_FLOOR`):
  clean instruments measure 88–100% exact on the real archive; the
  residual is positive-signed noise (collision-truncated walks,
  unmodeled events inside windows). Real drift collapses the share
  toward zero, so the floor loses no detection power. Mine
  detonations (0x45) and deposits (0x64) are modeled — detonations
  contaminate windows, deposits are absolute fuel readings.
- **`audit.py`** — `tankpit-audit --stamp`: prints one evidence row
  per claim (samples / exact / mismatch / PASS-FAIL), exits non-zero
  on any zero-sample or mismatching claim, and rewrites
  `fact_checked:` for pages whose validated claims ALL passed —
  stamp text is computed from validator output. `STAMPED_PAGES` maps
  page → owned claim ids explicitly.
- Claims covered (9): the 4 firing costs, walk, teleport, both hit
  damages, fuel capacity. Claim blocks carry `validator:` pointers.
  Not archive-validatable (no wire evidence stream): block-op-cost,
  mine-press flat cost (manual-capture only), radar-cost (never
  isolated — radar rides pickup/scan-heavy windows), deposit-floor,
  inventory-capacity (0x52-code-7 refusals not in the timeline),
  radar radius, reroute TTL — these stay pinned by unit tests and
  the Phase 1 claim gate.
- Verification: gate green (100% stmt+branch), and `make audit`
  green against the real archive — 9/9 claims, 21,395 clean samples:
  teleport 63/63 exact, single-hit 738/738, dual-hit 6/6, capacity
  18,649/18,649 within bound (1,463 AT cap), missile 6/6, single
  242/247, dual 863/932, homing 487/522, walk 204/232.
  [[game-economy]]'s fact_checked stamp is now machine-written.

## Phase 3 — live divergence (the wire grades every run)

The bot predicts each fuel/ammo delta from `physics/` before the wire
confirms it; the session scorecard gains `physics divergences: N`.
Nonzero → `make analyze` extracts each divergence as a candidate wiki
claim (new fact or bug). Includes double-entry fuel/ammo bookkeeping
in the ledger: every delta must be explained by a known action or
flagged. (This is what would have auto-decomposed the −45/−10
combat-tick mystery months earlier.) Touches the tick loop and
ledger; do it last, after Phases 1–2 make predictions cheap.[^2]

### Phase 3 design (locked 2026-07-21, pre-implementation)

**Instrument: an interval double-entry fuel book**
(`ledger/fuel_book.py`, pure functions over a typed book dict).
Between two absolute wire fuel readings the book accumulates ENTRIES
— each a predicted delta with a feasibility range, because live
windows contain events whose exact fuel effect is legitimately
unknowable at prediction time:[^2]

- exact debits: own shots by weapon (homing carries a −5/−10 split
  tolerance), radar 10, mine press 10, teleport floor(6*euclid) on
  the ACTUAL landing;
- ranged debits: own 0x47 walk echo = [0, path_tiles] (paths truncate
  on collision — Phase 2 finding); optional enemy hits = {0, −45} or
  {0, −90} per lone echo; own-mine detonation −45 optional;
- open credits: container pickups = [0, capacity − fuel].[^2]

**Reconciliation** happens at the single wire choke point
`update_world_state_from_fuel_total` (all of 0x2E sync / 0x44 gain /
0x64 deposit-total flow through it): measured residual must fall in
the entries' feasible interval; outside = a `physics_divergence`
diagnostic event carrying the residual and the window's entry list.
Strict windows (exact entries only) demand equality — the live
equivalent of the audit's clean windows.[^2]

**Entry sources**: wire-echo entries recorded in the
world_state_dispatch handlers (0x53 own/enemy, self 0x47, 0x43
pickup, 0x45 detonation); dispatch-side entries (radar, mine press,
teleport-landed with pre-position) recorded from the executor/ledger
outcome path. Ammo book v1: 0x49 snapshots must never exceed the
previous snapshot plus pickups nor undercut it by more than shots
fired — the consumption-equals-hit cross-check, live.[^2]

**Surfacing**: scorecard line `physics divergences: N` (counted from
events.jsonl by session_scorecard); `make analyze` lists each
divergence with its window as a candidate wiki claim. Verification:
gate + a soak whose only behavioral change is the new scorecard line;
divergences on a healthy run should be ~0.[^2]

### Phase 3 as-built (2026-07-21, core loop)

Implemented as designed with these notes:[^2]

- **`ledger/fuel_book.py`** — the interval double-entry book, pure
  functions over a typed dict, contract-enforced mutations
  (`FuelEntryContract`, `FuelReadingContract`), owned per-session by
  `WorldService.fuel_book`.
- **Entry sources wired**: own 0x53 echoes debit their physics cost
  exactly (homing ceiling −5 + a `homing_carry` [−5, 0] seeded into
  the next window); enemy 0x53 echoes are optional debits [−90, 0];
  self 0x47 walks are ranged [−path_tiles, 0]; 0x43 pickups are open
  credits [0, fuel_capacity(rank)]; 0x45 detonations optional [−45,
  0]; executor-side radar dispatch [−10, −10] and teleport dispatch
  [−(cost+18), −max(cost−18, 0)] (displacement drift bound).
- **Reconciliation** at `update_world_state_from_fuel_total` — the
  single choke point all three wire fuel channels flow through; an
  out-of-interval residual emits a `physics_divergence` DIAGNOSTIC
  with residual, feasible interval, entry kinds, and fact source.
- **Surfaced**: scorecard line `physics divergences: N` +
  `collect_scorecard_issues` entry pointing analysts at the events
  query; issue_report codecs round-trip the new field.
- **Live calibration (four soaks, 2026-07-21)**: 71 → 12 → 18 → 1
  divergences. Each round's residual signatures named the next fix:
  (1) per-sync windows mis-attribute because charges lag their cause
  echoes → the book judges at QUIET boundaries (zero-delta reading,
  no new entries; forced at 50 readings) — the live twin of the
  audit's episode method; (2) 0x44/0x64 fuel totals announce their
  own delta → credited exactly at the choke point; (3) the pickup
  credit only fired on emptied containers → moved above the
  partial-pickup branch; (4) teleport drift bound widened to ±6
  tiles. Final soak: heavy combat (67 hits), ONE divergence — a
  double radar debit in a single-entry block (charge-latency
  straggler), a legible candidate residual, not noise.
- **Ammo book (2026-07-21)**: `ledger/ammo_book.py` enforces
  consumption-equals-hit live — between 0x49 snapshots, weapon slots
  may fall by at most the own-shot echoes counted (misses consume
  nothing), the radar slot by at most the scans dispatched, and no
  slot may rise without a 0x67 gain; armor falls freely (incoming
  hits are unpredictable) but rises only with a gain. Infeasible
  deltas emit `physics_divergence` with `divergence_channel="ammo"`.
  Verification soak with BOTH books live: **zero divergences** across
  a full combat run (38 hits, 27 radars).
- **Deferred within Phase 3**: divergence-to-candidate-claim
  extraction in `make analyze` beyond the issue line — the next
  increment on this foundation.[^2]


## Phase 4 — the simulator (wiki-derived fake server) — SPEC 2026-07-21, not started

**Goal**: the production bot runs UNCHANGED against a simulated server
whose every rule cites a wiki claim. Planner changes get graded on
thousands of simulated sessions before one live run; any live
sim-vs-wire disagreement is automatically a candidate claim (Phase 3
divergence machinery grades the sim with the same instruments it
grades the real server).[^2]

### Architecture decision: wire-level fake server

The sim speaks REAL BYTES — framing, XOR, 0x2E tunneling — behind the
seams the codebase already has:[^2]

- **Inbound to bot**: the received-message buffer
  (`Bot._on_message_captured`, `CapturedMessage` dicts) — the same
  path `replay.engine` already drives headless.
- **Outbound from bot**: `command_sender.send_command_bytes`'s
  `send_ws_bytes` callback + `_test_hooks.CDPSessionProtocol` — the
  protocols tests already fake. A `SimTransport` implements them:
  commands leave the bot as production-encoded frames; the sim decodes
  them with the PRODUCTION decode path (frame split → XOR →
  `decode_message`), advances the world one law at a time, and answers
  with encoded server frames.[^2]

Why wire-level and not a state-level stub: the bot's decoders,
dispatch, world-state, books, and planner all run untouched, so a sim
bug and a decoder bug stay distinguishable; and sim sessions are
CAPTURES in the standard format — `make audit` and the replay engine
work on them for free.[^2]

### New code

- **`protocol` encoders** (the one real gap): we decode every server
  message but encode almost none. Each `decode_*` for a message the
  world-state consumes gains an `encode_*` sibling IN THE SAME FILE
  (layout knowledge stays in one place; the coding standard already
  wants encode/decode pairs). Inventory = exactly what
  `world_state_dispatch` + the tick loop consume: 0x21 identity, 0x2E
  short/long sync and its tunneled subtypes (0x53 shoot, 0x47
  movement, 0x43 pickup, 0x44 fuel gain, 0x64 deposit, 0x45/0x4B
  mines, 0x49 counts, 0x67 equipment gain, 0x58 remove, 0x41
  deactivation, 0x52 errors), 0x3E full status, 0x4C map data,
  0x46/0x4F radar, viewport/terrain frames, cache/overlay patches.
- **`src/tankpit_bot/sim/`** (production-typed, no mocks):
  - `world.py` — `SimWorldDict`: terrain (`TerrainMap`), containers,
    mines, tanks, tick counter. v1 seeding: from a real capture
    snapshot; generated worlds come after the spawn-distribution
    crack.
  - `server.py` — the tick processor (laws below), pure functions
    over the world dict.
  - `transport.py` — the seam adapter (`CDPSessionProtocol` +
    `send_ws_bytes` + message delivery into the buffer).
  - `policy.py` — `SimPolicy` protocol (world view → commands) for
    opponents. v1 policies: stationary dummy, scripted walker,
    capture-replay ghost. Enemy MINDS are not physics — the sim never
    hardcodes "realistic" enemies; it takes policies as plugins.[^2]

### The laws (every rule carries its wiki anchor)

1. **Global queue, 2 s tick**: commands queue and process at tick
   boundaries; wire flushes batch per tick ([[shoot-event-format]]
   queue model; log 2026-07-21 sync cadence). Within-round order was
   originally modeled as arrival order; measured 2026-07-25 as
   **ascending tank id** ([[game-rules]] §Combat rounds,
   1,820/1,825 archive bursts) and the sim now sorts the queue.
2. **Movement is instant** ([[walk-mechanics]]): deterministic
   quadrant-keyed pathfinder (vertical-first, NE quadrant
   horizontal-first; routes around terrain + enemy mines), full path
   billed 1/tile at processing, destination pickup same tick, 0x47
   echo carries the route.
3. **Shots** ([[weapon-selection]], [[shoot-event-format]],
   [[game-economy]]): resolve against the target TILE at processing
   (no range mechanic); server-side weapon selection (dual default,
   homing on same-tick mover, missile only vs obstructed ENEMY);
   terrain clips non-missile shots to the impact tile and still bills;
   damage 45/90/45/45 with armor absorbing at damage/45; victim billed
   instantly, shooter debit next tick; consumption-equals-hit.
4. **Homing reroute** with the ~12.9 s TTL after 0x58
   ([[shoot-event-format]], `REROUTE_TTL_MS`).
5. **Teleport** ([[teleport-mechanics]]): cost floor(6×euclid) on the
   ACTUAL landing; ring-1 displacement preference E→N→W; enemy-mine
   tiles block landing; landing auto-pickup.
6. **Mines** ([[mine-mechanics]]): 3×3 viewport-clipped placement,
   skips terrain/water/tanks, 1:1 enemy-mine exchange, two-packet
   cascade detonation, walking into an enemy mine −45.
7. **Fuel/equipment** ([[game-economy]], [[fuel-system]]):
   capacity 1000+100×rank, deposit floor 100, pickup volume
   semantics, duplicate pickup broadcasts.
8. **Radar/map** ([[radar-mechanics]], [[map-data-decode]]): 0x4F as
   CACHE DIFF against what the client has seen; 0x4C fuel-dot atlas +
   5-byte blips (no mines on the map).

Documented sim assumptions (revisit when measured): max single-click
walk distance (23 wire-confirmed; sim accepts any in-viewport click),
displacement south-preference/beyond-ring-1, container respawn (v1:
static seeded world, no respawn).[^2]

### Verification ladder (definition of done)

1. **Encoder round-trip**: `decode(encode(x)) == x` property tests
   PLUS corpus round-trip — every consumed server message in the
   archive re-encodes byte-identically.
2. **Divergence-zero soak**: the production bot plays a full session
   against the sim; fuel book + ammo book report **0 physics
   divergences**. This is the acceptance test — the Phase 3
   instruments cannot tell the sim from the real server.
3. **Audit cross-check**: `make audit` over sim-generated captures
   re-derives every archive-validatable claim exactly.
4. **Fidelity statement** on this page: 1:1 for every measured law;
   explicitly NOT 1:1 for spawn distributions (until cracked), enemy
   minds (by design), and the listed assumptions.[^2]

### Build order (one commit each, gate green throughout)

(a) encoders + round-trip corpus tests; (b) `sim/world.py` +
`server.py` with laws 1–3; (c) `transport.py` + bot-vs-sim smoke
session; (d) laws 4–8; (e) divergence-zero soak + audit cross-check +
fidelity statement.[^2]

### Step (a) as-built (2026-07-21): encoders — 72,916/72,916 corpus messages byte-identical

- **`protocol/encoders/` package** mirrors `decoders/`
  module-for-module (tank, movement, resources, combat, world,
  map_data, radar, session_events) — a separate package, not
  same-file placement, because `decoders/tank.py` already sits at the
  400-line ceiling. `container/encoders.py` covers the five
  container-only bodies. The radar encode trio moved from
  `decoders/radar.py` into `encoders/radar.py` so every encoder has
  one home (`tankpit_bot.protocol` still re-exports them).
- **`encoders/envelope.py`** is the keystone:
  `encode_message_payload` (top-level frame payload) and
  `encode_envelope_body` (0x2E body — subtype byte + payload for
  protocol messages, verbatim body for container messages), grouped
  if-chain dispatch mirroring `decoders/routing.py`. Match-statement
  dispatch was rejected by mypy's tagged-union narrowing; literal
  `==` if-chains narrow correctly.
- **Two decoders were provably lossy** and their TypedDicts gained
  the missing wire bits: `TankStatusDict.damage_state` (info-byte
  bits 2–3 — 223/244 corpus bodies nonzero) and `FuelGainDict.flag`
  (the raw byte behind ``is_free`` — one corpus body carries 0x2B).
  Corpus-constant bytes were NOT added as fields, just re-derived and
  documented: TankEntry's flags byte equals team (6/6), the sync
  has-fuel-bar byte is 1 (21,278/21,278), the 0x3F body is 1
  (1,166/1,166), the 0x5A no-mine overlay nibble is 8 (3,724 bodies).
- **Greedy skip-RLE encodings confirmed canonical**: the linear-cursor
  greedy emitter reproduces every 0x4C fuel-dot atlas (3,797) and
  every 0x5A viewport patch (3,724) byte-for-byte.
- **`make roundtrip`** (`tankpit-roundtrip`,
  `validate/roundtrip.py`): re-encodes every archived binary message;
  244 sessions, 72,916 messages, 28 families, **0 mismatches**; 9
  invalid frames counted, not judged. Lobby TEXT frames
  (`is_text_message`: `= + % * $ - ~ \` R`) are excluded — the
  outer `+`/`=` frames the old census counted as undecodable are
  plaintext room listings and profile rows, not binary messages.
- Gate green (4,639 tests, 100% stmt+branch); `make audit` unchanged
  at 11/11 claims.[^2]

### Step (b) as-built (2026-07-21/22): `sim/` — laws 1–3 live

New package `src/tankpit_bot/sim/`, layered and DI'd
(`TerrainMapProtocol` from `_test_hooks`, so tests drive the sim on
in-memory terrain):[^2]

- **`world.py`** — `SimWorldDict` (tanks / containers / mines / tick)
  with full encode/decode codecs and require_* validation; worlds
  seed from JSON snapshots.
- **`pathfind.py`** — the deterministic quadrant-keyed router:
  single-turn L with vertical-first legs (horizontal-first toward NE,
  the measured exception), the other L on obstruction, and a
  fixed-order BFS for forced staircases/detours; hard 256×256 map
  bounds.
- **`movement.py`** — law 2: route → relocate → bill 1/tile → resolve
  destination pickup (capacity-clamped) and enemy-mine detonation
  (−45), all in one call; `cant_go` / `insufficient_fuel` as typed
  outcomes. Enemy mines block interior routing; own-color mines are
  walkable; the destination itself may hold an enemy mine.
- **`combat.py`** — law 3: Bresenham ray clipping to the first rock
  or tank (water is not an obstruction), server-side weapon selection
  (dual default, homing for same-tick movers, missile only vs
  obstructed enemies and only when the slot is ready), damage table
  with armor absorption at damage/45, tier progression 0→3→2→1,
  deactivation at zero fuel, and the two-packet mine cascade.
- **`commands.py`** — typed decode of client `[!][type][cmd]` frames
  (move/shoot/teleport/radar/mine/map/pickups; unknowns preserved);
  `SimError` for anything outside the current build stage.
- **`server.py`** — law 1: global queue (arrival order as built at
  this step; ascending-tank-id since 2026-07-25 — see the round-order
  note below), everything processed
  and flushed per 2 s tick; shooter debits bill the NEXT tick
  (charge latency), victims instantly; emits decoded messages (0x47
  echoes, 0x2E fuel syncs, 0x43 pickups, 0x45 cascades, 0x52 command
  errors, 0x49 client snapshots, 0x41 kills) for the step-(c)
  transport to encode.
- 55 sim tests; gate green at 100% stmt+branch (4,690 tests).
  Radar/map/teleport/mine placement raise `SimError` until step (d)
  — explicit refusal, not silent stubs.[^2]

### Step (c) wire integration (2026-07-22): the production bot consumes sim bytes

`sim/transport.py` closes the loop the unit tests could not:
`encode_tick_payload` turns a tick's batch into length-prefixed 0x2E
envelope frames (step-(a) encoders + the session XOR table, framed
exactly as `process_received_message` ingests), and
`decode_client_payload` turns the bot's real `!`-command frames back
into typed commands. `SimServer.handshake()` emits the join burst
(own 0x3D/0x44/0x49, then 0x21+0x3D per living tank — the scenario
harness's `place_self`/`place_enemy` choreography, on the wire).[^2]

`tests/sim/test_integration.py` is the standing proof, all through
PRODUCTION code paths: the handshake establishes `self_state` and the
enemy registry via `sniffer.decoders.process_received_message`; a
`build_move_command` byte frame (XOR'd as the command sender
transmits it) drives the sim and the bot's believed position/fuel
equal sim ground truth after the tick — pickup included; and the real
planner `decide()` produces a HUNT/COLLECT decision from sim-fed
beliefs.[^2]

**Bugs the wiring caught that typed-dict testing could not:**
1. The sim long-form-synced EVERY fuel-changed tank; the production
   dispatcher treats any fuel-bearing 0x2E as SELF fuel (the real
   wire is per-recipient), so a victim's sync would have corrupted
   the bot's own fuel belief. Fixed: long form for the client only,
   short form otherwise — and a regression test pins it.
2. A 0x21 with an empty decoration field shifts the whole wire
   layout; the handshake must emit the full 4-byte field. Direct
   typed ingestion (the scenario harness) can never catch either.[^2]

### Step (c) completed (2026-07-22): the real bot plays the sim

`sim/session.py` closes the seam. `SimCDPSession` implements the same
`CDPSessionProtocol` the production tick loop talks to, answering
every `Runtime.evaluate` from SIM WORLD TRUTH — no canned values:[^2]

- the page-client snapshot query answers with a truthfully-built
  `PageClientSnapshotDict` (`client_present=True` and
  `ws_ready_state=1` because the sim link genuinely is the client and
  is genuinely open; `map_visible` tracks map-open/teleport commands;
  the JS-heap field maps are empty — the type's honest "not captured"
  form, which the alignment samplers already treat as absent);
- the injected websocket send (`atob('<b64>')`) decodes through the
  sim transport into typed commands and queues on the server;
- unmodeled expressions RAISE `EncodeError` — loud, never best-effort.[^2]

Wiring recipe (pinned by `tests/sim/test_session.py`): construct
`Bot(url, headless=True)` (no browser), `bot._on_magic_captured(magic)`
builds the command XOR table from the shared static key,
`bot._cdp = SimCDPSession(server, table)`, and sim batches are
delivered as base64 payloads into `bot._cdp_message_buffer` — the
exact shape `world_sync.drain_messages` consumes.[^2]

**The smoke test**: 12 rounds of the PRODUCTION `_tick_once` against
a seeded world. The bot toggled its equipment, radared, collected the
container, hunted, and fought the rank-8 enemy — and its believed
position and fuel equalled sim ground truth at the end. Two findings
from wiring the real loop:[^2]

1. The bot's actual opening move is **equipment toggling (cmd 114)**,
   which the sim didn't model — law added: the toggle flips the slot
   server-side and answers the documented `t + 5 bytes` 0x74 state.
2. A too-poor world (one container, weak enemy) ends the session the
   PRODUCTION way: after killing the enemy and draining the map, the
   COLLECT owner raised the real `SessionExitError`
   (`no_productive_collect`). Sim worlds must be seeded sustainably —
   and equipment containers are still absent from the world model, so
   the sim bot can restock fuel but never ammo.[^2]

### Step (d) as-built (2026-07-22): laws 5–8 — the full bot command set processes

`sim/actions.py` + tick-processor wiring:[^2]

- **Teleport (law 5)**: cost `floor(6 × euclid)` to the ACTUAL
  landing; ring-1 displacement E→N→W (S as the documented last-resort
  assumption; a sealed ring rejects with cant_go); other tanks and
  enemy mines block, own-color mines don't; landing auto-pickup via
  the shared `resolve_pickup`; rejections emit 0x52 with the
  map-close flag.
- **Radar (law 8, scan side)**: an available extra is consumed and
  covers the viewport (Chebyshev radius 8), else
  `free_radar_radius(rank)`; emits 0x4F (containers + mines in the
  square) and 0x46 (enemy-found); bills 10. Sim assumption: full-info
  scans, not cache diffs — semantically safe (the client dedups),
  not byte-faithful to the diff protocol.
- **Map open (law 8, map side)**: free; 0x4C with atlas-ordered fuel
  dots from live containers + living-tank blips; no mines, per the
  measured map contents.
- **Mine press (law 6)**: 10 flat; 3×3 centered on the placer, skips
  rock/water/tanks, trades 1:1 with enemy mines (0x45), places own
  (0x4B); mines are not inventory. Sim assumption: clipped to map
  bounds — the viewport-edge clip needs a per-client viewport model.
- Pickup-fuel/equipment clicks route through the move law (they are
  destination clicks on the wire). Only unknown command bytes still
  raise `SimError`.
- Gate: 4,719 tests, 100% stmt+branch; 84 sim tests.[^2]

~~Not yet implemented from the law list: **law 4** (homing reroute +
TTL) — the sim has no departure semantics yet~~ — RESOLVED 2026-07-22,
see the law-4 as-built below. Equipment containers / ferries /
movable blocks remain out of the world model.[^2]

### Law 4 as-built (2026-07-22): the viewport model, 0x58 departure, and the reroute TTL

The reroute law needed a real trigger for 0x58, which forced the
per-client viewport model the step-(d) assumptions listed as missing:[^2]

- **Viewport**: Chebyshev radius 8 around the client — the SAME
  constant the extra-radar scan already used (`VIEWPORT_RADIUS`,
  promoted from the radar law; [[radar-mechanics]]: extra = whole
  viewport). Tank POSITIONS are viewport-scoped on the wire:
  the join burst 0x3D-states only in-view tanks (identities 0x21 stay
  global), each tick diffs membership after relocations — exit emits
  **0x58 TankRemove** and starts the reroute clock, re-entry emits a
  fresh 0x3D. A deactivated tank just drops from the visible set (its
  exit is announced by 0x41, not 0x58).
- **Id-targeted resolution** (`sim/commands.py` already decoded the
  shoot command's `target_id`): an id-shot at a living VISIBLE tank
  reroutes the click to the tank's current tile before positional
  resolution — the queue-race conversion (a same-tick mover drawn
  from stale coordinates resolves as homing, not a miss). An id-shot
  at a DEPARTED tank keeps firing guaranteed homing hits — ammo
  debited, damage applied, position dark — while
  ``departed_age_ms <= REROUTE_TTL_MS`` (`physics.combat`, the
  machine-checked 12 000 ms midpoint of the measured [11.0, 13.0] s
  boundary); past the TTL the id no longer resolves and the shot is
  the measured free single miss with nothing debited. A shooter
  without a ready homing slot cannot reroute (the human analogue
  needs homing enabled).
- Server plumbing: `_removed_at` records the 0x58 tick;
  `advance_tick` prices the age in ticks × 2 000 ms for the shot
  processor.

Behavior check that fell out for free: with positions
viewport-scoped, the seam's seeded enemy (Chebyshev 10 from spawn)
is position-dark at join — and the production bot still finds and
engages it through map blips + teleport, the real gameplay loop.[^2]

### Equipment containers as-built (2026-07-22): the archive-mined grant law — and the bug it flushed out

The last world-model gap ("the sim bot can restock fuel but never
ammo") closed in four pieces, each forced by the previous:[^2]

1. **The grant law was archive-mined, not trusted** (crack-before-
   code): 1,154 ``0x67 -> next 0x49`` exact-pre pairs across 246
   sessions. One slot per grant; hard cap 25; stack rolls 5-9
   (dual/homing) and 2-4 (radar); slot choice RANDOM among deficient
   slots — the wiki's "deterministic most-behind" contract is
   falsified and rewritten ([[equipment-system]]). Plus 5 unexplained
   ``show_message=False`` multi-slot grants, all at radar=0.
   Sim assumption (documented in ``sim/equipment.py``): deterministic
   most-deficient slot with midpoint stacks (7 weapons / 3 radar) —
   distribution traded for reproducibility.
2. **The 0x5A viewport model**: the AI's equipment path gates on
   in-viewport walk-reachability, and 0x5A ViewportUpdate is the ONLY
   message that sets the viewport ([[viewport-shift-protocol]]) — the
   sim had never emitted one, so equipment candidates could never be
   actionable. The sim now sends origin-only 0x5A patches (entities
   empty — every sim entity is hidden-layer/radar-owned, and the
   production reset-then-apply sweep spares radar-sourced entries) at
   handshake and on every client relocation, window centered on the
   client (the real window scrolls; documented approximation).
3. **The empty-container rejection**: no wire message announces a
   consumed container (0x67 travels alone, 1,154/1,154) — the client
   learns by re-clicking and receiving 0x52 error 4. The sim
   validates pickup destinations and answers exactly that.
4. **A REAL production bug, sim-found**: the re-click at a consumed
   container the bot is STANDING ON completes instantly by
   ``position_reached`` — completions run before the in-flight error
   handler, so the code=4 orphans, the stale belief survives, and the
   bot re-clicks the ghost forever (the live DOM-consumer removal was
   deleted 2026-07-19 on the assumption the wire code-4 path would
   attribute; same-tile collects never wait). Fix in
   ``completions._maybe_complete_collection``: a pending 0x52 defers
   position-completion one phase so the error handler attributes it
   and deletes the belief. Regression-pinned in
   ``tests/bot/test_completion_events.py``.

End-to-end proof (`tests/sim/test_equipment.py`): the production bot,
seeded at 8 extra radars, radar-reveals the seeded equipment
(0x4F ``0xFFFF -> -1`` cache marker), walks both containers, takes
two grants over the real wire, eats the code-4 on each re-click,
deletes the beliefs, and returns to fuel collection.[^2]

### Scripted opponent as-built (2026-07-22): return fire — and the dead wiring it exposed

``sim/opponent.py`` is a deterministic aggressor (pure function of
the world tick: dodge / shoot / hold / shoot on a 4-beat, acting only
while the client is inside its own viewport radius). It is NOT a
model of enemy minds — those stay uncertified — it exists to run the
client-side channels a passive world never touches. Wiring it forced
two fixes:[^2]

1. **Per-recipient sweep**: 0x52 supervisor rejections, 0x67 gains,
   and the inventory-full error are PER-CONNECTION on the real wire.
   The sim had been appending them to its single batch regardless of
   the commanding tank — harmless while only the client acted, a
   belief-corrupting leak the moment the opponent moved (production
   treats any 0x67 as a SELF gain; same class as the step-(c)
   fuel-sync leak). All three now emit only for the client's own
   commands.
2. **A second REAL production catch — dead instrument wiring**:
   ``ledger.ammo_book.record_ammo_enemy_shot`` was defined, exported,
   unit-tested — and called from NOWHERE in production. The ammo
   book's armor rule bounds shield loss by ``2 x enemy_shots``, so
   with the counter frozen at 0 the FIRST armor-absorbed hit in a
   real fight would have raised a false ammo divergence. Unit tests
   could never catch it (they call the function directly); the
   fighting soak's positive control (``enemy_shots > 0``) failed
   instantly. Now wired at the 0x53 dispatch point next to the
   fuel book's enemy-hit entry.[^2]

The fighting soak (`tests/sim/test_soak.py`): 24 production rounds
under return fire — the enemy lands real duals, the client's fuel
pays real 90s, the fuel book absorbs them as enemy-hit feasibility
entries — and both books still judge ZERO divergences, with zero
``physics_divergence`` events in the captured stream.[^2]

Still out of the world model: the radar-zero emergency grant and
real enemy minds (the scripted opponent is a harness, not a model).
Ferries and movable blocks both landed 2026-07-22 — see their
as-builts below. With blocks in, the world model holds EVERY
documented entity class: tanks, fuel containers, equipment, mines,
ferries, and blocks.[^2]

### Movable blocks as-built (2026-07-22): law 6b — the full pickup/drop cycle

The wire-cracked block contract ([[movable-blocks]]) is executable:

- **One command, carry-state routed** (`CMD_BLOCK` 98 'b', now a
  named constant with `build_block_command`): empty-handed presses
  pick up a CARDINALLY adjacent block (0x42 direction = the measured
  compass letter), towing presses drop. Out-of-reach and refused
  drops answer the measured 0x52 code 1; teleport while towing
  refuses with the measured code 0 (three-for-three capture).
- **The shared enum end to end**: one block over static water is a
  walkable bridge (1), on land an obstacle (2), two on water stacked
  (3) — derived from context, emitted identically on 0x42
  ``obstacle_type``, 0x4A tile updates, and 0x5A ``terrain_type``
  (the dynamic-terrain patcher now serves ferries AND blocks with
  value-change repatching and explicit reverts).
- **Law interactions**: bridges route as ordinary ground (surface
  classifier), land/stacked blocks are impassable and CLIP
  non-missile shots (bridge exemption is a documented assumption);
  block tiles refuse teleport landings, are skipped by mine
  placement, and exclude container respawns; a land drop destroys
  ANY team's mine wire-silently (the measured friendly-fire
  refinement); block ops are FREE.
- **Seam proof**: the production ingestion composes sim block tiles
  into wire terrain (obstacle class) from real wire bytes, and the
  real command service round-trips the block press.[^2]

Assumptions documented in ``sim/blocks.py``: cardinal-adjacency
reach, refusal codes for unmeasured rejects, and the untowed
transient 0x4A pairs along a towed walk are not modeled. The BOT
still has no block planner awareness — that remains the open work
the [[movable-blocks]] page names.

### `make sim-run` as-built (2026-07-22): the free soak — real terrain, and production gap #4

``tankpit-sim-run`` (``sim/run.py``) promotes the seam to a CLI: the
real ``Bot`` plays a timed session against the sim on the REAL
``field01_r.gif`` terrain (actual mountains and water shape the
router, clipping, and displacement), with the scripted opponent
returning fire. Artifacts: probe-channel log/events
(``runs/probe/latest.sim.*`` — the live ``runs/bot`` archive stays
reserved for real-server evidence), the recorded wire as a standard
``CaptureSession`` under ``runs/sim/``, and the world's final state.
No server, no browser, no fuel spent.[^2]

The first real-terrain runs earned their keep immediately:[^2]

1. **Six containers drowned**: the naive (100, 100) scenario region
   is coastal on field01 — the bot starved among dots it could never
   reach. Now a law: ``_require_seeds_passable`` rejects any
   scenario seed on rock/water, loudly, and the shipped arena is a
   verified fully-open 21x21 clearing at (216, 108), pinned against
   the real GIF by a test.
2. **Corpse clicks are real behavior**: the opponent killed the bot
   and the production loop kept clicking — the real connection
   survives deactivation, so a dead CLIENT's commands now drop
   silently (dead harness tanks still raise ``SimError``).
3. **Production gap #4 — a killed bot ticks forever**: own-kill
   0x41s have been decoded since 2026-07-19, but nothing consumed
   them for self-death — the deactivated bot sat "waiting for radar
   results" until the round budget ran out. Now: the 0x41 dispatch
   records ``self_deactivated`` when the victim is self (dispatch
   must not throw — it also runs under replay), and the tick loop
   raises the new ``deactivated`` session exit. The wire replaces
   the DOM scrape as the self-death channel.
4. **The bot fights with armor OFF by policy** (desired_equipment is
   dual/homing/radar), so its effective HP is its fuel — and an
   out-of-ammo opponent still lands unlimited 45-fuel singles, so
   ammo seeding does not cap damage. The default scenario is tuned
   winnable around both facts; the reference 150-round run: fight →
   kill → refuel to cap → collect → ``no_viable_targets``, the
   production HUNT owner's clean end.[^2]

### Ferries as-built (2026-07-22): law 2b — surface routing over live wire terrain

The single-command surface contract ([[ferry-mechanics]], user
verbatim 2026-07-19) is now executable end to end:

- **World**: ``SimFerryDict`` — one dynamic water tile that moves
  with its rider.
- **Movement law** (``sim/movement.py``): routing is surface-gated —
  on land the router opens ground + ferry tiles (a water click is
  cant_go, exactly the measured "you can't reach that"); riding
  opens water + land. The FIRST queue-consuming transition truncates
  the move ON the transition tile: boarding stops on the ferry even
  when the click was beyond it, disembarking stops one step onto
  land with the ferry left on the last water tile; billed cost and
  the echoed 0x47 path cover only the tiles actually walked. The
  ferry ends every afloat move under its rider. Floating containers
  pick up normally from the water (the 2026-07-20 user contract).
- **Wire** (``SimServer._viewport_update``): ferries travel as 0x5A
  visible-layer entities (wire terrain 5) with explicit reverts
  (terrain 0 → the static map value) for vacated tiles, deferred
  until the window covers them. Integration caught the patch-grid
  border: the 0x5A grid is 18x18 with a one-tile margin around the
  16x16 window, so ``col = x - left + 1`` — the first seam delivery
  landed the ferry one tile off in both axes.
- **Proof over the seam**: the production ingestion
  (``update_viewport_entities`` → ``FerryAwareTerrain``) learns the
  sim's ferry tile through real wire bytes
  (``tests/sim/test_ferry.py``).[^2]

Not modeled: ``TERRAIN_FERRY_ROCK`` (7), multi-tile ferries (the sim
ferry is one tile), teleport landings on ferry tiles (still blocked
by static water), and ferry fuel costs beyond the standard walk
tile rate (assumption — unmeasured).[^2]

### Respawn as-built (2026-07-22): the world replenishes, players return

Two laws close the finite-world problem (every session used to end
when the map drained or the only enemy died):[^2]

- **Container respawn** (``sim/spawn.py``): the archive-mined law —
  population-seeking, ~1/min below the seeded equilibrium, always at
  fresh tick-derived passable tiles, wire-silent
  ([[game-economy]] "Container respawn dynamics" for the mining).
  Server targets fix at init from the seeded stock.
- **Opponent revival as a NEW tank id** (``sim/opponent.py`` +
  ``SimServer.announce_tank``): the first revival reused the killed
  id and the production bot — CORRECTLY — never re-engaged it: kill
  suppression and registry liveness never forgive a dead id, and
  they should not, because real respawns join with a NEW wire id
  (that is exactly what ``persistent_tank_id`` exists to bridge).
  The harness now activates a fresh id near the client (a
  corner-of-the-map respawn fails HUNT's affordability gates),
  announced by a mid-session 0x21 riding the next batch head. The
  reference CLI run then shows the full cycle: kill → re-acquire the
  respawned id → fight on → the documented radar death-spiral exit
  as the session's old age.

### Step (e) as-built (2026-07-22): the verdict — soak, negative control, audit cross-check

The Phase 3 instruments judge the sim, and the sim passes. Three
tests close the phase (`tests/sim/test_soak.py`,
`tests/sim/test_audit_crosscheck.py`, shared boot in
`tests/sim/seam.py`):[^2]

- **Divergence-zero soak**: 30 rounds of the production `_tick_once`
  under a stepped `SeamClock` (the scenarios-harness clock
  discipline, 1 s/round), events captured via
  `configure_bot_runtime_logging` + the fake filesystem. Positive
  controls first (commands crossed the seam, the fuel book judged
  windows, the ammo book anchored snapshots, events flowed), then the
  verdict: **zero divergences** in both book counters AND zero
  `physics_divergence` records in the captured `events.jsonl`.
- **Negative control — the detector has teeth**: a corrupted self
  fuel sync (+700 with no announcing gain) delivered through the REAL
  ingestion path, followed by a quiet reading so the block closes.
  The fuel book counts the divergence and `physics_divergence` lands
  in the event stream. A soak that can't fail is not evidence.
- **Audit cross-check**: `SimCDPSession` now records every frame in
  both directions (`wire_log`) and `build_capture_session` assembles
  the standard `CaptureSession` shape. A 40-round session written to
  a temp runs tree and fed to the real `collect_evidence`:
  walk-cost 1/1 exact, dual-shot-cost 20 samples / 19 exact,
  fuel-capacity 45/45 exact — every sampled claim at or above
  `EXACTNESS_FLOOR`, the audit's own gate. The single dual-shot
  "mismatch" is the measured charge latency splitting a burst's first
  echo and its debit across a window boundary — the same
  positive-signed noise shape the real archive shows, which is
  fidelity evidence in itself.[^2]

The instruments forced three catches before they went green:[^2]

1. **Sync cadence**: the sim emitted 0x2E syncs only when fuel
   changed; the measured wire broadcasts one per living tank every
   ~2 s regardless of activity ([[tank-freshness-model]]). Without
   quiet zero-delta readings the fuel book can NEVER close a block —
   the soak sat at zero judged windows until the sim matched the
   wire. `advance_tick` now syncs every living tank every tick.
2. **Handshake self-identity**: the audit's `wire_timeline` names
   self from the FIRST received 0x21 (the archive convention), but
   the sim join burst carried no own-identity message. The handshake
   now opens with the client's own 0x21, matching real choreography.
3. **Latent test-infra contamination** (pre-existing, exposed by the
   new tests changing xdist scheduling): the client-structure
   survey's once-per-session gate was never reset by the central
   isolation fixture, so any tick-loop test that emitted the survey
   poisoned later tick-loop tests on the same worker.
   `reset_client_structure_survey()` joined
   `_isolate_protocol_singletons`.

Behavior finding worth recording: over 40 rounds the production bot
NEVER walks — its collect style is 100 % teleport locomotion
(fuel-dot hops), so the walk-cost positive control needed one
scripted walk driven through the real command service.[^2]

**Fidelity statement.** The sim is certified by the same instruments
that watch the real server: (1) encoders byte-identical on
72,916/72,916 archive messages; (2) the unmodified production tick
loop plays full sessions over real wire bytes through the live CDP
seam; (3) the Phase 3 fuel/ammo books judge those sessions
divergence-free, with a negative control proving the detector fires;
(4) `make audit`'s archive validators re-derive the economy claims
from sim-generated wire at real-archive exactness. NOT covered by
this certification: enemy minds, radar cache-diff byte layout (sim
sends full-info scans), spawn placement distributions and volumes
(rate and freshness laws ARE mined — [[game-economy]]), the
teleport displacement search beyond ring 1 (the 2026-07-22 corpus
sweep measured a ~24 % ring-2/diagonal tail; the sim models ring-1
E→N→W→S then cant_go), and the equipment grant's randomness (sim
grants deterministically to the most-deficient slot with midpoint
stacks; the kill mercy bundle likewise uses the measured medians).
RETIRED from this list by the 2026-07-22 archive mining: the
S-displacement assumption (south is measured, 31 corpus samples),
the reroute TTL estimate (corpus-swept to [12.91, 12.93] s —
``REROUTE_TTL_MS`` = 12 920), the centered-viewport approximation
(3,387 bot-session samples put the at-rest tank at exactly offset
(8,8); the dispersion is client animation lag the sim rightly
lacks, and the bot only acts from rest-center, so the viewport-edge
mine clip cannot bite in bot play), and the radar-zero grant — no
longer a mystery but a MEASURED deterministic law (a kill at radar
zero grants a silent mercy bundle, 5/5 vs 0/254; implemented in the
sim, [[equipment-system]]). Gate at close: 4,730 tests, 100 %
stmt+branch (4,754 after the law-4 and equipment follow-ups).

### Shadow comparator as-built (2026-07-22): `make shadow` — the archive judges the sim

The inverse instrument of the seam soaks, closing the certification
loop from the other side: the soaks prove the BOT cannot tell the sim
from the real server; the shadow proves the SIM cannot be told apart
from the archive. `tankpit_bot/validate/shadow*.py`, CLI
`tankpit-shadow`, target `make shadow`.[^2]

Design rule: every validator imports its predictor FROM THE SIM
SOURCE — the same constants and predicates `SimServer` executes —
never a restated copy. A shadow mismatch therefore always means "the
sim and the real server disagree": a wiki gap or a sim bug, both
demanding investigation (same non-softening posture as `make audit`,
whose `EXACTNESS_FLOOR` gates the table). This graduates the one-off
2026-07-22 mining sweeps into a standing instrument: every future
live run lands in `runs/` and is automatically re-judged against the
sim's laws.[^2]

First full-archive run (245 decodable sessions):[^2]

| law | samples | exact | verdict |
|---|---|---|---|
| sync-cadence | 126 | 118 (94 %) | PASS — other-tank median 0x2E gap within 500 ms of `TICK_MS` |
| grant-invariants | 1,149 | 1,149 | PASS — one deficient slot, cap-25 clip, rolls 5-9 / 2-4 |
| kill-mercy-bundle | 283 | 283 | PASS — silent bundle iff radar zero, amounts in rolls |
| corpse-window | 17 | 17 | PASS — kill→0x58 gap = `CORPSE_WINDOW_TICKS × TICK_MS` |

The measured roll ranges moved into sim source as part of this build
(`sim/equipment.py`: `WEAPON_STACK_ROLL`, `RADAR_STACK_ROLL`,
`MERCY_BUNDLE_ROLLS`, `kill_grants_mercy()` — the deterministic sim
stacks are now DERIVED midpoints of the measured ranges, and the
server's mercy branch calls the shared predicate).[^2]

**Calibration discovery — the self-sync cadence anomaly.** The first
calibration sweep judged all tanks and failed 31 of 346; outlier
triage showed 23 of the 31 were the session's OWN tank, drifting to
3-4 s+ median gaps, while other-tank inlier medians pinned at
1981-2010 ms — the 2 s law, dead on. ~10 % of sessions show the self
drift; other tanks never do (their only outliers are
brief-observation noise). The law was scoped to non-self tanks by
measurement, the finding recorded in [[tank-freshness-model]], and
the sim's every-tick self-sync stands as a documented simplification
(the self tank's truth rides 0x44/0x64/0x49, so its 0x2E cadence is
evidently not load-bearing). Open question: what condition triggers
the sparse self schedule.

Corpse-window note: 17 clean samples vs the mining sweep's 37 —
the shadow's filters are stricter (any victim-id 0x2E sync between
kill and removal disqualifies the pair as slot reuse, and quits
disqualify via 0x29), and all 17 survivors sit inside ±1 s of
22.0 s. Gate at close: 4,852 tests, 100 % stmt+branch.[^2]

### Bot policy as-built (2026-07-24): the sixth shadow law — enemy minds start becoming physics

The practice-room twin's last uncertified layer (enemy minds) began
closing with the archive-mined bot policy ([[enemy-bot-behavior]]
§Corpus-mined policy): ``sim/bot_policy.py`` implements the mined
laws (stationary default; one next-tick ``weapon=0`` single at the
attacker's tile; teleport-off at 7/8 hits by rank) as a certified
MODEL — distinct from ``sim/opponent.py``, which stays a harness —
and ``validate/shadow_bot_laws.py`` adds ``bot-return-fire`` as the
sixth ``make shadow`` law, importing its constants from the sim
source. First full-archive run: 2,247 samples, 2,125 exact (94.6%),
PASS — and the sample count equals the independent mining script's
bot-shot population exactly. The shadow timeline gained names /
shots / positions extraction to feed it. Same day, the "refuel
anomaly" resolved into the SEVENTH law: practice bots REACTIVATE —
same id (fixed 36-slot roster), full fuel, at the 22 s corpse
boundary, respawned FAR from the corpse (user correction + 102
measured pairs all ≥ 24 tiles; 70/102 > 96). `SimServer` takes
`roster_ids` and runs `reactivate_practice_bot` when a roster corpse
clears; the `bot-reactivation` shadow law prices it at 39 samples /
35 exact, PASS. Still uncertified: the teleport-off destination and
the respawn placement DISTRIBUTION (both modeled as deterministic
scatter, documented), and ranks ≥ 2 (no such bot in the archive).[^2]

**Team aggro joined the model (2026-07-25):**
`note_hit_for_team_aggro` implements the sight-gated per-hit reflex
mined the day before ([[enemy-bot-behavior]] §Team aggro — 48
gang-up + 81 assist archive shots, all within `AGGRO_SIGHT_RADIUS`
= 8), and the `bot-return-fire` law was upgraded to judge all three
reflex classes: archive exactness rose 94.6% → **97.6%
(2,192/2,247)** as the former residual was recognized as lawful
aggro. The law-test round also fixed a self-justification bug in
the event walk (judgment now precedes hit recording). The model is
wired into sim sessions the same day: `make sim-run-practice`
(`sim/practice_room.py`) seeds a four-bot certified roster (gang-up
cluster + ally) driven by `decide_practice_bot`, with hits noted
from 0x53 emissions and corpse-window reactivation active. First
soak: the production bot deactivated in 21 rounds under gang-up
fire — the sim now reproduces the live multi-bot failure mode the
scripted harness could not. `sim/opponent.py` remains the default
`make sim-run` (deterministic kill path); practice mode is the
fidelity soak.[^2]

**Hunt-only-when-full contract (2026-07-25):** the practice-room
soak's 21-round gang-up death was traced to the 2026-07-13
cardinal-adjacent mode-selector override (never user-approved,
shipped undocumented inside commit 89ab2715 — see the log
post-mortem for the four-patch stack beneath it). User contract, now
law in `bot/ai/mode_controller.py` + `ai_strategy.py`: HUNT is a
privilege of a full tank — fuel ≥ `fuel_capacity(rank)`,
duals+homings at `inventory_capacity(rank)`, radars ≥ cap−5 — with
every bar rank-derived (the fixed `fuel_full_threshold` and resume
thresholds are deleted from config). Mid-fight breaks disengage to
COLLECT keeping the combat lock; `HUNT/ACQUIRE` returns to the
locked target after the restock (damage persists, so the
sortie-and-return cycle beats even a 3v1). Re-run proof:
`make sim-run-practice` went from deactivated-in-21-rounds to 86
rounds, kills on 510 + 511 + the scripted opponent, bot alive —
session ends only when the sim world runs out of collectible
equipment.[^2]

**Sim world model lost its supply law (2026-07-25):** the 1/min
population-seeking container replenishment in `sim/spawn.py` was
built on the 2026-07-22 "spawn" mining, which is now FALSIFIED —
all 605 "spawns" were our own exposures of pre-existing ≥500-volume
containers ([[game-economy]], [[map-data-decode]]). The practice-room
`no_productive_collect` starvation at round 86 was this model
poverty, not a bot defect.[^2]

**World rework as-built (2026-07-25, same day):** the spawner is
deleted (`sim/spawn.py` keeps only the deterministic tile pickers)
and the honest model is live. `SimContainerDict` gains ``dotted``;
`process_radar` reports every in-radius container (volume-0 included
— the wire's removal signal) and permanently dots ≥500 reveals
(`physics/map.py::MAP_DOT_MIN_VOLUME`, machine-checked claim);
`build_map_data` emits the DOTTED set, so the sim atlas over-promises
exactly like live. `sim/world_seed.py` seeds the static population
(620 dots at the ~40% hold rate / 900 hidden fuel on the measured
volume mix / 450 hidden equipment; placements skip the pre-(1,1)
region the 0x4C skip-RLE cannot encode) and carries THREE real
practice layouts mined from archive first-map snapshots
(`analysis_scripts/mine_practice_roster.py` — all 223 sessions show
the same 36-bot shape, ids 500-535, 9 per team, ranks 0-1);
`--practice` selects one by run stamp, spawns the client at its real
join position at full stock, and `PracticeRoomDriver` drives all 36
bots with the certified policy. Proof: the 150-round soak plays to
`rounds_exhausted` (bot alive at 1082 fuel, kills on 529 + 508
across the map, 109 pickups — no starvation), and the exposure miner
against the sim's own capture matches the archive signature 18/18
exposure-preceded / 0 unpreceded. Gate green (4,961 tests,
100%).[^2]

**Round-resolution order wired in (2026-07-25):** the same-day
measurement (`analysis_scripts/mine_round_order.py` — 1,820/1,825
archive multi-shooter bursts fire in ascending shooter id; the only
5 violations are our own sim captures) is now law in the sim:
`SimServer.advance_tick` sorts the per-tick queue by tank id before
processing ([[game-rules]] §Combat rounds). The stable sort keeps one
tank's own commands in arrival order. Two sim tests that leaned on
arrival order were re-anchored so the intended earlier actor carries
the lower id; a pinning test
(`test_round_resolution_orders_by_ascending_tank_id`) guards the
law. Gate green (4,955 tests, 100%); all 7 shadow laws still
PASS.[^2]

### Damage tier solved (2026-07-23): no healing exists — the tier is the fuel quartile

The "healing ladder" gap died to a user correction ("tanks dont heal…
fuel IS the health pool; mouse-over shade = HP") and a same-day
corpus fit: **19,658/19,658** long-form 0x2E syncs obey
``damage_tier = min(3, 4 * fuel // fuel_capacity(rank))`` — zero
exceptions, boundaries exactly at capacity quartiles
([[deactivation-format]], claim block + `physics/capacity.py`).
Consequences swept through the stack in one build:

- **Sim**: the hit-driven ``_DAMAGE_PROGRESSION`` state machine was
  the WRONG model — deleted; ``SimTankDict`` no longer stores a tier
  at all; every emission point derives it from fuel.
- **Bot**: the ``DAMAGE_*`` constants and finish-off ordering in
  ``bot/ai/threats.py`` were INVERTED (June's "counts down 0→3→2→1"
  misreading) — tier 0 is now correctly the kill-shot target, and an
  unknown tank defaults to tier 3 (assume healthy), not "full = 0".
- **Shadow**: fifth law ``damage-tier`` re-derives the quartile fit
  on every ``make shadow`` (19,658/19,658 on the archive).
- The controlled live "healing measurement" session is CANCELLED —
  nothing to measure.[^2]

## Parallel track (independent of phases): executor staleness audit — DONE 2026-07-21

RESOLVED — see [[executor-rejection-loops]] Resolution 2026-07-21:
all three validators deleted after unreachability proof; executor is
pure dispatch. Original framing kept below for the record.

[[executor-rejection-loops]] instances #2 and #3 remained: the
combat-anchor position-match and the pickup-race checks guard planner
CROSS-TICK state. End state (consistent with the mine-veto deletion):
decision-time validation complete enough that `executor.execute` is
pure dispatch. Migrate each check into the lock-continuation branches
that produce the stale state; delete the executor check; keep the
invariant that any veto-like refusal must change next-tick inputs.

## Constraints (non-negotiable, see [[coding-standards]])

No Any/cast/type-ignore/noqa; no mocks — `_test_hooks` save-restore
DI only; 100 % coverage; guard rules enforced; files < 400 lines;
no back-compat shims (move symbols, update all imports, delete the
old home); wiki updated + log entry with every phase; verbatim user
contracts preserved wherever quoted.[^2]

## Verification per phase

`make check` green (guard + ruff + mypy + tests + 100 % coverage),
then a 5-minute `make run` soak analyzed for behavior neutrality
(Phases 1–2 must be behavior-neutral; Phase 3 adds scorecard lines
only). Commit per phase with the soak evidence in the message —
follow the 2026-07-20 commit style (`6d2afdbe`, `3bd031f9`).[^2]

[^1]: Design conversation 2026-07-20: user framing "wiki as the source of truth... with 3 consumers" (code, archived wire evidence, live wire) and "no handwaving, no half assing it at all. the full complete process verified. quality." Phase ordering user-approved; Phase 1 explicitly agreed as the starting point.
[^2]: receipts for every design and as-built claim above, three-fold: (1) CODE — the blob-pinned trees in frontmatter (`src/tankpit_bot/physics`, `src/tankpit_bot/sim`, `src/tankpit_bot/validate`) plus `protocol/encoders/`, `ledger/fuel_book.py`/`ammo_book.py`, `scripts/physics_claims.py`, and the named `tests/sim/*` files — every symbol, constant, and law named above is greppable on disk, and design paragraphs describe the plan those trees implement (deviations recorded inline); (2) INSTRUMENTS — `make check` (gate/coverage), `make audit` (per-claim sample counts), `make roundtrip` (72,916-message corpus), `make shadow` (law table), `make sim-run` re-derive every number quoted above on demand; (3) HISTORY — the dated 2026-07-20/21/22 commits in git history and their wiki-log entries, plus soak artifacts under `runs/`.
