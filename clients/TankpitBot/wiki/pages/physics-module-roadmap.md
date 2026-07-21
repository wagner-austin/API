---
title: Physics Module Roadmap — Wiki as Executable Truth
tags: [architecture, roadmap, physics, wiki, enforcement]
related: [[terrain-composition]], [[game-economy]], [[executor-rejection-loops]], [[self-observing-architecture]], [[coding-standards]], [[movable-blocks]]
sources: [design session 2026-07-20 (user + AI), user-approved direction; mcps-workspace precedent PLAN_WIKI_AUDIT_SEARCH_MCP_REFACTOR.md]
fact_checked: 2026-07-20
confidence: high (Phases 1-2 IMPLEMENTED 2026-07-20/21; Phase 3 + executor track designed, not started)
---

# Physics Module Roadmap — Wiki as Executable Truth

**Status: Phase 1 implemented 2026-07-20, Phase 2 implemented
2026-07-21 (make audit green, 9/9 claims). Phase 3 and the executor
track remain designed-not-started.**
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
an archaeology project.

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
  reroute TTL (~12 s, boundary [11.0, 13.0]) only in the wiki.
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
- `combat.py`: reroute TTL constant (current estimate 12_000 ms,
  boundary [11.0, 13.0] s — mark estimate), consumption-equals-hit
  rule helpers.
- Terrain vocabulary stays in `state/types/constants.py` (already
  truth-named after 2026-07-20); physics may re-export.

**Claim binding**: each physics wiki page (game-economy,
teleport-mechanics, ferry-mechanics, weapon-selection,
movable-blocks) gets a fenced machine-readable claim block:
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
claim. Drift in either direction = red gate.

**Definition of done**: all economy/damage/capacity constants live in
`physics/`, zero magic game-numbers left in `bot/ai/` (guard-greppable),
claim checker wired into `make check`, 100 % coverage held.

### Phase 1 as-built (2026-07-20)

Implemented exactly as designed with these recorded deviations:

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
  `validate_hit_damage.py` (same window technique)

Claim blocks gain `validator:` and `evidence:` fields. `make audit`
runs every validator, reports per-claim sample counts, and rewrites
each page's `fact_checked:` from validator output — computed, not
hand-typed. A claim whose validator fails or whose evidence files
vanished is flagged like a stale hash (mcps wiki-check analogy).

### Phase 2 as-built (2026-07-21)

Implemented as `src/tankpit_bot/validate/` (not `tools/validate/` —
`src` is where mypy/ruff/coverage/guard already apply; follows the
`diagnostics/` precedent, CLI `tankpit-audit`, `make audit` target):

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
  (a walk drains across many windows, so single windows cannot price
  it: an episode runs from a walk-only window through event-free
  windows to a zero-delta quiet window; SINGLE-ECHO only, because a
  second 0x47 means a mid-walk re-command whose full commanded path
  was never fully stepped — the 2026-07-21 probe showed every
  multi-echo mismatch overcounts tiles, never fuel), hit damage
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
ledger; do it last, after Phases 1–2 make predictions cheap.

### Phase 3 design (locked 2026-07-21, pre-implementation)

**Instrument: an interval double-entry fuel book**
(`ledger/fuel_book.py`, pure functions over a typed book dict).
Between two absolute wire fuel readings the book accumulates ENTRIES
— each a predicted delta with a feasibility range, because live
windows contain events whose exact fuel effect is legitimately
unknowable at prediction time:

- exact debits: own shots by weapon (homing carries a −5/−10 split
  tolerance), radar 10, mine press 10, teleport floor(6*euclid) on
  the ACTUAL landing;
- ranged debits: own 0x47 walk echo = [0, path_tiles] (paths truncate
  on collision — Phase 2 finding); optional enemy hits = {0, −45} or
  {0, −90} per lone echo; own-mine detonation −45 optional;
- open credits: container pickups = [0, capacity − fuel].

**Reconciliation** happens at the single wire choke point
`update_world_state_from_fuel_total` (all of 0x2E sync / 0x44 gain /
0x64 deposit-total flow through it): measured residual must fall in
the entries' feasible interval; outside = a `physics_divergence`
diagnostic event carrying the residual and the window's entry list.
Strict windows (exact entries only) demand equality — the live
equivalent of the audit's clean windows.

**Entry sources**: wire-echo entries recorded in the
world_state_dispatch handlers (0x53 own/enemy, self 0x47, 0x43
pickup, 0x45 detonation); dispatch-side entries (radar, mine press,
teleport-landed with pre-position) recorded from the executor/ledger
outcome path. Ammo book v1: 0x49 snapshots must never exceed the
previous snapshot plus pickups nor undercut it by more than shots
fired — the consumption-equals-hit cross-check, live.

**Surfacing**: scorecard line `physics divergences: N` (counted from
events.jsonl by session_scorecard); `make analyze` lists each
divergence with its window as a candidate wiki claim. Verification:
gate + a soak whose only behavioral change is the new scorecard line;
divergences on a healthy run should be ~0.

### Phase 3 as-built (2026-07-21, core loop)

Implemented as designed with these notes:

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
  increment on this foundation.


## Parallel track (independent of phases): executor staleness audit

[[executor-rejection-loops]] instances #2 and #3 remain: the
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
contracts preserved wherever quoted.

## Verification per phase

`make check` green (guard + ruff + mypy + tests + 100 % coverage),
then a 5-minute `make run` soak analyzed for behavior neutrality
(Phases 1–2 must be behavior-neutral; Phase 3 adds scorecard lines
only). Commit per phase with the soak evidence in the message —
follow the 2026-07-20 commit style (`6d2afdbe`, `3bd031f9`).

[^1]: Design conversation 2026-07-20: user framing "wiki as the source of truth... with 3 consumers" (code, archived wire evidence, live wire) and "no handwaving, no half assing it at all. the full complete process verified. quality." Phase ordering user-approved; Phase 1 explicitly agreed as the starting point.
