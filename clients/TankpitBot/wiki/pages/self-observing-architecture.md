---
title: Self-Observing Bot Architecture
tags: [architecture, decisions, observability, contracts, ledger, memory]
related:
  - "[[bot-behavior-contract]]"
  - "[[coding-standards]]"
  - "[[combat-chase-bug]]"
source_paths:
  - "docs/handoffs/self-observing-bot-architecture.md"
source_git_blobs:
  "docs/handoffs/self-observing-bot-architecture.md": "e56a608cef109009be41d9c149a67068fba4d7f1"
fact_checked: "2026-07-18"
confidence: high
hubs: [architecture]
---

# Self-Observing Bot Architecture

The tank_pit bot has been organically built one action at a time. Each action grew its own diagnostic channel: `wire_complete` for scan/move/teleport/collect/map_open, `teleport_attempt` for teleports, `combat_feedback` for shots. Three parallel diagnostic fabrics with no shared contract, and one class of failure — client-side executor discards — that no channel covered at all. The 2026-07-06 20:47:31 live deadlock exposed the gap: 26 seconds of silent self-rejection because a bare `emit_ai("rejecting shoot at ...")` log line looked identical to a legitimate server rejection and no structured event correlated the discard to the planner's decision.

The corrected architecture treats the bot as a self-observing decision system with four layers, one cross-cutting contracts framework, and one philosophical principle: **fail hard on state entry, not soft on state observation.**

## Foundational principle

**The bot cannot enter a wrong state.** State transitions have contracts. Contract violations raise on the transition, not after N observations of the consequences.

- No repeated-failure detection. No "wait until N tries then complain."
- No anomaly thresholds over consequences. Anomalies are just late notifications of contract violations that should have raised earlier.
- Races vs bugs are structurally distinguished: if the planner had the evidence and used it correctly, a same-tick external change is a race → replan gracefully. If the planner should have checked but didn't → contract violation, raise.

The 20:47:31 deadlock under this architecture: at tick 1, the executor sees the tank position is not in the registry, raises `ShootTargetNotTrackedError`, session exits with a specific error naming the planner-generated command. The human fixes the planner. No retry counter. No anomaly detector. No 26-second wait.

## What the bot is missing today — the fifteen

Each item has the same shape: **missing** state / what it looks like **present** / what we have **today**.

### 1. Standardized WHAT (action outcomes) — ✅ Phase 2 (2026-07-18)

- **Missing:** silent discards, no correlation dispatch↔outcome, three parallel diagnostic channels.
- **Present:** unified `action_outcome` event per attempt (all six action kinds).
- **Today:** DONE — unified `action_outcome` fabric with per-kind typed emitters, executor discards included; the three legacy channels are deleted.

### 2. Standardized WHY (decision reasoning) — ✅ Phase 2 (2026-07-18)

- **Missing:** free-text `reason` strings, unstructured, dropped after log.
- **Present:** `reason_kind` (Literal) + `reason_context` (TypedDict) per Decision.
- **Today:** DONE — `reason_kind` (closed 17-value Literal) + `reason_context` on every behavior; free-text reasons deleted.

### 3. Predictions and their accuracy — ❌

- **Missing:** bot commits to actions without stating what it EXPECTS to happen. Can't tell when its world model is wrong.
- **Present:** before each decision, structured `PredictionDict` of expected outcome + specifics. After outcome, `matched_prediction: bool` computed. `prediction_accuracy_by_reason_kind` derived across runs.
- **Example:** when the bot teleports to purple-4 at (172,107), it doesn't predict "purple-4 will be at (172,107) when I land." So it can't learn that targets with recent teleport activity don't stay put.
- **Today:** no expected-hit-rate per weapon, per range, per target rank.

### 4. Alternatives considered — ❌

- **Missing:** planner picks target A, we never record that B and C were also candidates and why they lost.
- **Present:** `alternatives: list[RankedAlternativeDict]` on every Decision. Post-hoc: "was my ranking right? was the second choice better historically?"
- **Example:** HUNT picks red-8 as top threat, orange-4 was #2 at score 780, red-5 was #3 at 650. If red-8 always escapes and orange-4 always dies, the ranking is miscalibrated — but today we can't discover this because we didn't record the ranking.
- **Today:** acquisition scores aren't structured in a queryable form.

### 5. Confidence / uncertainty — ❌

- **Missing:** decisions are binary commitments. No "70% sure the container has 400 fuel" or "45% sure this target will hold still."
- **Present:** `Confidence[0,1]` field on every Fact and Decision; low-confidence triggers verification actions (extra radar, scan-on-landing).
- **Example:** the bot treats a container discovered 30 s ago the same as one discovered 3 s ago. Different confidences, same commitment.
- **Today:** the retired freshness TTL was a crude single-threshold proxy for this.

### 6. Provenance for facts — ⚠️ containers only

- **Missing:** when the bot believes "container at (150,150) has vol=400," it doesn't record whether that belief came from 0x5A viewport patch, 0x43 cache update, 0x4F radar reveal, or a game-log inference — so it can't reason about trust; every belief is treated as equally reliable.
- **Present:** every world-state field carries `source: FactSource + observed_ms + provenance_chain`. Planner can weight beliefs by source reliability.
- **Today:** `ContainerStateDict.source` and `refresh_kind` capture this for containers only. Not extended to tanks, mines, terrain, or self-tank state.
- **Example:** after the 30 s freshness TTL fix, the bot trusts any in-viewport container. But a container revealed by a stale radar cache and one confirmed by the current 0x5A sweep have different provenance — different trust.

### 7. Per-entity memory — ❌

- **Missing:** every enemy is treated identically. If Yuppler tends to marathon and Kirby tends to stand and fight, the bot doesn't record that pattern.
- **Present:** `per_entity_memory: dict[EntityId, EntityMemoryDict]` — tracked across ticks (and eventually across sessions).
- **Example:** Sigma's guide is entirely built on this — Type 1/2/3 target categorization from per-player observation.
- **Today:** `killed_tank_ids` and `blocked_combat_targets` are ad-hoc partial memory. Not general.

### 8. Cross-session persistence — ❌

- **Missing:** every session starts blank. No memory of yesterday's game, no map-specific hot spots, no learned enemy behaviors.
- **Present:** a session artifact (JSONL + SQLite index) that accumulates per-map and per-entity beliefs, re-loaded at bot start.
- **Example:** field01 (Practice room) is played every session. We could know its fuel-dot decay rate, hot combat zones, typical enemy density.
- **Today:** `docs/handoffs/` accumulates human-authored handoffs; nothing machine-consumable.

### 9. Causal chain — ⚠️ primitives landed (Phase 2)

- **Missing:** "I teleported → landed adjacent → fired dual → hit → homed off → target teleported" is a chain implied by the log, not queryable data.
- **Present:** every event carries `caused_by: list[EventId]`. Chains form a DAG; queries traverse.
- **Example:** to answer "when I get a kill, what typical sequence led there?" you eyeball the log. Should be a query.
- **Today:** every outcome and mode transition carries `caused_by` (its decision event id); `decision_record(id)` resolves it. Traversal API (`trace_backward`/`forward`) lands with its Phase 3 consumers.

### 10. Anomaly detection — REJECTED

- The whole category is a symptom of missing contracts. All would-be anomalies are contract violations that should raise at state entry, not after accumulating observations.
- The one exception: `PredictionModelDriftError` fires when the planner's stated prediction accuracy falls below its own committed threshold — that IS a contract on the planner ("your model is broken"), not anomaly detection over consequences.

### 11. Mode transitions — ✅ Phase 2 (2026-07-18)

- **Missing:** mode changes (COLLECT ↔ HUNT) happen via cascade fall-through, not first-class events. Why did we transition? Was it the right time?
- **Present:** `mode_transition` diagnostic on every mode change with `from_mode`, `to_mode`, `reason_kind`, world snapshot.
- **Today:** DONE — first-class `mode_transition` events with `from_mode`/`to_mode`/`reason_kind`/`caused_by`.

### 12. Time budgets — ❌

- **Missing:** no "I've been engaged with purple-4 for 30 s; should I disengage?" Time-aware self-control is minimal.
- **Present:** per-decision time budgets that expire and force re-evaluation. `TimeBudgetExpiredError` raises at expiry.
- **Example:** the 20:47:31 deadlock consumed 26 s on one target. A "30-second engagement budget" would have forced disengage.

### 13. Self-model — ❌

- **Missing:** the bot has no model of its own performance. Doesn't know that at range 3 it hits 90% of the time but at range 6 only 40%. Doesn't know its dot-relay landed-exact rate vs combat-landing rate.
- **Present:** aggregated `SelfModelDict` derived from the ledger, exposed on `DecideCtx`.
- **Example:** bot commits to homing pursuit not knowing its historical off-viewport hit rate is 20%. Would rather teleport-close.
- **Today:** `session_hit_count` / `session_miss_count` are the only self-observations, and they're not per-context.

### 14. Comparative baselines — ❌

- **Missing:** no session-to-session comparison. This session vs last session, this hour vs baseline, this map vs typical.
- **Present:** `runs_index.tsv` aggregation feeding a "compare-to-baseline" scorecard line.
- **Today:** `runs_index.tsv` exists but isn't consumed for comparison during play.

### 15. Feature-flag experiments — ❌

- **Missing:** no way to A/B test a hypothesis. "Would blocking targets after 2 rejections be better?" — only way to answer is deploy and see.
- **Present:** config-flag machinery so a fraction of sessions run variant A vs variant B, with paired outcome comparison.
- **Today:** config values are static.

## The four-layer architecture

```
┌────────────────────────────────────────────────────────────────┐
│ Layer 4: MEMORY (long-lived, cross-tick, cross-session)       │
│  Per-entity behavior models, per-map facts, session index,     │
│  self-model, feature-flag store                                │
└────────────────────────────────────────────────────────────────┘
                            ▲    │
┌────────────────────────────────────────────────────────────────┐
│ Layer 3: LEDGER (per-attempt, in-session)                     │
│  Decisions, Outcomes, ring buffer, causal chain,               │
│  mode transitions                                               │
└────────────────────────────────────────────────────────────────┘
                            ▲    │
┌────────────────────────────────────────────────────────────────┐
│ Layer 2: DECISION ENGINE (per-tick)                           │
│  Planners consume Facts + Memory + Ledger + SelfModel          │
│  Emit Decisions with evidence, predictions, alternatives,      │
│  confidence, time budgets                                       │
└────────────────────────────────────────────────────────────────┘
                            ▲    │
┌────────────────────────────────────────────────────────────────┐
│ Layer 1: FACTS (world model with metadata)                    │
│  Every fact: value + source + observed_ms + confidence +       │
│              provenance_chain                                   │
└────────────────────────────────────────────────────────────────┘
                            ▲
                     WIRE + LOGS
```

**Cross-cutting: `contracts/`** — `ContractError` base + `@enforce_contract` decorator + guard rule scanning for public mutations that skip enforcement.

## Phase roadmap

The full architecture is a multi-phase multi-session commitment. Each phase leaves the tree green and shippable.

| Phase | Deliverable | LoC | Sessions |
|---|---|---|---|
| 0 | Immediate deadlock fix: delete executor position-check | 200 | 1 |
| 1 | `contracts/` + `facts/` foundation | 800 | 2-3 |
| 1a | ✅ landed 2026-07-18: `contracts/` (error hierarchy, `require`, `@enforce_contract`) + `facts/` core (`Fact[T]`, `FactSource`, provenance, confidence ops) + guard rule in `scripts/contract_rules.py` | — | — |
| 1b | ✅ landed 2026-07-18: `ContainerStateDict` carries `confidence` + `provenance` (origin derived from `refresh_kind`); `facts/container_facts.py` projects `Fact[ContainerValueDict]` | — | — |
| 1c | ✅ landed 2026-07-18: `TankStateDict` carries `confidence` + `provenance`; `TankObservation.fact_source` records the exact wire channel at all 12 dispatch sites; `facts/tank_facts.py` projects `Fact[TankValueDict]` | — | — |
| 1d | ✅ landed 2026-07-18: `SelfStateDict`/`MineStateDict`/`TerrainTileDict`/`ViewportStateDict` carry the fact metadata flat (self/viewport/terrain also gained `observed_ms`); `FactSource` 18 → 23 (0x2B promotion, 0x44 fuel gain, 0x4A terrain update, 0x4B mine placement, 0x64 fuel total); `facts/world_facts.py` projects all four | — | — |
| 2 | ✅ landed 2026-07-18: `ledger/` core — unified outcome fabric (31+ typed outcomes incl. executor discards + `superseded`), typed `ReasonKind` decisions, Decision↔Outcome correlation via pending-pairing (`caused_by` on every outcome), `verify_outcome_invariant` session sweep, first-class mode transitions, scorecard per-outcome counters | 2500 | 4-6 |
| 2b | ✅ landed 2026-07-19: deterministic run audit (`tankpit-run-audit` in `make analyze`) — typed findings with severity + evidence over the ledger (kill uniqueness, unresolved decisions, retry loops, stalls, cadence gaps, exit forensics) plus capture replay cross-validation (current-decoder re-decode of every received frame, wire-vs-ledger channel diffs, undecoded-subtype canary, DOM-witness diff). The ratchet rule: any hand interpretation of a run becomes a check here or is explicitly rejected — same artifacts always produce the same verdicts | — | — |
| 3 | Decision enrichment: Predictions, Alternatives, Confidence, Time budgets | 1500 | 2-3 |
| 4 | `memory/` (per-entity + persistence + session start/end) | 2500 | 3-5 |
| 5 | Aggregation (self-model + baselines + experiments) | 1500 | 2-3 |

Total: ~9000 LoC production + ~13500 LoC tests + wiki. 14-21 sessions of focused work.

Detailed phase specs live in `docs/handoffs/self-observing-bot-architecture.md`.

### Phase 1a implementation notes (2026-07-18)

`src/tankpit_bot/contracts/` — `ContractError` hierarchy (`base.py`: `NoUnsourcedFactError`, `ConfidenceOutOfBoundsError`, `ProvenanceRootednessError`, each self-naming via `contract_name`), `require(condition, error, **details)` helper recording the caller's `file:line` (via `traceback.extract_stack` — the monorepo guard bans `import inspect`), and `@enforce_contract(contract)` decorator (`enforcement.py`). The `Contract` protocol is generic over a `ParamSpec`: a contract's `check` carries the same typed signature as the function it guards, so enforcement adds no type erasure (the guard also bans `object` in annotations, which rules out the untyped-kwargs-mapping design).

`src/tankpit_bot/facts/` — generic `Fact[T]` TypedDict (value + source + observed_ms + confidence + provenance), the 11-value `FactSource` literal, `SourceRefDict`/`ProvenanceChainDict` with encode/decode, and confidence ops (`combine_independent` noisy-OR, `combine_weighted`, `decay_by_age`). `make_fact`/`decode_fact` enforce all three contracts — a stored fact violating a contract fails at load.

Guard rule `scripts/contract_rules.py` (wired into `scripts/guard.py`, runs in `make lint`): public `apply_*`/`record_*`/`mutate_*`/`set_*`/`update_*` functions in `facts/`, `ledger/`, `memory/` must carry `@enforce_contract`.

Three documented deviations from the handoff spec: (1) `dom_registry_scrape` counts as an observation origin (the DOM is a second wire, not a derivation) — only `client_side_inference` requires provenance citations (`game_log_scrape` was also a source until 2026-07-19, when capture replay proved every game-log line is the client's rendering of an already-decoded wire message and the channel was retired to witness-only); (2) `make_fact` invokes its contract explicitly instead of via the decorator, because decorating a generic function erases its type variable under mypy — the decorator is the mechanism for the non-generic mutations of Phases 1b+; (3) the spec's field-presence and source-membership runtime checks are structural under the typed keyword-only constructor (and re-validated by `require_fact_source` on decode), so the runtime contract checks are the ones typing cannot express: `observed_ms >= 0`, confidence bounds, provenance rootedness, and source/origin coherence.

### Phase 1b/1c implementation notes (2026-07-18)

**Flat-carry retrofit (deviation from spec's nested `Fact[ContainerValueDict]`):** `ContainerStateDict` and `TankStateDict` keep their flat shape and gain the two missing Fact fields (`confidence: float`, `provenance: ProvenanceChainDict`); `facts/container_facts.py` and `facts/tank_facts.py` provide the true `Fact[T]` projections for Fact-consuming layers. Rationale: nesting the value under `["value"]` would touch ~200 construction sites and ~300 access sites across 68 files for zero information gain — the flat dict already carries every Fact field.

**Provenance sources are message-granular:** `FactSource` grew from the spec's 11 to 18 channels (added 0x21 TankInfo, 0x28 TankEntry, 0x3E TankStatus, 0x42 BuildPickup, 0x47 Movement, 0x48 EnemyDetect, `dom_registry_scrape`; renamed 0x2E to `wire_0x2E_tank_status_sync`) because the spec's list missed the channels that actually update the tank registry. Container origin derives mechanically from `refresh_kind` (`container_fact_source`); tank origin is passed explicitly by every dispatch site via the new `TankObservation.fact_source` field (0x53 shoot / 0x42 build-pickup / 0x47 movement pass through the parameterized `_update_tank_position`).

**Convergent decode defaults:** pre-1b/1c snapshots lacking the new keys decode to exactly what a contemporary encoder writes (confidence 1.0, provenance derived from refresh_kind / coarse source) — the same `_optional_int` precedent tank.py already used, no divergent state possible. Synthetic default fact-sources (`tank_default_fact_source`) exist only for direct constructor calls in tests; the production observation pipeline always supplies the true channel.

### Phase 1d implementation notes (2026-07-18)

Same flat-carry pattern as 1b/1c, completing the world-state coverage. Channel-threading highlights: self-position updates flow through `update_self_position(…, fact_source)` — the 0x47 waypoint path and 0x3D movement path each pass their own channel; fuel totals thread per dispatch arm (0x2E sync / 0x44 fuel gain / 0x64 fuel total); rank stamps `wire_0x2B_promotion`; a witnessed mine placement stamps `wire_0x4B_mine_placement` while radar/viewport/map mine sightings derive from their coarse source; terrain distinguishes 0x5A patch grids (default) from 0x4A terrain updates; the viewport's origin is constitutionally `wire_0x5A_viewport_patch` (the only message that sets it — see [[viewport-shift-protocol]]). `ViewportStateDict` construction now goes through `make_viewport_state` (previously raw dict literals). Self/viewport/terrain gained `observed_ms` (they had no timestamp at all); mines already had one.

Phase 1 is COMPLETE.

### Phase 2 implementation notes (2026-07-18)

**Correlation via pending-pairing, not parameter threading.** The bot's own invariant (at most one in-flight action per kind) is the pairing rule: `record_decision` (executor entry, every dispatchable command) registers the decision as its kind's pending causal parent; `emit_action_outcome` — the single low-level emission path — consumes it into `caused_by`. A re-dispatch of the same kind closes the unresolved predecessor with an explicit `superseded` outcome, so **every recorded decision resolves to exactly one outcome** except the ≤6 pending at shutdown. `verify_outcome_invariant()` (session end, `_emit_session_scorecard`) raises `LedgerInvariantError` on any decision that is neither resolved nor pending — possible only if a code path bypassed the fabric.

**Two more guard-enforced contracts:** `DecisionRecordContract` (score band, non-empty reason) on `record_decision`; `TeleportDispatchContract` on `record_teleport_dispatch`.

**Mode transitions** (blind spot #11): every mode flip at the tick-loop persist point emits a `mode_transition` event with the driving decision's `reason_kind` and `caused_by` (0 for the non-dispatching manual-hold path).

**Session end** now emits `session_outcome_counts` per action kind (from the rings) and `session_unresolved_decisions`, and the scorecard/issue report carry `action_outcome_counts` (`"kind:outcome"` tallies) — the per-outcome counters replacing ad-hoc hit/miss/reject views.

**Deferred to Phase 3 (consumers don't exist yet, and dead API is banned):** the rich causal-chain traversal (`trace_backward`/`trace_forward`) and `DecideCtx` ledger views — the primitives are in place (`decision_record(id)` lookup, `caused_by` on every outcome and transition, `recent_outcomes`/`outcome_counts` ring queries).

Next: Phase 3 (Decision enrichment — predictions, alternatives, confidence, time budgets).

## Why we're building this

The 20:47:31 deadlock is one instance of a class. The class is "decisions and outcomes live in disjoint observability channels." Every future bug of that class hides the same way. Fixing the specific bug closes one instance; installing the architecture closes the class.

The 15 items above are not features — they are the shape of what "a bot that can learn from itself" requires. Each item names a specific blind spot the bot has today. The 20:47:31 deadlock hit at least four of them (1, 3, 10, 12) simultaneously. Any single item, present, would have surfaced the bug within seconds instead of hiding it for 26 seconds.

## Related pages

- [[bot-behavior-contract]] — the current contract for AI behavior (updated in each phase)
- [[coding-standards]] — style guide, ban list, testing patterns
- [[combat-chase-bug]] — the closest prior architectural bug, closed by decoupling planner from executor
- [[game-economy]] — wire-verified action costs; the substrate the ledger records above

## Provenance

Design conversation: 2026-07-06 session, driven by user's observation "we were missing standardized WHY, and WHAT — what else are we missing?" Full architectural spec in `docs/handoffs/self-observing-bot-architecture.md`.
