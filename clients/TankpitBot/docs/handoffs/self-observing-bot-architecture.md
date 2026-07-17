# Handoff: Self-Observing Bot Architecture

**Branch:** `combat-rework`
**Date authored:** 2026-07-06
**Author of this handoff:** prior AI session (design + wiki); user drove the philosophy
**For:** the next AI implementing this
**Read first:** `wiki/index.md`, then `wiki/pages/self-observing-architecture.md`, then this file end-to-end.

---

## Ban list (non-negotiable, per user directive)

This project runs a strict style. Interpret aggressively; when in doubt, read `wiki/pages/coding-standards.md`.

- **No** `Any`, `object` cast, `type: ignore`, `.pyi`, `noqa`, `TYPE_CHECKING`.
- **No** fallbacks, best-effort recovery, `try/except` in core logic. Exceptions PROPAGATE. Boundary validation only, at parse/decode edges.
- **No** back-compat shims, wrappers, legacy code, deprecated re-exports, type aliases (`X = int`).
- **No** mocks in tests. Use the `_test_hooks.py` DI pattern (see `feedback_di_save_and_restore_not_monkeypatch.md` in `.claude/projects/.../memory/`).
- **No** monkeypatching (guard-banned).
- **100%** statement + branch coverage. All tests exercise real code paths.
- **Google-style docstrings** with `Args:` / `Returns:` / `Raises:` sections.
- Every `TypedDict` requires `encode_*` / `decode_*` functions using `platform_core.json_utils.require_*` validators.
- Every module (`services`-style) requires a `_test_hooks.py` where DI applies.
- No weak assertions (`assert x is not None`, `assert x`, `assert len(x) > 0`, etc.). Use `if x is None: raise AssertionError(...)`.
- Files under 400 lines where possible. **No monoliths** — clear separation of concerns per module.

---

## The philosophy that drives every design decision

**Fail hard on state entry, not soft on state observation.**

- The bot cannot enter a wrong state. State transitions have **contracts**. Contract violations raise on the transition, not after N observations of the consequences.
- No repeated-failure detection. No "wait until N tries then complain." No anomaly thresholds.
- Races vs bugs are distinguished structurally: if the planner had the evidence and used it correctly, a same-tick external change is a race → replan gracefully. If the planner should have checked but didn't → contract violation, raise.
- Every mutation, every decision, every dispatch is gated by a contract. Contracts are first-class.

The 2026-07-06 20:47:31 deadlock (26 s of client-side self-rejections, zero wire dispatches, silent) is the motivating incident. Under this architecture the deadlock is a `ShootTargetNotTrackedError` at tick 1, session exits with a specific error naming the planner-generated command, human fixes the planner. No retry counter, no anomaly detector, no gradual degradation.

---

## What the bot is missing today (the fifteen)

Each item has the same shape: **missing** / **present** / **today**. Full table lives in `wiki/pages/self-observing-architecture.md`. Summarized here for context:

| # | Item | Status | Layer |
|---|---|---|---|
| 1 | Standardized WHAT (action outcomes) | ⚠️ partial | Ledger |
| 2 | Standardized WHY (decision reasoning) | ❌ | Ledger |
| 3 | Predictions + accuracy | ❌ | Decision |
| 4 | Alternatives considered | ❌ | Decision |
| 5 | Confidence / uncertainty | ❌ | Facts + Decision |
| 6 | Provenance for facts | ⚠️ containers only | Facts |
| 7 | Per-entity memory | ❌ | Memory |
| 8 | Cross-session persistence | ❌ | Memory |
| 9 | Causal chain | ❌ | Ledger |
| 10 | Anomaly detection | REJECTED — replaced by contracts | Contracts |
| 11 | Mode transitions | ⚠️ free-text | Ledger |
| 12 | Time budgets | ❌ | Reasoning |
| 13 | Self-model | ❌ | Memory |
| 14 | Comparative baselines | ❌ | Memory |
| 15 | Feature-flag experiments | ❌ | Reasoning |

---

## The four-layer architecture

```
┌────────────────────────────────────────────────────────────────┐
│ Layer 4: MEMORY (long-lived, cross-tick, cross-session)       │
│  - Per-entity behavior models (Yuppler marathons; Kirby holds) │
│  - Per-map facts (Field01 fuel-dot decay rate, hot zones)     │
│  - Session-index for baselines                                 │
│  - Feature-flag config store                                   │
└────────────────────────────────────────────────────────────────┘
                            ▲    │
                            │    ▼ read/write
┌────────────────────────────────────────────────────────────────┐
│ Layer 3: LEDGER (per-attempt, in-session)                     │
│  - Decisions: reason_kind, reason_context, alternatives,       │
│               prediction, confidence, evidence_facts,          │
│               time_budget                                       │
│  - Outcomes: kind, context, matched_prediction, caused_by      │
│  - Ring buffer: in-memory, planner-queryable per action_kind   │
│  - Causal chain: DAG of events                                 │
│  - Mode transitions as first-class events                      │
└────────────────────────────────────────────────────────────────┘
                            ▲    │
                            │    ▼ produce/consume
┌────────────────────────────────────────────────────────────────┐
│ Layer 2: DECISION ENGINE (per-tick)                           │
│  - Planners consume Facts + Memory + Ledger.recent + SelfModel │
│  - Emit Decisions with evidence, predictions, alternatives,    │
│    confidence, time budgets                                     │
│  - Contracts on every decision produced                        │
└────────────────────────────────────────────────────────────────┘
                            ▲    │
                            │    ▼ read
┌────────────────────────────────────────────────────────────────┐
│ Layer 1: FACTS (world model with metadata)                    │
│  - Every fact: value + source + observed_ms + confidence +     │
│                provenance_chain                                 │
│  - Contracts: no unsourced facts, confidence bounded, etc.     │
└────────────────────────────────────────────────────────────────┘
                            ▲
                            │ construct from
                        WIRE + LOGS
```

**Cross-cutting: `contracts/`** — `ContractError` base + `@enforce_contract` decorator + guard rule scanning for public mutations that skip enforcement.

---

## Module layout

Each directory follows the same convention: `__init__.py` re-exports the public surface, `_types.py` holds shared types, `_contracts.py` holds layer invariants, `_test_hooks.py` holds DI seams, per-kind files hold the specifics.

```
src/tankpit_bot/
    contracts/
        __init__.py
        base.py                     # ContractError hierarchy
        enforcement.py              # @enforce_contract decorator + registry
        _test_hooks.py

    facts/                          # Layer 1
        __init__.py
        fact.py                     # class Fact[T]
        source.py                   # FactSource Literal
        provenance.py               # ProvenanceChainDict
        confidence.py               # Confidence[0,1] arithmetic
        _contracts.py
        _test_hooks.py

    memory/                         # Layer 4
        __init__.py
        entity/
            __init__.py
            record.py               # EntityMemoryDict
            behavior_model.py       # BehaviorModelDict
            engagement_history.py   # EngagementRecordDict list
        map/
            __init__.py
            record.py               # MapMemoryDict
            hot_zones.py
            fuel_dot_decay.py
        session/
            __init__.py
            start_state.py          # SessionStartDict
            index.py                # runs_index.tsv writer/reader
            self_model.py           # SelfModelDict aggregate
        store/
            __init__.py
            jsonl.py                # append-only writes
            sqlite.py               # indexed reads
        _contracts.py
        _test_hooks.py

    ledger/                         # Layer 3
        __init__.py
        types/
            __init__.py
            outcome.py              # ActionOutcome Literal
            reason.py               # ReasonKind base
            attempt_id.py           # AttemptId counter
            event_id.py             # EventId counter
        decision/
            __init__.py
            record.py               # DecisionDict
            alternatives.py         # RankedAlternativeDict
            prediction.py           # PredictionDict
            evidence.py             # EvidenceRef
        outcome/
            __init__.py
            _emit.py                # core _emit + attempt_id counter
            scan.py                 # Scan outcome types + emit helpers
            move.py
            teleport.py
            collect.py
            map_open.py
            shoot.py
        ring/
            __init__.py
            per_kind.py             # ring buffer per ActionKind
            queries.py              # PlannerLedgerView
        causal/
            __init__.py
            chain.py                # DAG
            queries.py              # trace_backward/forward
        mode_transition/
            __init__.py
            record.py
        _contracts.py
        _test_hooks.py

    reasoning/                      # Cross-layer reasoning primitives
        __init__.py
        prediction_check.py         # matched_prediction: bool + delta
        confidence.py               # combine/decay/threshold
        alternatives_gen.py         # ranking helpers
        time_budget.py              # Budget tracking, expiry raises
        experiment/
            __init__.py
            flag.py                 # FeatureFlagDict
            cohort.py               # deterministic cohort selection
        _test_hooks.py

    bot/                            # Existing surface, refactored to use above
        ai/                         # Planners produce structured Decisions
        ...                         # (existing files, incrementally migrated)
```

---

## Phase 0: Immediate deadlock fix (do first, ship independently)

**Motivation:** the live-run 2026-07-06 20:47:31 deadlock left the bot stuck for 26 s. The full architecture is a multi-month project; this phase closes the specific bug now.

**Scope:** ~200 LoC changed. Ships before Phase 1 lands.

### The bug

`bot/executor.py::_is_valid_shoot` has a position-match clause that self-rejects any shoot command whose `target_x/y` doesn't match the target tank's registry position. Under the `_clamp_aim_into_viewport` mechanic (`bot/ai/combat_strategy.py:43`), the aim tile is deliberately different from the target's registry position — the server picks homing from `target_id` and the seeker tracks the true target. The position clause treats this deliberate drift as a stale-target race and blocks every clamped homing shot.

Wire-verified: at 20:47:29 the bot dispatched `shoot(172,107, id=512)` — aim was viewport-legal because purple-4 was still at (172,107) at planner tick. Between dispatch and server-resolve, purple-4 teleported to (166,120). Server accepted the aim (viewport-legal), picked homing (target_id + off-adjacent), homing tracked purple-4 to (166,120). Hit at 20:47:31. That is the exact mechanic the clamp exists to trigger — server checks the AIM tile, then homing seeks via `target_id` wherever the target actually is.

### The fix

**File:** `src/tankpit_bot/bot/executor.py`

Delete lines 274-283 (the position-match clause):

```python
if tank["x"] != command["target_x"] or tank["y"] != command["target_y"]:
    emit_ai(
        "rejecting shoot at (%d,%d): target id=%d moved to (%d,%d)",
        command["target_x"],
        command["target_y"],
        command["target_id"],
        tank["x"],
        tank["y"],
    )
    return False
```

Keep the tank-existence clause (`tank is None` → return False). That one is a real race guard; if the tank vanishes from the registry between planner-decide and dispatch, we can't shoot it and the wire will crash us if we try.

Update the docstring to explain: `target_id` is the truth channel; aim is a viewport-legal hint the server uses to route to homing. Aim ≠ tank position is intentional under the clamp mechanic.

### Test

**File:** `tests/bot/test_executor.py`

Add a test that a clamped shoot command (aim ≠ tank position, tank exists) dispatches through `_is_valid_shoot` returning True. The existing test verifying "target_id not tracked → False" stays. Delete (or invert) any test asserting position-mismatch → False.

### Wiki

**File:** `wiki/pages/bot-behavior-contract.md` §3.3

Update the "Viewport-clamped aim (2026-07-03)" row to note that the executor position-check has been retired and drift is intentional. The 20:47:31 deadlock scenario is now documented in the anti-patterns section.

**File:** `wiki/log.md`

Append entry: `## [YYYY-MM-DD] fix | Retire executor position-match for shoot; unblock clamped homing pursuit (Phase 0 of self-observing architecture)`.

### Verification

`make check` green. Test the fix live via `make run` — the previous 20:47:31 scenario should now produce actual wire dispatches with `OUR_SHOT weapon=3` events (or `command_error 0x52 code 0` if the server refuses, in which case the existing `combat_feedback == "rejected"` handling in `bot/ai/combat_strategy.py:597` fires).

---

## Phase 1: Contracts + Facts foundation

**Scope:** ~800 LoC. Blocks all subsequent phases; land this before starting Phase 2.

### `contracts/`

- `contracts/base.py` — `ContractError` base class hierarchy:
  ```
  class ContractError(Exception):
      contract_name: str
      violated_at: str        # file:line
      violation_details: ContractViolationDict
  ```
- `contracts/enforcement.py` — `@enforce_contract(SpecificContract())` decorator + module-level `require(condition, error_class, **details)` helper.
- Every specific contract subclasses `ContractError` and names itself: `ShootTargetNotTrackedError`, `NoUnsourcedFactError`, `DecisionMissingReasonError`, `TimeBudgetExpiredError`, `PredictionModelDriftError`, `MissingOutcomeError`, `EntityIdStableViolationError`, `ConfidenceOutOfBoundsError`, `LedgerInvariantError`, etc.
- Guard rule at `scripts/guard/contract_rules.py` — scans `facts/`, `ledger/`, `memory/` for public mutation functions (name matches `apply_*`, `record_*`, `mutate_*`, `set_*`, `update_*`) that lack `@enforce_contract`. Fails `make lint`.

### `facts/`

Replace ad-hoc world-state field access with `Fact[T]`:

```python
class Fact[T](TypedDict):
    value: T
    source: FactSource
    observed_ms: int
    confidence: float
    provenance: ProvenanceChainDict
```

- `facts/source.py` — `FactSource = Literal["wire_0x2E_tank_status", "wire_0x5A_viewport_patch", "wire_0x43_cache_update", "wire_0x4F_radar_response", "wire_0x4C_map_data", "wire_0x3D_movement", "wire_0x41_deactivation", "wire_0x53_shoot_event", "wire_0x52_supervisor", "game_log_scrape", "client_side_inference"]`.
- `facts/provenance.py` — `ProvenanceChainDict` = origin + list of derived-from `SourceRefDict`. Contract: every derivation must reference at least one prior source.
- `facts/confidence.py` — `Confidence[0.0, 1.0]` arithmetic (combine independent, weighted combine, exponential decay by age). Contract: `ConfidenceOutOfBoundsError` at any operation producing out-of-range.

**Contracts (`facts/_contracts.py`):**
- `NoUnsourcedFactContract` — every Fact construction requires source, observed_ms, confidence, provenance. Raise on missing.
- `ConfidenceInBoundsContract` — [0.0, 1.0] enforced at Fact construction and confidence operations.
- `ProvenanceRootednessContract` — non-derived Facts must have a wire-originating source; derived Facts must cite prior sources.

**Migration strategy for existing world state:**

The bulk of world state is in `WorldStateDict` and its constituents (`SelfStateDict`, `TankStateDict`, `ContainerStateDict`, etc.). Migration is incremental:

- Phase 1a: introduce `Fact[T]` types alongside existing raw types. New code uses `Fact[T]`, existing code untouched.
- Phase 1b: retrofit `ContainerStateDict` (already has `source` + `refresh_kind` + `timestamp_ms` — closest to Fact-shaped) to be a `Fact[ContainerValueDict]`.
- Phase 1c: retrofit `TankStateDict` next.
- Phase 1d: `SelfStateDict`, `MineStateDict`, terrain, viewport.

Each substep is a self-contained refactor; `make check` stays green throughout.

### Tests

Every new type gets encode/decode via `platform_core.json_utils.require_*`. Round-trip tests. Contract violation tests: every `raise` path is exercised by a test that constructs the failing state and asserts the specific error.

### Verification gate

`make check` green with 100% coverage on `contracts/` and `facts/`. Guard rule passing. No regressions in existing tests.

---

## Phase 2: Ledger core

**Scope:** ~2500 LoC. Enables the ledger-derived features in Phases 3-5.

### The lift

Three parallel diagnostic mechanisms exist today for action completion:

- `runtime_logging.emit_wire_complete(action_kind, signal, duration_ms)` — used by scan/move/teleport/collect/map_open completions (`bot/completions.py`, `bot/tick_loop_actions.py`)
- `diagnostics/teleport_attempts.emit_teleport_attempt_outcome(status)` — used by teleport HFSM completion path
- `bot/tick_loop._get_combat_feedback` — returns `"hit"/"miss"/"rejected"/""` for shots; not a diagnostic

Three names, three shapes, no shared contract. The 20:47:31 deadlock hid because client-side discards emitted only `emit_ai(...)` free text — the fourth channel that no consumer knew existed.

Consolidate into `ledger/outcome/` with one diagnostic kind (`action_outcome`) split into per-kind files:

- `ledger/outcome/scan.py` — 3 outcomes: `scan_radar_complete`, `scan_stall_timeout`, `scan_command_rejected`
- `ledger/outcome/move.py` — 5 outcomes including `move_discarded_unwalkable` (new, was silent)
- `ledger/outcome/teleport.py` — 5 outcomes including `teleport_discarded_unwalkable`
- `ledger/outcome/collect.py` — 5 outcomes including `collect_discarded_no_container`
- `ledger/outcome/map_open.py` — 3 outcomes
- `ledger/outcome/shoot.py` — 5 outcomes including `shoot_discarded_no_tank`

Each per-kind file:
- Its own `<Kind>Outcome` Literal narrowed union
- Per-outcome `TypedDict` for the event payload (strict fields per outcome, no sentinels for unrelated fields)
- Per-outcome `emit_<kind>_<outcome>(...)` helper with strictly typed required args
- `_types.py` in `ledger/outcome/` re-exports the `ActionOutcome` union of all six per-kind unions

**Attempt IDs:** monotonic per `ActionKind`, incremented by `_emit.py::_next_attempt_id(action_kind)`. Reset via `reset_action_outcome_tracking()` in the shared test-isolation fixture.

### Decisions

`ledger/decision/record.py` defines `DecisionDict`:

```
class DecisionDict(TypedDict):
    event_id: EventId              # monotonic across all events
    tick_id: int
    action_kind: ActionKind
    command: BotCommand            # the wire command
    reason_kind: ReasonKind        # per-action-kind literal, one of the reason enums
    reason_context: ReasonContextDict  # per-reason-kind typed dict
    alternatives: list[RankedAlternativeDict]  # even if 1-long
    prediction: PredictionDict
    confidence: float
    evidence_facts: list[EvidenceRef]  # FactIds the decision relied on
    time_budget_ms: int            # 0 if none; positive if bounded
    caused_by: list[EventId]       # prior events this decision responds to
```

Contract: `DecisionCompletenessContract` at every planner-return. Missing `reason_kind`, `prediction`, `alternatives`, `evidence_facts` → `DecisionIncompleteError`.

**Reasoning enums per action kind live in `ledger/decision/reason/`:**

```
ledger/decision/reason/
    _types.py                      # base ReasonKind type + registry
    scan.py                        # ScanReasonKind + per-reason context TypedDicts
    move.py
    teleport.py                    # e.g. teleport_dot_relay, teleport_combat_landing_close
    collect.py                     # e.g. collect_lock_continuation, collect_opportunistic_viewport
    map_open.py
    shoot.py                       # e.g. shoot_cardinal_dual, shoot_engaged_stay_put,
                                   #      shoot_engaged_off_viewport, shoot_fresh_acquire
```

Planners produce `Decision` via typed constructors. `emit_ai("reason=...")` free-text is deleted; `reason` field of `BehaviorScoreDict` is replaced by `reason_kind` (Literal) + `reason_context` (TypedDict).

### Outcomes

At outcome time, resolve the `Decision.event_id` via `caused_by` and record:

```
class OutcomeDict(TypedDict):
    event_id: EventId
    tick_id: int
    action_kind: ActionKind
    outcome: ActionOutcome
    outcome_context: OutcomeContextDict  # per-kind payload
    matched_prediction: bool             # derived from Decision.prediction vs actual
    caused_by: list[EventId]             # includes the Decision.event_id
```

Contract: `OutcomeInvariantContract` — every planner-produced `Decision.event_id` must have exactly one downstream `OutcomeDict` referencing it in `caused_by`. Session-end sweep raises `LedgerInvariantError` on any orphan Decision.

### Ring buffer

`ledger/ring/per_kind.py` — bounded ring per `ActionKind`, size 128 (config). Every outcome append appends to its kind's ring. `PlannerLedgerView.recent(kind, n)` returns the last N (Decision, Outcome) joined records. `by_reason(kind, reason_kind)` filters.

### Causal chain

`ledger/causal/chain.py` — DAG of Events. Every event has `event_id` + `caused_by: list[EventId]`. `chain.trace_backward(event_id)` returns the full causal path back to root wire events. `chain.trace_forward(event_id)` returns descendants. Contract: `CausalChainRootednessContract` — every event either roots at a wire signal or references a prior event.

### Mode transitions

`ledger/mode_transition/record.py` — every mode change is a first-class event:

```
class ModeTransitionDict(TypedDict):
    event_id: EventId
    tick_id: int
    from_mode: BehaviorMode
    to_mode: BehaviorMode
    reason_kind: ModeTransitionReasonKind
    reason_context: ModeTransitionReasonContextDict
    caused_by: list[EventId]
    evidence_facts: list[FactId]
```

Every `bot/ai/ai_strategy.py::decide` returning a decision that flips `mode` writes a ModeTransition record. Contract: `ModeTransitionMustHaveReasonContract`, `ModeTransitionMustCiteEvidenceContract`.

### DecideCtx exposure

`bot/ai/context.py::DecideCtx` gains typed views:

```
class DecideCtx:
    # existing fields...
    facts: FactsView       # from facts/
    memory: MemoryView     # from memory/
    ledger: LedgerView     # from ledger/
    reasoning: ReasoningPrimitives  # from reasoning/
```

Every planner mode owner reads from these views. `ctx.ledger.recent(action_kind="shoot", n=5)` replaces implicit state inference.

### Migration

Files touched:
- `runtime_logging.py` — delete `emit_wire_complete`
- `bot/completions.py` — replace 5 `emit_wire_complete` calls with per-kind `emit_*_*` helpers
- `bot/tick_loop_actions.py` — replace 3 `emit_wire_complete` calls
- `diagnostics/teleport_attempts.py` — DELETE. Migrate `record_teleport_dispatch` + `emit_teleport_attempt_outcome` into `ledger/outcome/teleport.py`
- `bot/tick_loop.py::_get_combat_feedback` — emit `ledger/outcome/shoot.py` alongside existing `combat_feedback` state
- `bot/executor.py::_is_valid_*` guards — replace `return False` with `raise <Specific>Error` (the four discard outcomes become contract violations)
- `bot/ai/*.py` — planners produce `DecisionDict` with all required fields. `TickDecisionDict.behavior.reason` (free text) is deleted; `reason_kind` (Literal) + `reason_context` (TypedDict) replace it.
- `session_scorecard.py` + `issue_report.py` — consume the unified `action_outcome` diagnostic; per-outcome counters replace ad-hoc hit/miss/reject.

### Contracts

- `LedgerInvariantContract` — every planner-produced Decision has exactly one matching Outcome by session end.
- `AttemptIdMonotonicContract` — per-kind attempt counter increments strictly by 1 per emit.
- `EventIdMonotonicContract` — event_id strictly monotonic across all events.
- `DecisionCompletenessContract` — every field required.
- `CausalChainRootednessContract` — every event rooted or referenced.

### Tests

- Per-outcome emit tests (26 outcomes × their required fields).
- Invariant tests: for each action kind, produce one Decision through the executor and wire path, assert exactly one Outcome fires, assert attempt_id increments, assert causal_by references the Decision.
- Contract violation tests: every contract's raise path is exercised by a test that constructs the failing state and asserts the specific error type.

### Verification gate

`make check` green with 100% coverage. All 26 outcomes tested. Every contract's raise tested. Grep sweep for `emit_wire_complete`, `emit_teleport_attempt_outcome`, `combat_feedback` classifier returns — zero hits in production code (some may remain in wiki/docs as historical references).

---

## Phase 3: Decision enrichment

**Scope:** ~1500 LoC. Adds the WHY-side observability. Requires Phase 2 (Decision/Outcome records).

### Predictions (item #3)

`ledger/decision/prediction.py`:

```
class PredictionDict(TypedDict):
    predicted_outcome: ActionOutcome
    predicted_specifics: PredictedSpecificsDict  # per-kind narrowed
    prediction_confidence: float

# Per-kind predicted specifics live in ledger/decision/reason/<kind>.py
class ShootPredictedSpecificsDict(TypedDict):
    expected_weapon: Literal["dual", "homing"]
    expected_hit: bool
    expected_elapsed_ms_min: int
    expected_elapsed_ms_max: int
```

Every `DecisionDict.prediction` populated by the planner. At outcome time, `reasoning/prediction_check.py::matched_prediction(decision, outcome) -> bool` computes match; result stored on `OutcomeDict.matched_prediction`.

Contract: `PredictionRequiredContract`. Contract: `PredictionModelDriftError` raised when `prediction_accuracy_by_reason[reason_kind]` drops below configured threshold across last N attempts (this is the ONE anomaly-like check, and it fires at the exact moment the model is empirically broken, not via arbitrary N=4 counting).

### Alternatives (item #4)

`ledger/decision/alternatives.py`:

```
class RankedAlternativeDict(TypedDict):
    action_kind: ActionKind
    target_id: int
    target_x: int
    target_y: int
    score: float
    ruled_out_reason: RuledOutReasonKind
```

Every `DecisionDict.alternatives: list[RankedAlternativeDict]` — chosen is index 0, others ranked with structured `ruled_out_reason`. Contract: `AlternativesRequiredContract` — even 1-long list required (single-candidate is a valid case; the list documents "there was only one option").

Planner mode owners populate: `bot/ai/threats.py::analyze_threats` and `bot/ai/combat_strategy.py::select_new_combat_target` already do implicit ranking; they now emit structured ranking as part of the Decision.

### Confidence on decisions (item #5)

`DecisionDict.confidence: float`. Contract: `ConfidenceInBoundsContract`. Confidence combines from `evidence_facts` — the planner declares "my confidence in this decision" and cites the facts. `reasoning/confidence.py::combine(*facts)` produces default combined confidence; planner can override.

Low-confidence decisions can trigger verification actions (e.g., extra radar before commit). Threshold logic lives in each mode owner.

### Time budgets (item #12)

`reasoning/time_budget.py`:

```
class TimeBudgetDict(TypedDict):
    budget_kind: TimeBudgetKind      # "engagement", "acquisition", "hop_planning"
    started_ms: int
    max_ms: int
    scope_key: str                    # "target=512" or "mode=COLLECT"
```

`DecisionDict.time_budget_ms` = remaining ms at decision time. Zero = unbounded. Positive = enforced.

Contract: `TimeBudgetExpiredError` raises when a Decision fires against an expired budget for its scope. Enforcement in `bot/tick_loop.py::_tick_once` before executor dispatch.

Budgets registered via `reasoning/time_budget.py::start_budget(kind, scope_key, max_ms)`. Registered budgets survive across ticks; expiry is checked at every Decision produced for the matching scope.

### Contracts

- `PredictionRequiredContract`
- `AlternativesRequiredContract`
- `DecisionConfidenceRequiredContract`
- `TimeBudgetMonotonicityContract` — budgets can only extend, never shrink retroactively
- `TimeBudgetExpiredError` — raise when Decision fires past expiry

### Tests

- Every planner mode owner produces Decisions with all four enrichment fields; missing fields raise the specific contract error.
- `matched_prediction` computation tests per outcome.
- Time budget expiry test drives a repeated-engagement scenario past the budget and asserts `TimeBudgetExpiredError`.

### Verification gate

`make check` green with 100% coverage on all new fields and contracts. Every planner branch exercised for missing-field violations.

---

## Phase 4: Memory

**Scope:** ~2500 LoC. Adds long-lived beliefs. Requires Phase 1 (Facts) + Phase 2 (Ledger events).

### Provenance for facts (item #6)

Already scaffolded in Phase 1 via `Fact[T].provenance`. Phase 4 extends provenance to every world-state field, not just `ContainerStateDict`.

Complete migration of `TankStateDict`, `SelfStateDict`, `MineStateDict`, `TerrainMapProtocol` cells, and `ViewportStateDict` to `Fact[T]`-wrapped shapes. Every wire decoder (`sniffer/world_state_*.py`) constructs Facts with correct source.

### Per-entity memory (item #7)

`memory/entity/`:

```
class BehaviorModelDict(TypedDict):
    teleport_frequency_per_min: float
    marathon_score: float           # tendency to marathon under attack
    hold_ground_score: float        # tendency to stand and fight
    friendly_calls_score: float     # summons color-team backup?
    equipment_richness_score: float

class EngagementRecordDict(TypedDict):
    started_ms: int
    ended_ms: int
    ended_reason: Literal["killed", "escaped", "we_disengaged", "we_died"]
    our_shots_fired: int
    our_shots_hit: int
    fuel_spent: int
    causal_root_event_id: EventId   # links to ledger's causal chain

class EntityMemoryDict(TypedDict):
    entity_id: int                   # tank_id (stable across ticks in session)
    entity_kind: Literal["player", "bot"]
    display_name: str
    first_seen_ms: int
    last_seen_ms: int
    observation_count: int
    behavior_model: BehaviorModelDict
    engagement_history: list[EngagementRecordDict]
    kill_by_us_count: int
    death_at_our_hands_count: int
```

Updated tick-by-tick from observed behavior. Serialized to per-session then aggregated at session end into cross-session store.

Contract: `EntityIdStableContract` — same `entity_id` + `display_name` across ticks in a session is the same entity. Mid-session divergence raises `EntityIdStableViolationError`.

**AI usage:** `DecideCtx.memory.entity(id=512)` returns the current `EntityMemoryDict`. Planners use `.behavior_model.marathon_score` to influence acquisition ranking.

### Cross-session persistence (item #8)

`memory/store/`:

- `jsonl.py` — append-only writes. Session end appends per-entity + per-map diffs.
- `sqlite.py` — indexed reads. On session start, `store.load(map_id=map_id)` populates the in-session cache.

Contract: `PersistenceIntegrityContract` — every write is transactionally consistent (SQLite write-ahead log). Partial writes at process kill raise `PersistenceIntegrityError` on next load; startup requires manual reconciliation.

**Location:** persistent store under `runs/memory/` (git-ignored) — per-map subdirs holding SQLite files.

### Mode transition storage (item #11)

Already scaffolded in Phase 2. Phase 4 additionally aggregates: `SelfModelDict.mode_transition_rates_per_min` computed from ModeTransition history.

### Session start/end (item #14, foundation)

`memory/session/start_state.py`:

```
class SessionStartDict(TypedDict):
    session_id: str
    started_ms: int
    map_id: int
    map_image: str
    rank: int
    fuel_at_ready: int
    dual_shots: int
    missile_shots: int
    homing_shots: int
    extra_radars: int
    armor_shields: int
    self_x: int
    self_y: int
    active_feature_flags: dict[str, str]  # from Phase 5
```

Captured on first ready tick of every session. Written to `memory/session/index.py` for cross-session lookup.

Contract: `SessionStartCompletenessContract` — every field required. Session artifacts without a start state raise on load.

### Tests

- Provenance migration: every wire decoder constructs Facts with the correct source; retrofitted fields round-trip through encode/decode.
- Entity memory: tick-by-tick behavior model updates verified against known enemy patterns.
- Cross-session persistence: write/read round-trip, transactional consistency on simulated crash.
- Session start capture: every field populated on first ready tick.

### Verification gate

`make check` green with 100% coverage. Persistent store tests use a scratch directory (per `_test_hooks.py`). No dependency on network or filesystem beyond scratch.

---

## Phase 5: Aggregation + self-observation

**Scope:** ~1500 LoC. Turns Memory into a self-improving system.

### Self-model (item #13)

`memory/session/self_model.py`:

```
class SelfModelDict(TypedDict):
    hit_rate_by_range: dict[int, float]
    hit_rate_by_weapon: dict[Literal["dual", "missile", "homing"], float]
    hit_rate_by_reason_kind: dict[ShootReasonKind, float]
    teleport_landed_exact_rate: float
    teleport_landed_exact_rate_by_reason: dict[TeleportReasonKind, float]
    prediction_accuracy_by_reason: dict[ReasonKind, float]
    time_to_first_kill_median_ms: int
    per_mode_dispatch_counts: dict[BehaviorMode, int]
    per_action_kind_success_rate: dict[ActionKind, float]
```

Populated at session start from cross-session store. Updated in-session as ledger fills. Contract: `SelfModelConsistencyContract` — rates sum to samples consistently.

**AI usage:** planners consult `ctx.memory.self_model.hit_rate_by_reason_kind[ShootReasonKind.shoot_engaged_off_viewport]` to decide whether to attempt or teleport-close.

### Comparative baselines (item #14)

`memory/session/index.py`:

- Cross-session index of `(SessionStartDict, SessionEndDict, per-kind aggregate outcomes)`.
- Query: `baseline(start_signature) -> BaselineDict` — historical median performance for sessions with the same starting signature.
- Session in-progress: compare current session's metrics against baseline; emit `session_vs_baseline` diagnostic every N ticks.

Comparability without a reset routine is achieved via **normalization** on `(rank, map_id, starting-inventory-band)`. Reset routine (bot ends session at anchor tile with full fuel + baseline inventory) is documented as Phase 5b (optional).

### Feature-flag experiments (item #15)

`reasoning/experiment/`:

```
class FeatureFlagDict(TypedDict):
    flag_id: str
    variants: tuple[str, ...]
    cohort_assignment: Literal["hash_session_id", "hash_wire_seed"]

class ExperimentAssignmentDict(TypedDict):
    session_id: str
    flag_id: str
    assigned_variant: str
```

Deterministic cohort assignment at session start via `reasoning/experiment/cohort.py::assign(flag_id, session_id)`. Every Decision stamps active variants in a new `DecisionDict.active_flags` field. Post-hoc scorecard groups outcomes by variant.

Contract: `ExperimentAssignmentStableContract` — same session_id + flag_id must always produce the same variant. Non-determinism raises.

### Anomaly detection (item #10) — CONFIRMED REJECTED

Per user directive: no anomaly-based repeated-failure detection. All would-be anomalies are handled as contract violations at state entry. Explicitly excluded from this phase.

The one exception: `PredictionModelDriftError` (Phase 3) — this is NOT anomaly detection over consequences, it's a contract on the planner's self-model: "you claimed to know X, wire says you don't, your model is broken." Raise on evidence, not on retry counting.

### Tests

- Self-model derivation tests: given a fixture ledger, computed rates match expected values.
- Baseline comparison test: current-session metrics correctly compared against a fixture cross-session store.
- Feature flag determinism: same session_id + flag_id always produces the same variant.

### Verification gate

`make check` green with 100% coverage. Cross-session store tests use scratch directories. Full ledger-to-self-model pipeline exercised end-to-end.

---

## Ordering constraints

- Phase 0 is independent, ships first, closes the specific bug.
- Phase 1 blocks Phases 2-5 (every layer needs Facts + Contracts).
- Phase 2 blocks Phases 3-5 (Ledger records are the input to Decision enrichment, Memory aggregation, Self-model).
- Phases 3 and 4 can be developed in parallel after Phase 2. Phase 3 adds Decision fields (in-tick); Phase 4 adds Memory (across ticks).
- Phase 5 requires Phases 2, 3, 4 (aggregates from all three).

Timing (rough guide, not commitments): each phase is a multi-session commitment.

- Phase 0: 1 session (immediate fix + wiki + test)
- Phase 1: 2-3 sessions (contracts framework + Facts core + guard rule)
- Phase 2: 4-6 sessions (biggest lift; ledger migration is cross-cutting)
- Phase 3: 2-3 sessions (decision-field additions, mostly planner code)
- Phase 4: 3-5 sessions (memory persistence, per-entity models)
- Phase 5: 2-3 sessions (aggregation views)

Total: 14-21 sessions of focused work. If the user says "unlimited time, tokens, context," they mean multi-week engagement, not one afternoon.

---

## Verification gate (every phase)

- `make check` green: guard + ruff + mypy + tests + 100% statement + branch coverage
- No new banned symbols (grep sweep before commit)
- All contracts have raise-path tests
- Wiki updated for every phase: page under `wiki/pages/`, hub link updated, index count bumped, `wiki/log.md` entry
- Handoff document appended with lessons-learned + surprises encountered

---

## Ban list — repeated because it matters

- No `Any`, no `cast`, no `type: ignore`, no `.pyi`, no `noqa`, no `TYPE_CHECKING`
- No fallbacks, no best-effort, no `try/except` in core logic
- No back-compat shims, no wrappers, no legacy code, no type aliases (`X = int`)
- No mocks in tests
- No monkeypatching (guard-banned)
- No monolithic files — clear separation of concerns per module
- 100% statement + branch coverage
- Google-style docstrings with `Args:` / `Returns:` / `Raises:`
- Every `TypedDict` requires `encode_*` / `decode_*` using `require_*` validators
- Every module needs `_test_hooks.py` where DI applies
- No weak assertions; use `if x is None: raise AssertionError(...)`

Interpret aggressively. When in doubt, read `wiki/pages/coding-standards.md` and the CLAUDE.md files.

---

## Where the wiki tracks this

- `wiki/pages/self-observing-architecture.md` — vision, four layers, fifteen items, phase overview
- `wiki/pages/bot-behavior-contract.md` — updated §3.3 for the executor position-check retirement (Phase 0), §4.1 for the ledger invariant (Phase 2)
- `wiki/log.md` — one entry per phase landing, one entry for major architectural decisions
- `docs/handoffs/self-observing-bot-architecture.md` — this document; update with each phase's lessons

---

## What NOT to do

- Don't build any phase without landing Phase 0 first. The deadlock is a live bug; ship the one-line fix now.
- Don't add anomaly-based retry counters anywhere. Every would-be anomaly is a contract violation.
- Don't leave silent discards. Every executor guard that returns False must raise a specific error unless it's a documented race (see Phase 0 for the race vs bug distinction).
- Don't build the monolithic `action_outcome.py` I sketched in a prior session. Split from the start into per-kind files under `ledger/outcome/`.
- Don't split just for the sake of splitting. `_types.py` and `_emit.py` are shared; per-kind files own only their kind. Six kinds, six files, one shared core.

---

# Phase 0 addendum: seven bug patterns from live-run analysis (2026-07-06 22:37-22:40)

The 2026-07-06 22:37 `make run` produced a 4-minute session with 2 kills, 1 blocked target, and 20+ wasted teleports circling a live cardinal-adjacent enemy. Full log analysis distilled seven bug patterns that all belong in Phase 0. Each is 10-30 LoC, all deliverable alongside the position-check delete, all become fail-hard contract violations under the architecture from later phases.

The live-run mechanism (fully traced): the bot entered HUNT under-armed (16 duals, 4 homings vs cap 25/25 at private), engaged orange-4 (killed successfully because it was cardinal-adjacent, no clamp needed), dot-relayed to orange-8, teleported to (46,159), landed adjacent to orange-8 at (47,159) — and then for 26 seconds, instead of firing the free cardinal shot, the bot's COLLECT owner opportunistically dispatched fuel pickups on nearby containers, each pickup partial-transferred to overflow the tank at 1100 cap, the server code=5-rejected the *transferred* container, marked it `failed_pickup`, COLLECT exhausted, yielded to HUNT (because "fuel healthy"), HUNT re-teleported to (46,159) at −46 fuel per hop, repeat. Five cycles blacklisted five nearby fuel containers, burned ~230 fuel, and never fired the free shot. The loop broke only when every nearby fuel container was blacklisted and COLLECT literally had no local action, then HUNT ran and fired.

## The seven bugs

### Bug 0.1 — Executor position-check self-rejects clamped homing shots

The originally-scoped Phase 0 bug. `bot/executor.py::_is_valid_shoot` lines 274-283 reject any shoot command whose `target_x/y` doesn't match the tank's registry position. Under the clamp mechanic (`_clamp_aim_into_viewport`) that drift is intentional. Fix: delete the position-match clause; keep the tank-existence clause. See main Phase 0 section above.

### Bug 0.2 — `_select_and_pickup_fuel` doesn't predict overflow

**Symptom:** 4 consecutive `pickup_fuel` dispatches at fuel 1040/1054/1062/1054 (all within 46 of cap 1100). Each transfer filled to exactly 1100 and drew `code=5`. Each container marked `failed_pickup`.

**Root cause:** `bot/ai/collect_mode.py::_select_and_pickup_fuel` gates on `if ctx.fuel >= fuel_capacity(ctx.self_state["rank"]): return None` — strict inequality. At fuel=1054, gate passes, dispatch happens. Zero prediction of the transfer.

**Contract:** `PickupWouldOverfillContract` — refuse dispatch when the sum of current fuel + walk cost + minimum(container.volume, headroom) would exceed capacity. Formula:

```
def would_overfill(ctx, container) -> bool:
    cap = fuel_capacity(ctx.self_state["rank"])
    walk_cost = manhattan(bot_pos, container_pos) * FUEL_PER_WALK_TILE
    headroom = cap - ctx.fuel
    return ctx.fuel + walk_cost + min(container.volume, headroom) > cap
```

At fuel=1054, cap=1100, walk=1, container.volume=386: `1054 + 1 + min(386, 46) = 1101 > 1100` → refuse. Container untouched, unblacklisted.

Violation raises `PickupWouldOverfillError` at planner time.

### Bug 0.3 — `failed_pickups` conflates "empty" with "at-capacity race"

**Symptom:** Every container that partially transferred fuel then drew `code=5` was marked `failed_pickup=1`. Downstream filters treat these as blacklisted despite the containers holding hundreds of fuel each.

**Root cause:** `bot/tick_loop_actions.py::_clear_command_error` calls `increment_container_failed_pickups(...)` on ANY collect rejection. Empty containers (real reject, container gone) and at-capacity-race containers (partial transfer then reject) get the same treatment. The wire evidence to distinguish them exists — a same-tick fuel delta from the same coordinate proves the container is not empty — but the code doesn't discriminate.

**Contract:** `FailedPickupSemanticsContract` — the `failed_pickups` counter increments only when the pickup produced NO fuel/equipment delta. When a same-tick fuel gain from the same tile is present, the code=5 is an `at_capacity_race` outcome, not a `blacklisted_empty` outcome. Distinguish structurally:

```
class CollectRejectionKind(Enum):
    empty              # container truly gone; blacklist
    tank_full_race     # partial transfer + code=5; do NOT blacklist
    inventory_full     # code=7; blacklist for equipment
    illegal_geometry   # code=0; blacklist per position
```

Fires from the same 0x52 code=5 handler; the discriminator is "did fuel/inventory change in the same wire window."

### Bug 0.4 — "Yield to hunt at healthy fuel" ignores ammo state

**Symptom:** COLLECT yielded to HUNT while inventory was well below cap (duals 12/25, homings 3/25). HUNT then teleported the under-armed bot into an engagement that couldn't finish, exhausting duals and homings, hitting the stationary-miss classifier (bug 0.6), and blocking a live target.

**Root cause:** `bot/ai/collect_mode.py::decide_collect_mode` at the end of the cascade checks only fuel:

```python
if ctx.fuel > ctx.config["fuel_low_threshold"]:
    return None    # yield to hunt
```

The check was added 2026-07-06 for a different scenario (bot stuck in COLLECT with nothing to do at full fuel + full ammo). It fires here at full fuel + partial ammo, which is not the intended condition.

**Contract:** `HuntEntryFullInventoryContract` — mode transition to HUNT (or any yield-to-hunt gesture) is forbidden unless inventory is at cap.

```
def hunt_entry_permitted(ctx) -> bool:
    rank = ctx.self_state["rank"]
    cap = inventory_capacity(rank)
    radar_min = combat_radar_min(rank)
    return (
        ctx.inventory["dual_shots"]["count"]   == cap
        and ctx.inventory["homing_shots"]["count"] == cap
        and ctx.inventory["extra_radars"]["count"] >= radar_min
    )
```

Where the rank-derived constants live in `state/rank_formulas.py`:

```
def inventory_capacity(rank: int) -> int:
    """Main-map inventory cap per slot.

    Verified: recruit=20, +5/rank. See wiki/pages/game-rules.md.
    """
    return 20 + 5 * rank

def combat_radar_min(rank: int) -> int:
    """Minimum extra radars for combat-readiness.

    User rule: "5 below full" — inventory_capacity(rank) - 5.
    """
    return inventory_capacity(rank) - 5
```

Live values by rank:

| Rank | Cap/slot | HUNT-entry duals | HUNT-entry homings | HUNT-entry radars |
|---|---|---|---|---|
| Recruit (0) | 20 | 20 | 20 | 15 |
| Private (1) | 25 | 25 | 25 | 20 |
| Corporal (2) | 30 | 30 | 30 | 25 |
| Sergeant (3) | 35 | 35 | 35 | 30 |
| Lieutenant (4) | 40 | 40 | 40 | 35 |
| Captain (5) | 45 | 45 | 45 | 40 |
| Major (6) | 50 | 50 | 50 | 45 |
| Colonel (7) | 55 | 55 | 55 | 50 |
| General (8) | 60 | 60 | 60 | 55 |

Violation raises `HuntEntryInventoryTooLowError` at the yield-to-hunt decision. The mode selector is forbidden from picking HUNT until COLLECT restores inventory to cap.

### Bug 0.5 — HUNT re-teleports without checking cardinal-shot availability

**Symptom:** 5+ cycles where HUNT teleported to (46,159), landed at (47,159) with orange-8 at (46,159) cardinally adjacent, but COLLECT ran first and dispatched a fuel pickup. Free cardinal shot ignored on every landing.

**Root cause:** The mode selector's ordering runs COLLECT before HUNT unconditionally. When COLLECT has any actionable pickup (fuel_pickup in this case, even one that violates bug 0.2's contract), it dispatches. HUNT never gets the tick.

**Contract:** `KillPriorityContract` — the mode selector's first check every tick is "does the bot have a cardinal combat shot on a live wire-fresh target?" If yes → HUNT this tick, regardless of COLLECT candidates or inventory reserves. A free adjacent shot is worth more than any refill action; even a single dual advances the kill.

```
def mode_selector(ctx) -> Literal["COLLECT", "HUNT"]:
    if has_cardinal_combat_shot(ctx.self_state, live_threats):
        return "HUNT"    # override
    if hunt_entry_permitted(ctx) and enemies_visible():
        return "HUNT"
    return "COLLECT"
```

Note the ordering: cardinal shot override fires BEFORE the inventory-readiness check. Even under-armed, if the enemy is one tile away, take the shot.

Violation (mode=COLLECT while a cardinal shot is available) raises `KillPrioritySkippedError`.

### Bug 0.6 — Stationary-miss classifier fires on ammo-exhaustion single shots

**Symptom:** Twice in the run — orange-8 at 22:39:36 and purple-9 at 22:40:41 — the bot fired `weapon=0` (server-picked single, forced because dual/homing were exhausted or disabled), the classifier read `weapon=0 + stationary target = afterimage → block target`. Both times the target was a live tank at the aim tile.

**Root cause:** `bot/ai/combat_strategy.py::_combat_shoot`'s miss branch treats `weapon=0` unconditionally as "server resolved against empty ground because target isn't there." That's true when we HAD dual or homing enabled and the server still picked single (implying the target was gone). It's false when we had zero duals and zero homings enabled — the server routed to single because it was the only weapon left, and single missed for a range/damage reason, not an afterimage reason.

**Contract:** `StationaryMissClassifierContract` — the stationary-miss-block action requires that at the moment of the shot, at least one damaging weapon (dual or homing) was enabled and had inventory > 0. Without that precondition the `weapon=0 + stationary` outcome is `ammo_exhaustion_miss`, not `afterimage_confirmed`. The right response is to disengage and refill, not to block the target.

```
def stationary_miss_should_block(ctx, target) -> bool:
    if not (dual_available(ctx) or homing_available(ctx)):
        return False   # ammo_exhaustion_miss — do not block a live target
    if not target_stationary(target, prior_shot):
        return False
    return True
```

Violation (blocking a live target on ammo-exhaustion single) raises `WrongfulTargetBlockError`.

### Bug 0.7 — No equipment-hop path when inventory is under-armed and full fuel

**Symptom:** Bot at fuel=1100 (max), duals 12/25, homings 3/25, 13 tracked equipment containers in `world.containers` at other viewports, zero equipment in current viewport. Cascade yields to HUNT because "fuel healthy" and "no local equipment." Bot never teleports to a tracked equipment location.

**Root cause:** `bot/ai/resource_search.py::make_resource_search_hop` hops to fuel dots (from the 0x4C map_data atlas), not to tracked equipment containers. The `find_nearest_equipment` in `bot/ai/equipment_search.py` filters to the current viewport only. There's no function that returns equipment candidates across the whole tracked map, and no cascade step that consumes such a function.

Fuel dots exist because deposits produce yellow dots on 0x4C map_data. Equipment containers produce no map-dot signal — they're only revealed by radar. The two atlases are structurally different.

**Contract + missing capability:** `CollectMustProduceProductiveActionContract` — when COLLECT is entered (or continues), and no cascade step from the cascade-of-six produces a decision, the cascade MUST fall through to a shortfall-driven hop. If nothing productive exists on the tracked map (equipment nor fuel dot), the session exits with `no_productive_collect`.

New capability in `bot/ai/equipment_search.py`:

```
def find_all_tracked_equipment(world: WorldStateDict) -> list[ContainerStateDict]:
    """Every tracked equipment container, ignoring current viewport bounds.

    Used by the equipment-hop cascade step to pick a distant teleport target
    when local equipment is exhausted.
    """
    return [c for c in world["containers"].values() if not c["is_fuel"] and c["failed_pickups"] == 0]
```

New cascade step in `bot/ai/collect_mode.py`, inserted between forage and fuel-dot-hop:

```
def _hop_toward_equipment(ctx, base_state) -> TickDecisionDict | None:
    if _inventory_at_cap(ctx):
        return None
    candidates = find_all_tracked_equipment(ctx.world)
    if not candidates:
        return None
    target = _pick_nearest_teleport_reachable_affordable(ctx, candidates)
    if target is None:
        return None
    return teleport_decision_to(target.x, target.y, reason_kind="collect_equipment_hop")
```

**Reachability caveat** discovered during log analysis: `blocked_walk` (walk-reachability from current position) is not the same as teleport-reachability. A container at (128,126) surrounded by water is `blocked_walk` from (131,126) but reachable by teleport-landing on (128,127) — one tile east, on ground, adjacent to the container. The `_pick_nearest_teleport_reachable_affordable` helper must use teleport-landing-tile validity, not walk-graph connectivity.

**Confidence caveat:** a container tracked from a radar scan 5 minutes ago may already have been picked up by another player. No wire signal confirms distant container consumption. Under the self-observing architecture this becomes a `Fact[T].confidence` decay (item #5 of the fifteen); for Phase 0 the pragmatic version accepts stale-belief risk and pays the wasted-hop cost if the container is gone.

Violation (COLLECT cascade returns None while inventory below cap AND tracked equipment exists) raises `MissingCollectActionError`.

## Contract summary

| # | Name | Layer | Detects |
|---|---|---|---|
| 0.1 | (existing position-check delete) | Executor | Clamped homing self-rejection |
| 0.2 | `PickupWouldOverfillContract` | Decision Engine | Fuel pickup that would exceed cap |
| 0.3 | `FailedPickupSemanticsContract` | Ledger / mutations | Container blacklist from at-cap race |
| 0.4 | `HuntEntryFullInventoryContract` | Mode selector | HUNT while under-armed |
| 0.5 | `KillPriorityContract` | Mode selector | COLLECT while cardinal kill available |
| 0.6 | `StationaryMissClassifierContract` | Combat strategy | Block live target on ammo-exhaustion |
| 0.7 | `CollectMustProduceProductiveActionContract` | Collect cascade | Yield without exhausting equipment-hop path |

Plus the new `state/rank_formulas.py` entries:

- `inventory_capacity(rank) = 20 + 5 * rank`
- `combat_radar_min(rank) = inventory_capacity(rank) - 5`
- `DUALS_DISENGAGE_FLOOR = 5` (constant, rank-independent)
- `HOMINGS_DISENGAGE_FLOOR = 5` (constant, rank-independent)

Plus the new state field:

- `AIStateDict.engagement_paused_target_id: int` — set when a mid-combat disengage fires; HUNT resumes on this target after COLLECT restocks to cap. Distinct from `combat_target_id` (active lock) and `blocked_combat_targets` (blacklisted).

## Rough sizing

Each bug: 10-30 LoC change + 20-40 LoC tests. Phase 0 total (with the position-check delete): ~200 LoC production + ~350 LoC tests + wiki. One session's work for a focused AI.

Land all seven before starting Phase 1. Under the architecture from Phases 1-5 they eventually become their proper structural forms (contracts, invariants, ring-buffer-consumed diagnostics). Phase 0 lands them as focused patches to the current code with fail-hard error types.

## Container-tracking mechanics (from log audit)

Understanding `world.containers` matters for Bug 0.7. The dict accumulates across the session. Every radar scan adds containers via the 0x4F radar response and the extra-radar envelope (16x16 visible + 1-tile fringe = 18x18 reveal). Containers are removed when:

- Bot picks one up (`container_consumed` wire signal)
- Radar re-scans the container's viewport and the 0x4F response OMITS the container (omission-prune logic)

Containers are NOT removed when:

- Another player picks them up while we're elsewhere (no wire signal)
- Time passes (freshness TTL was retired 2026-07-06)

So `world.containers` accumulates "everything we've ever seen and haven't personally consumed or re-scanned as absent." Distant containers may be stale. Belief confidence decays over time in the target architecture; Phase 0 accepts this as-is.

Diagnostic breakdown of the `equipment` line (`collect_mode.py::describe_container_search`):

- `total` — count of `is_fuel=False` entries in `world.containers`
- `nearby` — of those, count inside `viewport_visible_bounds`
- `actionable` — of nearby, count that pass `is_collection_reachable_in_viewport` AND `failed_pickups == 0`
- `blocked` — of nearby, count that failed the reachability check (walk-blocked)
- `low_volume` — count below `minimum_volume` (equipment always has volume=0; this is a fuel-specific filter that reads as 0 for equipment)
- `nearest` — the closest tracked equipment with its blocking reason (or "none")

The Phase 0 equipment-hop path uses `total` (or a `find_all_tracked_equipment` walk of `world.containers` directly), not `nearby`. The current filter to `nearby` was designed for a different question ("what can I pick up without teleporting").

