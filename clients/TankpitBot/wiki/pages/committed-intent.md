---
title: Committed Intent
tags: [architecture, bot, planner, intent]
related:
  - "[[bot-behavior-contract]]"
  - "[[executor-rejection-loops]]"
  - "[[larder-plan]]"
  - "[[flag-triage-20260729]]"
source_paths:
  - "src/tankpit_bot/bot/ai/intent.py"
  - "src/tankpit_bot/bot/ai/collect_mode.py"
source_git_blobs:
  "src/tankpit_bot/bot/ai/intent.py": "74bcd19f55ef923ab28a40c02993d19fb7a7cacc"
  "src/tankpit_bot/bot/ai/collect_mode.py": "81c893b10c143da3c9db8f316d912c225454373b"
fact_checked: "2026-07-30"
confidence: high
hubs: [architecture]
---

# Committed intent: plans that survive the tick boundary

The bot re-derives its whole decision every ~2 s tick, so historically
nothing forced tick N+1 to finish what tick N started. Whenever an
input flickered (fuel drop changed a cost gate, a lock read "no longer
executable" for one tick, the map aged), the next derivation produced
a DIFFERENT plan[^1] — and the switch itself cost a tick, always at the
worst moment (under fire, mid-duel). The point-latches shipped through
2026-07-30 (break latch, clearance latch, F12 deferred-teleport
survival) are each this same law — "a decision must survive the tick
boundary" — applied at one measured flicker point.

This layer makes the law structural instead of per-incident[^1]. Ruling
receipt: the user's 2026-07-30 challenge — "i'm worried we're papering
over issues... not addressing the underlying uncertainty that the bot
had" — and the s8-2 flag (run bot-20260730-025337, 03:00:00): an
escape hop landed ON its locked equipment, and the next derivation
selected a fresh teleport TO THE TILE THE TANK WAS STANDING ON,
deferring a map open for a zero-distance jump, because nothing asked
whether the committed plan's purpose was already served.

## Design

One question opens every decision pass[^2]: **is there a committed plan,
and is it still valid?** If it completes right here, finish it — no
cascade, no re-scoring. Only when a plan completes or explicitly
invalidates does full re-derivation run, and every release is logged
with a specific reason so churn is measurable in the events stream
instead of silent.

The plan's persistent representation is state that already exists —
no new fields, no migration[^3]:

| Plan | Persistent fields | Owner module |
|---|---|---|
| Collect (harvest a container) | `resource_target_kind/x/y` + `suppress_landing_scan` | `bot/ai/intent.py` (phase 1, SHIPPED) |
| Hunt (close on / pursue a target) | `combat_target_id/x/y` + break latch | phase 2, open |
| Clearance (shoot mine, then collect) | `mine_clearance_aim_key/_shot_ms` | phase 2, open |

`bot/ai/intent.py` is the single owner of collect-plan SEMANTICS;
raw field mutation stays in `context.py` (`set_resource_target` /
`clear_resource_target`)[^3]. The API:

- `current_collect_plan(ai_state)` — the held plan as a typed
  `CollectPlanDict` (kind, target_x, target_y), or `None`.
- `plan_completes_here(plan, x, y)` — Manhattan ≤ 1 (the auto-pick
  reach, [[fuel-system]]): the continuation is a single action, so
  the plan must be finished, never re-derived away.
- `validate_collect_plan(ai_state, world)` — per-tick validity at
  `DecideCtx` construction (lifted from `normalize_resource_target`):
  target exists and is pursuable, else released with a reason.
- `release_collect_plan(ai_state, reason=...)` — the ONLY sanctioned
  release path. Emits a `plan_released` diagnostic every time a held
  plan is dropped.

## Release-reason vocabulary (closed)

`tank_at_capacity` (completion — the fuel plan's purpose is served),
`superior_candidate`, `not_executable`, `landing_scan_reset`,
`walk_for_fuel_override`, `target_gone`, `target_not_pursuable`,
`kind_invalid`[^4]. A new release site means a new documented reason in
`intent.PLAN_RELEASE_REASONS`, not an invented string[^4].

Churn query: count `diagnostic_kind=plan_released` per reason per
run[^5]. High `superior_candidate` counts = lock thrash; `not_executable`
clusters = the F6 passability family; anything at all during a fight
window = ticks spent re-planning under fire.

## Wired-in continuity (phase 1)

- **Under-fire escape** (`_escape_under_fire_decision`)[^6]: before any
  hop selection, a held plan that completes here is finished — the
  pickup IS the escape continuation (one action, no added exposure).
  This is the s8-2 fix at the root: the landing tick now serves the
  hop's purpose instead of re-deriving a self-teleport.
- **Own-ground gate** (`_hop_toward_equipment`): a landing equal to
  the current position is never travel (cost-0 candidates
  structurally win cost ranking, which is exactly how s8-2's
  self-teleport got selected). Declined candidates tally as
  `own_ground` in the `hop_declined` diagnostic.
- **Hold on transient inexecutability** (F22, found by this layer's
  own events within 40 minutes of shipping): the lock continuations
  hold the plan when `walk_or_teleport` returns None transiently
  (mid-map-open, momentary blockage, water-boxed awaiting a ferry)
  and release `not_executable` only on the server-confirmed
  move-failed mark. Run bot-20260730-032x ticks 361/366/371: three
  releases whose targets were re-locked and served 2-3 ticks later
  — the plan was never invalid, the executor was busy.

## Phase 2 (open, coordinate via [[flag-triage-20260729]])

- Hunt plans: the close/pursuit intent (combat lock + break latch)
  expressed through the same validity/reason shape — subsumes the
  s8-3 mid-duel `find_target` map open (F12 family) by making "the
  engaged in-view target IS the plan" explicit.
- Clearance plan: "clear mine at (x,y) then collect (x2,y2)" as one
  plan with two legs, replacing the aim-key latch.
- Supersede visibility: hop selectors overwrite a held lock via
  `set_resource_target` without a release event; route replacement
  through intent so plan HANDOFFS are also visible.

[^1]: [synthesis] — the motivating behaviour, recorded from live runs rather than measured in a test. The specific receipt is the s8-2 flag on run `bot-20260730-025337` (03:00:00) described in [[flag-triage-20260729]]: an escape hop landed on its own locked equipment and the next derivation selected a fresh teleport to it. `src/tankpit_bot/bot/ai/intent.py:23-26` records the same rationale in the module doc — every release emits a `plan_released` diagnostic "so plan churn is measurable in the events stream instead of silent".
[^2]: `src/tankpit_bot/bot/ai/intent.py` — the four-function API that implements the question: `current_collect_plan(ai_state)` at `:174`, `plan_completes_here(plan, self_x, self_y)` at `:198`, `release_collect_plan(...)` at `:215`, and `validate_collect_plan(...)` at `:251`, the last "run at ``DecideCtx`` construction" per the module doc at `:20-21`.
[^3]: `src/tankpit_bot/bot/ai/intent.py:28-30` — module doc, verbatim: "Raw field mutation stays in :mod:`tankpit_bot.bot.ai.context` (``set_resource_target`` / ``clear_resource_target``); this layer adds meaning, not a parallel mechanism." Both mutators exist at `src/tankpit_bot/bot/ai/context.py:183` and `:164` respectively, and carry the plan's persistent fields, so no new state was introduced.
[^4]: `src/tankpit_bot/bot/ai/intent.py:72-81` — `PLAN_RELEASE_REASONS: tuple[PlanReleaseReason, ...]` is the closed tuple `("tank_at_capacity", "superior_candidate", "not_executable", "landing_scan_reset", "walk_for_fuel_override", "target_gone", "target_not_pursuable", "kind_invalid")`, matching this section's list exactly and in order; its docstring at `:82-84` names it a "Closed vocabulary of plan-release reason codes". `release_collect_plan`'s `reason` parameter is documented against it at `:232`.
[^5]: `src/tankpit_bot/bot/ai/intent.py:23-26,88` — every release "emits a ``plan_released`` diagnostic with a specific reason code, so plan churn is measurable in the events stream instead of silent", and `plan_released` events are grouped "by this field". The churn interpretations below (lock thrash, the F6 passability family, re-planning under fire) are this page's reading of that data, not a recorded measurement.
[^6]: `src/tankpit_bot/bot/ai/collect_mode.py:110` — `def _escape_under_fire_decision(...)`, called at `:400` before the hop-selection path.
