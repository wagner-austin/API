---
title: ClearGBM — the decorative-knob class: five config fields that never reached training
tags: [ml, cleargbm, covenant_ml, bugs, negative-result, class-weighting]
related:
  - "[[cleargbm-f32-score-narrowing-reverted]]"
  - "[[cleargbm-perf-leaf-wise-growth]]"
  - "[[cleargbm-leaf-normalized-benchmarking]]"
source_paths:
  - libs/covenant_ml/src/covenant_ml/backends/cleargbm/backend.py
  - libs/covenant_ml/src/covenant_ml/backends/cleargbm/config_resolution.py
  - libs/covenant_ml/src/covenant_ml/optimizer/objectives/cleargbm_objective.py
  - libs/cleargbm_rs/src/training/train.rs
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-22_scale_pos_weight.md
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-22_knob_closure.md
fact_checked: "2026-08-22"
confidence: high
hubs: [libs]
---

# ClearGBM — the decorative-knob class: five config fields that never reached training

On 2026-08-22 a single question ("why does the config say reg_lambda=1.0?")
unraveled a five-member bug class: **config fields that types, 100% coverage,
and completion-asserting tests all validated while their values went
nowhere.** Every member is fixed or removed; this page is the institutional
memory so the class is recognized on sight next time.

## The five members

1. **Backend hardcodes** (`996bf364`) — `backends/cleargbm/backend.py` built
   the native config with `reg_alpha=0.0, reg_lambda=0.0, n_jobs=1,
   max_features=None, monotonic_constraints=None` regardless of the caller's
   `ClearGBMConfig`. Every CV asked for `reg_lambda=1.0` and trained
   unregularized while its LightGBM comparator got its regularization. Fix:
   pass everything through (`config_resolution.py` owns the shape
   translations). Measured: +0.2 AUC pts on the rw_matches CV.
2. **`scale_pos_weight` decorative** (`c14dc11f`) — the backend computed the
   class weight, logged "Auto-calculated scale_pos_weight for ClearGBM", and
   returned it in every `TrainOutcome` — but neither cleargbm layer had any
   weighting mechanism. Every imbalanced comparison ran weighted LightGBM vs
   unweighted ClearGBM. Fix: weighted binary log loss end to end in
   `cleargbm_rs` (gradients `w*(p-1)`, hessians `w*p(1-p)`, weighted base
   score and early-stopping loss), bit-identical at `w=1.0`. Measured: +1.3
   AUC pts on rw_matches (0.7365 → 0.7492).
3. **`max_features` dropped at the Rust boundary** (`ef6d6794`) — accepted by
   the Python config, silently discarded before Rust. Fix: implemented as a
   real per-split feature budget (per-node subset from a stream-free
   `(seed, round, node_id)` derivation, so `None` stays bit-identical).
   Measured neutral at 0.8 on rw_matches — a working dial, not a default.
4. **`track_contributions` unimplementable** (`68b88a72`) — a trainer flag
   for a capability that lives in the covenant_ml explainer, post-hoc over
   saved model JSON. Removed from every config surface; the wire parser
   tolerates-and-drops the stray key from old clients.
5. **Sweep-save crash** (`3b746e32`) — `save_best_model` demanded
   `float_params["reg_alpha"]` although the ClearGBM search space never
   samples regularization, so **every ClearGBM optimize sweep crashed at the
   save step** and the stored "optimal configs" were relics of an older code
   state. Found only by running the sweep end to end.

Adjacent, same session (`73dd637c`): the optimizer objective trained every
trial **unweighted** while the backend weights the final model — the sweep
tuned an objective production never ran. Trials now derive the same
auto-computed weight via the backend's own `_compute_class_weight`.

## What the honored semantics changed

- rw_matches production CV: 0.7345 (buggy) → 0.7365 (reg honored) → **0.7492**
  (weight applied), vs weighted LightGBM 0.7299 on the identical protocol.
- taiwan re-sweep: TPE found the **same** best hyperparameters, weighted they
  score 0.9364 vs the stale 0.9320.
- us re-sweep: weighted best 0.8155 vs the stale 0.8737 — **class weighting
  costs ranking AUC on us across the whole search space** (verified by a
  weight-1.0 control at 0.8326 and by bit-identity of unweighted training on
  this very corpus). The old number described a model production never
  trained. Weighting is a policy the backend imposes (always `neg/pos`);
  where AUC is the objective it is not a free win.
- Model artifacts predating the two new required config fields
  (`scale_pos_weight`, `max_features`) do not load, by the same no-default
  policy `growth_strategy` set. All stored artifacts were retrained
  (`516968e0`, `3b746e32`); the pre-fix active model had been silently
  unloadable since the leaf-wise land.

## The detector

Types pass (the value is well-typed), coverage passes (the hardcoded line
executes), completion tests pass (training runs). The only test that catches
this class is a **knob-sensitivity test**: train twice with different values
and assert the model changed — or, for wiring, open the saved model and
assert its config records the caller's value. Both patterns now guard
`growth_strategy`, `num_leaves`, `scale_pos_weight`, `max_features`,
`reg_lambda` and the objective's trial weight. The invariant that holds
after this session: **every field a ClearGBM config can state is either
honored by training or does not exist.**
