# 2026-08-22 — P3 Landing A: learned NaN direction verified; per-tree column sampling

Agent-board task `6e71afae` (ClearGBM program charter P3, first landing).
Two of the phase's three data-realism items: the missing-value routing
audit and the `colsample_bytree` per-tree column sampler. Native
categorical splits remain (Landing B).

## 1. Learned NaN direction — the spec's premise was wrong

The P3 spec assumed a "fixed NaN bin policy" needing replacement. The
audit found the LightGBM `default_direction` mechanism ALREADY
implemented: the split search evaluates every candidate with the NaN
partition on the left and on the right and keeps the higher gain
(`NanDirection::{Left, Right}` in the split kernel, `nan_goes_left` on
every internal node, prediction routes by it). What was missing was
proof, so the landing is verification, not code:

- **The stump discriminator** (`train_nan_tests.rs`): feature values
  `[1, 2, 3, 4]` labelled `[0, 0, 1, 1]` plus four NaN rows. With the NaN
  rows labelled positive, the only pure single split is 2|3 with NaN
  routed RIGHT — a depth-1 stump must learn `nan_goes_left = false`, and
  does. Mirroring the missing rows' labels flips the requirement to LEFT
  — and the learned flag flips to `true`. No fixed always-left or
  always-right policy fits both datasets, and a stump has no second
  split to compensate with; predictions are asserted to land bit-exactly
  in the leaf sharing the missing rows' label.
- End to end at real depth, the missing rows' probability lands on the
  correct side of 0.5 on both mirrored datasets.

## 2. colsample_bytree — one mask per round, composed with max_features

New required-with-null config field `colsample_bytree: float | None`,
validated in **(0, 1) exclusive**: `None` is the only spelling of "all
features", so `1.0` — a second spelling of the same meaning — is
rejected at the config boundary rather than silently equivalent.

- Budget: `k_tree = max(1, floor(fraction * n_features))` (the row
  subsampling convention), computed by the directly-testable
  `tree_column_budget` — including its `n_features > u32::MAX` rejection,
  the same ceiling row subsampling imposes on `n_samples`.
- Determinism is stream-free: the tree mask is a pure function of
  `(random_state, round)` through `TREE_MIX = 0xC2B2_AE3D_27D4_EB4F`
  (xxhash64 prime-2), deliberately distinct from the per-node
  `NODE_MIX` golden-ratio constant so the two derivations can never
  collide into one RNG stream — and nothing advances the run RNG that
  row subsampling reads.
- Composition (the LightGBM semantics): when both axes are set, the
  per-node `max_features` draw selects WITHIN the tree's sampled set,
  with the per-split budget capped at that pool's size. Histograms are
  still built for every feature (sibling subtraction needs complete
  parent histograms); only the split search is restricted.
- Knob sensitivity: 0.5 changes the model under both growers, is
  deterministic across runs, and composes visibly with `max_features`.

## Equivalence gate: PASS, byte-for-byte

The four-arm benchmark under the colsample-capable crate reproduces the
2026-08-22 knob-identity manifest exactly — 112/112 quality values and
leaf counts across all 16 (model, variant, seed) arms, seeds 42–45,
LightGBM/XGBoost anchors identical.
Manifest: `BENCHMARK_MANIFEST_2026-08-22_p3_colsample_identity.json`.

## Artifact retrains (18-field serde break), all EXACTLY reproduced

The serialized config grew to 18 fields; by the honesty policy old
artifacts refuse to load, so all five service artifacts retrained under
the new schema and reproduced their recorded numbers bit-for-bit at the
reported precision:

| artifact | metric | value | trees |
|---|---|---|---|
| active_cgbm (rw_matches) | val/test AUC | 0.7790 / 0.7142 | best_round 16, spw 1.655 |
| taiwan production model | val AUC | 0.9451 | 98, spw 29.994 |
| us production model | val AUC | 0.7848 | 14, spw 14.077 |
| taiwan sweep best (5 trials) | best val AUC | 0.9364 | — |
| us sweep best (5 trials) | best val AUC | 0.8155 | — |

## Surface

- cleargbm_rs: `GradientBoostingConfig.colsample_bytree`;
  `select_tree_features` / `tree_column_budget` in
  `tree/feature_subsample.rs`; `BuildTreeInput.tree_feature_mask`
  composed by both growers; serde field 18; pyo3 extractor requires the
  key (absent ≠ null).
- cleargbm (Python): TypedDict field, `require_open_unit_float`
  validator, encode/decode, `_config_to_rust_dict` forwards it.
- covenant_ml: `ClearGBMConfig.colsample_bytree`, both backends forward
  it, the SHAP decoder reads it back from stored artifacts.
- covenant-radar-api: wire parser accepts float-or-null
  (absence tolerated as null at the HTTP boundary; strictness lives at
  the Rust boundary, matching `max_features` precedent).

## Gates

cargo fmt / clippy `-D warnings` / 1444 tests / 100.00% segment
coverage; cleargbm 221 passed, 100.00%; covenant_ml 2431 passed,
100.00%; covenant-radar-api 2586 passed, 100.00%. Three over-cap tree
test files split (builder_tests 1111 → 374+328+427; leafwise 634 →
helpers+behavior+errors; error_tests 629 → main+hooks).
