---
title: ClearGBM missing-value routing is learned; per-tree column sampling
tags: [ml, cleargbm, missing-values, colsample, roadmap-p3]
related:
  - "[[cleargbm-program-charter]]"
  - "[[cleargbm-objective-seam]]"
  - "[[cleargbm-sample-weights]]"
  - "[[cleargbm-decorative-knob-class]]"
source_paths:
  - libs/cleargbm_rs/src/tree/feature_subsample.rs
  - libs/cleargbm_rs/src/training/tests/train_nan_tests.rs
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-22_p3_colsample_nan.md
source_git_blobs:
  "libs/cleargbm_rs/src/tree/feature_subsample.rs": a8206fd50aaff22d73b5df7671c65c4c387e4f9d
  "libs/cleargbm_rs/src/training/tests/train_nan_tests.rs": c7b4a1f8584580354c14990d5b33965b60eeb45b
  "libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-22_p3_colsample_nan.md": c876dfab8595a8b0959dd994ec60e53ceb8d78f0
fact_checked: "2026-08-22"
confidence: high
hubs: [libs]
---

# ClearGBM missing-value routing is learned; per-tree column sampling

P3 Landing A of the [[cleargbm-program-charter]] roadmap (board task
`6e71afae`); categoricals remain as Landing B. Four-arm identity gate
reproduced the knob-identity manifest 112/112 byte-for-byte; the five
service artifacts retrained under the 18-field schema and reproduced
every recorded number (active 0.7790/0.7142, taiwan 0.9451/98 trees +
sweep 0.9364, us 0.7848/14 trees + sweep 0.8155).

## NaN direction: already learned — the spec premise was wrong

The P3 spec assumed a fixed NaN-bin policy needing replacement. The
audit found LightGBM's `default_direction` mechanism already present:
the split search tries every candidate with the NaN partition on BOTH
sides and keeps the higher gain; `nan_goes_left` lives on every internal
node and prediction routes by it. The landing is therefore proof, not
code — `train_nan_tests.rs` pins it with a depth-1 stump discriminator:

- `[1,2,3,4]` labelled `[0,0,1,1]` + four NaN rows labelled positive →
  the only pure split is 2|3 with NaN RIGHT; the stump learns
  `nan_goes_left = false` and the NaN row lands bit-exactly in the
  positive leaf.
- Mirror the missing rows' labels → the same split needs NaN LEFT; the
  learned flag flips to `true`. No fixed policy fits both, and a stump
  has no second split to compensate with.

Do not re-audit this: the earlier all-finite-0/all-NaN-1 construction
CANNOT discriminate — mirrored labels there produce the same tree with
negated leaf values, so the direction flags never flip. The 0,0,1,1
threshold construction is what makes the flag observable.

## colsample_bytree: one mask per round, composed with max_features

Required-with-null `colsample_bytree: float | None` on every config
surface (Rust serde field 18, pyo3 required key, Python TypedDicts,
covenant_ml config, radar-api wire). Validated **(0, 1) exclusive** —
`None` is the only spelling of "all features", so `1.0` is rejected as a
second spelling of the same meaning.

- `k_tree = max(1, floor(f * n_features))` via the directly-testable
  `tree_column_budget` (u32 ceiling like row subsampling).
- Stream-free determinism: the mask is a pure function of
  `(random_state, round)` mixed through `TREE_MIX = 0xC2B2_AE3D_27D4_EB4F`
  (xxhash64 prime-2) — deliberately distinct from the per-node
  `NODE_MIX` golden ratio so the two derivations can never collide; the
  run RNG that row subsampling reads never advances.
- Composition is LightGBM's: the per-node `max_features` draw selects
  WITHIN the tree's mask, budget capped at the pool size. Histograms are
  still built for every feature (sibling subtraction needs complete
  parents); only the split search is restricted.
- Old artifacts refuse to load by the honesty policy (18th field is
  required); the wire parser tolerates an absent key as null at the HTTP
  boundary only, matching the `max_features` precedent.

## File-size splits that rode along

`builder_tests.rs` (1111) → node-helper tests + `builder_build_tests` +
`builder_build_edge_tests`; `leafwise_tests.rs` (634) → shared
`leafwise_helpers` + behavior + error files; `error_tests.rs` (629) →
internal-function errors + `error_hook_tests`. All tree test files now
sit under the 600-line cap.
