---
title: ClearGBM native categorical splits — many-vs-many by gradient order
tags: [ml, cleargbm, categorical, roadmap-p3]
related:
  - "[[cleargbm-program-charter]]"
  - "[[cleargbm-nan-direction-and-colsample]]"
  - "[[cleargbm-histogram-split-path]]"
source_paths:
  - libs/cleargbm_rs/src/split/categorical.rs
  - libs/cleargbm_rs/src/binning/categorical.rs
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-23_p3_categorical_splits.md
source_git_blobs:
  "libs/cleargbm_rs/src/split/categorical.rs": d8dbb80a66c730cc6f77fe02009d47892f53a423
  "libs/cleargbm_rs/src/binning/categorical.rs": 06ca5aa38121cfa53988f182d7c4c65f5405597a
  "libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-23_p3_categorical_splits.md": 7b0fdb3f3d54616e915df1d865cdcc6146829e0c
fact_checked: "2026-08-23"
confidence: high
hubs: [libs]
---

# ClearGBM native categorical splits — many-vs-many by gradient order

P3 Landing B of the [[cleargbm-program-charter]] (board task `6e71afae`),
completing the phase. Features declared in `categorical_features` split
by SET MEMBERSHIP over integer category codes instead of by threshold:
per node, non-empty categories sort by gradient/hessian ratio (Fisher
1958's sufficient ordering for an optimal binary partition under convex
loss) and a prefix scan finds the best subset, NaN tried on both sides.
Identity gate 112/112 with the axis off; five artifacts retrained
exactly under the 19-field-config / 10-field-node schema.

## Semantics that differ from LightGBM, on purpose

- **No `cat_smooth`**: the sort key floors near-zero hessians at EPSILON
  instead of applying LightGBM's smoothing prior (default 10). No
  smoothing constant exists in the config, so none is silently applied —
  adding it later is a stated knob, not a hidden default.
- **No `max_cat_threshold` cap and no rare-category overflow bin**: the
  full sorted order is scanned, and a feature with more distinct codes
  than `max_bins` is an ERROR naming both counts, never a silent
  grouping.

## Shapes

- Config: `categorical_features` required-with-null everywhere. Rust and
  cleargbm-python carry strictly ascending INDICES (one canonical
  spelling); covenant_ml and the Covenant-Radar wire carry column NAMES,
  resolved like monotonic constraints — unknown name = error.
- `SplitDecision` enum: `Threshold{split_bin}` XOR
  `CategorySubset{left_bins: CategoryBinSet}` (256-bit bin mask). No
  placeholder values in either direction.
- Node: `categories_goes_left` (sorted raw codes, model-wire field 10;
  `threshold` is null on such nodes). Prediction: member → left,
  anything else (other/unseen/non-integer values) → right, NaN → learned
  direction. `-0.0` normalizes to `0.0` at both binning and predict.
- Pairing: monotonic constraint on a categorical feature rejected at
  train time; indices bounds-checked against `n_features`.
- SHAP: the covenant_ml decoder REFUSES categorical models (the path
  explainer walks thresholds) — explicit error, not mis-attribution.

## The discriminating fixture

Codes `[0,1,2,3]` labelled `[1,0,1,0]`: alternating in code order, so no
single threshold separates it, but the subset `{0,2}` vs `{1,3}` does. A
categorical stump learns exactly that partition (structural assert on the
root's code set); the numeric stump provably cannot. Keep this fixture
for any future categorical regression testing — contiguous-label
fixtures cannot tell a subset split from a threshold.
