---
title: ClearGBM objective seam — regression joins binary, and leads its first benchmark
tags: [ml, cleargbm, regression, objectives, roadmap-p1]
related:
  - "[[cleargbm-program-charter]]"
  - "[[cleargbm-decorative-knob-class]]"
  - "[[cleargbm-leaf-normalized-benchmarking]]"
source_paths:
  - libs/cleargbm_rs/src/training/labels.rs
  - libs/cleargbm_rs/src/training/train.rs
  - libs/cleargbm_rs/src/losses/squared_error.rs
  - libs/covenant_ml/src/covenant_ml/backends/cleargbm/regressor.py
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-22_p1_objective_regression.md
fact_checked: "2026-08-22"
confidence: high
hubs: [libs]
---

# ClearGBM objective seam — regression joins binary, and leads its first benchmark

P1 of the [[cleargbm-program-charter]] roadmap, landed 2026-08-22
(board task `a65aa1f0`). The core gained an `Objective` enum —
`binary_log_loss` | `squared_error` — behind one seam: base score,
per-round gradients/hessians, early-stopping evaluation loss, prediction
transform. Squared error is init = mean(y), gradient = prediction − y,
hessian = 1, eval loss = MSE. This is the seam every later objective
(multiclass softmax, LambdaMART ranking — P4) rides.

## Design facts a later session needs

- **The objective is a required config and model-JSON field** with no
  default, wire spelling identical on both boundaries
  (`"binary_log_loss"` / `"squared_error"`) — the `growth_strategy`
  policy. Artifacts predating the field do not load; everything stored
  was retrained the same day.
- **`scale_pos_weight` is objective-paired** (`Option<f64>`): `Some(w)`
  required under binary (1.0 stated explicitly for unweighted), `None`
  required under squared error — the `num_leaves` pairing shape. A class
  weight on a regression config is a rejected config, not an ignored one.
- **Labels are typed at the entry**: `TrainingLabels::Binary(&[u8])` |
  `Continuous(&[f64])`, with `ValidationData` bundling validation
  features+labels so features-without-labels is unrepresentable.
  `resolve_objective` (training/labels.rs) checks objective/label pairing
  once and produces a `ResolvedObjective` the boosting loop matches on
  totally — no unreachable arms, no per-round re-checks. Continuous
  labels must be finite (NaN/inf rejected with the offending index).
- **`predict_proba` is rejected for squared-error models** — their raw
  scores ARE the predictions (identity transform), and the error names
  `predict_raw` as the answer. Two pyo3 entries
  (`train_gradient_boosting_rs` i64 labels /
  `train_gradient_boosting_regression_rs` f64 targets); entry and config
  objective must agree, rejected in both directions.
- **Removed as legacy** while in the seam: the model-level `n_classes`
  field (constant 2, objective-derivable, meaningless for regression —
  P4 reintroduces a real class count as a validated pairing) and the
  dead pre-weighting unweighted gradient/hessian exports.

## The equivalence gate (binary path)

The four-arm benchmark under the seam reproduces the 2026-08-22
knob-identity manifest byte-for-byte: 56/56 cleargbm quality values and
leaf counts identical across seeds 42–45, LightGBM/XGBoost anchors
identical. Manifest:
`libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-08-22_p1_objective_identity.json`.
Every recorded manifest remains valid. The production retrain confirmed
it end to end: the rw_matches active model retrained to numerically
identical metrics (val 0.7790 / test 0.7142, weight 1.6552, 16 trees) —
only the artifact schema moved.

## First regression scoreboard entry: ClearGBM leads

`financial_distress` (registry-verified Kaggle corpus, 3,672 × 83,
heavy-tailed continuous target), 5 seeds, all arms through their
covenant_ml RegressorBackend on identical `regression_split` partitions,
matched hyperparameters (300 rounds, lr 0.05, depth 6 / 31 leaves,
early stop 30):

| arm | mean test RMSE | mean test R² | mean wall |
|---|---|---|---|
| cleargbm@leaf_wise | **1.8023** | **0.4463** | **0.27 s** |
| cleargbm@depth_wise | 1.8036 | 0.4454 | 0.28 s |
| xgboost | 1.8291 | 0.4317 | 25.9 s |
| lightgbm | 1.8637 | 0.3928 | 0.33 s |

Best or tied-best on 4/5 seeds; degrades least on the tail-risk split
(seed 44). Notably the fastest arm on this small corpus — including
beating LightGBM's wall clock. Caveat stated in the record: one corpus,
small; P6's onboarding (weather, RustedWarfare value models,
metabolomics/BVOC) turns the single win into a standing scoreboard.
Manifest: `BENCHMARK_MANIFEST_2026-08-22_p1_regression_quality.json`;
narrative: `BENCHMARK_RESULTS_2026-08-22_p1_objective_regression.md`.

## The consumer surface

- cleargbm (Python): `train_gradient_boosting_regression`, `Objective`
  literal + `OBJECTIVES` + `require_objective`; config TypedDict carries
  `objective` and nullable `scale_pos_weight`.
- covenant_ml: `cleargbm_reg` RegressorBackend registered beside
  `lightgbm_reg`/`xgboost_reg` (shared regression_split and search
  spaces; JSON artifact self-describes its objective). The shap decoder
  reads the objective tag and nullable weight from payloads.
- covenant-radar-api: no code changes — it consumes cleargbm only
  through covenant_ml. All three stored artifacts (active rw_matches
  model, taiwan, us) retrained under the new schema with reproduced
  numbers (taiwan best 0.9364; us best 0.8155).
