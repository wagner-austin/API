# 2026-08-22 — P1: the objective seam lands; ClearGBM's first regression benchmark, and it leads

Agent-board task `a65aa1f0` (ClearGBM program charter P1). The core gains an
`Objective` enum — `binary_log_loss` | `squared_error` — behind one seam:
base score, per-round gradients/hessians, early-stopping evaluation loss,
prediction transform. Squared error is init = mean(y), gradient =
prediction − y, hessian = 1, eval = MSE. Labels are typed at the entry
(`TrainingLabels::Binary(&[u8])` / `Continuous(&[f64])`), and
`resolve_objective` makes an objective/label mismatch unrepresentable past
the boundary.

## Config honesty, extended to the objective axis

- `objective` is a REQUIRED config and model-JSON field with no default —
  the `growth_strategy` policy. Artifacts predating the field do not load;
  all stored models retrain.
- `scale_pos_weight` became objective-PAIRED (`Option<f64>`): must be set
  under `binary_log_loss` (1.0 stated explicitly for unweighted), must be
  unset under `squared_error`, which has no positive class to weight —
  the `num_leaves` pairing shape, enforced at construction and at both
  pyo3 entries in both directions.
- `predict_proba` on a squared-error model is REJECTED (its raw scores are
  predictions, not log-odds); the error names `predict_raw` as the answer.
- Removed as legacy while in the seam: the model-level `n_classes` field
  (constant 2, derivable from the objective, meaningless for regression —
  P4's multiclass reintroduces a real class count as a validated pairing)
  and the dead pre-weighting `binary_log_loss_gradients/hessians` exports.

## Equivalence gate (binary path): PASS, byte-for-byte

The four-arm benchmark under the objective-seam crate reproduces the
2026-08-22 knob-identity manifest exactly: 56/56 quality values and leaf
counts on the cleargbm arms identical across all seeds, LightGBM/XGBoost
anchors identical (no environment drift). Manifest:
`BENCHMARK_MANIFEST_2026-08-22_p1_objective_identity.json`. The binary
arm of the seam executes the exact historical operation sequence; every
recorded manifest remains valid.

## The measurement: regression quality on financial_distress

First entry of the regression objective into the standing benchmark.
Corpus: `financial_distress` from the verified dataset registry (Kaggle;
3,672 rows, 83 features, continuous distress score, heavy-tailed).
Protocol: every arm trains through its covenant_ml RegressorBackend on
identical data; all arms share `regression_split` (0.6/0.2/0.2) at the
same seed, so partitions are identical per seed. Matched hyperparameters:
300 rounds, lr 0.05, depth 6 (leaf-wise arms budgeted at 31 leaves),
min 20 samples/leaf, reg_lambda 1.0, early stopping 30 on validation.
Seeds 42–46. Manifest:
`BENCHMARK_MANIFEST_2026-08-22_p1_regression_quality.json`.

| arm | mean test RMSE | mean test R² | mean test MAE | mean wall |
|---|---|---|---|---|
| **cleargbm@leaf_wise** | **1.8023** | **0.4463** | 0.6202 | **0.27 s** |
| cleargbm@depth_wise | 1.8036 | 0.4454 | 0.6212 | 0.28 s |
| xgboost | 1.8291 | 0.4317 | 0.6140 | 25.9 s |
| lightgbm | 1.8637 | 0.3928 | 0.6350 | 0.33 s |

ClearGBM leads RMSE and R² on both growth arms, and is best or tied-best
on four of five seeds. Seed 44 is the corpus's tail-risk split (every arm
craters; ClearGBM degrades least: RMSE 4.686 vs LightGBM 4.706, XGBoost
4.795). Wall clock: the cleargbm arms are the FASTEST of the four on this
corpus — including beating LightGBM — a small-corpus regime where
ClearGBM's lean per-fit overhead wins. (The XGBoost wall number is its
backend's defaults on CPU and is reported, not tuned; quality is the gate
here, and its quality trails on 4 of 5 seeds regardless.)

Caveat, stated rather than hidden: one corpus, small, heavy-tailed. The
registry currently holds exactly one verified regression dataset; P6's
corpus onboarding (weather, RustedWarfare value models, metabolomics/BVOC)
is what turns this single win into a standing regression scoreboard.

## Surface changes

- cleargbm_rs: `train_gradient_boosting` takes typed labels +
  `ValidationData`; new pyo3 entry `train_gradient_boosting_regression_rs`
  (f64 targets); entry and config objective must agree (rejected in both
  directions). `n_classes` accessor and pyo3 fn removed.
- cleargbm (Python): `train_gradient_boosting_regression`, `Objective`
  literal + `OBJECTIVES` + `require_objective`; `predict_raw` documented
  as the regression inference function.
- covenant_ml: `cleargbm_reg` RegressorBackend registered beside
  `lightgbm_reg`/`xgboost_reg` (shared search spaces, JSON artifact with
  the objective tag inside); shap decoder reads the objective and the
  nullable weight from payloads.

Gates at land: cleargbm_rs 1390 tests / clippy -D warnings / 100.00%
segment coverage; cleargbm 212 tests / 100.00% coverage; covenant_ml
2431 tests / 100.00% coverage; all linters and guards green.
