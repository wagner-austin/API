# P4 Landing A — multiclass softmax (2026-08-23)

The `multiclass_softmax` objective landed end to end: K-class labels, one
tree per class per boosting round (round-major storage), LightGBM-style
uncentered log-prior base scores, the softmax gradient with Friedman's
K/(K-1) hessian rescale, whole-round early stopping on weighted multiclass
log loss, and a prediction trio (raw score matrix, softmax probabilities,
argmax classes with ties to the lowest index).

## Serde breaks (round-3 artifact retrain)

Two required-with-null fields entered the wire this landing:

- config field 20: `n_classes` — `Some(k >= 2)` iff the objective is
  `multiclass_softmax`, `null` under every other objective; the pairing is
  enforced at the Rust boundary.
- model field 6: `class_base_predictions` — exactly one of
  `base_prediction` / `class_base_predictions` is non-null, decided by the
  objective.

Every stored artifact predating the fields refuses to load (stated or
nothing), so all three service artifacts were retrained and reproduced
their recorded numbers exactly:

| artifact | expected | reproduced |
|---|---|---|
| rw_matches `active_cgbm.json` | val 0.7790 / test 0.7142 / best 16 / spw 1.655 | identical |
| taiwan `taiwan_cleargbm_model.json` | val 0.9451 / 98 trees / spw 29.994 | identical |
| us `us_cleargbm_model.json` | val 0.7848 / 14 trees / spw 14.077 | identical |

The taiwan/us optimal-config JSONs are round-2 optuna sweep summaries
(best_value 0.9364 / 0.8155); they carry no crate-serialized state and were
left as recorded.

## Identity gate — the single-score path is bit-unchanged

The multiclass landing refactored shared training setup
(`training/setup.rs`), split the objective enums out of `config.rs`, and
made the model's base score an enum. The four-arm benchmark
(cleargbm, cleargbm@leaf_wise, lightgbm, xgboost x seeds 42-45) reproduces
the 2026-08-22 knob-identity manifest **112/112 non-timing values
byte-for-byte** (96 quality values + 16 mean-leaves values).

Manifest: `BENCHMARK_MANIFEST_2026-08-23_p4_multiclass_identity.json`.

## Multiclass quality vs LightGBM

New harness: `covenant_ml.benchmarking.multiclass_quality` +
`scripts/benchmark_cleargbm_multiclass.py`. Deterministic synthetic corpus
(6000 rows x 8 features, 5 overlapping uniform-noise clusters,
class-interleaved rows), 75/25 deterministic split, matched
hyperparameters (100 rounds, depth 4, lr 0.1, 64 bins, min 20 rows/leaf,
no subsampling, single thread), seeds 42-45.

| seed | cleargbm log_loss | lightgbm log_loss | cleargbm acc | lightgbm acc |
|---|---|---|---|---|
| 42 | **0.659225** | 0.664991 | 0.6287 | 0.6327 |
| 43 | **0.654349** | 0.661044 | 0.6400 | 0.6407 |
| 44 | **0.673719** | 0.674521 | 0.6260 | 0.6280 |
| 45 | **0.643014** | 0.645967 | 0.6487 | 0.6453 |

ClearGBM's held-out log loss is below LightGBM's on all four seeds;
accuracy is at parity (within ±0.004, split 1-3 by seed). The softmax
objective is quality-competitive, not merely wired.

Manifest: `BENCHMARK_MANIFEST_2026-08-23_p4_multiclass_quality.json`.

## Gates at landing

- cleargbm_rs: 1539 tests, 100.00% segment coverage, clippy `-D warnings`
  clean (unsafe forbidden, no `?`/unwrap/panic/`as`/allow), all files
  <= 600 lines.
- cleargbm: 243 tests, 100.00% coverage (multiclass surface split into
  `ensemble_multiclass.py`; `_types_config.py` split out of
  `_types_model.py` at the 600-line ceiling).
- covenant_ml: 2455 tests, 100.00% coverage (SHAP decoder refuses
  multiclass models at the model level; the converter refuses a null
  scalar base).
- covenant-radar-api: 2588 tests, 100.00% coverage, zero source changes —
  the service wire is binary-only and `n_classes` never crosses it.
