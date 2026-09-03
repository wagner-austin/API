---
title: ClearGBM multiclass softmax — K trees per round on the objective seam
tags: [ml, cleargbm, multiclass, roadmap-p4]
related:
  - "[[cleargbm-program-charter]]"
  - "[[cleargbm-objective-seam]]"
  - "[[cleargbm-sample-weights]]"
source_paths:
  - libs/cleargbm_rs/src/training/train_multiclass.rs
  - libs/cleargbm_rs/src/losses/multiclass.rs
  - libs/cleargbm/src/cleargbm/ensemble_multiclass.py
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-23_p4_multiclass.md
source_git_blobs:
  "libs/cleargbm_rs/src/training/train_multiclass.rs": 4f018cc0bf8c73f7ad6420e5458121c89619c6b6
  "libs/cleargbm_rs/src/losses/multiclass.rs": 873d9528cc19002c22d953b4be6219e32b66505c
  "libs/cleargbm/src/cleargbm/ensemble_multiclass.py": 77a0f7074a9631bee999ddea6be06d787a6f8b4f
  "libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-23_p4_multiclass.md": 95d251a138924c672d178232f29f3b84869825a5
fact_checked: "2026-08-23"
confidence: high
hubs: [libs]
---

# ClearGBM multiclass softmax — K trees per round on the objective seam

P4 Landing A of the [[cleargbm-program-charter]] (board task `1fedf1e3`).
The `multiclass_softmax` objective trains `n_classes` trees per boosting
round — one per class against its softmax gradient — reusing the existing
tree builder unchanged over class-major gradient slices. Identity gate:
the single-score path reproduced the knob-identity manifest 112/112
byte-for-byte through the refactor; quality gate: held-out log loss BELOW
LightGBM's multiclass on all four seeds of the synthetic corpus, accuracy
at parity (`BENCHMARK_MANIFEST_2026-08-23_p4_multiclass_quality.json`).

## The pairing (config honesty applied to K)

`n_classes` is required-with-null everywhere the config exists (Rust serde
field 20, cleargbm TypedDict, the 21-key Rust-boundary dict): an int >= 2
iff the objective is `multiclass_softmax`, null otherwise, and
`scale_pos_weight` must be null under multiclass (rows are weighted via
`sample_weight`, not a two-class ratio). The pairing is enforced once, at
the Rust boundary. The covenant-radar wire is binary-only and does NOT
carry the field — zero service changes.

## Mechanics (LightGBM-parity choices, each deliberate)

- **Class-major buffers**: scores/gradients/hessians live in one flat
  `Vec` indexed `class * n_samples + row`, so each class's tree consumes a
  contiguous slice (LightGBM's layout; XGBoost is row-major).
- **Base scores**: uncentered per-class log priors, weighted
  (LightGBM-style; XGBoost mean-centers). The model stores them as
  `class_base_predictions` (model serde field 6) — exactly one of
  scalar/per-class base is non-null, decided by the objective.
- **Hessian rescale**: `K/(K-1) * p * (1-p)` (Friedman's factor, LightGBM
  parity). `PROB_EPSILON = 1e-15` clips the softmax before the log.
- **Rounds, not trees**: one row-subsample per round shared by the K
  trees; feature-mask seeds mix the GLOBAL tree index (`round*K + class`);
  early stopping evaluates weighted multiclass log loss per round and
  truncates whole rounds (`(best+1)*K` trees) — the stored tree count is
  always a multiple of K, trees round-major (`tree t -> class t % K`).
- **Prediction trio**: raw `(n, K)` score matrix, max-subtracted softmax
  probabilities, argmax classes with ties to the LOWEST index. The
  single-score predictors reject multiclass models and vice versa, and the
  SHAP decoder refuses a multiclass payload at the model level (SHAP's
  walker is single-output).

## Where things live

Rust: `training/train_multiclass.rs` (the K-tree loop),
`losses/multiclass.rs` (softmax/log-prior/log-loss/grad-hess),
`training/setup.rs` (shared prep both trainers call),
`ResolvedTraining::{SingleScore, Multiclass}` keeps unreachable arms out
of both loops. Python: `cleargbm.ensemble_multiclass` (train + trio),
split from `ensemble.py` as a different contract; `_types_config.py`
split out of `_types_model.py` at the 600-line ceiling. The quality
harness is `covenant_ml.benchmarking.multiclass_quality` +
`scripts/benchmark_cleargbm_multiclass.py`.

P4's second half — LambdaMART ranking — builds on this landing's seam
plus P2's weights; the lambda formulation is pinned in the tech-wiki
(Burges 2010 + LightGBM rank objective pages, captured 2026-08-23).
