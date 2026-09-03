---
title: ClearGBM histogram-based split finding
tags: [ml, gradient-boosting, cleargbm, rust]
related:
  - "[[monorepo-discipline]]"
source_paths:
  - libs/cleargbm_rs/src/histogram/mod.rs
  - libs/cleargbm_rs/src/split/mod.rs
  - libs/cleargbm_rs/src/tree/builder.rs
  - libs/cleargbm_rs/src/tree/histograms.rs
  - libs/cleargbm_rs/src/training/single_score_rounds.rs
  - libs/cleargbm_rs/src/training/config_rules.rs
  - libs/cleargbm_rs/src/losses/initial_prediction.rs
  - libs/cleargbm_rs/src/losses/sigmoid_arr.rs
  - libs/cleargbm_rs/src/binning/feature_bins.rs
  - libs/cleargbm/src/cleargbm/_rust.py
  - libs/cleargbm/src/cleargbm/ensemble.py
  - libs/cleargbm/src/cleargbm/_types_config.py
  - libs/cleargbm/docs/VALIDATION_REPORT_2026-07-20.md
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md
source_git_blobs:
  "libs/cleargbm_rs/src/histogram/mod.rs": c5f91b51728ff6113c38acdf9b894591fb7ea27c
  "libs/cleargbm_rs/src/split/mod.rs": 2fae8441aba469e55b2f8cec444f74a4db85999f
  "libs/cleargbm_rs/src/tree/builder.rs": 8fefb405a694475e8297a81923291f879afbcb3d
  "libs/cleargbm_rs/src/tree/histograms.rs": 7a2f44ae25b941d81d7cf08c2b0a6ae6d6363952
  "libs/cleargbm_rs/src/training/single_score_rounds.rs": 7fb0f35e74dd69934978bdd4ea51f640ba47bc50
  "libs/cleargbm_rs/src/training/config_rules.rs": a5020894551ab7042b5d6f40789469560e052cae
  "libs/cleargbm_rs/src/losses/initial_prediction.rs": 8470134c2a68db9daae4455b96a55df32064afff
  "libs/cleargbm_rs/src/losses/sigmoid_arr.rs": e73feb1e3d0604709332eb3885539d27062e006b
  "libs/cleargbm_rs/src/binning/feature_bins.rs": 75cdd19ed161603d031e2bef9e5edf51373094e5
  "libs/cleargbm/src/cleargbm/_rust.py": 5f8ba08dec7197ffe2a203a44385d3337f0b47db
  "libs/cleargbm/src/cleargbm/ensemble.py": d855968ef0fd5be83716ae0a331765004dee690f
  "libs/cleargbm/src/cleargbm/_types_config.py": 478cf51b7bcdd0e00b630f358397a1d4d4138e44
  "libs/cleargbm/docs/VALIDATION_REPORT_2026-07-20.md": c601efb92c2f596c97e60fdddade03d9dc1fa379
  "libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md": 3f95cc90aaf624a9f250c4b5f892c570ac27c4df
fact_checked: "2026-09-03"
confidence: high
hubs: [libs]
---

# ClearGBM histogram-based split finding

Split finding runs entirely in Rust. The Python surface (`cleargbm.ensemble.train_gradient_boosting`) validates inputs, marshals config, and hands off to `cleargbm_rs.train_gradient_boosting_rs` for the whole training loop [^1].

## O(K) complexity claim

Rust's `find_best_split_from_histogram` scans `n_regular_bins - 1` bins with prefix sums for gradient sums, hessian sums, and counts. Wall-time is bounded by K (= `max_bins`), independent of the number of samples in the node. Empirically measured on random inputs at K=64: 100× more samples produces 0.92× the split-scan time [^2].

`max_bins` has **no default** in `GradientBoostingConfig` — it is a required parameter, validated to `2 <= max_bins <= 255` (`libs/cleargbm_rs/src/training/config_rules.rs:54` and `:64`, the upper bound being the `u8` bin index; the checks moved out of `config.rs` since this was written). Bin edges + per-sample bin assignments are precomputed once per training run and reused across every tree [^3]. **Corrected 2026-08-05:** this sentence read "`max_bins` is 64 by default in `GradientBoostingConfig`". No such default exists: `GradientBoostingConfigDict` declares `max_bins: int` as a required key (`libs/cleargbm/src/cleargbm/_types_config.py:180`; it lived in `_types_model.py` when this correction was first written), and every `64` in the tree is a script or test default — `libs/cleargbm/scripts/autotune.py:73`, `libs/cleargbm/scripts/benchmark.py:140`, `libs/cleargbm/tests/conftest.py:14`. **The likely origin of the error is `libs/cleargbm/scripts/benchmark.py:215`, which labels its own sweep entry `"max_bins=64 (default)"`** — a mislabel in the script, since the library has no default to be. See [[cleargbm-perf-uint8-histogram-bins]], which correctly attributes 64 to the harness rather than the library.

## Sibling subtraction

For each node split, ClearGBM builds a histogram only for the smaller child; the larger child's histogram is derived by subtraction from the parent (`libs/cleargbm_rs/src/tree/histograms.rs:396::compute_child_histograms` in Rust — moved out of `builder.rs` by the `0fdb63f7` builder/nodes/histograms split; `libs/cleargbm_rs/src/histogram/mod.rs:193::subtract_histogram` for the primitive). This is the standard LightGBM 2× histogram-building speedup [^4]. Numerically verified to floating-point precision (max_abs_diff ~ 1e-15 per bin on gradient sums, hessian sums, and counts) in the pre-refactor Python-fallback path [^5]; the Rust code is exercised by cargo tests + the covenant_ml integration tests.

## Loss function

Binary log-loss with the textbook `p - y` gradient and `p * (1 - p)` hessian. Initial prediction is `log(p / (1-p))` where p is the training positive rate [^7]. Sigmoid clips input to `[-500, 500]` to keep the exponent finite [^8].

**Corrected 2026-09-03: the standalone loss module this section cited is gone, and the formula it gave was incomplete.** `libs/cleargbm_rs/src/losses/derivatives.rs` was deleted in `7f55994d`, the commit that introduced the Objective seam. Per-round derivatives are no longer free functions in `losses/`; they are inline arms of a `ResolvedObjective` match inside the training loop, so that one enum drives base score, gradients, hessians, early-stopping loss and prediction transform together [^6].

The arithmetic also carries a term the old sentence omitted. The gradient is not written as `p - y`: it is `p - 1.0` on positives and `p` on negatives, which is the same number, **multiplied by `scale_pos_weight` on the positive class** — and the hessian is scaled the same way. A reader who took "textbook `p - y`" literally would predict the wrong gradient on any imbalanced run with `scale_pos_weight` set [^6].

## Rust-only architecture (as of 2026-07-21)

There is exactly one compute path: Rust. The Python side calls `cleargbm._rust.train_gradient_boosting_rs`, which is pinned to the compiled extension's `train_gradient_boosting_rs` symbol via a Protocol-typed `__import__` [^9]. The Python-fallback code (`_hooks_*.py`, `_rust_adapters.py`, `parallel.py`, `split.py`, `tree.py`, `histogram.py`, `losses.py`) that previously existed alongside the Rust binding was deleted in Phase C of the Rust-only refactor. See [`libs/cleargbm/docs/RUST_ONLY_REFACTOR.md`](../../libs/cleargbm/docs/RUST_ONLY_REFACTOR.md).

## Correctness fix history

### reg_lambda drift on histogram path (2026-07-20)

Prior to 2026-07-20, the histogram-runtime `_compute_split_gain` (then in Python) had no `reg_lambda` parameter, so the histogram runtime path silently ignored L2 regularization on split gain (only leaf values got regularized). Silent at default `reg_lambda = 0.0`; asymmetric with LightGBM / XGBoost at `reg_lambda > 0`. Fixed by threading `reg_lambda` into the Python-side `_compute_split_gain` and `find_best_split_from_histogram`, with regression coverage. The whole Python path was then retired in Phase C; the Rust `compute_split_gain` at `libs/cleargbm_rs/src/split/mod.rs:337` has always applied `reg_lambda` correctly, so the fix carried forward automatically [^10]. (The Rust function lost its leading underscore at some point after this section was written; it is `compute_split_gain`, not `_compute_split_gain`.)

L1 (`reg_alpha`) is **not** applied to split gain on either path — it is leaves-only[^11]. LightGBM's `lambda_l1` does affect splits[^12]; that's a design question for a future PR, not a fix on the same footing.

### np.ascontiguousarray copy on histogram sample_bins (2026-07-20, superseded)

The pre-refactor histogram Rust adapter received a strided column-slice from a 2D `sample_bins` layout and had to `np.ascontiguousarray` on every call — real cost on every histogram build. The full-loop Rust training call (which is the only path post-refactor) assembles its own layout in Rust, so the copy is gone. See [`libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md`](../../libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md) for the measured impact: 14% faster fit + 10× lower run-to-run variance[^13].

---

[^1]: `libs/cleargbm/src/cleargbm/ensemble.py::train_gradient_boosting` line 189 calls `_validate_training_inputs`; line 190 marshals config via `_config_to_rust_dict`; line 192 dispatches to `train_gradient_boosting_rs` which resolves to `cleargbm_rs.train_gradient_boosting_rs`. (Cited as 148/149/151 until 2026-09-03; the file grew above them.)
[^2]: `libs/cleargbm/docs/VALIDATION_REPORT_2026-07-20.md` — check 5 (O(K) empirical, ratio 0.92× across 10k → 1M samples at K=64). Measured on the then-Python `find_best_split_from_histogram`; the Rust implementation shares the algorithm.
[^3]: `libs/cleargbm_rs/src/binning/feature_bins.rs:182::precompute_feature_bins` — called once per training run at the start of `train_gradient_boosting`; feature bins are threaded through the tree-building loop for every tree.
[^4]: `libs/cleargbm_rs/src/tree/builder.rs` — builds histogram for smaller child, calls `subtract_histogram(parent, smaller)` to derive the larger sibling.
[^5]: `libs/cleargbm/docs/VALIDATION_REPORT_2026-07-20.md` — check 4 (sibling subtraction, max_abs_diff = 1.78e-15 / 3.55e-15 per bin on gradient sums / hessian sums).
[^6]: `libs/cleargbm_rs/src/training/single_score_rounds.rs:124` — the `ResolvedObjective::Binary` arm, whose weightless branch computes `scale_pos_weight * (p - 1.0)` for `y == 1` and `p` otherwise, with hessians `scale_pos_weight * (p * (1.0 - p))` and `p * (1.0 - p)`; the weighted branch multiplies each by the sample weight `w`. `ResolvedObjective::SquaredError` follows at `:198`. The deleted module is recoverable at `git show 7f55994d^:libs/cleargbm_rs/src/losses/derivatives.rs` — cited as history, not as a current path, since `source-path-exists` correctly refuses a file absent from HEAD.
[^7]: `libs/cleargbm_rs/src/losses/initial_prediction.rs:45::binary_log_loss_initial_prediction` — `(p_positive / (1.0 - p_positive)).ln()`.
[^8]: `libs/cleargbm_rs/src/losses/sigmoid_arr.rs:8` — clips to `[-500.0, 500.0]` before `1 / (1 + exp(-x))`.
[^9]: `libs/cleargbm/src/cleargbm/_rust.py` line 398 — `_native_mod: types.ModuleType = __import__("cleargbm_rs")`; line 400 pins `train_gradient_boosting_rs: _TrainProto = _native_mod.train_gradient_boosting_rs`. (Cited as 177/179 until 2026-09-03; the Protocol block above them has since grown to cover the regression, multiclass, ranking and continue-training entry points.) The extension was reached as `cleargbm_rs.cleargbm_rs` until 2026-08-17, when maturin's `python-source` and the hand-written forwarder above it were removed (commit ea7835d2); it is now the top-level module.
[^10]: `libs/cleargbm_rs/src/split/mod.rs:337::compute_split_gain` — applies `reg_lambda` to `H_L`, `H_R`, and `H_total` in the gain formula per the standard XGBoost / LightGBM form.
[^11]: `libs/cleargbm_rs/src/tree/builder.rs:385,459,482` [synthesis] — every consumer of `config.reg_alpha()` in the crate is a `compute_leaf_value(g_sum, h_sum, config.reg_alpha(), config.reg_lambda())` call at those three sites (cited as 283/337/360 until 2026-09-03; still exactly three call sites, re-located). A crate-wide grep for `reg_alpha` across `libs/cleargbm_rs/src/` (re-run 2026-09-03) returns only these leaf-value calls plus config plumbing (`training/config.rs` field + validation, `pyo3_module/training_fns.rs` dict read, `training/serde_impl.rs` round-trip, tests) — no occurrence inside `split/mod.rs` or any gain computation.
[^12]: LightGBM upstream, [`src/treelearner/feature_histogram.hpp`](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp) — `GetSplitGains` is templated on `USE_L1` (`:759`) and dispatches to `GetLeafGainGivenOutput<USE_L1>` (`:792`); the L1 soft-threshold `ThresholdL1(double s, double l1)` is defined at `:711` and applied in the leaf-output formula at `:724` (`-ThresholdL1(sum_gradients, l1) / (sum_hessians + l2)`). Call sites in `src/treelearner/feature_histogram.cpp:222,324,503,654`. Verified 2026-07-31 against the local checkout at `~/PROJECTS/LightGBM`; cited by URL because that repository is outside this workspace root and so cannot appear in `source_paths`.
[^13]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md:12` — verbatim: "**Speed improved** — cleargbm's fit-time dropped from **7.87s ± 1.32s** to **6.88s ± 0.13s** (14% faster, 10× lower variance). The hook indirection removed in Phase C was costing ~1 second per training run and dominating the run-to-run variance." Restated in the results table at `:36` and the variance analysis at `:38`.
