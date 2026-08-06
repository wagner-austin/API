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
  - libs/cleargbm_rs/src/losses/derivatives.rs
  - libs/cleargbm_rs/src/losses/sigmoid_arr.rs
  - libs/cleargbm/src/cleargbm/_rust.py
  - libs/cleargbm/src/cleargbm/ensemble.py
  - libs/cleargbm/docs/VALIDATION_REPORT_2026-07-20.md
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md
fact_checked: "2026-07-21"
confidence: high
hubs: [libs]
---

# ClearGBM histogram-based split finding

Split finding runs entirely in Rust. The Python surface (`cleargbm.ensemble.train_gradient_boosting`) validates inputs, marshals config, and hands off to `cleargbm_rs.train_gradient_boosting_rs` for the whole training loop [^1].

## O(K) complexity claim

Rust's `find_best_split_from_histogram` scans `n_regular_bins - 1` bins with prefix sums for gradient sums, hessian sums, and counts. Wall-time is bounded by K (= `max_bins`), independent of the number of samples in the node. Empirically measured on random inputs at K=64: 100× more samples produces 0.92× the split-scan time [^2].

`max_bins` is 64 by default in `GradientBoostingConfig`, and bin edges + per-sample bin assignments are precomputed once per training run and reused across every tree [^3].

## Sibling subtraction

For each node split, ClearGBM builds a histogram only for the smaller child; the larger child's histogram is derived by subtraction from the parent (`libs/cleargbm_rs/src/tree/histograms.rs:199::compute_child_histograms` in Rust — moved out of `builder.rs` by the `0fdb63f7` builder/nodes/histograms split; `libs/cleargbm_rs/src/histogram/mod.rs::subtract_histogram` for the primitive). This is the standard LightGBM 2× histogram-building speedup [^4]. Numerically verified to floating-point precision (max_abs_diff ~ 1e-15 per bin on gradient sums, hessian sums, and counts) in the pre-refactor Python-fallback path [^5]; the Rust code is exercised by cargo tests + the covenant_ml integration tests.

## Loss function

Binary log-loss with the textbook `p - y` gradient and `p * (1 - p)` hessian, implemented in `libs/cleargbm_rs/src/losses/derivatives.rs` [^6]. Initial prediction is `log(p / (1-p))` where p is the training positive rate [^7]. Sigmoid clips input to `[-500, 500]` to keep the exponent finite [^8].

## Rust-only architecture (as of 2026-07-21)

There is exactly one compute path: Rust. The Python side calls `cleargbm._rust.train_gradient_boosting_rs`, which is pinned to the compiled extension's `train_gradient_boosting_rs` symbol via a Protocol-typed `__import__` [^9]. The Python-fallback code (`_hooks_*.py`, `_rust_adapters.py`, `parallel.py`, `split.py`, `tree.py`, `histogram.py`, `losses.py`) that previously existed alongside the Rust binding was deleted in Phase C of the Rust-only refactor. See [`libs/cleargbm/docs/RUST_ONLY_REFACTOR.md`](../../libs/cleargbm/docs/RUST_ONLY_REFACTOR.md).

## Correctness fix history

### reg_lambda drift on histogram path (2026-07-20)

Prior to 2026-07-20, the histogram-runtime `_compute_split_gain` (then in Python) had no `reg_lambda` parameter, so the histogram runtime path silently ignored L2 regularization on split gain (only leaf values got regularized). Silent at default `reg_lambda = 0.0`; asymmetric with LightGBM / XGBoost at `reg_lambda > 0`. Fixed by threading `reg_lambda` into the Python-side `_compute_split_gain` and `find_best_split_from_histogram`, with regression coverage. The whole Python path was then retired in Phase C; the Rust `_compute_split_gain` at `libs/cleargbm_rs/src/split/mod.rs` has always applied `reg_lambda` correctly, so the fix carried forward automatically [^10].

L1 (`reg_alpha`) is **not** applied to split gain on either path — it is leaves-only[^11]. LightGBM's `lambda_l1` does affect splits[^12]; that's a design question for a future PR, not a fix on the same footing.

### np.ascontiguousarray copy on histogram sample_bins (2026-07-20, superseded)

The pre-refactor histogram Rust adapter received a strided column-slice from a 2D `sample_bins` layout and had to `np.ascontiguousarray` on every call — real cost on every histogram build. The full-loop Rust training call (which is the only path post-refactor) assembles its own layout in Rust, so the copy is gone. See [`libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md`](../../libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md) for the measured impact: 14% faster fit + 10× lower run-to-run variance[^13].

---

[^1]: `libs/cleargbm/src/cleargbm/ensemble.py::train_gradient_boosting` line 148 calls `_validate_training_inputs`; line 149 marshals config via `_config_to_rust_dict`; line 151 dispatches to `train_gradient_boosting_rs` which resolves to `cleargbm_rs.cleargbm_rs.train_gradient_boosting_rs`.
[^2]: `libs/cleargbm/docs/VALIDATION_REPORT_2026-07-20.md` — check 5 (O(K) empirical, ratio 0.92× across 10k → 1M samples at K=64). Measured on the then-Python `find_best_split_from_histogram`; the Rust implementation shares the algorithm.
[^3]: `libs/cleargbm_rs/src/binning/feature_bins.rs::precompute_feature_bins` — called once per training run at the start of `train_gradient_boosting`; feature bins are threaded through the tree-building loop for every tree.
[^4]: `libs/cleargbm_rs/src/tree/builder.rs` — builds histogram for smaller child, calls `subtract_histogram(parent, smaller)` to derive the larger sibling.
[^5]: `libs/cleargbm/docs/VALIDATION_REPORT_2026-07-20.md` — check 4 (sibling subtraction, max_abs_diff = 1.78e-15 / 3.55e-15 per bin on gradient sums / hessian sums).
[^6]: `libs/cleargbm_rs/src/losses/derivatives.rs::binary_log_loss_gradients` — `result[i] = y_pred[i] - y_float`; `::binary_log_loss_hessians` — `p_clipped * (1.0 - p_clipped)`.
[^7]: `libs/cleargbm_rs/src/losses/initial_prediction.rs::binary_log_loss_initial_prediction` — `(p_positive / (1.0 - p_positive)).ln()`.
[^8]: `libs/cleargbm_rs/src/losses/sigmoid_arr.rs` — clips to `[-500.0, 500.0]` before `1 / (1 + exp(-x))`.
[^9]: `libs/cleargbm/src/cleargbm/_rust.py` line 177 — `_native_mod = __import__("cleargbm_rs.cleargbm_rs", fromlist=["cleargbm_rs"])`; line 179 pins `train_gradient_boosting_rs: _TrainProto = _native_mod.train_gradient_boosting_rs`.
[^10]: `libs/cleargbm_rs/src/split/mod.rs::_compute_split_gain` — applies `reg_lambda` to `H_L`, `H_R`, and `H_total` in the gain formula per the standard XGBoost / LightGBM form.
[^11]: `libs/cleargbm_rs/src/tree/builder.rs:283,337,360` [synthesis] — every consumer of `config.reg_alpha()` in the crate is a `compute_leaf_value(g_sum, h_sum, config.reg_alpha(), config.reg_lambda())` call at those three sites. A crate-wide grep for `reg_alpha` across `libs/cleargbm_rs/src/` (2026-07-31) returns only these leaf-value calls plus config plumbing (`training/config.rs` field + validation, `pyo3_module/training_fns.rs` dict read, `training/serde_impl.rs` round-trip, tests) — no occurrence inside `split/mod.rs` or any gain computation.
[^12]: LightGBM upstream, [`src/treelearner/feature_histogram.hpp`](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp) — `GetSplitGains` is templated on `USE_L1` (`:759`) and dispatches to `GetLeafGainGivenOutput<USE_L1>` (`:792`); the L1 soft-threshold `ThresholdL1(double s, double l1)` is defined at `:711` and applied in the leaf-output formula at `:724` (`-ThresholdL1(sum_gradients, l1) / (sum_hessians + l2)`). Call sites in `src/treelearner/feature_histogram.cpp:222,324,503,654`. Verified 2026-07-31 against the local checkout at `~/PROJECTS/LightGBM`; cited by URL because that repository is outside this workspace root and so cannot appear in `source_paths`.
[^13]: `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-21.md:12` — verbatim: "**Speed improved** — cleargbm's fit-time dropped from **7.87s ± 1.32s** to **6.88s ± 0.13s** (14% faster, 10× lower variance). The hook indirection removed in Phase C was costing ~1 second per training run and dominating the run-to-run variance." Restated in the results table at `:36` and the variance analysis at `:38`.
