---
title: ClearGBM histogram-based split finding
tags: [ml, gradient-boosting, cleargbm]
related: [[monorepo-discipline]]
sources:
  - libs/cleargbm/src/cleargbm/histogram.py
  - libs/cleargbm/src/cleargbm/parallel.py
  - libs/cleargbm/src/cleargbm/tree.py
  - libs/cleargbm/src/cleargbm/split.py
  - libs/cleargbm/src/cleargbm/losses.py
  - libs/cleargbm/docs/VALIDATION_REPORT_2026-07-20.md
fact_checked: 2026-07-20
confidence: high
---

# ClearGBM histogram-based split finding

ClearGBM has **two independent split-finding paths**. Only the histogram path runs at training time.

## The two paths

- **Histogram (default, LightGBM-style)** — `libs/cleargbm/src/cleargbm/histogram.py::find_best_split_from_histogram`. Called from `libs/cleargbm/src/cleargbm/parallel.py::_find_best_histogram_split_with_cache` and `::_select_best_split`, which are the two entry points reached from `libs/cleargbm/src/cleargbm/tree.py::_build_tree_with_histograms`. This is what `train_gradient_boosting` uses [^1].
- **Exact (sorted, XGBoost-style)** — `libs/cleargbm/src/cleargbm/split.py::find_best_split`. Retained for testing and small-node cases; not invoked from the runtime tree-building loop [^2].

Both paths implement L2 regularization via a `reg_lambda` parameter on `_compute_split_gain` (`histogram.py:229`, `split.py:79`). The histogram path did not until 2026-07-20 — see [Correctness note](#correctness-note-reg_lambda-2026-07-20).

## O(K) complexity claim

`find_best_split_from_histogram` scans `n_regular_bins - 1` bins with prefix sums for gradient sums, hessian sums, and counts (`histogram.py:383-416`) [^3]. Wall-time is bounded by K (= `max_bins`), independent of the number of samples in the node. Empirically measured on random inputs at K=64: 100× more samples produces 0.92× the split-scan time [^4].

`max_bins` is 64 by default in `GradientBoostingConfig` (`libs/cleargbm/src/cleargbm/_types_model.py`), and the resulting bin edges + per-sample bin assignments are precomputed once per training run in `ensemble.py::train_gradient_boosting` (line 363) via `precompute_feature_bins(x_train, config["max_bins"])` and then reused across every tree [^5].

## Sibling subtraction

For each node split, ClearGBM builds a histogram only for the smaller child; the larger child's histogram is derived by subtraction from the parent (`tree.py::_compute_child_histograms`, `histogram.py::subtract_histogram`). This is the standard LightGBM 2× histogram-building speedup [^6]. Numerically verified to floating-point precision (max_abs_diff ~ 1e-15 per bin on gradient sums, hessian sums, and counts) [^4].

## Loss function

Binary log-loss with the textbook `p - y` gradient and `p * (1 - p)` hessian (`libs/cleargbm/src/cleargbm/losses.py::BinaryLogLoss`, delegating to `_hooks_loss.py::_default_binary_log_loss_*`) [^7]. Initial prediction is `log(p / (1-p))` where p is the training positive rate [^8]. Sigmoid clips input to `[-500, 500]` to keep the exponent finite [^9].

## Rust vs Python fallback

Every math primitive is routed through a hook indirection layer (`_hooks_loss.py`, `_hooks_sigmoid.py`, `_hooks_histogram.py`, `_hooks_binning.py`, `_hooks_prediction.py`, `_hooks_ensemble.py`). Default hooks are pure-Python; calling `cleargbm._rust_adapters.use_rust_backend()` at process start rewires them to the Rust core in `libs/cleargbm_rs/` [^10]. Tests exercise the Python defaults; production callers activate Rust for throughput. The Python fallback is a correct reference, not a performance target.

## Correctness note (reg_lambda, 2026-07-20)

Prior to 2026-07-20, `histogram.py::_compute_split_gain` had no `reg_lambda` parameter, so the histogram runtime path silently ignored L2 regularization on split gain (only leaf values got regularized via `split.py::_compute_leaf_value`). Silent at default `reg_lambda = 0.0`; asymmetric with LightGBM / XGBoost at `reg_lambda > 0`. Fixed by threading `reg_lambda` into `_compute_split_gain` and `find_best_split_from_histogram`, and by passing `config["reg_lambda"]` from both call sites in `parallel.py`. Regression coverage in `libs/cleargbm/tests/test_histogram.py::TestComputeSplitGainRegLambda` [^11].

L1 (`reg_alpha`) is **not** currently applied to split gain in either path (both apply it only to leaf values via `_compute_leaf_value`). LightGBM's `lambda_l1` does affect splits; that's a design question for a future PR, not a fix on the same footing.

---

[^1]: `libs/cleargbm/src/cleargbm/ensemble.py::train_gradient_boosting` line 363 calls `precompute_feature_bins`; line 380 calls `build_tree` which delegates to `_build_tree_with_histograms`; that function calls `_find_best_histogram_split_with_cache` which calls `find_best_split_from_histogram`. The exact path in `split.py` is never reached from this chain.
[^2]: `grep -r find_best_split libs/cleargbm/src/` — 0 hits from `tree.py` or `ensemble.py`, only from `split.py`'s own module and tests.
[^3]: `libs/cleargbm/src/cleargbm/histogram.py:383-416` — outer loop over `range(n_regular_bins - 1)`; inner NaN-direction loop is 2 iterations. Total: 2 · (n_regular_bins - 1) gain evaluations, each O(1).
[^4]: `libs/cleargbm/docs/VALIDATION_REPORT_2026-07-20.md` — check 4 (sibling subtraction, max_abs_diff = 1.78e-15 / 3.55e-15 per bin); check 5 (O(K) empirical, ratio 0.92× across 10k → 1M samples at K=64).
[^5]: `libs/cleargbm/src/cleargbm/ensemble.py:362-363` — `feature_bins = precompute_feature_bins(x_train, config["max_bins"])` called once outside the `for tree_idx in range(n_estimators):` loop; the same `feature_bins` object is passed to every `build_tree` call at line 386.
[^6]: `libs/cleargbm/src/cleargbm/tree.py::_compute_child_histograms` lines 284-341 — picks the smaller of `left_indices` / `right_indices`, builds a histogram for it via `build_histogram`, derives the larger sibling via `subtract_histogram(parent_hist, smaller_hist)`.
[^7]: `libs/cleargbm/src/cleargbm/_hooks_loss.py::_default_binary_log_loss_gradients` line 152: `result = y_pred - y_float`. `::_default_binary_log_loss_hessians` line 178: `result = p_clipped * (1.0 - p_clipped)`.
[^8]: `libs/cleargbm/src/cleargbm/_hooks_loss.py::_default_binary_log_loss_initial_prediction` line 212: `return math.log(p_positive / (1.0 - p_positive))`.
[^9]: `libs/cleargbm/src/cleargbm/_hooks_sigmoid.py::_default_sigmoid` line 59: `x_clipped = max(-500.0, min(500.0, x))` before `1 / (1 + exp(-x))`.
[^10]: `libs/cleargbm/src/cleargbm/_rust_adapters.py::use_rust_backend` — reassigns each `_*_backend` module attribute from the pure-Python default to a Rust binding. Idempotent at process startup.
[^11]: `libs/cleargbm/tests/test_histogram.py::TestComputeSplitGainRegLambda` (5 tests) + `TestFindBestSplitFromHistogram::test_reg_lambda_changes_reported_gain` (regression guard on the end-to-end call). Fix landed 2026-07-20 alongside `docs/VALIDATION_REPORT_2026-07-20.md`.
