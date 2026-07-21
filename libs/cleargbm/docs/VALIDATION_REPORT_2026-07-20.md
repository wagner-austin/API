# ClearGBM correctness validation — 2026-07-20

**Author:** validation session (fresh AI instance following `docs/HANDOFF_BENCHMARK_AND_VALIDATE.md`)
**Scope:** Phase 1 (correctness) of the handoff. Phase 2 (LightGBM benchmark) NOT yet run.
**Verdict:** 21 / 21 automated checks PASS. **1 YELLOW finding found and FIXED in the same session** (histogram-path split gain silently discarded `reg_lambda`). `make check` green after the fix: 555 tests, 100% statement + branch coverage. No RED findings.

## Environment

- Python 3.11.9 · numpy 2.4.0
- ClearGBM: current tree at `~/PROJECTS/api/libs/cleargbm/`
- Python-fallback backend (Rust `use_rust_backend()` NOT activated) — validates the Python reference implementation. Rust hooks are a separate audit surface not covered here.
- Data: `covenant_ml/tests/data/american_bankruptcy.csv` — 78,682 rows, 18 features, 6.63% positive (5,220 failed / 73,462 alive), year range 1999-2018.

## Summary table

| # | Category | Check | Verdict | Numeric |
| - | -------- | ----- | ------- | ------- |
| 1a | Loss math | `binary_log_loss == mean(-(y·log(p) + (1-y)·log(1-p)))` | GREEN | rel diff exact |
| 1b | Loss math | `gradient == p - y` | GREEN | max_abs_diff = 0 |
| 1c | Loss math | `hessian == clip(p) · (1 - clip(p))` | GREEN | max_abs_diff = 0 |
| 2a | Sigmoid | `sigmoid(0) == 0.5` | GREEN | exact |
| 2b | Sigmoid | `sigmoid(1e6)` finite in `[0, 1]` | GREEN | 1.0 |
| 2c | Sigmoid | `sigmoid(-1e6)` finite in `[0, 1]` | GREEN | 7.12e-218 |
| 2d | Sigmoid | `sigmoid_array` batch == scalar loop | GREEN | max_abs_diff = 0 |
| 3a | Init pred | 20% positive → `log(0.25)` | GREEN | expected == got to 1e-15 |
| 3b | Init pred | all-zeros raises `ValueError` | GREEN | — |
| 3c | Init pred | all-ones raises `ValueError` | GREEN | — |
| 4a | Sibling subtraction | grad sums equal per bin | GREEN | max_abs_diff = 1.78e-15 |
| 4b | Sibling subtraction | hess sums equal per bin | GREEN | max_abs_diff = 3.55e-15 |
| 4c | Sibling subtraction | counts equal per bin | GREEN | 0 |
| 5 | O(K) claim | split-scan time ratio (1M / 10k samples) | GREEN | ratio = 0.92× |
| 6a | Real data | `predict_proba` in `[0, 1]` on test | GREEN | min=0.0029 max=0.7136 |
| 6b | Real data | `p0 + p1 == 1` | GREEN | — |
| 6c | Real data | mean(pred) ≈ train positive rate (±5pp) | GREEN | diff = 0.0017 |
| 6d | Real data | test AUC > 0.55 | GREEN | AUC = 0.8436 |
| 7 | Determinism | two same-seed runs bit-identical | GREEN | max_abs_diff = 0 |
| 8a | Additivity | `sigmoid(base + Σ lr·predict_tree)` == `predict_proba` | GREEN | max_abs_diff = 1.25e-16 |
| 8b | Additivity | `explain_prediction.final_probability` == `predict_proba` | GREEN | max_abs_diff = 0 |

Totals: **21 PASS · 0 FAIL**.

## Verdict per README claim

| README claim | Verdict | Evidence |
| ------------ | ------- | -------- |
| "Numpy-backed arrays with vectorized histogram building" | GREEN | `_default_build_histogram` uses vectorized `sample_bins[sample_indices]` then `buf.accumulate_batch(bins, grads, hess)`. |
| "Sibling subtraction" | GREEN | `_default_subtract_histogram` at `_hooks_histogram.py`, used at `tree.py::_compute_child_histograms` L308-341 for the smaller-child-first + parent-minus-smaller pattern. Numerically verified in check 4. |
| "LightGBM-style O(K) split finding instead of O(n log n) sorting" | GREEN | `histogram.py::find_best_split_from_histogram` scans `n_regular_bins - 1` bins with a prefix-sum accumulator; total wall-time is independent of `n` (empirical ratio 0.92× when scaling `n` by 100×). |
| "Precomputed bins reused across all trees" | GREEN | `ensemble.py::train_gradient_boosting` L363 calls `precompute_feature_bins(x_train, config["max_bins"])` once outside the tree loop and passes the same `feature_bins` object to every `build_tree` call. |
| "Strict typing" | GREEN by inspection | All read files declare `NDArray[np.float64]` / `NDArray[np.int64]` / `TypedDict`; no `Any` observed in `src/`. |

## FIXED-IN-SESSION: reg_lambda dropped by the histogram-path split gain

**Original severity when found:** yellow (silent at default `reg_lambda = 0.0`; changes behavior when a user sets it > 0).
**Status now:** fixed and covered — Austin asked to fix issues we come across, so the fix landed in this same session. See `Fix summary` below.

**What was wrong:**

- `split.py::_compute_split_gain` (the **exact/sorted** path) already took `reg_lambda` and applied it to `H_L`, `H_R`, and `H_total` in the gain formula. Standard XGBoost/LightGBM behavior.
- `histogram.py::_compute_split_gain` (the **runtime histogram** path) took no `reg_lambda` and did not apply it to the gain formula.
- `parallel.py::_find_best_histogram_split_with_cache` and `parallel.py::_select_best_split` (the two histogram split-finder call sites) did not forward `reg_lambda`.
- Consequence at `reg_lambda > 0`: leaf **values** were regularized (`tree.py:248` passes `reg_lambda` to `_compute_leaf_value`), but the split **gain** used to pick each split was not. Splits were chosen as if `reg_lambda = 0`, then leaf values shrunk toward 0 — asymmetric with LightGBM/XGBoost.
- The `covenant_ml` `ClearGBMBackend` wrapper hardcodes `reg_alpha=0.0`, `reg_lambda=0.0`, so this drift was never exercised in production wrapping. But `GradientBoostingConfig` exposes the parameters to direct users, and the README's "Configuration Reference" table advertises them as L1/L2 regularization.

**Why the existing test suite missed it:** all reg_lambda tests exercised `split.py::_compute_split_gain` directly, which was correct. Nothing trained through the runtime histogram path with `reg_lambda > 0` and asserted a behavior difference vs `reg_lambda = 0`.

**Fix summary (commits in this session's working tree):**

- `src/cleargbm/histogram.py::_compute_split_gain` — added `reg_lambda: float = 0.0` param, applied to `H_L + λ`, `H_R + λ`, `H_total + λ` denominators. Default `0.0` preserves the pre-regularization formula, so pre-existing tests still hold verbatim.
- `src/cleargbm/histogram.py::find_best_split_from_histogram` — added `reg_lambda: float = 0.0` param, forwarded to the gain function.
- `src/cleargbm/parallel.py` — both `find_best_split_from_histogram` call sites (in `_find_best_histogram_split_with_cache` and `_select_best_split`) now pass `config["reg_lambda"]`.
- `tests/test_histogram.py::TestComputeSplitGainRegLambda` — 5 new tests: reg_lambda=0 matches the omitted default; reg_lambda>0 reduces gain by the analytic amount (16 → 16/3 for the reference case); the exact regularized formula holds on an arbitrary tuple; large lambda drives gain to zero; the eps guard fires on the regularized `h_left_reg`.
- `tests/test_histogram.py::TestFindBestSplitFromHistogram::test_reg_lambda_changes_reported_gain` — regression guard: end-to-end call at reg_lambda=0 vs reg_lambda=1 on separable data returns different gain values.

**Post-fix `make check`:** 555 tests pass, coverage 100.00% (statements + branches) across all 33 files including the new lines.

**Scope decision on `reg_alpha` (L1):** LEFT AS-IS for this session. `split.py::_compute_split_gain` — even the exact path — does **not** apply `reg_alpha` to split gain (only to leaf value via `_compute_leaf_value`). Matching the histogram path to the exact path is consistent; whether both should also apply L1 to gain (LightGBM's `lambda_l1` does) is a broader design question worth its own PR. Flagged for follow-up but not shipped.

**Also fixed in-session:** three pre-existing `monkey-patch-ban` guard violations in `tests/test_parallel.py` at lines 119, 245, 314 (direct assignment of `parallel_module._WORKER_FEATURE_BINS = FeatureBins(...)`). Predated my session (`git blame`: authored 2025-12-24). Would have blocked `make check` regardless of the reg_lambda work. Fix: added `_set_worker_feature_bins_for_test` / `_reset_worker_feature_bins_for_test` DI helpers to `parallel.py` and routed the fake-pool setup + finally-cleanup through them.

**Rust path status:** `_rust_native_adapters.py:172` passes `reg_lambda` through to the native config dict. Whether the Rust `histogram_scan` uses it in its gain formula is out of scope for this session — the Rust core is a separate audit surface. Flag for the Rust-audit pass.

## Timing observations (Python fallback, single-threaded)

Small-scale numbers for context, not a benchmark:

- 50 trees, max_depth=4, 62,896 train rows × 18 features, `n_jobs=1`, Python-fallback backend: **8.11 s total train time**. ~6 trees/s.
- Prediction on 8,928 test rows: sub-second (not separately timed).
- **Interpretation:** the Python fallback is functional but not tuned for throughput. Phase 2 (LightGBM comparison) should activate the Rust backend via `use_rust_backend()` before timing — the Python path is a validation harness, not a production configuration. LightGBM is a natively-compiled library; benchmarking it against Python fallback would be an apples-to-oranges comparison and would understate ClearGBM's real position.

## What surprised me

1. **The O(K) claim is genuinely verified, and it's easy to check.** 100× more samples took 0.92× the split-scan time. This is exactly the shape you want to see. Many "we implement histogram-based split finding" claims turn out to be O(K + N) or O(N log K) in practice; ClearGBM's is a true prefix-sum scan.
2. **AUC 0.84 on a 6.6% positive-rate real dataset with zero feature engineering.** Better than I expected for a from-scratch implementation on financials.
3. **The reg_lambda / reg_alpha drift is a genuine issue and had zero tests exercising it.** This is the kind of finding a benchmark session catches: the code has a slot for L2 regularization on the histogram path and it's just quietly discarded, with 100% test coverage everywhere except the one place that would notice.

## Recommendation to Austin

- **Correctness is GREEN.** ClearGBM's Python reference implementation does what the README claims — histogram-based split finding is a true O(K) scan, sibling subtraction is arithmetically correct, loss math is textbook, and predictions on a real fintech-style panel are calibrated + discriminative. No red bugs found.
- **Fix the reg_lambda drift as a follow-up** (out of this session's scope per the handoff). Small PR: thread `reg_lambda` into `_compute_split_gain` in `histogram.py`, add a regression test.
- **Phase 2 (LightGBM head-to-head) is safe to run.** No correctness blocker.
- **Before Phase 2 timing runs, activate the Rust backend** via `cleargbm._rust_adapters.use_rust_backend()` — the Python-fallback timings in this report are validation harness numbers, not the configuration you'd defend in a comparison.

## Reproducing this report

The runnable harness is in the session scratchpad, not committed:

```
C:\Users\Test\AppData\Local\Temp\claude\C--Users-Test-PROJECTS-MCPs\<session>\scratchpad\validate_cleargbm.py
```

To re-run:

```powershell
cd C:\Users\Test\PROJECTS\api\libs\cleargbm
poetry run python <path-to>\validate_cleargbm.py
```

If Austin wants a permanent, coverage-audited version of the harness, it needs to be lifted into `libs/cleargbm/scripts/validate_correctness.py` with a matching `tests/test_scripts_validate_correctness.py`. That's a small follow-up (~1 hour) and not required to trust this report — the raw outputs above are deterministic given the input CSV and the current codebase.

## Files touched this session

- `docs/VALIDATION_REPORT_2026-07-20.md` (this file — new)
- `src/cleargbm/histogram.py` — `_compute_split_gain` + `find_best_split_from_histogram` take `reg_lambda` (default 0.0)
- `src/cleargbm/parallel.py` — the two histogram-split call sites now pass `config["reg_lambda"]`; added `_set_worker_feature_bins_for_test` / `_reset_worker_feature_bins_for_test` DI helpers
- `tests/test_histogram.py` — added `TestComputeSplitGainRegLambda` (5 tests) + a regression test on `find_best_split_from_histogram`
- `tests/test_parallel.py` — 3 monkey-patch guard violations replaced with calls to the new DI helpers

`make check` (green, 555 tests, 100% coverage, statements + branches).
