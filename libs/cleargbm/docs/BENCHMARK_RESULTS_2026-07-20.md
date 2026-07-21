# ClearGBM vs LightGBM benchmark — 2026-07-20

**Author:** benchmark session following `docs/HANDOFF_BENCHMARK_AND_VALIDATE.md`.
**Environment:** Windows-10-10.0.26200-SP0, Python 3.11.9, numpy 2.3.5, sklearn 1.7.2, lightgbm 4.6.0, cleargbm_rs 0.1.0.
**Dataset:** `libs/covenant_ml/tests/data/american_bankruptcy.csv` — 78,682 rows, 18 features, 8,971 unique companies, 6.63% positive class. SHA-256 in the manifest.
**Manifest (machine-readable):** `docs/BENCHMARK_MANIFEST_2026-07-20.json` — every run, seed, metric, config, environment field.
**`make check`:** green after all fixes (555 tests, 100.00% statement + branch coverage).

## TL;DR

- **Quality: ClearGBM ties LightGBM** on this dataset. AUC-ROC 0.687 vs 0.683 (LightGBM +0.004), AUC-PR 0.138 vs 0.142 (ClearGBM +0.004). All differences within seed noise (std ~0.02). Prediction Pearson correlation 0.93.
- **Speed: ClearGBM is 8.7× slower than LightGBM at its best** (`train_gradient_boosting_native`, full Rust loop). At 7.87s vs 0.91s for a 200-tree ensemble on 55k rows.
- **Speed via covenant_ml wrapper: 74× slower** because `ClearGBMBackend.train` calls the hook-routed path, not the native one. FFI overhead only — same math, same accuracy.
- **Two fixes shipped in this session as prerequisites:** (1) `reg_lambda` correctly applied to histogram-path split gain; (2) `np.ascontiguousarray` in the Rust histogram adapter (was fatal — no training could complete via `use_rust_backend()`).

## Methodology (why the numbers can be trusted)

Rigorous by design, in response to the Phase 1 finding that a naive panel split leaked identity across train/test.

1. **Company-disjoint split.** `american_bankruptcy` is panel data — same company appears in multiple years. Splitting by year (Phase 1's approach) let the model learn company identity, inflating AUC to 0.84. This benchmark partitions the 8,971 **unique companies** into 70/15/15 and follows each company's rows into its assigned split. **Result: AUC drops from 0.84 to 0.68. That's the honest number.**
2. **Multiple seeds.** 3 seeds (42, 43, 44). Every metric reported as mean ± std, not one lucky number.
3. **Baselines that contextualize AUC.**
   - `baseline_majority` = always predict train's positive rate → AUC-ROC=0.500 by construction.
   - `logistic_reg` = sklearn `LogisticRegression(class_weight="balanced", max_iter=1000)` on standardized features → shows what a linear model on this data can achieve.
4. **Multiple metrics.** AUC-ROC, AUC-PR (essential at 6.6% positive rate), log-loss, Brier score, accuracy at 0.5 threshold, mean predicted probability, and a 10-quantile calibration slope. Any one metric alone is misleading; the full row tells the story.
5. **Matched hyperparameters.** Both GBMs: `n_estimators=200`, `max_depth=6`, `learning_rate=0.05`, `max_bins=64`, `min_data_in_leaf=20`, `reg_alpha=0`, `reg_lambda=0`, `n_jobs=1`. LightGBM `num_leaves = 2**max_depth - 1 = 63` to prevent it from growing wider than depth allows. Both use deterministic RNG (LightGBM `deterministic=True`; ClearGBM sets `random_state`).
6. **Timing on identical hardware, single-threaded.** Wall-clock via `time.perf_counter`; `num_threads=1` on LightGBM and `n_jobs=1` on ClearGBM. No warm-up run at benchmark time — fit_time is first invocation. That penalizes LightGBM slightly (Python import + JIT), but the ratio doesn't shift meaningfully.
7. **Agreement analysis.** Feature-importance Spearman rank correlation (do they agree on what's important?), per-sample prediction Pearson correlation (do they agree on individual predictions?), top-1000-risky-sample overlap (do they identify the same at-risk companies?).
8. **Three ClearGBM run modes** — see next section for why this matters.

## The three ClearGBM run modes (why one number would be misleading)

ClearGBM has two training entry points and two backend states, giving three interesting configurations:

| Mode | Entry point | Backend | What this represents |
| ---- | ----------- | ------- | -------------------- |
| Python fallback | `train_gradient_boosting` | pure Python | Sanity + correctness reference (Phase 1). Not run here (`make check` covers it). |
| Hook-routed Rust | `train_gradient_boosting` | Rust hooks | **What `covenant_ml.ClearGBMBackend.train` uses today.** Python drives the tree/boosting loop; every math primitive crosses FFI to Rust. |
| Native full-loop Rust | `train_gradient_boosting_native` | native | ClearGBM at its **speed ceiling**. Entire training loop lives in Rust — one FFI call for all 200 trees. |

Reporting only the hook-routed number would understate ClearGBM. Reporting only the native number would overstate what a covenant_ml consumer gets today. Both are shipped in the results.

## Results (mean ± std over 3 seeds)

### Quality metrics

| Model | AUC-ROC | AUC-PR | log-loss | Brier | accuracy_at_0.5 |
| ----- | ------- | ------ | -------- | ----- | --------------- |
| baseline_majority | 0.5000 ± 0.0000 | 0.0636 ± 0.0086 | 0.2371 ± 0.0223 | 0.0596 ± 0.0074 | 0.9364 ± 0.0086 |
| logistic_reg | 0.6787 ± 0.0195 | 0.1281 ± 0.0130 | 0.6738 ± 0.0119 | 0.2384 ± 0.0018 | 0.3720 ± 0.0103 |
| **lightgbm** | **0.6871 ± 0.0212** | 0.1376 ± 0.0154 | **0.2294 ± 0.0271** | 0.0587 ± 0.0070 | 0.9345 ± 0.0088 |
| cleargbm_hook | 0.6825 ± 0.0185 | **0.1416 ± 0.0181** | 0.2298 ± 0.0268 | **0.0584 ± 0.0069** | 0.9350 ± 0.0086 |
| cleargbm_native | 0.6825 ± 0.0185 | **0.1416 ± 0.0181** | 0.2298 ± 0.0268 | **0.0584 ± 0.0069** | 0.9350 ± 0.0086 |

**LightGBM edges AUC-ROC by 0.004; ClearGBM edges AUC-PR by 0.004; log-loss / Brier essentially tied.** Every gap is smaller than the seed-to-seed std, so this is a statistical tie. cleargbm_hook and cleargbm_native produce the same metrics (they run the same math on the same inputs); the tiny difference below the 4th decimal is FP-order noise (Pearson prediction correlation between them = 0.9999, not exactly 1.0).

**Accuracy-at-0.5 is a trap here** — the majority-class predictor scores 0.9364 by never predicting positive. Anything below ~0.94 on this dataset is not usable at a 0.5 threshold without calibration. Which brings us to:

### Calibration

Calibration slope (linear regression of observed frequency on predicted probability, over 10 quantile bins; ideal = 1.0):

| Model | Mean predicted probability | Calibration slope |
| ----- | -------------------------- | ----------------- |
| baseline_majority | 0.0669 ± 0.0025 | nan (constant prediction) |
| logistic_reg | 0.4720 ± 0.0035 | 0.20 ± 0.04 (badly miscalibrated by class_weight="balanced") |
| lightgbm | 0.0644 ± 0.0007 | 0.68 ± 0.08 |
| cleargbm_hook | 0.0644 ± 0.0005 | 0.67 ± 0.09 |
| cleargbm_native | 0.0644 ± 0.0005 | 0.67 ± 0.09 |

**Both GBMs are under-calibrated (slope ~0.68) and neither is fit for direct use at a threshold.** Both would benefit from Platt scaling / isotonic regression on the val set. This is a property of the dataset (small positive class, short history per company) more than the models.

### Speed

| Model | fit_time (s) | predict_time on ~12k rows (s) | vs LightGBM |
| ----- | ------------ | ------------------------------ | ----------- |
| logistic_reg | 0.12 ± 0.01 | 0.003 | 0.13× |
| **lightgbm** | **0.91 ± 0.02** | 0.020 | **1.00×** |
| cleargbm_native | 7.87 ± 1.32 | 0.059 | **8.7× slower** |
| cleargbm_hook | 67.06 ± 3.87 | 0.115 | **74× slower** |

At its best (`train_gradient_boosting_native`), ClearGBM trains at 25 trees/s vs LightGBM's 220. On the path a covenant_ml consumer takes today (hook-routed), it trains at 3 trees/s. That gap is not from bad math — cleargbm_native and cleargbm_hook produce Pearson-correlation-0.9999 predictions with matched metrics. It's pure FFI overhead: each Python-loop iteration marshals gradient/hessian/bin arrays across the boundary and back, hundreds of thousands of times per training run.

### Agreement

| Comparison | Metric | Value |
| ---------- | ------ | ----- |
| ClearGBM vs LightGBM | feature-importance Spearman rank correlation (18 features) | 0.78 |
| ClearGBM vs LightGBM | per-sample prediction Pearson correlation | 0.93 |
| ClearGBM vs LightGBM | top-1000 risky-sample overlap | 768 / 1000 = 76.8% |
| cleargbm_hook vs cleargbm_native | per-sample prediction Pearson correlation | 0.9999 (essentially identical) |

The two GBMs agree on **who is risky** (77% overlap in top-1000) but rank them slightly differently. Their feature-importance rankings agree on 78% of the ordering — meaningfully aligned, not identical.

## Why the speed gap exists (specific causes)

LightGBM has 8+ years of production hardening. Concrete gaps ClearGBM has today:

1. **Histogram bin dtype: int64 vs uint8.** LightGBM caps `max_bin ≤ 255` and stores bins as `uint8`, fitting 8× more values per cache line. ClearGBM stores bins as `int64` (`libs/cleargbm/src/cleargbm/histogram.py::bin_samples`) with no dtype-density optimization. Fixing this is a small refactor but touches the Rust binding shape.
2. **Row-major sample_bins.** ClearGBM stores `sample_bins` shape `(n_samples, n_features)` in C-order. Per-feature histogram scans stride through memory instead of scanning contiguously; every scan takes an unnecessary `np.ascontiguousarray` copy (~500KB for 62k samples at int64). Column-major storage — `(n_features, n_samples)` — would make each scan contiguous with zero copies.
3. **No SIMD in the histogram accumulator.** LightGBM has hand-tuned AVX2/AVX-512 accumulators. ClearGBM's Rust histogram writes `grad_sums[bin] += gradient` in a scalar loop.
4. **Depth-first tree growth, not leaf-wise.** ClearGBM's `_build_tree_with_histograms` uses a stack-based depth-first traversal (`libs/cleargbm/src/cleargbm/tree.py:203`). LightGBM defaults to leaf-wise (best-first): expand the leaf with highest gain regardless of depth. Leaf-wise usually reaches similar quality with fewer splits — a wall-clock win at matched `n_estimators`.
5. **Feature parallelism.** LightGBM builds histograms across features in parallel by default. ClearGBM has `n_jobs > 1` in the hook path but no equivalent in `train_gradient_boosting_native`.
6. **covenant_ml wrapper picks the hook path.** `libs/covenant_ml/src/covenant_ml/backends/cleargbm/backend.py::ClearGBMBackend.train` calls `train_gradient_boosting` (line 332), not `train_gradient_boosting_native`. Switching it — when `use_rust_backend()` has been called — gets the 8.5× speedup with zero other changes. That's the highest-ROI single fix.

## Fixes shipped in this session as prerequisites for a defensible benchmark

### 1. `reg_lambda` was silently discarded on the histogram runtime path

Detailed in `docs/VALIDATION_REPORT_2026-07-20.md`. `histogram.py::_compute_split_gain` didn't take `reg_lambda`; `split.py`'s did. Leaf values were regularized; split gains weren't. Silent at default `reg_lambda=0.0` (the config in this benchmark), but a real bug at `reg_lambda > 0`. Fixed: threaded through both call sites in `parallel.py`; 5+1 regression tests in `tests/test_histogram.py::TestComputeSplitGainRegLambda`.

### 2. Rust histogram adapter crashed on non-contiguous column slices

Discovered running Phase 2 with `use_rust_backend()` active: **every** attempt to train raised `ValueError: bins: The given array is not contiguous`. Root cause: `parallel.py::_find_best_histogram_split_sequential` (and its cache/select twin) extracts `feat_bins = feature_bins.sample_bins[:, feat_idx]` — a strided view into the 2D C-order array — and passes it to the Rust binding, which requires C-contiguous. The Python fallback tolerated it silently (numpy handles strided fine); Rust does not.

**Meaning: `use_rust_backend()` was fatal to any real training run before this fix.** All Rust adapter tests still passed because their test inputs happened to already be contiguous — a gap in the test coverage, not the implementation. `make check` reports 100% branch coverage but the branch that mattered was never exercised on real inputs.

**Fix**: `sample_bins_c = np.ascontiguousarray(sample_bins)` at the entry of `_rust_adapters.py::_rust_build_histogram` (line 457). No-op if already contiguous; single copy if not. Doesn't touch Python-fallback path.

**Follow-up worth doing (not in this session):** store `sample_bins` in column-major (Fortran) order so column slices are contiguous by construction, and the `ascontiguousarray` is truly a no-op. That's the "gap #2" fix from above.

### 3. Pre-existing `monkey-patch-ban` guard violations (unblocked make check)

Three `parallel_module._WORKER_FEATURE_BINS = FeatureBins(...)` direct assignments in `tests/test_parallel.py` (authored 2025-12-24). Replaced with `_set_worker_feature_bins_for_test` / `_reset_worker_feature_bins_for_test` DI helpers on `parallel.py`.

## What surprised me

1. **The panel-split leakage was worse than I'd guessed.** Phase 1's AUC = 0.84 → this benchmark's AUC = 0.68. Same model, same code, same dataset — the only change was making the split respect company boundaries. That's a 16 AUC-point spread from a leakage pattern many practitioners wouldn't audit for. Anyone benchmarking against this dataset needs a company-disjoint split or they're benchmarking their own memorization.
2. **cleargbm_hook and cleargbm_native produced identical metrics to 4 decimals.** I expected FP-order differences to be visible at metric level. They weren't — Pearson correlation 0.9999 between hook and native predictions, which sanity-checks that the FFI-routed and native-loop implementations compute the same math. Good design.
3. **`use_rust_backend()` was totally broken in real training** and yet 555 tests + 100% branch coverage passed. The contiguity issue only surfaces at the boundary between real 2D `sample_bins` layouts and the Rust binding. Tests constructed contiguous 1D histograms in isolation and never hit it. That's a coverage-line vs coverage-behavior gap worth noting for the roadmap.
4. **ClearGBM slightly wins on AUC-PR.** At a 6.6% positive rate, AUC-PR is the metric that matters. ClearGBM's 0.142 vs LightGBM's 0.138 — within noise, but consistent across seeds. Not a story worth telling on a marketing page, but interesting.
5. **The wrapper leaves 8.5× on the table.** covenant_ml's `ClearGBMBackend` will happily use Rust for every histogram/loss primitive but keep the boosting loop in Python. Switching to `train_gradient_boosting_native` is a single-line change with an immediate 8.5× throughput win, and it produces the same predictions (Pearson 0.9999).

## Recommendations

**For ClearGBM as-is on this dataset:**

- **Don't sell speed.** ClearGBM at its best is 9× slower than LightGBM. That is not going to close without SIMD + uint8 bins + leaf-wise + column-major — a real engineering push, not a tweak. Sell interpretability + typing + testability.
- **Do sell quality.** On this dataset ClearGBM ties LightGBM. Any concern that "LightGBM will always beat a from-scratch impl" is empirically unsupported here.
- **Fix `ClearGBMBackend`.** Make it use `train_gradient_boosting_native` when the Rust backend has been activated. Free 8.5×.

**For the ClearGBM library:**

- **Ship the column-major layout of `sample_bins`.** Removes the `np.ascontiguousarray` cost in the Rust path.
- **Consider uint8 histogram bins.** LightGBM's `max_bin ≤ 255` is a well-established default; ClearGBM's int64 costs 8× cache-line density.
- **Add a real-input Rust adapter test.** The contiguity bug slipped through 100% coverage because the tests fabricated contiguous inputs. A test that trains a whole tree through `use_rust_backend()` on realistic data would have caught it.
- **Update the README benchmark table** to reflect the empirical picture. Draft in `docs/BENCHMARK_RESULTS_2026-07-20.md`; a proposed README diff follows.

**For this dataset (`american_bankruptcy`):**

- **Anyone benchmarking on it must use a company-disjoint split.** Panel splits leak identity through the company column. AUC drops from 0.84 to 0.68 when the leak closes; that's a 16-point spread on the same code.
- **Neither GBM is calibrated at 0.68 slope.** Any downstream user should Platt-scale or isotonic-fit on val before applying a threshold.

## Reproducibility

- **Full log:** `docs/BENCHMARK_MANIFEST_2026-07-20.json` — every run's metrics, seeds, dataset SHA-256 (`cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`), environment, config.
- **Script (throwaway, not committed):** `<session-scratchpad>/benchmark_vs_lightgbm.py` — 400 lines, self-contained.
- **To re-run:** activate covenant_ml's poetry venv (`cd libs/covenant_ml && poetry install`), install `cleargbm_rs` wheel from `libs/cleargbm_rs/target/wheels/`, then `poetry run python <script>`.
- **Total wall time:** ~4 minutes (LightGBM ×3 + cleargbm_hook ×3 + cleargbm_native ×3 + baselines).

## Files touched this session (across Phases 1 and 2)

Modified:
- `libs/cleargbm/src/cleargbm/histogram.py` — `reg_lambda` on split gain
- `libs/cleargbm/src/cleargbm/parallel.py` — thread `config["reg_lambda"]`; DI helpers `_set_worker_feature_bins_for_test`
- `libs/cleargbm/src/cleargbm/_rust_adapters.py` — `np.ascontiguousarray` on histogram input
- `libs/cleargbm/tests/test_histogram.py` — 6 new regression tests
- `libs/cleargbm/tests/test_parallel.py` — use DI helpers instead of direct global mutation

Created:
- `libs/cleargbm/docs/VALIDATION_REPORT_2026-07-20.md`
- `libs/cleargbm/docs/BENCHMARK_RESULTS_2026-07-20.md` (this file)
- `libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-20.json`
- `wiki/pages/cleargbm-histogram-split-path.md` (linked from `wiki/hubs/libs.md`; `wiki/index.md` + `wiki/log.md` updated)

`make check` on `libs/cleargbm/`: 555 tests pass, 100.00% statement + branch coverage, all guards clean.
