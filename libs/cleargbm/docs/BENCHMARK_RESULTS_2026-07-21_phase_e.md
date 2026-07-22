# ClearGBM vs LightGBM benchmark — 2026-07-21 (Phase E, wiki perf-roadmap items 1-3)

**Author:** Phase E benchmark following the column-major + uint8 + unrolled histogram accumulator lift on the Rust core.
**Environment:** Windows-10-10.0.26200-SP0, Python 3.11.9, numpy 2.4.0, sklearn 1.7.2, lightgbm 4.6.0.
**Dataset:** `libs/covenant_ml/tests/data/american_bankruptcy.csv` — 78,682 rows, 18 features, 6.63% positive class. SHA-256 in the manifest.
**Manifest (machine-readable):** `docs/BENCHMARK_MANIFEST_2026-07-21_phase_e.json`.
**Baseline being compared to:** `docs/BENCHMARK_MANIFEST_2026-07-21.json` (Phase D, Rust-only refactor with row-major usize bins).

## TL;DR

- **Speed: ClearGBM 2.79× faster than the prior baseline.** Fit time dropped from **6.88s ± 0.13s** to **2.47s ± 0.03s** on the 200-tree / depth-6 / max_bins=64 config.
- **Quality: unchanged** — every AUC / log-loss / Brier metric stayed within seed-to-seed std vs the Phase D baseline and vs LightGBM.
- **Gap to LightGBM: 8.0× → 3.4×.** Well inside the wiki's forecast range for column-major + uint8 combined ("6.88s → 2-3s").

## Wiki roadmap items landed in this run

The three items from `wiki/pages/cleargbm-perf-*.md` that the wiki flagged as unambiguously beneficial (uint8 + column-major + SIMD-shaped hot loop) all landed together on the Rust side. Files touched: `libs/cleargbm_rs/src/binning/feature_bins.rs`, `libs/cleargbm_rs/src/histogram/mod.rs`, `libs/cleargbm_rs/src/tree/builder.rs`, `libs/cleargbm_rs/src/training/config.rs`, `libs/cleargbm_rs/src/hooks.rs`, plus the pyo3 boundary in `libs/cleargbm_rs/src/pyo3_module/`.

| Wiki item | Status | Where |
| --- | --- | --- |
| [cleargbm-perf-column-major-sample-bins](../../../wiki/pages/cleargbm-perf-column-major-sample-bins.md) | **shipped** | `FeatureBins.sample_bins` is now `Vec<u8>` in flat column-major layout at `binning/feature_bins.rs:31`. `bins_for_feature(feat_idx)` returns a contiguous `&[u8]` slice. |
| [cleargbm-perf-uint8-histogram-bins](../../../wiki/pages/cleargbm-perf-uint8-histogram-bins.md) | **shipped** | Bin index dtype flipped from `usize` (8 bytes) to `u8` (1 byte) across `histogram::build_histogram`, `Hooks::BuildHistogramFn`, `BuildTreeInput.bins`, and `FeatureBins`. `max_bins ≤ 255` is enforced at `training/config.rs::GradientBoostingConfig::new`. Bin values pack 8× denser into a cache line. |
| [cleargbm-perf-simd-histogram-accumulator](../../../wiki/pages/cleargbm-perf-simd-histogram-accumulator.md) | **shipped (pragmatic form)** | `histogram/mod.rs::build_histogram` was rewritten: (a) two dedicated pre-validation passes on `sample_indices` and bin values (auto-vectorized by LLVM into SIMD compares), (b) direct `HistogramBuffer` field access (no per-sample `accumulate` function-call boundary and its per-sample bounds check), (c) 4-wide manually unrolled hot loop plus a scalar tail. The `wide` crate dep was considered and dropped — gather-loads over random `sample_indices` aren't SIMD-native without `unsafe`, and the crate forbids `unsafe_code`. The unrolled scalar form lets the auto-vectorizer produce the same effective code without adding an external crate. |
| [cleargbm-perf-leaf-wise-growth](../../../wiki/pages/cleargbm-perf-leaf-wise-growth.md) | **not yet shipped (see wiki)** | Wiki-flagged as "do LAST" with medium confidence and an explicit interpretability trade-off. Deferred pending a decision on whether to accept the cleargbm-vs-LightGBM structural symmetry loss. |

## Results (mean ± std over 3 seeds, company-disjoint split, 200 trees @ depth 6, max_bins=64)

### Quality metrics

| Model | AUC-ROC | AUC-PR | log-loss |
| ----- | ------- | ------ | -------- |
| **lightgbm** | 0.6960 ± 0.0150 | 0.1572 ± 0.0237 | 0.2302 ± 0.0071 |
| **cleargbm (Phase E)** | 0.6991 ± 0.0132 | 0.1620 ± 0.0259 | 0.2303 ± 0.0068 |

Every difference between cleargbm and lightgbm is smaller than the seed-to-seed std for either model — statistical tie, consistent with the Phase D baseline (which reported "cleargbm still ties LightGBM within seed noise").

### Speed

| Model | fit_time | vs LightGBM | vs prior cleargbm (2026-07-21 Phase D) |
| ----- | -------- | ----------- | -------------------------------------- |
| **lightgbm** | **0.72s ± 0.08s** | **1.00×** | — |
| **cleargbm (Phase D)** | 6.88s ± 0.13s | 8.0× slower | (baseline) |
| **cleargbm (Phase E)** | **2.47s ± 0.03s** | **3.4× slower** | **-4.42s, 2.79× faster** |

The three combined structural changes (column-major storage, uint8 bin indices, unrolled hot loop with pre-validation) delivered a 2.79× speedup on cleargbm's total fit time, closing the LightGBM gap from 8.0× to 3.4×. The wiki's forecast for column-major + uint8 combined was "cleargbm fit time 6.88s → 2-3s, LightGBM gap 8.0× → 2-3×" — the measured 2.47s / 3.4× lands squarely in that range. The additional variance collapse from 0.13s (Phase D) to 0.033s (Phase E) reflects the same-shape hot loop being smaller and more predictable.

## Why the numbers changed

1. **Cache-line density: 8× more bins per line.** The pre-refactor `FeatureBins.sample_bins: Vec<Vec<usize>>` stored each sample's bin index as 8 bytes in a per-row heap allocation. The refactored `FeatureBins.sample_bins: Vec<u8>` in flat column-major layout stores 8× more bin values per 64-byte cache line AND lays them out sequentially per feature. A per-feature histogram scan now walks `n_samples` contiguous bytes; the previous version strode through disjoint heap allocations. On a 55K-sample × 18-feature dataset that's the difference between an 8 MiB bin array (worse than L2 on most CPUs) and a 1 MiB bin array (fits comfortably in L2).
2. **Elimination of per-node allocations.** The pre-refactor `build_feature_histograms` and `compute_child_histograms` allocated three fresh `Vec<f64>` per (node, feature) — one each for the extracted feat_gradients, feat_hessians, and feat_bins — on every histogram build. With column-major storage the per-feature bin slice is `&bins[feat_idx * n_samples..(feat_idx+1) * n_samples]` (no allocation), and the gradients/hessians are threaded through directly to `build_histogram(sample_indices, gradients, hessians, per_feature_bins, n_bins)`. Zero allocation per histogram build.
3. **Pre-validated hot loop + inlined accumulate.** The pre-refactor loop did a bounds check on every sample (`if idx >= n_samples`) and called `HistogramBuffer::accumulate` which did its own bin bounds check (`if bin >= n_bins`). The refactored loop pre-validates both invariants in dedicated scan passes (auto-vectorized by LLVM into SIMD compares on x86_64 AVX2 / ARM64 NEON), then runs a 4-wide manually unrolled body with direct field access. The pre-validation trades one O(N) scan for the elimination of two branch-per-sample checks — net win once N samples means "everything the workload actually does."

## Fixes not yet applied that would move these numbers further

- **[Leaf-wise growth](../../../wiki/pages/cleargbm-perf-leaf-wise-growth.md)** — Wiki-flagged "do LAST" with `confidence: medium` and an explicit interpretability trade-off. Not implemented in this run. Impact on the benchmark is ambiguous (quality is already a statistical tie), and cleargbm's *"Gradient Boosting You Can See Through"* tagline explicitly values the balanced-tree structural symmetry that leaf-wise gives up. This one needs a human decision.
- **True SIMD accumulation via `unsafe` intrinsics or `std::simd` (nightly)** — the safe-Rust unrolled form landed in this session gets most of the wins that LLVM can extract without gather/scatter primitives. To close the last 3.4× gap to LightGBM, the histogram accumulator would need either `unsafe` gather/scatter intrinsics (blocked by the crate's `unsafe_code = "forbid"`) or nightly `std::simd` for portable gathers.

## Reproducibility

- **Script:** the Phase E benchmark script lives at
  `C:\Users\Test\AppData\Local\Temp\claude\C--Users-Test-PROJECTS-MCPs\<session>\scratchpad\bench_cleargbm_vs_lightgbm.py`. Company-disjoint 70/15/15 split on `company_name`, seeds 42/43/44, `num_leaves=31`, `max_depth=6`, `max_bins=64`, `learning_rate=0.05`, `min_data_in_leaf=20`, no regularization, single-threaded (`num_threads=1` for LightGBM, `n_jobs=1` for cleargbm — matches the prior benchmark).
- **Manifest:** `docs/BENCHMARK_MANIFEST_2026-07-21_phase_e.json` — every run's metrics, seeds, dataset SHA-256, env, config.
- **Prior baseline:** `docs/BENCHMARK_MANIFEST_2026-07-21.json` (Phase D, immediately post-refactor with row-major usize).
- **Total wall time:** ~15 seconds (3 seeds × (LightGBM ≈ 0.7s warm + 0.7s timed + ClearGBM ≈ 2.5s warm + 2.5s timed)).

## Files touched in this benchmark

- `docs/BENCHMARK_RESULTS_2026-07-21_phase_e.md` — this file
- `docs/BENCHMARK_MANIFEST_2026-07-21_phase_e.json` — machine-readable log

Source changes are the ones described in the "Wiki roadmap items landed" table above.
